//! Comparing a tree of goldens the generator wrote with the Rust side:
//! every `<annotator>/<run>/<set>/<id>.golden` under a root is read, the
//! run opened once, its input text annotated, the result compared and
//! the offset invariants of every layer the Rust side wrote checked.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use turbo_goldens::invariants::{Violation, check_document};
use turbo_goldens::{Golden, GoldenError, segmentation_shapes};

use crate::resources::{Resources, Unavailable};
use crate::runs::{self, Outcome, Run};

/// What one golden gave.
#[derive(Clone, Debug, PartialEq)]
pub struct Finding {
    pub annotator: String,
    pub run: String,
    pub set: String,
    pub id: String,
    pub outcome: Outcome,
    /// Offset invariants the Rust layers broke on this text.
    pub violations: Vec<Violation>,
}

impl Finding {
    pub fn run_name(&self) -> String {
        format!("{}/{}", self.annotator, self.run)
    }

    pub fn input_name(&self) -> String {
        format!("{}/{}", self.set, self.id)
    }

    /// Whether this finding needs looking into: a divergence, or a broken
    /// invariant.
    pub fn is_problem(&self) -> bool {
        self.outcome.is_divergence() || !self.violations.is_empty()
    }
}

/// Everything a walk found.
#[derive(Debug, Default)]
pub struct Report {
    pub findings: Vec<Finding>,
    /// Runs the tree has that could not be opened, with why.
    pub unavailable: BTreeMap<String, Unavailable>,
}

impl Report {
    pub fn problems(&self) -> impl Iterator<Item = &Finding> {
        self.findings.iter().filter(|f| f.is_problem())
    }

    /// The findings tallied per run and outcome kind, then every problem
    /// in full, then the documented differences one line each.
    pub fn render(&self) -> String {
        let mut out = String::new();
        let mut per_run: BTreeMap<String, [usize; 5]> = BTreeMap::new();
        for f in &self.findings {
            let t = per_run.entry(f.run_name()).or_default();
            match &f.outcome {
                Outcome::Match => t[0] += 1,
                Outcome::Documented(_) => t[1] += 1,
                Outcome::Divergence(_) => t[2] += 1,
                Outcome::Skipped(_) => t[3] += 1,
            }
            if !f.violations.is_empty() {
                t[4] += 1;
            }
        }
        let _ =
            writeln!(out, "{:<64} {:>6} {:>6} {:>6} {:>6} {:>6}", "run", "match", "docum", "diverg", "skip", "invar");
        for (run, t) in &per_run {
            let _ = writeln!(out, "{run:<64} {:>6} {:>6} {:>6} {:>6} {:>6}", t[0], t[1], t[2], t[3], t[4]);
        }
        for (run, why) in &self.unavailable {
            let _ = writeln!(out, "{run}: not compared: {why}");
        }
        let problems: Vec<&Finding> = self.problems().collect();
        let _ = writeln!(out, "\n{} goldens compared, {} problems", self.findings.len(), problems.len());
        for f in problems {
            let _ = writeln!(out, "\n{} on {}: {}", f.run_name(), f.input_name(), f.outcome);
            for v in &f.violations {
                let _ = writeln!(out, "  invariant: {v}");
            }
        }
        let documented: Vec<&Finding> =
            self.findings.iter().filter(|f| matches!(f.outcome, Outcome::Documented(_))).collect();
        if !documented.is_empty() {
            let _ = writeln!(out, "\n{} documented differences", documented.len());
            for f in documented {
                let _ = writeln!(out, "{} on {}: {}", f.run_name(), f.input_name(), f.outcome);
            }
        }
        out
    }
}

/// Reads a golden with every shape the generator writes. A lemma golden
/// whose lemma is half a surrogate pair does not read; that is reported
/// as a skipped finding, not an error.
pub fn read_golden(path: &Path) -> Result<Golden, GoldenError> {
    Golden::read_with(path, &segmentation_shapes())
}

/// Opens each run once, by name.
pub struct RunCache<'r> {
    res: &'r Resources,
    open: BTreeMap<(String, String), Result<Box<dyn Run>, Unavailable>>,
}

impl<'r> RunCache<'r> {
    pub fn new(res: &'r Resources) -> RunCache<'r> {
        RunCache { res, open: BTreeMap::new() }
    }

    pub fn get(&mut self, annotator: &str, run: &str) -> &Result<Box<dyn Run>, Unavailable> {
        let key = (annotator.to_string(), run.to_string());
        if !self.open.contains_key(&key) {
            let opened = runs::open(annotator, run, self.res);
            self.open.insert(key.clone(), opened);
        }
        &self.open[&key]
    }
}

/// Checks one golden file against the Rust side: the input text is read
/// from `inputs/<set>/<id>.txt`.
pub fn check_golden(path: &Path, inputs: &Path, cache: &mut RunCache<'_>) -> Result<Finding, String> {
    let golden = match read_golden(path) {
        Ok(g) => g,
        Err(e) => {
            let (annotator, run, set, id) = parts(path)?;
            let outcome = Outcome::Skipped(format!("the golden does not read: {e}"));
            return Ok(Finding { annotator, run, set, id, outcome, violations: Vec::new() });
        }
    };
    let (annotator, run, _, _) = parts(path)?;
    let input = &golden.header.input;
    let (set, id) = (input.set.clone(), input.id.clone());
    let text_path = inputs.join(&set).join(format!("{id}.txt"));
    let text = std::fs::read_to_string(&text_path).map_err(|e| format!("{}: {e}", text_path.display()))?;
    let (outcome, violations) = match cache.get(&annotator, &run) {
        Ok(r) => {
            let checked = r.check(&text, &golden);
            let violations = checked.doc.as_ref().map_or_else(Vec::new, |d| check_document(d).1);
            (checked.outcome, violations)
        }
        Err(e) => (Outcome::Skipped(format!("the run is not available: {e}")), Vec::new()),
    };
    Ok(Finding { annotator, run, set, id, outcome, violations })
}

/// `<annotator>/<run>/<set>/<id>.golden` from the path's last four parts.
fn parts(path: &Path) -> Result<(String, String, String, String), String> {
    let names: Vec<String> = path
        .iter()
        .rev()
        .take(4)
        .map(|p| p.to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let [annotator, run, set, file] = names.as_slice() else {
        return Err(format!("{}: not <annotator>/<run>/<set>/<id>.golden", path.display()));
    };
    let id = file.strip_suffix(".golden").ok_or_else(|| format!("{}: not a .golden file", path.display()))?;
    Ok((annotator.clone(), run.clone(), set.clone(), id.to_string()))
}

/// Every `.golden` file under `root`, sorted.
pub fn golden_files(root: &Path) -> Result<Vec<PathBuf>, String> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))? {
            let entry = entry.map_err(|e| format!("{}: {e}", dir.display()))?;
            let p = entry.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "golden") {
                out.push(p);
            }
        }
    }
    out.sort();
    Ok(out)
}

/// Compares every golden under `root` whose annotator the differential
/// covers, and whose `(annotator, run)` the `wanted` filter accepts, with
/// the Rust side, reading the inputs from `inputs`. `progress` is told
/// every file, so a long run can show where it is.
pub fn compare_tree(
    root: &Path,
    inputs: &Path,
    res: &Resources,
    wanted: impl Fn(&str, &str) -> bool,
    mut progress: impl FnMut(usize, usize, &Path),
) -> Result<Report, String> {
    let files: Vec<PathBuf> = golden_files(root)?
        .into_iter()
        .filter(|p| {
            let (annotator, run, ..) = parts(p).unwrap_or_default();
            !runs::NOT_COVERED.contains(&annotator.as_str()) && wanted(&annotator, &run)
        })
        .collect();
    let mut cache = RunCache::new(res);
    let mut report = Report::default();
    let total = files.len();
    for (i, path) in files.iter().enumerate() {
        progress(i, total, path);
        let finding = check_golden(path, inputs, &mut cache)?;
        report.findings.push(finding);
    }
    for ((annotator, run), opened) in &cache.open {
        if let Err(e) = opened {
            report.unavailable.insert(format!("{annotator}/{run}"), e.clone());
        }
    }
    Ok(report)
}
