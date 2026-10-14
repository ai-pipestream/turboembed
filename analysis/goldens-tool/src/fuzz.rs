//! `turbo-goldens fuzz`: a seeded corpus through the JVM generator and the
//! Rust annotators, compared, and every divergence reduced to a minimal
//! input.
//!
//! The corpus comes from `turbo_goldens::fuzz`; each text is written into
//! several input sets, so every run the generator has for those sets
//! takes part (`Runs.java` decides which runs read which set). The
//! container runs with `--skip-annotators` for the annotators the
//! differential does not cover. `turbo_fuzz::compare_tree` compares what
//! it wrote with the Rust side and checks the offset invariants; each
//! problem is then reduced (`turbo_fuzz::reduce`) with the container as
//! the oracle, one run per round, and the minimal text written as a
//! candidate input beside a report naming the run it fails.

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use turbo_fuzz::differential::{RunCache, check_golden};
use turbo_fuzz::{Finding, Outcome, Report, Resources, reduce};
use turbo_goldens::fuzz::{corpus, parse_input_id};

use crate::{docker, models, paths, recipe};

/// The input sets a generated text is written into: between them, every
/// annotator the differential covers runs on each text. `en` brings the
/// language detector, the taggers, the lemmatizers, the chunker and the
/// English sentence and token models; `edge` every tokenizer, segmenter,
/// sentence detector and the WordPiece encoder; the others their own
/// annotators.
pub const DEFAULT_SETS: [&str; 7] = ["en", "edge", "norm", "num", "money", "quantity", "sentence-fixture"];

/// The annotators the container is told to skip: the differential does
/// not cover them (`turbo_fuzz::runs::NOT_COVERED`).
pub fn skipped_annotators() -> String {
    turbo_fuzz::runs::NOT_COVERED.join(",")
}

/// The options of `fuzz`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Options {
    pub seed: u64,
    pub count: u64,
    /// Where the inputs, the JVM's goldens, the report and the candidates
    /// go; `target/turbo-goldens/fuzz/s<seed>-<count>` by default.
    pub out: Option<PathBuf>,
    pub models: PathBuf,
    pub image: String,
    pub sets: Vec<String>,
    /// How many problems to reduce; 0 reduces none.
    pub reduce: usize,
    /// The most container runs one reduction may take.
    pub rounds: usize,
    /// Compare and reduce what an earlier run left under `out` without
    /// writing the inputs or running the generator again.
    pub reuse: bool,
    /// Reduce the documented differences too, to minimal examples of each.
    pub reduce_documented: bool,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            seed: 1,
            count: 200,
            out: None,
            models: paths::scratch_dir().join("models"),
            image: recipe::IMAGE_TAG.to_string(),
            sets: DEFAULT_SETS.iter().map(|s| s.to_string()).collect(),
            reduce: 10,
            rounds: 40,
            reuse: false,
            reduce_documented: false,
        }
    }
}

impl Options {
    pub fn parse(args: &[String]) -> Result<Options, String> {
        let mut opts = Options::default();
        let mut it = args.iter();
        while let Some(flag) = it.next() {
            let mut value = || it.next().ok_or_else(|| format!("{flag} needs a value"));
            let number = |v: &str| v.parse::<u64>().map_err(|_| format!("{flag}: {v} is not a number"));
            match flag.as_str() {
                "--seed" => opts.seed = number(value()?)?,
                "--count" => opts.count = number(value()?)?,
                "--out" => opts.out = Some(PathBuf::from(value()?)),
                "--models" => opts.models = PathBuf::from(value()?),
                "--image" => opts.image = value()?.clone(),
                "--sets" => {
                    let sets: Vec<String> = value()?.split(',').filter(|s| !s.is_empty()).map(str::to_string).collect();
                    for s in &sets {
                        if !recipe::INPUT_SETS.contains(&s.as_str()) {
                            return Err(format!("--sets: {s} is not an input set of the recipe"));
                        }
                    }
                    if sets.is_empty() {
                        return Err("--sets needs at least one set".into());
                    }
                    opts.sets = sets;
                }
                "--reduce" => opts.reduce = number(value()?)? as usize,
                "--no-reduce" => opts.reduce = 0,
                "--rounds" => opts.rounds = number(value()?)? as usize,
                "--reuse" => opts.reuse = true,
                "--reduce-documented" => opts.reduce_documented = true,
                _ => return Err(format!("unknown argument {flag}")),
            }
        }
        if opts.count == 0 {
            return Err("--count must be at least 1".into());
        }
        Ok(opts)
    }

    /// The output directory.
    pub fn out_dir(&self) -> PathBuf {
        self.out
            .clone()
            .unwrap_or_else(|| paths::scratch_dir().join("fuzz").join(format!("s{}-{}", self.seed, self.count)))
    }
}

/// What a run of `fuzz` produced.
#[derive(Debug)]
pub struct Summary {
    pub out: PathBuf,
    pub inputs: usize,
    pub goldens: usize,
    pub problems: usize,
    pub candidates: Vec<PathBuf>,
}

/// Writes the corpus into `inputs/<set>/<id>.txt` for every set.
pub fn write_inputs(inputs: &Path, seed: u64, count: u64, sets: &[String]) -> Result<usize, String> {
    let texts = corpus(seed, count);
    for set in sets {
        let dir = inputs.join(set);
        fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        for t in &texts {
            let path = dir.join(format!("{}.txt", t.id));
            fs::write(&path, &t.text).map_err(|e| format!("{}: {e}", path.display()))?;
        }
    }
    Ok(texts.len())
}

/// An empty directory at `path`, whatever was there.
fn fresh_dir(path: &Path) -> Result<(), String> {
    let _ = fs::remove_dir_all(path);
    fs::create_dir_all(path).map_err(|e| format!("{}: {e}", path.display()))
}

/// How many inputs an earlier run wrote: the texts of its first set
/// (every set holds the same texts).
fn count_inputs(inputs: &Path) -> Result<usize, String> {
    let read = |dir: &Path| fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()));
    let mut sets: Vec<PathBuf> =
        read(inputs)?.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.is_dir()).collect();
    sets.sort();
    let Some(set) = sets.first() else { return Err(format!("--reuse: {} holds no input set", inputs.display())) };
    Ok(read(set)?.filter_map(|e| e.ok()).filter(|e| e.path().extension().is_some_and(|x| x == "txt")).count())
}

/// Runs the whole command.
pub fn run(opts: &Options) -> Result<Summary, String> {
    let out = opts.out_dir();
    let inputs = out.join("inputs");
    let jvm = out.join("jvm");
    let n = if opts.reuse {
        if !jvm.is_dir() || !inputs.is_dir() {
            return Err(format!("--reuse: {} holds no inputs and jvm directories of an earlier run", out.display()));
        }
        count_inputs(&inputs)?
    } else {
        fresh_dir(&out)?;
        let n = write_inputs(&inputs, opts.seed, opts.count, &opts.sets)?;
        eprintln!(
            "fuzz: {n} inputs of seed {} written into {} sets under {}",
            opts.seed,
            opts.sets.len(),
            inputs.display()
        );
        models::fetch(&opts.models)?;
        models::check_fixtures(&paths::fixtures_dir())?;
        fresh_dir(&jvm)?;
        let skipped = skipped_annotators();
        eprintln!("fuzz: running the generator in {} (skipping {skipped})", opts.image);
        let started = std::time::Instant::now();
        docker::generate_with(
            &opts.image,
            &["--skip-annotators", &skipped],
            &inputs,
            &paths::rates_dir(),
            &opts.models,
            &paths::fixtures_dir(),
            &jvm,
        )?;
        eprintln!("fuzz: the generator took {:.0} s", started.elapsed().as_secs_f64());
        n
    };
    let res = Resources::new(&opts.models);
    let mut shown = 0;
    let report = turbo_fuzz::compare_tree(
        &jvm,
        &inputs,
        &res,
        |_, _| true,
        |i, total, _| {
            if i >= shown {
                eprintln!("fuzz: comparing {i}/{total}");
                shown += 2000;
            }
        },
    )?;
    let rendered = report.render();
    let report_path = out.join("report.txt");
    fs::write(&report_path, &rendered).map_err(|e| format!("{}: {e}", report_path.display()))?;
    print!("{rendered}");
    let problems: Vec<&Finding> = report.problems().collect();
    let mut candidates = Vec::new();
    if opts.reduce > 0 {
        candidates = reduce_problems(opts, &out, &inputs, &res, &report)?;
    }
    println!("fuzz: report in {}", report_path.display());
    Ok(Summary { out, inputs: n, goldens: report.findings.len(), problems: problems.len(), candidates })
}

/// Reduces the first `opts.reduce` problems, one input per run at most,
/// and writes each minimal text as `candidates/<id>-<annotator>-<run>.txt`
/// with a `.report` beside it.
fn reduce_problems(
    opts: &Options,
    out: &Path,
    inputs: &Path,
    res: &Resources,
    report: &Report,
) -> Result<Vec<PathBuf>, String> {
    let dir = out.join("candidates");
    fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut seen_runs = Vec::new();
    let mut written = Vec::new();
    let wanted =
        |f: &&Finding| f.is_problem() || (opts.reduce_documented && matches!(f.outcome, Outcome::Documented(_)));
    for f in report.findings.iter().filter(wanted) {
        if written.len() >= opts.reduce {
            break;
        }
        let run_name = f.run_name();
        if seen_runs.contains(&run_name) {
            continue;
        }
        seen_runs.push(run_name.clone());
        let original_path = inputs.join(&f.set).join(format!("{}.txt", f.id));
        let original = fs::read_to_string(&original_path).map_err(|e| format!("{}: {e}", original_path.display()))?;
        eprintln!("fuzz: reducing {} on {} ({} bytes)", run_name, f.input_name(), original.len());
        let documented = matches!(f.outcome, Outcome::Documented(_));
        let mut oracle = ContainerOracle { opts, res, scratch: out.join("reduce"), finding: f, documented, round: 0 };
        let reduced = reduce(&original, &mut oracle, opts.rounds)?;
        let stem = format!("{}-{}-{}", f.id, f.annotator, f.run.replace('+', "_"));
        let text_path = dir.join(format!("{stem}.txt"));
        fs::write(&text_path, &reduced.text).map_err(|e| format!("{}: {e}", text_path.display()))?;
        let mut note = String::new();
        let _ = writeln!(note, "run: {run_name}");
        let _ = writeln!(note, "input: {} (seed {}, {} bytes)", f.input_name(), opts.seed, original.len());
        let _ = writeln!(
            note,
            "reduced: {} bytes in {} container runs over {} candidates",
            reduced.text.len(),
            reduced.rounds,
            reduced.candidates
        );
        let _ = writeln!(note, "outcome on the full input: {}", f.outcome);
        if documented {
            let _ = writeln!(note, "reduced as a documented difference (--reduce-documented), not a divergence");
        }
        for v in &f.violations {
            let _ = writeln!(note, "invariant: {v}");
        }
        if let Some((seed, index)) = parse_input_id(&f.id) {
            let _ =
                writeln!(note, "reproduce: turbo-goldens fuzz --seed {seed} --count {} --sets {}", index + 1, f.set);
        }
        let note_path = dir.join(format!("{stem}.report"));
        fs::write(&note_path, note).map_err(|e| format!("{}: {e}", note_path.display()))?;
        eprintln!("fuzz: {} reduced to {} bytes: {}", f.input_name(), reduced.text.len(), text_path.display());
        written.push(text_path);
    }
    Ok(written)
}

/// The container as the reducer's oracle: each batch of candidate texts
/// is written as inputs of the finding's set, the generator runs
/// `--only-runs <annotator>/<run>` over them, and each candidate fails
/// when its golden compares as a problem.
struct ContainerOracle<'a> {
    opts: &'a Options,
    res: &'a Resources,
    scratch: PathBuf,
    finding: &'a Finding,
    /// Whether a documented difference counts as the failure looked for.
    documented: bool,
    round: usize,
}

impl turbo_fuzz::Oracle for ContainerOracle<'_> {
    fn fails(&mut self, texts: &[String]) -> Result<Vec<bool>, String> {
        self.round += 1;
        let dir = self.scratch.join(format!("round-{}", self.round));
        fresh_dir(&dir)?;
        let inputs = dir.join("inputs").join(&self.finding.set);
        fs::create_dir_all(&inputs).map_err(|e| format!("{}: {e}", inputs.display()))?;
        for (k, t) in texts.iter().enumerate() {
            let p = inputs.join(format!("c{k:04}.txt"));
            fs::write(&p, t).map_err(|e| format!("{}: {e}", p.display()))?;
        }
        let jvm = dir.join("jvm");
        fresh_dir(&jvm)?;
        let only = self.finding.run_name();
        eprintln!("fuzz: reduction round {} with {} candidates", self.round, texts.len());
        docker::generate_with(
            &self.opts.image,
            &["--only-runs", &only],
            &dir.join("inputs"),
            &paths::rates_dir(),
            &self.opts.models,
            &paths::fixtures_dir(),
            &jvm,
        )?;
        let mut cache = RunCache::new(self.res);
        let mut out = Vec::with_capacity(texts.len());
        for k in 0..texts.len() {
            let golden = jvm
                .join(&self.finding.annotator)
                .join(&self.finding.run)
                .join(&self.finding.set)
                .join(format!("c{k:04}.golden"));
            if !golden.is_file() {
                out.push(false);
                continue;
            }
            let found = check_golden(&golden, &dir.join("inputs"), &mut cache)?;
            out.push(found.is_problem() || (self.documented && matches!(found.outcome, Outcome::Documented(_))));
        }
        let _ = fs::remove_dir_all(&dir);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_parse_with_defaults_and_checks() {
        let o = Options::parse(&[]).unwrap();
        assert_eq!((o.seed, o.count, o.reduce), (1, 200, 10));
        assert_eq!(o.sets, DEFAULT_SETS);
        assert_eq!(o.image, recipe::IMAGE_TAG);
        let args: Vec<String> =
            ["--seed", "9", "--count", "3", "--sets", "edge,en", "--no-reduce", "--image", "x:y", "--out", "/o"]
                .iter()
                .map(|s| s.to_string())
                .collect();
        let o = Options::parse(&args).unwrap();
        assert_eq!((o.seed, o.count, o.reduce), (9, 3, 0));
        assert_eq!(o.sets, ["edge", "en"]);
        assert_eq!(o.image, "x:y");
        assert_eq!(o.out_dir(), PathBuf::from("/o"));
        assert_eq!(Options::parse(&[]).unwrap().out_dir(), paths::scratch_dir().join("fuzz/s1-200"));
        let bad = |a: &[&str]| Options::parse(&a.iter().map(|s| s.to_string()).collect::<Vec<_>>()).unwrap_err();
        assert!(bad(&["--seed"]).contains("needs a value"));
        assert!(bad(&["--seed", "x"]).contains("not a number"));
        assert!(bad(&["--count", "0"]).contains("at least 1"));
        assert!(bad(&["--sets", "nope"]).contains("not an input set"));
        assert!(bad(&["--what"]).contains("unknown argument"));
        assert!(Options::parse(&["--reuse".to_string()]).unwrap().reuse);
    }

    #[test]
    fn the_inputs_are_written_once_per_set_with_the_corpus_ids() {
        let dir = std::env::temp_dir().join(format!("turbo-goldens-fuzz-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let n = write_inputs(&dir, 5, 3, &["edge".to_string(), "en".to_string()]).unwrap();
        assert_eq!(n, 3);
        for set in ["edge", "en"] {
            let mut names: Vec<String> =
                fs::read_dir(dir.join(set)).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
            names.sort();
            assert_eq!(names, ["s5-00000.txt", "s5-00001.txt", "s5-00002.txt"]);
        }
        assert_eq!(fs::read_to_string(dir.join("en/s5-00002.txt")).unwrap(), corpus(5, 3)[2].text);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_skipped_annotators_are_recipe_annotators() {
        for a in skipped_annotators().split(',') {
            assert!(recipe::ANNOTATORS.contains(&a), "{a}");
        }
    }
}
