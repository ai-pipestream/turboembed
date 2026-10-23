//! `turbo-bench qa`, the command line of `turbo_bench::qa`.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use turbo::{TURBO_PRECISION_EXACT, TURBO_PRECISION_FASTEST, TURBO_PRECISION_MODEL};
use turbo_bench::{Result, git};

use crate::cli::{Opts, USAGE};

/// `turbo-bench qa`: extractive QA questions per second by context length
/// on one device, written as a QA record.
pub(crate) fn run(args: &[String]) -> Result<()> {
    let mut o = Opts::parse(args)?;
    let bundle = o.take("--bundle").ok_or("--bundle is required")?;
    let questions = o.take("--questions").ok_or("--questions is required")?;
    let precision = match o.take("--precision").as_deref().unwrap_or("exact") {
        "model" => TURBO_PRECISION_MODEL,
        "fastest" => TURBO_PRECISION_FASTEST,
        "exact" => TURBO_PRECISION_EXACT,
        p => return Err(format!("--precision {p}: model, fastest or exact")),
    };
    let plan = turbo_bench::qa::QaPlan {
        bundle: PathBuf::from(bundle),
        device: o.take("--device").unwrap_or_else(|| "cpu".into()),
        precision,
        batch: o.number("--batch")?.unwrap_or(8),
        max_batch: o.number("--max-batch")?.unwrap_or(0),
        questions: PathBuf::from(questions),
        per_bin: o.number("--per-bin")?.unwrap_or(256) as usize,
        warmup: o.number("--warmup")?.unwrap_or(2),
        iterations: o.number("--iterations")?.unwrap_or(10),
    };
    let repo = o.take("--repo").map_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join(".."), PathBuf::from);
    let out = o.take("--out");
    if let Some(unknown) = o.0.keys().next() {
        return Err(format!("{unknown}: not a qa option\n{USAGE}"));
    }
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    turbo_bench::check_not_testdata(&plan.bundle, &[workspace.join("testdata"), repo.join("testdata")])?;
    let before = git::provenance(&repo)?;
    turbo_bench::check_build(&before.commit, turbo_bench::BUILD_COMMIT, turbo_bench::BUILD_CHANGES)?;
    let r = turbo_bench::qa::measure(&plan, &before, turbo_bench::utc(SystemTime::now()))?;
    let after = git::provenance(&repo)?;
    if after != before {
        return Err(format!("the working tree moved from {} to {} during the run", before.commit, after.commit));
    }
    let dir = out.map_or_else(|| Path::new(&after.top).join(turbo_bench::qa::QA_DIR), PathBuf::from);
    let path = turbo_bench::qa::write(&r, &dir)?;
    for b in &r.bins {
        eprintln!(
            "{} {}: contexts of {}..{} tokens: {} questions, {} windows, p50 {:.2} ms a pass, {:.1} questions/s",
            r.device.backend,
            r.device.name,
            b.min_tokens,
            if b.max_tokens == 0 { "".into() } else { b.max_tokens.to_string() },
            b.questions,
            b.windows,
            b.p50_ms,
            b.questions_per_second
        );
    }
    println!("{}", path.display());
    Ok(())
}
