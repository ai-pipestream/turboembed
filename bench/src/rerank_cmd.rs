//! `turbo-bench rerank`, the command line of `turbo_bench::rerank`.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use turbo::{TURBO_PRECISION_EXACT, TURBO_PRECISION_FASTEST, TURBO_PRECISION_MODEL};
use turbo_bench::{Result, git};

use crate::cli::{Opts, USAGE};

/// `turbo-bench rerank`: cross-encoder pairs per second on one device,
/// with TEI's /rerank beside it, written as a rerank record.
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
    let tei = match (o.take("--tei-image"), o.take("--tei-model"), o.flag("--no-tei")) {
        (Some(image), Some(model), false) => Some((image, PathBuf::from(model))),
        (None, None, true) => None,
        _ => return Err("give --tei-image and --tei-model, or --no-tei".into()),
    };
    let plan = turbo_bench::rerank::RerankPlan {
        bundle: PathBuf::from(bundle),
        device: o.take("--device").unwrap_or_else(|| "cpu".into()),
        precision,
        questions: PathBuf::from(questions),
        requests: o.number("--requests")?.unwrap_or(64) as usize,
        passages: o.number("--passages")?.unwrap_or(16) as usize,
        warmup: o.number("--warmup")?.unwrap_or(2),
        iterations: o.number("--iterations")?.unwrap_or(10),
        tei,
    };
    let repo = o.take("--repo").map_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join(".."), PathBuf::from);
    let out = o.take("--out");
    if let Some(unknown) = o.0.keys().next() {
        return Err(format!("{unknown}: not a rerank option\n{USAGE}"));
    }
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    turbo_bench::check_not_testdata(&plan.bundle, &[workspace.join("testdata"), repo.join("testdata")])?;
    let before = git::provenance(&repo)?;
    turbo_bench::check_build(&before.commit, turbo_bench::BUILD_COMMIT, turbo_bench::BUILD_CHANGES)?;
    let r = turbo_bench::rerank::measure(&plan, &before, turbo_bench::utc(SystemTime::now()))?;
    let after = git::provenance(&repo)?;
    if after != before {
        return Err(format!("the working tree moved from {} to {} during the run", before.commit, after.commit));
    }
    let dir = out.map_or_else(|| Path::new(&after.top).join(turbo_bench::rerank::RERANK_DIR), PathBuf::from);
    let path = turbo_bench::rerank::write(&r, &dir)?;
    eprintln!(
        "{} {}: {} requests of {} passages: p50 {:.1} ms a pass, {:.1} pairs/s",
        r.device.backend, r.device.name, r.requests, r.passages, r.turbo.p50_ms, r.turbo.pairs_per_second
    );
    if let Some(t) = &r.tei {
        eprintln!(
            "TEI: p50 {:.1} ms a pass, {:.1} pairs/s; the library {:.2}x; largest score difference {:e}",
            t.side.p50_ms, t.side.pairs_per_second, t.turbo_speedup, t.max_abs_score_diff
        );
    }
    println!("{}", path.display());
    Ok(())
}
