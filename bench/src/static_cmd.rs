//! `turbo-bench static`, the command line of `turbo_bench::statics`.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use turbo::{TURBO_PRECISION_FASTEST, TURBO_PRECISION_MODEL};
use turbo_bench::{Result, git};

use crate::cli::{Opts, USAGE};

/// `turbo-bench static`: a static bundle's texts per second on the CPU,
/// and the peers', written as a static record.
pub(crate) fn run(args: &[String]) -> Result<()> {
    let mut o = Opts::parse(args)?;
    let bundle = o.take("--bundle").ok_or("--bundle is required")?;
    let texts = o.take("--texts").ok_or("--texts is required")?;
    let precision = match o.take("--precision").as_deref().unwrap_or("model") {
        "model" => TURBO_PRECISION_MODEL,
        "fastest" => TURBO_PRECISION_FASTEST,
        p => return Err(format!("--precision {p}: model or fastest")),
    };
    let mut sets = Vec::new();
    for s in texts.split(',') {
        let (name, path) = s.split_once('=').ok_or_else(|| format!("--texts {s}: NAME=FILE"))?;
        sets.push((name.to_owned(), PathBuf::from(path)));
    }
    let batches = o
        .take("--batches")
        .unwrap_or_else(|| "1,32,256,1024".into())
        .split(',')
        .map(|b| b.trim().parse::<u32>().map_err(|e| format!("--batches {b}: {e}")))
        .collect::<Result<Vec<_>>>()?;
    let repo = o.take("--repo").map_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join(".."), PathBuf::from);
    let peers = match (o.take("--peers"), o.flag("--no-peers")) {
        (Some(w), false) => Some((repo.join(turbo_bench::statics::PEERS_SCRIPT), PathBuf::from(w))),
        (None, true) => None,
        _ => return Err("give --peers <dir peers.sh setup wrote> or --no-peers".into()),
    };
    let plan = turbo_bench::statics::StaticPlan { bundle: PathBuf::from(bundle), precision, sets, batches, peers };
    let out = o.take("--out");
    if let Some(unknown) = o.0.keys().next() {
        return Err(format!("{unknown}: not a static option\n{USAGE}"));
    }
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    turbo_bench::check_not_testdata(&plan.bundle, &[workspace.join("testdata"), repo.join("testdata")])?;
    let before = git::provenance(&repo)?;
    turbo_bench::check_build(&before.commit, turbo_bench::BUILD_COMMIT, turbo_bench::BUILD_CHANGES)?;
    let r = turbo_bench::statics::measure(&plan, &before, turbo_bench::utc(SystemTime::now()))?;
    let after = git::provenance(&repo)?;
    if after != before {
        return Err(format!("the working tree moved from {} to {} during the run", before.commit, after.commit));
    }
    let dir = out.map_or_else(|| Path::new(&after.top).join(turbo_bench::statics::STATIC_DIR), PathBuf::from);
    let path = turbo_bench::statics::write(&r, &dir)?;
    for x in &r.turbo {
        eprintln!("turbo {} batch {}: {:.0} texts/s", x.set, x.batch, x.texts_per_second);
    }
    for p in &r.peers {
        for x in &p.rates {
            eprintln!("{} {} batch {}: {:.0} texts/s", p.name, x.set, x.batch, x.texts_per_second);
        }
        if let Some(a) = &p.vectors {
            eprintln!(
                "{}: {} of {} rows to the bit, {} refused, min cosine {}",
                p.name, a.bit_equal, a.rows, a.refused, a.min_cosine
            );
        }
        if let Some(why) = &p.not_run {
            eprintln!("{}: {why}", p.name);
        }
    }
    println!("{}", path.display());
    Ok(())
}
