//! `turbo-bench static`: a static bundle's texts per second on the CPU,
//! beside the static-model programs of bench/static-peers on the same
//! machine and the same texts (docs/static.md, "Speed").
//!
//! Each text set is a JSON array of strings. The library and every peer
//! time it the same way: the texts in order, batch by batch, one warm
//! pass, then at least two seconds three times, the best of the three in
//! texts a second, at `max_length` with L2 normalization. Each peer also
//! writes its vectors for the first set, which are compared with the
//! library's (Model2Vec's to the bit, core/tests/static_parity.rs).
//!
//! These records sit under benchmarks/static/, apart from
//! benchmarks/records/: they are not compiled into the library and decide
//! no capability cell.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use turbo::record::{self, Device, Library, Machine};
use turbo::{TURBO_NORMALIZE_L2, TURBO_TRUNCATE_MODEL, turbo_embed_options};

use crate::Result;
use crate::api::{Runtime, field};
use crate::git::Provenance;

/// The version of `StaticRecord`.
pub const STATIC_RECORD_VERSION: u32 = 1;

/// Where static records go, under the repository.
pub const STATIC_DIR: &str = "benchmarks/static/";

/// The peers `peers.sh run` knows: each one's name, the version
/// bench/static-peers/README.md pins, and its program there.
pub const PEERS: [(&str, &str, &str); 7] = [
    ("statembed", "statembed 1.0.0", "rs/src/bin/statembed.rs"),
    ("statembed-rayon", "statembed 1.0.0 with rayon", "rs/src/bin/statembed.rs, built with rayon"),
    ("model2vec-rs", "model2vec-rs 1567eaa", "rs/src/bin/m2vrs.rs"),
    ("go-potion", "go-potion v0.1.2", "go/main.go"),
    (
        "model2vec-zig",
        "model2vec-zig ed5b443 (0.2.0)",
        "zig/src/main.zig; it has no batch API, so a batch above 1 runs every text over one thread per CPU",
    ),
    ("model2vec", "Model2Vec 3ef2bf2", "py/bench.py model2vec"),
    ("st", "sentence-transformers 6.1.0 StaticEmbedding", "py/bench.py st"),
];

/// The script the peers run through, as a record names it.
pub const PEERS_SCRIPT: &str = "bench/static-peers/peers.sh";

/// What to measure, as the command line gave it.
#[derive(Debug, Clone)]
pub struct StaticPlan {
    pub bundle: PathBuf,
    pub precision: u32,
    /// Each text set's name and file.
    pub sets: Vec<(String, PathBuf)>,
    pub batches: Vec<u32>,
    /// peers.sh and the directory its setup wrote (`W`); none times the
    /// library alone.
    pub peers: Option<(PathBuf, PathBuf)>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StaticRecord {
    pub record_version: u32,
    pub recorded_at: String,
    pub machine: Machine,
    pub device: Device,
    pub library: Library,
    pub precision: String,
    pub model_id: String,
    pub revision: String,
    pub manifest_sha256: String,
    pub sets: Vec<TextSet>,
    pub batches: Vec<u32>,
    pub turbo: Vec<Rate>,
    pub peers: Vec<Peer>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextSet {
    pub name: String,
    pub texts: u32,
    pub sha256: String,
}

/// The best of three runs of at least two seconds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rate {
    pub set: String,
    pub batch: u32,
    pub texts_per_second: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Peer {
    pub name: String,
    pub version: String,
    /// Its program under bench/static-peers, and how it batches when that
    /// differs from the library.
    pub procedure: String,
    /// Every command run for it, each as its argv, with the text set as
    /// `<set:NAME>` and the vectors file as `<vectors>`.
    pub commands: Vec<Vec<String>>,
    pub rates: Vec<Rate>,
    pub vectors: Option<Agreement>,
    /// Why it was not run, or ran without a rate; null when it ran.
    pub not_run: Option<String>,
}

/// A peer's vectors for a set against the library's.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Agreement {
    pub set: String,
    pub rows: u32,
    /// Rows equal to the library's to the bit.
    pub bit_equal: u32,
    /// Rows with non-finite values: texts the peer refused.
    pub refused: u32,
    /// The lowest cosine over the rows it did not refuse.
    pub min_cosine: f64,
}

/// Time `pass` over `n` texts: one warm pass, then the best of three runs
/// of at least two seconds, in texts a second.
pub fn best(n: usize, pass: &mut dyn FnMut() -> Result<()>) -> Result<f64> {
    pass()?;
    let mut top = 0f64;
    for _ in 0..3 {
        let (start, mut done) = (Instant::now(), 0usize);
        while start.elapsed() < Duration::from_secs(2) {
            pass()?;
            done += n;
        }
        top = top.max(done as f64 / start.elapsed().as_secs_f64());
    }
    Ok(top)
}

/// The rates in a peer's output: one line ending `batch <b>: <n> texts/s`
/// for each of `batches`, in order. A program that runs every batch above
/// 1 alike names it `all`.
pub fn rates(set: &str, out: &str, batches: &[u32]) -> Vec<Rate> {
    let lines = out.lines().filter_map(|l| {
        let (_, rest) = l.rsplit_once(" batch ")?;
        let (b, tps) = rest.split_once(": ")?;
        Some((b, tps.strip_suffix(" texts/s")?.parse::<f64>().ok()?))
    });
    lines
        .zip(batches)
        .filter(|((b, _), want)| b.parse() == Ok(**want) || (**want > 1 && b.starts_with("all")))
        .map(|((_, tps), &batch)| Rate { set: set.into(), batch, texts_per_second: tps })
        .collect()
}

/// `got` (raw little-endian F32 rows) against the library's `want`.
pub fn agreement(set: &str, want: &[f32], got: &[u8], dim: usize) -> Result<Agreement> {
    if got.len() != want.len() * 4 {
        return Err(format!("{} bytes of vectors, the set makes {}", got.len(), want.len() * 4));
    }
    let got: Vec<f32> = got.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
    let (mut bit_equal, mut refused, mut min_cosine) = (0, 0, 1f64);
    for (w, g) in want.chunks(dim).zip(got.chunks(dim)) {
        if g.iter().any(|v| !v.is_finite()) {
            refused += 1;
            continue;
        }
        if w.iter().zip(g).all(|(a, b)| a.to_bits() == b.to_bits()) {
            bit_equal += 1;
        }
        let dot: f64 = w.iter().zip(g).map(|(&a, &b)| f64::from(a) * f64::from(b)).sum();
        let norm = |v: &[f32]| v.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>().sqrt();
        let (nw, ng) = (norm(w), norm(g));
        // Two zero vectors (a text with no tokens) agree.
        let c = if nw == 0.0 && ng == 0.0 {
            1.0
        } else if nw == 0.0 || ng == 0.0 {
            0.0
        } else {
            dot / (nw * ng)
        };
        min_cosine = min_cosine.min(c);
    }
    Ok(Agreement { set: set.into(), rows: (want.len() / dim) as u32, bit_equal, refused, min_cosine })
}

pub fn measure(plan: &StaticPlan, p: &Provenance, recorded_at: String) -> Result<StaticRecord> {
    if plan.sets.is_empty() || plan.batches.is_empty() || plan.batches.contains(&0) {
        return Err("--texts and --batches must name at least one set and batches above 0".into());
    }
    let rt = Runtime::create()?;
    let index = rt.find("cpu")?;
    let d = rt.device_info(index)?;
    let model = rt.context(index)?.load(&plan.bundle)?;
    let info = model.info()?;
    let dim = info.dim as usize;
    let mut o: turbo_embed_options = unsafe { std::mem::zeroed() };
    o.struct_size = size_of::<turbo_embed_options>() as u32;
    o.truncate = TURBO_TRUNCATE_MODEL;
    o.normalize = TURBO_NORMALIZE_L2;
    let mut sets = Vec::new();
    let mut texts = Vec::new();
    for (name, path) in &plan.sets {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let t: Vec<String> = serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
        sets.push(TextSet { name: name.clone(), texts: t.len() as u32, sha256: turbo::bundle::sha256_hex(&bytes) });
        texts.push(t);
    }
    let mut turbo = Vec::new();
    let mut first = Vec::new();
    for &b in &plan.batches {
        let s = model.session(b, 0, plan.precision)?;
        for ((name, _), t) in plan.sets.iter().zip(&texts) {
            let views: Vec<&str> = t.iter().map(String::as_str).collect();
            let mut vectors = vec![0f32; views.len() * dim];
            let tps = best(views.len(), &mut || {
                for (chunk, out) in views.chunks(b as usize).zip(vectors.chunks_mut(b as usize * dim)) {
                    s.embed_texts_into(chunk, &o, out)?;
                }
                Ok(())
            })?;
            if first.is_empty() {
                first = vectors;
            }
            turbo.push(Rate { set: name.clone(), batch: b, texts_per_second: tps });
        }
    }
    let name = field(&info.model_id);
    let short = name.rsplit('/').next().unwrap_or(&name).to_owned();
    let peers = match &plan.peers {
        Some((script, w)) => PEERS.iter().map(|&p| peer(plan, script, w, &short, p, &first, dim)).collect(),
        None => Vec::new(),
    };
    Ok(StaticRecord {
        record_version: STATIC_RECORD_VERSION,
        recorded_at,
        machine: Machine { arch: field(&d.arch), host_cpu: field(&d.name), os: std::env::consts::OS.into() },
        device: Device {
            backend: field(&d.backend),
            kind: crate::kind_name(d.kind).into(),
            name: field(&d.name),
            vendor: field(&d.vendor),
            driver_version: field(&d.driver_version),
            runtime_version: field(&d.runtime_version),
            memory_total: d.memory_total,
        },
        library: Library {
            version: crate::api::version().split(' ').next().unwrap_or("").into(),
            build: crate::api::version(),
            commit: p.commit.clone(),
            pushed_to: p.pushed_to.clone(),
            settings: Vec::new(),
        },
        precision: record::precision_name(plan.precision).ok_or("precision is not a TURBO_PRECISION_* value")?.into(),
        model_id: name,
        revision: field(&info.revision),
        manifest_sha256: field(&info.manifest_sha256),
        sets,
        batches: plan.batches.clone(),
        turbo,
        peers,
    })
}

/// One peer through `peers.sh run`: its rates on every set, then its
/// vectors for the first set against the library's.
fn peer(
    plan: &StaticPlan,
    script: &Path,
    w: &Path,
    model: &str,
    (name, version, procedure): (&str, &str, &str),
    first: &[f32],
    dim: usize,
) -> Peer {
    let batches = plan.batches.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
    let run = |set: &str, texts: &Path, batches: &str, out: Option<&Path>| {
        let mut c = Command::new("bash");
        c.arg(script).args(["run", name, model]).arg(texts).arg(batches).env("W", w);
        if let Some(out) = out {
            c.arg(out);
        }
        let mut argv: Vec<String> =
            [PEERS_SCRIPT, "run", name, model, &format!("<set:{set}>"), batches].map(String::from).to_vec();
        if out.is_some() {
            argv.push("<vectors>".into());
        }
        (c.output(), argv)
    };
    let mut p = Peer {
        name: name.into(),
        version: version.into(),
        procedure: procedure.into(),
        commands: Vec::new(),
        rates: Vec::new(),
        vectors: None,
        not_run: None,
    };
    for (set, path) in &plan.sets {
        let (out, argv) = run(set, path, &batches, None);
        p.commands.push(argv);
        // Some programs print their rates on stderr.
        match out {
            Ok(o) => {
                let text = String::from_utf8_lossy(&o.stdout).into_owned() + &String::from_utf8_lossy(&o.stderr);
                let got = rates(set, &text, &plan.batches);
                if got.is_empty() {
                    p.not_run = Some(format!("{set}: no rate ({}): {}", o.status, text.lines().last().unwrap_or("")));
                }
                p.rates.extend(got);
            }
            Err(e) => p.not_run = Some(format!("{PEERS_SCRIPT}: {e}")),
        }
    }
    if p.rates.is_empty() {
        return p;
    }
    let (set, path) = &plan.sets[0];
    let file = std::env::temp_dir().join(format!("turbo-bench-static-{}-{name}.f32", std::process::id()));
    let (out, argv) = run(set, path, "-", Some(&file));
    p.commands.push(argv);
    if out.is_ok_and(|o| o.status.success()) {
        match std::fs::read(&file).map_err(|e| e.to_string()).and_then(|got| agreement(set, first, &got, dim)) {
            Ok(a) => p.vectors = Some(a),
            Err(e) => p.not_run = Some(format!("its vectors: {e}")),
        }
    }
    let _ = std::fs::remove_file(&file);
    p
}

/// The record's file name: machine, precision, model, manifest and commit.
pub fn file_name(r: &StaticRecord) -> String {
    let slug = |s: &str| {
        let mut out = String::new();
        for c in s.chars() {
            if c.is_ascii_alphanumeric() {
                out.push(c.to_ascii_lowercase());
            } else if !out.is_empty() && !out.ends_with('-') {
                out.push('-');
            }
        }
        out.trim_end_matches('-').to_owned()
    };
    let mut model = slug(r.model_id.rsplit('/').next().unwrap_or(""));
    model.truncate(32);
    format!(
        "{}.{}.static.{}.{}-{}.{}.json",
        slug(&r.machine.arch),
        slug(&r.device.backend),
        slug(r.precision.strip_prefix("PRECISION_").unwrap_or(&r.precision)),
        model.trim_end_matches('-'),
        r.manifest_sha256.get(..8).unwrap_or(""),
        r.library.commit.get(..12).unwrap_or("")
    )
}

/// What a committed static record must be: made from a pushed commit, a
/// finite positive rate for every set and batch the library ran, and no
/// host path in its text.
pub fn check(r: &StaticRecord) -> Result<()> {
    if r.record_version != STATIC_RECORD_VERSION {
        return Err(format!("record_version {}, expected {STATIC_RECORD_VERSION}", r.record_version));
    }
    let c = &r.library.commit;
    if c.len() != 40 || !c.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(format!("library.commit {c} is not 40 hex"));
    }
    if r.library.pushed_to.is_empty() {
        return Err("library.pushed_to is empty: made from a commit on no branch of origin".into());
    }
    let ok = |x: &Rate| x.texts_per_second.is_finite() && x.texts_per_second > 0.0;
    if r.turbo.len() != r.sets.len() * r.batches.len() || !r.turbo.iter().all(ok) {
        return Err("the library needs a positive rate for every set and batch".into());
    }
    for p in &r.peers {
        if !p.rates.iter().all(ok) || (p.rates.is_empty() && p.not_run.is_none()) {
            return Err(format!("peer {}: a rate is not positive, or it has none and no reason", p.name));
        }
    }
    let text = serde_json::to_string(r).map_err(|e| e.to_string())?.replace("\\\\", "\\");
    if let Some(home) = record::HOST_PATHS.iter().find(|h| text.contains(*h)) {
        return Err(format!("the record holds a host path ({home}...)"));
    }
    Ok(())
}

/// Write the record under `dir`, checked; a record is never replaced.
pub fn write(r: &StaticRecord, dir: &Path) -> Result<PathBuf> {
    check(r)?;
    let mut bytes = serde_json::to_vec_pretty(r).map_err(|e| e.to_string())?;
    bytes.push(b'\n');
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let path = dir.join(file_name(r));
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(|e| format!("{}: {e}; a record is never replaced", path.display()))?;
    std::io::Write::write_all(&mut f, &bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(path)
}

#[cfg(test)]
#[path = "statics_tests.rs"]
mod tests;
