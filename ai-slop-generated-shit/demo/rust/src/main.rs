//! Turbo Rust demo: load a bundle on the best device, embed sentences, and
//! print their cosine similarities through the safe Rust API.
//!
//! ```text
//! cargo run --manifest-path demo/rust/Cargo.toml -- [--provider-lib <so>] \
//!     [--provider <id> --ordinal <n>] --bundle <dir> text...
//! ```

use std::path::PathBuf;
use std::process::ExitCode;

use turbo::provider::{EmbedOptions, ModelDesc, SessionDesc};
use turbo::{Context, ContextDesc, DeviceSelector, RuntimeDesc, SelectPolicy};

struct Args {
    provider_lib: Option<String>,
    provider: Option<String>,
    ordinal: Option<u32>,
    bundle: Option<PathBuf>,
    texts: Vec<String>,
}

fn parse() -> Result<Args, String> {
    let mut a = Args { provider_lib: None, provider: None, ordinal: None, bundle: None, texts: Vec::new() };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().ok_or_else(|| format!("{name} needs a value"));
        match arg.as_str() {
            "--provider-lib" => a.provider_lib = Some(value("--provider-lib")?),
            "--provider" => a.provider = Some(value("--provider")?),
            "--ordinal" => a.ordinal = Some(value("--ordinal")?.parse().map_err(|e| format!("--ordinal: {e}"))?),
            "--bundle" => a.bundle = Some(PathBuf::from(value("--bundle")?)),
            _ => a.texts.push(arg),
        }
    }
    if a.bundle.is_none() || a.texts.is_empty() {
        return Err("usage: turbo-demo-rust [--provider-lib <so>] [--provider <id> --ordinal <n>] --bundle <dir> text...".into());
    }
    if a.provider.is_some() != a.ordinal.is_some() {
        return Err("--provider and --ordinal go together; omit both for AUTO".into());
    }
    Ok(a)
}

fn run(a: Args) -> Result<(), String> {
    let rt = turbo::create_runtime(RuntimeDesc {
        provider_paths: a.provider_lib.iter().cloned().collect(),
        ..Default::default()
    })
    .map_err(|e| format!("runtime: {e}"))?;
    let selector = match (&a.provider, a.ordinal) {
        (Some(p), Some(o)) => DeviceSelector { policy: SelectPolicy::Explicit, provider_id: p.clone(), ordinal: o, ..Default::default() },
        _ => DeviceSelector::default(), // AUTO: never a CPU
    };
    let idx = rt.select(&selector).map_err(|e| format!("select device: {e}"))?;
    let info = rt.device(idx).map_err(|e| e.to_string())?.info;
    println!("device: {} ({}:{}, {:?}, runtime {})", info.name, info.provider_id, info.ordinal, info.kind, info.runtime_version);
    let ctx = Context::create(rt, idx, &ContextDesc::default()).map_err(|e| format!("context: {e}"))?;
    let bundle = a.bundle.expect("checked");
    let model = ctx.load_model(&bundle, &ModelDesc::default()).map_err(|e| format!("load {}: {e}", bundle.display()))?;
    let mi = model.info();
    println!("model: {} dim={} max_seq={} provider={} fully_accelerated={}", mi.model_id, mi.dim, mi.max_seq, mi.provider_id, mi.stages.fully_accelerated());
    let session = model
        .create_session(&SessionDesc { max_batch: a.texts.len() as u32, max_seq: mi.max_seq.min(128), ..Default::default() })
        .map_err(|e| format!("session: {e}"))?;
    let refs: Vec<&str> = a.texts.iter().map(String::as_str).collect();
    session.write_text(&refs, &EmbedOptions::default()).map_err(|e| format!("write_text: {e}"))?;
    let r = session.run(&Default::default()).map_err(|e| format!("run: {e}"))?;
    let out = r.output(0).map_err(|e| e.to_string())?;
    let dim = out.shape[1] as usize;
    let placement = out.placement();
    let mut bytes = vec![0u8; out.logical_bytes().map_err(|e| e.to_string())? as usize];
    r.read(0, &mut bytes).map_err(|e| format!("read: {e}"))?;
    let floats: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let rows: Vec<&[f32]> = floats.chunks(dim).collect();
    println!("embeddings: {} x {dim} (placement {placement:?})", rows.len());
    println!("cosine similarity:");
    for (i, row) in rows.iter().enumerate() {
        let line: String = rows.iter().map(|other| format!(" {:6.3}", cosine(row, other))).collect();
        println!("{line}  {}", a.texts[i]);
    }
    Ok(())
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn main() -> ExitCode {
    match parse().and_then(run) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(1)
        }
    }
}
