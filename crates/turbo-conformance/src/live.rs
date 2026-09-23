//! Live-provider selection for the hardware tests in `tests/live_*.rs`.
//!
//! The live tests load a real provider library against real bundles and
//! skip, printing the reason, when the environment does not name one:
//!
//! - `TURBO_LIVE_LIB`: path to the provider library (`libturbo_provider_*.so`).
//! - `TURBO_LIVE_PROVIDER`: the provider id it registers (`openvino`, `cuda`).
//! - `TURBO_LIVE_ORDINAL` (optional): device ordinal. Default: the provider's
//!   CPU device when it has one, else ordinal 0.
//! - `TURBO_LIVE_BUNDLE`: an `all-MiniLM-L6-v2` bundle (embedding tests).
//! - `TURBO_LIVE_RERANK_BUNDLE`, `TURBO_LIVE_CLASSIFY_BUNDLE`,
//!   `TURBO_LIVE_NER_BUNDLE`: task bundles (see `tests/live_tasks.rs`).
//! - `TURBO_REFERENCE_DIR` (optional): directory of reference vectors when a
//!   test binary is copied to another machine.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use turbo::{
    Capability, Context, ContextDesc, DType, DeviceInfo, DeviceKind, DeviceSelector, Modality, RuntimeDesc,
    SelectPolicy, Task,
};

/// A selected live device.
pub struct Live {
    /// Context on the selected device.
    pub ctx: Arc<Context>,
    /// Provider id.
    pub provider: String,
    /// Selected device.
    pub device: DeviceInfo,
    /// The device's `EMBED x TEXT` capability cell.
    pub embed: Capability,
}

impl Live {
    /// True when the selected device is a GPU (results are expected on the device).
    pub fn gpu(&self) -> bool {
        matches!(self.device.kind, DeviceKind::Gpu | DeviceKind::IGpu)
    }

    /// True when the device advertises `bit`.
    pub fn has_cap(&self, bit: u64) -> bool {
        self.device.caps & bit == bit
    }

    /// Cosine floor the embedding checks hold this device to, against the
    /// FP32 reference vectors. An FP32 (or unstated) compute dtype is held
    /// to 0.9995. A quantized dtype is held to the floor the suite owns for
    /// this provider and dtype (`testdata/reference_embeddings/quantized_floors.json`,
    /// set from a committed receipt), and the device must also state a
    /// measured floor in its capability cell that is no higher than the
    /// suite's; a provider cannot set its own gate, and one the suite has
    /// no floor for fails until a receipt adds one.
    pub fn embed_cosine_floor(&self) -> f32 {
        match self.embed.dtype {
            None | Some(DType::F32) => 0.9995,
            Some(dtype) => {
                let path = reference_dir().join("quantized_floors.json");
                let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
                let table: serde_json::Value = serde_json::from_str(&text).expect("quantized_floors.json");
                let key = format!("{dtype:?}");
                let suite = table
                    .get(&self.provider)
                    .and_then(|p| p.get(&key))
                    .and_then(|c| c.get("floor"))
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or_else(|| {
                        panic!(
                            "{} computes EMBED in {dtype:?} but {} has no floor for it; add one from a receipt",
                            self.provider,
                            path.display()
                        )
                    }) as f32;
                assert!(
                    self.embed.cosine_floor > 0.0 && self.embed.cosine_floor <= suite,
                    "{} reports cosine_floor {} for {dtype:?}; the suite's floor from the receipt is {suite}, and a \
                     provider must state a measured floor no higher than that",
                    self.provider,
                    self.embed.cosine_floor
                );
                suite
            }
        }
    }
}

fn env(name: &str) -> Option<String> {
    match std::env::var(name) {
        Ok(v) if !v.is_empty() => Some(v),
        _ => None,
    }
}

/// Select the live device named by the environment, or `None` (with a
/// printed reason) when the environment names no provider.
pub fn live() -> Option<Live> {
    let Some(lib) = env("TURBO_LIVE_LIB") else {
        eprintln!("skipping: TURBO_LIVE_LIB is not set");
        return None;
    };
    let Some(provider) = env("TURBO_LIVE_PROVIDER") else {
        eprintln!("skipping: TURBO_LIVE_PROVIDER is not set");
        return None;
    };
    let rt = turbo::create_runtime(RuntimeDesc { provider_paths: vec![lib], ..Default::default() })
        .unwrap_or_else(|e| panic!("load the live provider: {e}"));
    assert!(rt.failures().is_empty(), "provider failures: {:?}", rt.failures());
    let devices: Vec<_> = rt.devices().into_iter().filter(|d| d.info.provider_id == provider).collect();
    assert!(!devices.is_empty(), "provider `{provider}` enumerated no devices");
    for d in &devices {
        eprintln!(
            "{provider} device ordinal {} kind {:?} `{}` runtime {} driver {} caps {:#x}",
            d.info.ordinal, d.info.kind, d.info.name, d.info.runtime_version, d.info.driver_version, d.info.caps
        );
    }
    let ordinal = match env("TURBO_LIVE_ORDINAL") {
        Some(v) => v.parse::<u32>().expect("TURBO_LIVE_ORDINAL is a device ordinal"),
        None => devices.iter().find(|d| d.info.kind == DeviceKind::Cpu).map(|d| d.info.ordinal).unwrap_or(0),
    };
    let idx = rt
        .select(&DeviceSelector {
            policy: SelectPolicy::Explicit,
            provider_id: provider.clone(),
            ordinal,
            ..Default::default()
        })
        .expect("select the live device");
    let device = rt.device(idx).expect("selected device").info;
    eprintln!("selected: {} (ordinal {ordinal})", device.name);
    let embed = rt.capability(idx, Task::Embed, Modality::Text).expect("EMBED x TEXT capability");
    let ctx = Context::create(rt, idx, &ContextDesc::default()).expect("context");
    Some(Live { ctx, provider, device, embed })
}

/// A bundle directory from `var`, or `None` with a printed reason.
pub fn bundle(var: &str) -> Option<PathBuf> {
    match env(var) {
        Some(v) => Some(PathBuf::from(v)),
        None => {
            eprintln!("skipping: {var} is not set");
            None
        }
    }
}

/// Directory of reference vectors: `TURBO_REFERENCE_DIR` or the repository's
/// `testdata/reference_embeddings`.
pub fn reference_dir() -> PathBuf {
    env("TURBO_REFERENCE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/reference_embeddings"))
}

/// Cosine similarity.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}
