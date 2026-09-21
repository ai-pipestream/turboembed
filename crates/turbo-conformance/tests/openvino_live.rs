//! Live checks for the OpenVINO provider against a real MiniLM bundle.
//!
//! Skipped (with a printed reason) unless both are set:
//! - `TURBO_OPENVINO_LIB`: path to `libturbo_provider_openvino.so`
//! - `TURBO_OPENVINO_BUNDLE`: a bundle with an `onnx` or `openvino_ir` artifact
//!   for `sentence-transformers/all-MiniLM-L6-v2` (see `tools/turbo-bundle`)
//!
//! Optional: `TURBO_OPENVINO_ORDINAL` selects a device ordinal (default: the
//! provider's CPU device, which is the last GPU ordinal + 1).
//!
//! Reference vectors come from `testdata/reference_embeddings/ort_cuda_minilm_*.json`,
//! produced by ONNX Runtime CUDA in FP32 on the same ONNX export.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use turbo::abi;
use turbo::{
    Context, ContextDesc, DeviceKind, DeviceSelector, EmbedOptions, ModelDesc, Placement, RuntimeDesc, SelectPolicy,
    SessionDesc, Truncate,
};

struct Live {
    ctx: Arc<Context>,
    bundle: PathBuf,
    gpu: bool,
}

fn live() -> Option<Live> {
    let lib = match std::env::var("TURBO_OPENVINO_LIB") {
        Ok(v) if !v.is_empty() => v,
        _ => {
            eprintln!("skipping: TURBO_OPENVINO_LIB is not set");
            return None;
        }
    };
    let bundle = match std::env::var("TURBO_OPENVINO_BUNDLE") {
        Ok(v) if !v.is_empty() => PathBuf::from(v),
        _ => {
            eprintln!("skipping: TURBO_OPENVINO_BUNDLE is not set");
            return None;
        }
    };
    let rt = turbo::create_runtime(RuntimeDesc { provider_paths: vec![lib], ..Default::default() })
        .unwrap_or_else(|e| panic!("load the openvino provider: {e}"));
    assert!(rt.failures().is_empty(), "provider failures: {:?}", rt.failures());
    let devices: Vec<_> = rt.devices().into_iter().filter(|d| d.info.provider_id == "openvino").collect();
    assert!(!devices.is_empty(), "the openvino provider enumerated no devices");
    for d in &devices {
        eprintln!(
            "openvino device ordinal {} kind {:?} `{}` runtime {} driver {} caps {:#x}",
            d.info.ordinal, d.info.kind, d.info.name, d.info.runtime_version, d.info.driver_version, d.info.caps
        );
    }
    let ordinal = match std::env::var("TURBO_OPENVINO_ORDINAL") {
        Ok(v) => v.parse::<u32>().expect("TURBO_OPENVINO_ORDINAL"),
        Err(_) => devices.iter().find(|d| d.info.kind == DeviceKind::Cpu).expect("an openvino CPU device").info.ordinal,
    };
    let idx = rt
        .select(&DeviceSelector {
            policy: SelectPolicy::Explicit,
            provider_id: "openvino".into(),
            ordinal,
            ..Default::default()
        })
        .expect("select the openvino device");
    let info = rt.device(idx).unwrap().info;
    let gpu = matches!(info.kind, DeviceKind::Gpu | DeviceKind::IGpu);
    let ctx = Context::create(rt, idx, &ContextDesc::default()).expect("context");
    Some(Live { ctx, bundle, gpu })
}

#[derive(serde::Deserialize)]
struct Golden {
    text: String,
    vector: Vec<f32>,
}

fn golden(name: &str) -> Golden {
    // `TURBO_REFERENCE_DIR` lets a copied test binary find the goldens on another machine.
    let dir = std::env::var("TURBO_REFERENCE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/reference_embeddings"));
    let path = dir.join(format!("ort_cuda_minilm_{name}.json"));
    serde_json::from_str(&std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))).unwrap()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn embed(
    live: &Live,
    texts: &[&str],
    opts: &EmbedOptions,
    max_batch: u32,
) -> (Vec<Vec<f32>>, turbo::SessionStats, Placement) {
    let model = live.ctx.load_model(&live.bundle, &ModelDesc::default()).expect("load MiniLM");
    assert_eq!(model.info().dim, 384);
    assert_eq!(model.info().provider_id, "openvino");
    let session =
        model.create_session(&SessionDesc { max_batch, max_seq: 256, ..Default::default() }).expect("session");
    session.write_text(texts, opts).expect("write_text");
    let r = session.run(&Default::default()).expect("run");
    let out = r.output(0).unwrap();
    assert_eq!(out.shape, vec![texts.len() as u64, 384]);
    let placement = out.placement();
    let mut bytes = vec![0u8; out.logical_bytes().unwrap() as usize];
    r.read(0, &mut bytes).unwrap();
    let floats: Vec<f32> = bytes.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    drop(r);
    let stats = session.stats().unwrap();
    (floats.chunks(384).map(|c| c.to_vec()).collect(), stats, placement)
}

#[test]
fn openvino_minilm_matches_the_reference_vectors() {
    let Some(live) = live() else { return };
    let names = ["short", "medium", "empty", "long_truncation"];
    let goldens: Vec<Golden> = names.iter().map(|n| golden(n)).collect();
    let texts: Vec<&str> = goldens.iter().map(|g| g.text.as_str()).collect();
    let (vecs, stats, placement) = embed(&live, &texts, &EmbedOptions::default(), 8);
    for (i, (v, g)) in vecs.iter().zip(&goldens).enumerate() {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "{}: norm {norm}", names[i]);
        let c = cosine(v, &g.vector);
        eprintln!("{}: cosine vs ORT CUDA reference = {c:.6}", names[i]);
        assert!(c > 0.9995, "{}: cosine {c} below the FP32 floor", names[i]);
    }
    eprintln!("stats: {stats:?}, output placement {placement:?}");
    if live.gpu {
        assert_eq!(placement, Placement::Device, "GPU results stay on the device");
        assert!(stats.h2d_bytes > 0 && stats.d2h_bytes == 0, "reads go through result_read, not the run: {stats:?}");
    } else {
        assert_eq!(placement, Placement::Host);
        assert_eq!(stats.h2d_bytes, 0);
    }
}

#[test]
fn openvino_batch_rows_equal_single_runs() {
    let Some(live) = live() else { return };
    let a = "the cat sat on the mat";
    let b = "an entirely different sentence about gpus";
    let (both, _, _) = embed(&live, &[a, b], &EmbedOptions::default(), 4);
    let (only_a, _, _) = embed(&live, &[a], &EmbedOptions::default(), 4);
    let (only_b, _, _) = embed(&live, &[b], &EmbedOptions::default(), 4);
    assert!(cosine(&both[0], &only_a[0]) > 0.99999);
    assert!(cosine(&both[1], &only_b[0]) > 0.99999);
    assert!(cosine(&both[0], &both[1]) < 0.9, "unrelated sentences should not be near-identical");
}

#[test]
fn openvino_truncation_policy_is_enforced() {
    let Some(live) = live() else { return };
    let model = live.ctx.load_model(&live.bundle, &ModelDesc::default()).unwrap();
    let session = model.create_session(&SessionDesc { max_batch: 2, max_seq: 16, ..Default::default() }).unwrap();
    let long = "token ".repeat(100);
    let e = session.write_text(&[&long], &EmbedOptions { truncate: Truncate::None, ..Default::default() }).unwrap_err();
    assert_eq!(e.code(), abi::TURBO_E_CAPACITY);
    session
        .write_text(
            &["a b c d e f g h i j k l m n o p q r s t u v w x y z"],
            &EmbedOptions { truncate: Truncate::Right, ..Default::default() },
        )
        .unwrap();
    let r = session.run(&Default::default()).unwrap();
    let mut right = vec![0u8; 384 * 4];
    r.read(0, &mut right).unwrap();
    drop(r);
    session
        .write_text(
            &["a b c d e f g h i j k l m n o p q r s t u v w x y z"],
            &EmbedOptions { truncate: Truncate::Left, ..Default::default() },
        )
        .unwrap();
    let r = session.run(&Default::default()).unwrap();
    let mut left = vec![0u8; 384 * 4];
    r.read(0, &mut left).unwrap();
    assert_ne!(left, right, "left and right truncation keep different tokens");
    let e =
        session.write_text(&["x"], &EmbedOptions { pooling: turbo::Pooling::Cls, ..Default::default() }).unwrap_err();
    assert_eq!(e.code(), abi::TURBO_E_UNSUPPORTED_OPTION);
    assert_eq!(e.field(), EmbedOptions::FIELD_POOLING);
}
