//! Live embedding checks for any provider against a real MiniLM bundle.
//!
//! Selection and skipping are described in `turbo_conformance::live`. The
//! reference vectors in `testdata/reference_embeddings/ort_cuda_minilm_*.json`
//! were produced by ONNX Runtime CUDA in FP32 on the same ONNX export, so an
//! FP32 provider is held to cosine 0.9995 against them.

use turbo::abi;
use turbo::{EmbedOptions, ModelDesc, Placement, SessionDesc, Truncate};
use turbo_conformance::live::{bundle, cosine, live, reference_dir, Live};

#[derive(serde::Deserialize)]
struct Golden {
    text: String,
    vector: Vec<f32>,
}

fn golden(name: &str) -> Golden {
    let path = reference_dir().join(format!("ort_cuda_minilm_{name}.json"));
    serde_json::from_str(&std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))).unwrap()
}

fn setup() -> Option<(Live, std::path::PathBuf)> {
    let live = live()?;
    let bundle = bundle("TURBO_LIVE_BUNDLE")?;
    Some((live, bundle))
}

fn embed(
    live: &Live,
    bundle: &std::path::Path,
    texts: &[&str],
    opts: &EmbedOptions,
    max_batch: u32,
) -> (Vec<Vec<f32>>, turbo::SessionStats, Placement) {
    let model = live.ctx.load_model(bundle, &ModelDesc::default()).expect("load MiniLM");
    assert_eq!(model.info().dim, 384);
    assert_eq!(model.info().provider_id, live.provider);
    let session =
        model.create_session(&SessionDesc { max_batch, max_seq: 256, ..Default::default() }).expect("session");
    session.write_text(texts, opts).expect("write_text");
    let r = session.run(&Default::default()).expect("run");
    let out = r.output(0).unwrap();
    let dim = if opts.output_dim == 0 { 384 } else { opts.output_dim as usize };
    assert_eq!(out.shape, vec![texts.len() as u64, dim as u64]);
    let placement = out.placement();
    let mut bytes = vec![0u8; out.logical_bytes().unwrap() as usize];
    r.read(0, &mut bytes).unwrap();
    let floats: Vec<f32> = bytes.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    drop(r);
    let stats = session.stats().unwrap();
    (floats.chunks(dim).map(|c| c.to_vec()).collect(), stats, placement)
}

#[test]
fn live_minilm_matches_the_reference_vectors() {
    let Some((live, bundle)) = setup() else { return };
    let names = ["short", "medium", "empty", "long_truncation"];
    let goldens: Vec<Golden> = names.iter().map(|n| golden(n)).collect();
    let texts: Vec<&str> = goldens.iter().map(|g| g.text.as_str()).collect();
    let (vecs, stats, placement) = embed(&live, &bundle, &texts, &EmbedOptions::default(), 8);
    for (i, (v, g)) in vecs.iter().zip(&goldens).enumerate() {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "{}: norm {norm}", names[i]);
        let c = cosine(v, &g.vector);
        eprintln!("{}: cosine vs ORT CUDA reference = {c:.6}", names[i]);
        assert!(c > 0.9995, "{}: cosine {c} below the FP32 floor", names[i]);
    }
    eprintln!("stats: {stats:?}, output placement {placement:?}");
    if live.gpu() {
        assert_eq!(placement, Placement::Device, "GPU results stay on the device");
        assert!(stats.h2d_bytes > 0 && stats.d2h_bytes == 0, "reads go through result_read, not the run: {stats:?}");
    } else {
        assert_eq!(placement, Placement::Host);
        assert_eq!(stats.h2d_bytes, 0);
    }
}

#[test]
fn live_batch_rows_equal_single_runs() {
    let Some((live, bundle)) = setup() else { return };
    let a = "the cat sat on the mat";
    let b = "an entirely different sentence about gpus";
    let (both, _, _) = embed(&live, &bundle, &[a, b], &EmbedOptions::default(), 4);
    let (only_a, _, _) = embed(&live, &bundle, &[a], &EmbedOptions::default(), 4);
    let (only_b, _, _) = embed(&live, &bundle, &[b], &EmbedOptions::default(), 4);
    assert!(cosine(&both[0], &only_a[0]) > 0.99999);
    assert!(cosine(&both[1], &only_b[0]) > 0.99999);
    assert!(cosine(&both[0], &both[1]) < 0.9, "unrelated sentences should not be near-identical");
}

#[test]
fn live_truncation_policy_is_enforced() {
    let Some((live, bundle)) = setup() else { return };
    let model = live.ctx.load_model(&bundle, &ModelDesc::default()).unwrap();
    let session = model.create_session(&SessionDesc { max_batch: 2, max_seq: 16, ..Default::default() }).unwrap();
    let long = "token ".repeat(100);
    let e = session.write_text(&[&long], &EmbedOptions { truncate: Truncate::None, ..Default::default() }).unwrap_err();
    assert_eq!(e.code(), abi::TURBO_E_CAPACITY);
    let alphabet = "a b c d e f g h i j k l m n o p q r s t u v w x y z";
    session.write_text(&[alphabet], &EmbedOptions { truncate: Truncate::Right, ..Default::default() }).unwrap();
    let r = session.run(&Default::default()).unwrap();
    let mut right = vec![0u8; 384 * 4];
    r.read(0, &mut right).unwrap();
    drop(r);
    session.write_text(&[alphabet], &EmbedOptions { truncate: Truncate::Left, ..Default::default() }).unwrap();
    let r = session.run(&Default::default()).unwrap();
    let mut left = vec![0u8; 384 * 4];
    r.read(0, &mut left).unwrap();
    assert_ne!(left, right, "left and right truncation keep different tokens");
}

#[test]
fn live_pooling_override_follows_the_capability_bit() {
    let Some((live, bundle)) = setup() else { return };
    let text = "pooling override check";
    let (mean, _, _) = embed(&live, &bundle, &[text], &EmbedOptions::default(), 2);
    let model = live.ctx.load_model(&bundle, &ModelDesc::default()).unwrap();
    let session = model.create_session(&SessionDesc { max_batch: 2, max_seq: 32, ..Default::default() }).unwrap();
    let cls = EmbedOptions { pooling: turbo::Pooling::Cls, ..Default::default() };
    if live.has_cap(abi::TURBO_CAP_OPT_POOLING_OVERRIDE) {
        session.write_text(&[text], &cls).expect("the bit is set, so the option is honored");
        let r = session.run(&Default::default()).unwrap();
        let mut bytes = vec![0u8; 384 * 4];
        r.read(0, &mut bytes).unwrap();
        let v: Vec<f32> = bytes.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        assert!(cosine(&v, &mean[0]) < 0.999, "CLS pooling must differ from mean pooling");
    } else {
        let e = session.write_text(&[text], &cls).unwrap_err();
        assert_eq!(e.code(), abi::TURBO_E_UNSUPPORTED_OPTION);
        assert_eq!(e.field(), EmbedOptions::FIELD_POOLING);
    }
}

#[test]
fn live_result_exports_a_native_handle_on_gpus() {
    let Some((live, bundle)) = setup() else { return };
    if !live.gpu() {
        eprintln!("skipping: selected device is not a GPU");
        return;
    }
    let model = live.ctx.load_model(&bundle, &ModelDesc::default()).unwrap();
    let session = model.create_session(&SessionDesc { max_batch: 1, max_seq: 32, ..Default::default() }).unwrap();
    session.write_text(&["export me"], &EmbedOptions::default()).unwrap();
    let r = session.run(&Default::default()).unwrap();
    let buffer = r.buffer(0).unwrap();
    let kind = match live.provider.as_str() {
        "cuda" => turbo::HandleKind::CudaPtr,
        "openvino" => turbo::HandleKind::ClMem,
        other => panic!("no native handle kind known for provider `{other}`"),
    };
    let h = buffer.export(kind).expect("device result exports its native handle");
    assert_eq!(h.kind, kind);
    assert_ne!(h.handle, 0);
    assert_eq!(buffer.export(turbo::HandleKind::HostPtr).unwrap_err().code(), abi::TURBO_E_UNSUPPORTED);
}
