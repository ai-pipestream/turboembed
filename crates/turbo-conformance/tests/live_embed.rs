//! Live embedding checks for any provider against a real MiniLM bundle.
//!
//! Selection and skipping are described in `turbo_conformance::live`. The
//! reference vectors in `testdata/reference_embeddings/ort_cuda_minilm_*.json`
//! were produced by ONNX Runtime CUDA in FP32 on the same ONNX export, so an
//! FP32 provider is held to cosine 0.9995 against them; a quantized device is
//! held to the floor its capability cell states (`Live::embed_cosine_floor`)
//! and, like every device, to a ranking gate on the STS pair corpus.

use turbo::abi;
use turbo::{DType, EmbedOptions, ModelDesc, OutputDType, Placement, SessionDesc, Truncate};
use turbo_conformance::live::{bundle, cosine, live, reference_dir, Live};
use turbo_conformance::read_f32;

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
    // A fixed-shape artifact (a HEF) caps the session at its frame length.
    let max_seq = model.info().max_seq.min(256);
    let session = model.create_session(&SessionDesc { max_batch, max_seq, ..Default::default() }).expect("session");
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
    let floor = live.embed_cosine_floor();
    eprintln!("cosine floor for this device: {floor} (compute dtype {:?})", live.embed.dtype);
    for (i, (v, g)) in vecs.iter().zip(&goldens).enumerate() {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "{}: norm {norm}", names[i]);
        let c = cosine(v, &g.vector);
        eprintln!("{}: cosine vs ORT CUDA reference = {c:.6}", names[i]);
        assert!(c > floor, "{}: cosine {c} below the device's floor {floor}", names[i]);
    }
    eprintln!("stats: {stats:?}, output placement {placement:?}");
    // A device that advertises TURBO_CAP_DEVICE_RESULT keeps the result
    // there and moves nothing back inside the run; one that does not (a
    // CPU, or a runtime that hands back host memory) reports HOST.
    if live.has_cap(abi::TURBO_CAP_DEVICE_RESULT) {
        assert_eq!(placement, Placement::Device, "results stay on the device");
        assert_eq!(stats.d2h_bytes, 0, "reads go through result_read, not the run: {stats:?}");
        if live.gpu() {
            assert!(stats.h2d_bytes > 0, "inputs were uploaded: {stats:?}");
        }
    } else {
        assert_eq!(placement, Placement::Host);
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
    if !live.gpu() || !live.has_cap(abi::TURBO_CAP_DEVICE_RESULT) {
        eprintln!("skipping: selected device is not a GPU with device-resident results");
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

#[test]
fn live_output_dim_is_honored_only_for_the_bundle_truncate_dims() {
    let Some((live, dir)) = setup() else { return };
    let model = live.ctx.load_model(&dir, &ModelDesc::default()).unwrap();
    let dim = model.info().dim;
    let allowed = model.bundle().contract().truncate_dims.clone();
    let session = model.create_session(&SessionDesc { max_batch: 1, max_seq: 32, ..Default::default() }).unwrap();
    // A dimension the bundle does not list is never produced: the core
    // rejects it as unsupported when the bit is clear and as an invalid
    // argument when the bit is set, and names field 7 either way.
    let absent = (1..dim).find(|d| !allowed.contains(d)).expect("a dimension the bundle does not list");
    let e = session
        .write_text(&["truncation check"], &EmbedOptions { output_dim: absent, ..Default::default() })
        .unwrap_err();
    let want = if live.has_cap(abi::TURBO_CAP_OPT_OUTPUT_DIM) {
        abi::TURBO_E_INVALID_ARGUMENT
    } else {
        abi::TURBO_E_UNSUPPORTED_OPTION
    };
    assert_eq!(e.code(), want, "output_dim {absent} is not in truncate_dims {allowed:?}: {e}");
    assert_eq!(e.field(), EmbedOptions::FIELD_OUTPUT_DIM, "the rejection names output_dim: {e}");
    // The model's own dimension is always accepted; it is not a truncation.
    session.write_text(&["truncation check"], &EmbedOptions { output_dim: dim, ..Default::default() }).unwrap();
    let r = session.run(&Default::default()).unwrap();
    assert_eq!(r.output(0).unwrap().shape, vec![1, dim as u64], "output_dim == dim is the full vector");
    drop(r);
    // A listed dimension, if the bundle has one, is produced exactly.
    for d in allowed {
        session.write_text(&["truncation check"], &EmbedOptions { output_dim: d, ..Default::default() }).unwrap();
        let r = session.run(&Default::default()).unwrap();
        assert_eq!(r.output(0).unwrap().shape, vec![1, d as u64], "a listed truncate_dim must be produced");
    }
}

#[test]
fn live_output_dtype_follows_the_capability_bit() {
    let Some((live, dir)) = setup() else { return };
    let model = live.ctx.load_model(&dir, &ModelDesc::default()).unwrap();
    let session = model.create_session(&SessionDesc { max_batch: 1, max_seq: 32, ..Default::default() }).unwrap();
    // MODEL and F32 name the same f32 result every provider here produces.
    for dtype in [OutputDType::Model, OutputDType::F32] {
        session
            .write_text(&["dtype check"], &EmbedOptions { output_dtype: dtype, ..Default::default() })
            .unwrap_or_else(|e| panic!("{dtype:?} is the model's own output dtype: {e}"));
        let r = session.run(&Default::default()).unwrap();
        assert_eq!(r.output(0).unwrap().dtype(), DType::F32, "{dtype:?} produces an f32 result");
    }
    for dtype in [OutputDType::F16, OutputDType::I8] {
        let opts = EmbedOptions { output_dtype: dtype, ..Default::default() };
        match session.write_text(&["dtype check"], &opts) {
            Ok(()) => assert!(
                live.has_cap(abi::TURBO_CAP_OPT_OUTPUT_DTYPE),
                "{dtype:?} was honored without TURBO_CAP_OPT_OUTPUT_DTYPE"
            ),
            Err(e) => {
                assert_eq!(e.code(), abi::TURBO_E_UNSUPPORTED_OPTION, "{dtype:?}: {e}");
                assert_eq!(e.field(), EmbedOptions::FIELD_OUTPUT_DTYPE, "the rejection names output_dtype: {e}");
            }
        }
    }
}

#[test]
fn live_two_sessions_on_one_model_produce_the_same_vectors() {
    let Some((live, dir)) = setup() else { return };
    let texts = ["the first sentence", "an unrelated second sentence about gpus"];
    let (expected, _, _) = embed(&live, &dir, &texts, &EmbedOptions::default(), 4);
    let model = live.ctx.load_model(&dir, &ModelDesc::default()).unwrap();
    let a = model.create_session(&SessionDesc { max_batch: 2, max_seq: 64, ..Default::default() }).unwrap();
    let b = model.create_session(&SessionDesc { max_batch: 2, max_seq: 64, ..Default::default() }).unwrap();
    // Interleaved so the second session runs both before and after the first.
    for (label, session) in [("b", &b), ("a", &a), ("b again", &b)] {
        session.write_text(&texts, &EmbedOptions::default()).expect("write");
        let r = session.run(&Default::default()).unwrap();
        let rows = read_f32(&r, 0);
        for (i, want) in expected.iter().enumerate() {
            let got = &rows[i * want.len()..(i + 1) * want.len()];
            let c = cosine(got, want);
            assert!(c > 0.9999, "session {label} row {i}: cosine {c} against the single-session vector");
        }
    }
}

#[test]
fn live_a_batch_of_mixed_lengths_equals_the_single_runs() {
    let Some((live, dir)) = setup() else { return };
    // Eight rows from 3 to 200 tokens (single-piece words plus [CLS] and
    // [SEP]), so the batch is padded to the longest row and every shorter
    // row must still match its own run.
    let lexicon = ["the", "cat", "sat", "on", "a", "mat", "and", "ran"];
    let words = [1usize, 5, 13, 29, 61, 97, 148, 198];
    let texts: Vec<String> =
        words.iter().map(|&n| (0..n).map(|w| lexicon[w % lexicon.len()]).collect::<Vec<_>>().join(" ")).collect();
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let model = live.ctx.load_model(&dir, &ModelDesc::default()).expect("load MiniLM");
    let dim = model.info().dim as usize;
    let max_seq = model.info().max_seq.min(256);
    let session = model.create_session(&SessionDesc { max_batch: 8, max_seq, ..Default::default() }).expect("session");
    session.write_text(&refs, &EmbedOptions::default()).expect("write the batch");
    let r = session.run(&Default::default()).expect("run the batch");
    assert_eq!(r.output(0).unwrap().shape, vec![refs.len() as u64, dim as u64], "one vector per row");
    let batched = read_f32(&r, 0);
    drop(r);
    for (i, text) in refs.iter().enumerate() {
        session.write_text(&[text], &EmbedOptions::default()).expect("write one row");
        let r = session.run(&Default::default()).expect("run one row");
        let single = read_f32(&r, 0);
        let c = cosine(&batched[i * dim..(i + 1) * dim], &single);
        assert!(c > 0.9999, "row {i} ({} words): batched vs single cosine {c}", words[i]);
    }
}

#[test]
fn live_one_session_serves_every_shape_it_is_asked_for_in_any_order() {
    let Some((live, dir)) = setup() else { return };
    let model = live.ctx.load_model(&dir, &ModelDesc::default()).expect("load MiniLM");
    let dim = model.info().dim as usize;
    let session =
        model.create_session(&SessionDesc { max_batch: 4, max_seq: 128, ..Default::default() }).expect("session");
    let long = "sequence ".repeat(100);
    let wide = [long.as_str(), "a second long row", "third", "x"];
    // A wide, long batch first, then a single short row, then the wide
    // batch again: a session that caches anything sized for the first run
    // fails on the second and produces a different answer on the third.
    session.write_text(&wide, &EmbedOptions::default()).expect("write the wide batch");
    let first = read_f32(&session.run(&Default::default()).expect("run the wide batch"), 0);
    session.write_text(&["x"], &EmbedOptions::default()).expect("write one short row");
    let short = read_f32(&session.run(&Default::default()).expect("run one short row after a wider one"), 0);
    assert_eq!(short.len(), dim, "the short run produces one vector");
    session.write_text(&wide[..2], &EmbedOptions::default()).expect("write a narrower batch");
    session.run(&Default::default()).expect("run a narrower batch");
    session.write_text(&wide, &EmbedOptions::default()).expect("write the wide batch again");
    let again = read_f32(&session.run(&Default::default()).expect("run the wide batch again"), 0);
    assert_eq!(first, again, "the same input must give the same vectors whatever shapes ran in between");
    let c = cosine(&first[3 * dim..], &short);
    assert!(c > 0.9999, "row 3 of the batch and the same text run alone differ: cosine {c}");
}

#[test]
fn live_a_held_result_makes_a_run_on_another_thread_busy() {
    let Some((live, dir)) = setup() else { return };
    let model = live.ctx.load_model(&dir, &ModelDesc::default()).unwrap();
    let session = model.create_session(&SessionDesc { max_batch: 1, max_seq: 32, ..Default::default() }).unwrap();
    session.write_text(&["busy check"], &EmbedOptions::default()).unwrap();
    // The lease is held for the whole of the other thread's attempt, so the
    // outcome is decided by the contract, not by timing.
    let result = session.run(&Default::default()).unwrap();
    let barrier = std::sync::Barrier::new(2);
    std::thread::scope(|scope| {
        let session = &session;
        let barrier = &barrier;
        let other = scope.spawn(move || {
            barrier.wait();
            let run = session.run(&Default::default());
            let write = session.write_text(&["x"], &EmbedOptions::default());
            (run.err().map(|e| e.code()), write.err().map(|e| e.code()))
        });
        barrier.wait();
        let (run, write) = other.join().expect("the other thread finished");
        assert_eq!(run, Some(abi::TURBO_E_BUSY), "a run while a result is leased must be TURBO_E_BUSY");
        assert_eq!(write, Some(abi::TURBO_E_BUSY), "a write while a result is leased must be TURBO_E_BUSY");
    });
    drop(result);
    session.run(&Default::default()).expect("the session recovers once the lease is returned");
}

#[test]
fn live_one_session_from_two_threads_is_busy_or_correct_never_wrong() {
    let Some((live, dir)) = setup() else { return };
    let text = "one session, two threads";
    let (expected, _, _) = embed(&live, &dir, &[text], &EmbedOptions::default(), 1);
    let model = live.ctx.load_model(&dir, &ModelDesc::default()).unwrap();
    let session = model.create_session(&SessionDesc { max_batch: 1, max_seq: 32, ..Default::default() }).unwrap();
    let barrier = std::sync::Barrier::new(2);
    let outcomes: Vec<(usize, usize)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..2)
            .map(|_| {
                let session = &session;
                let barrier = &barrier;
                let expected = &expected[0];
                scope.spawn(move || {
                    barrier.wait();
                    let (mut ok, mut busy) = (0usize, 0usize);
                    for _ in 0..20 {
                        match session.write_text(&[text], &EmbedOptions::default()) {
                            Ok(()) => {}
                            Err(e) => {
                                assert_eq!(e.code(), abi::TURBO_E_BUSY, "a concurrent write is BUSY or fine: {e}");
                                busy += 1;
                                continue;
                            }
                        }
                        match session.run(&Default::default()) {
                            Ok(r) => {
                                let c = cosine(&read_f32(&r, 0), expected);
                                assert!(c > 0.9999, "a run that succeeded returned a wrong vector (cosine {c})");
                                ok += 1;
                            }
                            Err(e) => {
                                assert_eq!(e.code(), abi::TURBO_E_BUSY, "a concurrent run is BUSY or fine: {e}");
                                busy += 1;
                            }
                        }
                    }
                    (ok, busy)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("thread")).collect()
    });
    let ok: usize = outcomes.iter().map(|o| o.0).sum();
    let busy: usize = outcomes.iter().map(|o| o.1).sum();
    eprintln!("two threads on one session: {ok} completed, {busy} rejected with TURBO_E_BUSY");
    assert_eq!(ok + busy, 40, "every attempt either completed or was rejected");
    assert!(ok > 0, "the session must still serve the thread that holds it");
}

#[test]
fn live_host_pointer_import_follows_the_capability_bit() {
    let Some(live) = live() else { return };
    let mut mine: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let ptr = mine.as_mut_ptr();
    let desc = turbo::BufferDesc::packed(Placement::Host, turbo::DType::F32, &[mine.len() as u64]).unwrap();
    let handle = turbo::NativeHandle { kind: turbo::HandleKind::HostPtr, handle: ptr as u64, aux: 0, offset: 0 };
    if !live.has_cap(abi::TURBO_CAP_HOST_PTR_IMPORT) {
        let e = live.ctx.import(&desc, &handle).unwrap_err();
        assert_eq!(e.code(), abi::TURBO_E_UNSUPPORTED, "the bit is clear, so the import must be refused");
        return;
    }
    let buffer = live.ctx.import(&desc, &handle).expect("the bit is set, so the import is honored");
    assert_eq!(buffer.host_ptr().expect("host placement").as_ptr() as u64, handle.handle, "import wraps the pointer");
    // Nothing was copied: the caller still owns the memory, and a write
    // through it is what a read of the buffer returns.
    // SAFETY: `mine` outlives `buffer` and nothing else writes to it here.
    unsafe { ptr.add(2).write(42.0) };
    let mut back = vec![0u8; desc.bytes as usize];
    buffer.read_to_host(&mut back).unwrap();
    let floats: Vec<f32> = back.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    assert_eq!(floats, vec![1.0, 2.0, 42.0, 4.0]);
    drop(buffer);
    assert_eq!(mine[2], 42.0, "releasing an imported buffer must not free the caller's memory");
}

#[derive(serde::Deserialize)]
struct StsPair {
    score: f32,
    text_a: String,
    text_b: String,
}

/// Ranks of `values` (1-based, ties averaged).
fn ranks(values: &[f32]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap());
    let mut out = vec![0.0; values.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && values[idx[j + 1]] == values[idx[i]] {
            j += 1;
        }
        let rank = (i + j) as f64 / 2.0 + 1.0;
        for &k in &idx[i..=j] {
            out[k] = rank;
        }
        i = j + 1;
    }
    out
}

fn spearman(a: &[f32], b: &[f32]) -> f64 {
    let (ra, rb) = (ranks(a), ranks(b));
    let n = ra.len() as f64;
    let (ma, mb) = (ra.iter().sum::<f64>() / n, rb.iter().sum::<f64>() / n);
    let cov: f64 = ra.iter().zip(&rb).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let va: f64 = ra.iter().map(|x| (x - ma).powi(2)).sum();
    let vb: f64 = rb.iter().map(|y| (y - mb).powi(2)).sum();
    cov / (va * vb).sqrt()
}

/// The ranking statistics above decide whether a device passes or fails the
/// STS gate below, so they are pinned here with hand-checked cases. These
/// cases need no hardware and run in every `cargo test`, unlike the gate.
#[test]
fn ranks_are_one_based_and_average_their_ties() {
    assert_eq!(ranks(&[]), Vec::<f64>::new(), "an empty sample has no ranks");
    assert_eq!(ranks(&[42.0]), vec![1.0], "ranks are 1-based, not 0-based");
    assert_eq!(ranks(&[10.0, 20.0, 30.0]), vec![1.0, 2.0, 3.0], "ranks follow the values, not the positions");
    assert_eq!(ranks(&[30.0, 20.0, 10.0]), vec![3.0, 2.0, 1.0]);
    assert_eq!(ranks(&[5.0, 5.0]), vec![1.5, 1.5], "a two-way tie takes the mean of ranks 1 and 2");
    assert_eq!(ranks(&[1.0, 2.0, 2.0, 3.0]), vec![1.0, 2.5, 2.5, 4.0], "a tie must not shift the ranks after it");
    assert_eq!(ranks(&[7.0, 7.0, 7.0]), vec![2.0, 2.0, 2.0], "an all-tie sample is the mean rank throughout");
    assert_eq!(ranks(&[2.0, 1.0, 2.0, 1.0]), vec![3.5, 1.5, 3.5, 1.5], "ties need not be adjacent in the input");
    // Averaged ties keep the rank sum at n(n+1)/2, which is what makes the
    // correlation comparable across samples with different tie counts.
    for values in [vec![3.0f32, 1.0, 2.0], vec![1.0, 1.0, 2.0, 2.0, 2.0], vec![0.5; 7]] {
        let n = values.len() as f64;
        let sum: f64 = ranks(&values).iter().sum();
        assert!((sum - n * (n + 1.0) / 2.0).abs() < 1e-9, "rank sum of {values:?} is {sum}, not n(n+1)/2");
    }
}

#[test]
fn spearman_matches_the_rank_difference_formula() {
    let ascending = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let descending = [5.0f32, 4.0, 3.0, 2.0, 1.0];
    assert!((spearman(&ascending, &ascending) - 1.0).abs() < 1e-12, "an identical ranking is +1");
    let shifted = [10.0f32, 20.0, 30.0, 40.0, 50.0];
    assert!((spearman(&ascending, &shifted) - 1.0).abs() < 1e-12, "only the order matters, not the scale");
    assert!((spearman(&ascending, &descending) + 1.0).abs() < 1e-12, "a reversed ranking is -1");

    // Tie-free case, against 1 - 6*sum(d^2) / (n * (n^2 - 1)): the rank
    // differences are -1, 1, -1, 1, 0, so rho = 1 - 24/120 = 0.8.
    let b = [2.0f32, 1.0, 4.0, 3.0, 5.0];
    assert!((spearman(&ascending, &b) - 0.8).abs() < 1e-12, "expected 0.8, got {}", spearman(&ascending, &b));

    // Tied case, worked by hand: ranks [1,2,3,4,5] against [1,2,3.5,5,3.5]
    // give covariance 8 and variances 10 and 9.5, so rho = 8 / sqrt(95).
    let tied = [5.0f32, 6.0, 7.0, 8.0, 7.0];
    let expected = 8.0 / 95.0f64.sqrt();
    let got = spearman(&ascending, &tied);
    assert!((got - expected).abs() < 1e-12, "expected {expected}, got {got}");

    // The gate reads one number, so the statistic must not depend on which
    // side the human scores are passed on.
    assert!((got - spearman(&tied, &ascending)).abs() < 1e-12, "spearman is symmetric in its arguments");
}

/// Ranking gate on the committed STS pair corpus: the Spearman correlation
/// between the device's pair cosines and the human scores must stay above
/// 0.85. An FP32 MiniLM scores about 0.94 here; the INT8 Hailo HEF about
/// 0.94 as well, which is why absolute cosine (above) and ranking are
/// gated separately.
#[test]
fn live_ranking_on_sts_pairs_holds() {
    let Some((live, bundle)) = setup() else { return };
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/corpus/sts-pairs.jsonl");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let pairs: Vec<StsPair> =
        text.lines().filter(|l| !l.trim().is_empty()).map(|l| serde_json::from_str(l).unwrap()).collect();
    assert!(pairs.len() >= 50, "corpus has {} pairs", pairs.len());
    let model = live.ctx.load_model(&bundle, &ModelDesc::default()).unwrap();
    let max_seq = model.info().max_seq.min(128);
    let session = model.create_session(&SessionDesc { max_batch: 16, max_seq, ..Default::default() }).unwrap();
    let embed_all = |texts: Vec<&str>| -> Vec<Vec<f32>> {
        let mut out = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(16) {
            session.write_text(chunk, &EmbedOptions::default()).unwrap();
            let r = session.run(&Default::default()).unwrap();
            let mut bytes = vec![0u8; r.output(0).unwrap().logical_bytes().unwrap() as usize];
            r.read(0, &mut bytes).unwrap();
            let floats: Vec<f32> = bytes.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            out.extend(floats.chunks(384).map(|c| c.to_vec()));
        }
        out
    };
    let a = embed_all(pairs.iter().map(|p| p.text_a.as_str()).collect());
    let b = embed_all(pairs.iter().map(|p| p.text_b.as_str()).collect());
    let cosines: Vec<f32> = a.iter().zip(&b).map(|(x, y)| cosine(x, y)).collect();
    let scores: Vec<f32> = pairs.iter().map(|p| p.score).collect();
    let rho = spearman(&cosines, &scores);
    eprintln!("Spearman(cosine, score) over {} STS pairs = {rho:.4}", pairs.len());
    assert!(rho > 0.85, "ranking gate: Spearman {rho:.4} <= 0.85");
}
