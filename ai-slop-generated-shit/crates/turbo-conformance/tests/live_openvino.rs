//! Live checks that are specific to the OpenVINO provider.
//!
//! Selection and skipping follow `turbo_conformance::live`: every test here
//! returns, printing the reason, unless `TURBO_LIVE_PROVIDER` is `openvino`
//! and `TURBO_LIVE_LIB` names the provider library. The embedding tests also
//! need `TURBO_LIVE_BUNDLE` (an `all-MiniLM-L6-v2` bundle with an `onnx` or
//! `openvino_ir` artifact).
//!
//! These cover what the provider-agnostic files cannot:
//!
//! - the two OpenVINO devices differ in run-to-run reproducibility, so the
//!   suite checks each one against the `TURBO_CAP_DETERMINISTIC` bit it
//!   reports rather than assuming either answer (`live_embed.rs` compares
//!   vectors for exact equality, which only a deterministic device can pass);
//! - `caps_of`/`offers` in `providers/openvino/src/provider.cpp` give a
//!   different capability set per device kind, including the NPU devices
//!   OpenVINO lists but this provider does not qualify;
//! - the limit and bad-input paths the provider rejects before it reaches
//!   OpenVINO, which the mock-bundle groups of the suite cannot reach for a
//!   provider that serves only real bundles.
//!
//! Recorded on an x86_64 host with an Intel Arc B70 (Battlemage) as ordinal
//! 0 and an AMD Ryzen 9 9950X CPU as ordinal 1, OpenVINO 2026.3.1, driver
//! 26.05.037020: the GPU device is not bit-reproducible and the CPU device is.

use turbo::abi;
use turbo::{
    CapStatus, Context, ContextDesc, DeviceKind, DeviceSelector, EmbedOptions, Modality, ModelDesc, PromptRole,
    RuntimeDesc, SelectPolicy, SessionDesc, Task, TokenBatch,
};
use turbo_conformance::live::{live, Live};
use turbo_conformance::read_f32;

/// Largest absolute difference between two runs of the same input that is
/// still floating-point reduction order rather than a wrong answer. MiniLM
/// vectors are L2 normalized, so every component is within -1..1 and this is
/// roughly sixteen times the FP32 spacing at 0.1.
const REPRO_TOLERANCE: f32 = 2e-6;

/// Skip unless the live provider is `openvino`.
fn openvino() -> Option<Live> {
    let live = live()?;
    if live.provider != "openvino" {
        println!("not applicable: TURBO_LIVE_PROVIDER is `{}`, not `openvino`", live.provider);
        return None;
    }
    Some(live)
}

/// The bundle directory `var` names, for a task the device under test
/// offers. A device whose capability cell does not offer the task prints
/// `not applicable` and the case returns; a device that does offer it with
/// no bundle configured is a configuration error and panics naming the
/// variable, so a live run never skips a case it could have run (the rule
/// `Target::offered` applies in `crates/turbo-conformance/src/lib.rs`).
fn bundle_for(live: &Live, task: Task, var: &str) -> Option<std::path::PathBuf> {
    let cell = live
        .ctx
        .runtime()
        .capability(live.ctx.device_index(), task, Modality::Text)
        .unwrap_or_else(|e| panic!("{task:?} x TEXT capability of `{}`: {e}", live.device.name));
    if matches!(cell.status, CapStatus::Unsupported | CapStatus::Planned) {
        println!(
            "not applicable: {} device {} (`{}`) does not offer {task:?} for Text (capability {:?})",
            live.provider, live.device.ordinal, live.device.name, cell.status
        );
        return None;
    }
    match std::env::var(var) {
        Ok(v) if !v.is_empty() => Some(std::path::PathBuf::from(v)),
        _ => panic!(
            "{} device {} (`{}`) offers {task:?} but {var} is not set; point it at a bundle of that kind",
            live.provider, live.device.ordinal, live.device.name
        ),
    }
}

fn setup() -> Option<(Live, std::path::PathBuf)> {
    let live = openvino()?;
    let dir = bundle_for(&live, Task::Embed, "TURBO_LIVE_BUNDLE")?;
    Some((live, dir))
}

fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "compared runs must have the same length");
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

/// The provider says whether a device repeats itself bit for bit. Hold it to
/// that: a device claiming `TURBO_CAP_DETERMINISTIC` must return identical
/// bits for an identical batch on every repeat, and a device that does not
/// claim it must both stay within floating-point reduction noise and
/// actually vary, because a device that never varies is under-reporting and
/// the bit belongs back in `caps_of`.
#[test]
fn live_openvino_repeats_back_the_determinism_claim() {
    let Some((live, dir)) = setup() else { return };
    let claims = live.has_cap(abi::TURBO_CAP_DETERMINISTIC);
    let cell_says = live.embed.deterministic;
    assert_eq!(
        claims, cell_says,
        "device `{}` reports TURBO_CAP_DETERMINISTIC={claims} but its EMBED x TEXT cell says deterministic={cell_says}",
        live.device.name
    );

    let model = live.ctx.load_model(&dir, &ModelDesc::default()).expect("load MiniLM");
    let session =
        model.create_session(&SessionDesc { max_batch: 4, max_seq: 64, ..Default::default() }).expect("session");
    let texts = ["a short sentence", "another one that is a little longer", "third", "x"];
    let mut first: Option<Vec<f32>> = None;
    let mut worst = 0.0f32;
    let mut varied = false;
    for i in 0..20 {
        session.write_text(&texts, &EmbedOptions::default()).expect("write_text");
        let v = read_f32(&session.run(&Default::default()).expect("run"), 0);
        match &first {
            None => first = Some(v),
            Some(f) => {
                if claims {
                    assert_eq!(
                        *f,
                        v,
                        "device `{}` claims TURBO_CAP_DETERMINISTIC but repeat {i} of the same batch differs by {:e}",
                        live.device.name,
                        maxdiff(f, &v)
                    );
                }
                worst = worst.max(maxdiff(f, &v));
                varied |= *f != v;
            }
        }
    }
    eprintln!(
        "device `{}` claims deterministic={claims}; worst repeat difference over 20 runs {worst:e}",
        live.device.name
    );
    if !claims {
        assert!(
            worst <= REPRO_TOLERANCE,
            "device `{}` repeats the same batch to only {worst:e}, past the {REPRO_TOLERANCE:e} that reduction order \
             explains; the vectors are wrong, not merely unreproducible",
            live.device.name
        );
        assert!(
            varied,
            "device `{}` does not claim TURBO_CAP_DETERMINISTIC, yet 20 repeats of one batch were bit-identical; \
             re-measure it and set the bit in caps_of (providers/openvino/src/provider.cpp) if it now holds",
            live.device.name
        );
    }
}

/// One session asked for a wide batch, then a single row, then the wide
/// batch again must answer the wide batch the same way both times. A
/// provider that sized anything to the first request, or that let a narrower
/// run leave stale columns behind, fails here. This is `live_embed.rs`'s
/// shape-order case held to the reproducibility the device claims, so it
/// still runs on a device that is not bit-reproducible.
#[test]
fn live_openvino_shape_order_does_not_change_the_vectors() {
    let Some((live, dir)) = setup() else { return };
    let model = live.ctx.load_model(&dir, &ModelDesc::default()).expect("load MiniLM");
    let dim = model.info().dim as usize;
    let session =
        model.create_session(&SessionDesc { max_batch: 4, max_seq: 128, ..Default::default() }).expect("session");
    let long = "sequence ".repeat(100);
    let wide = [long.as_str(), "a second long row", "third", "x"];
    let run = |texts: &[&str]| -> Vec<f32> {
        session.write_text(texts, &EmbedOptions::default()).expect("write_text");
        read_f32(&session.run(&Default::default()).expect("run"), 0)
    };

    let first = run(&wide);
    assert_eq!(first.len(), wide.len() * dim, "the wide batch produces one vector per row");
    let short = run(&["x"]);
    assert_eq!(short.len(), dim, "the single row produces one vector");
    run(&wide[..2]);
    let again = run(&wide);

    // The floor the device itself repeats to, measured on this session, so a
    // shape-order bug is separated from the device's own run-to-run noise.
    let repeat = run(&wide);
    let noise = maxdiff(&again, &repeat);
    let after_shapes = maxdiff(&first, &again);
    eprintln!(
        "device `{}`: same shape back to back {noise:e}, wide batch either side of other shapes {after_shapes:e}",
        live.device.name
    );
    if live.has_cap(abi::TURBO_CAP_DETERMINISTIC) {
        assert_eq!(first, again, "a deterministic device must answer the wide batch identically both times");
    } else {
        assert!(
            after_shapes <= REPRO_TOLERANCE,
            "the wide batch answered differently ({after_shapes:e}) after a 1-row and a 2-row run went through the \
             same session; the device's own repeat noise is {noise:e}"
        );
    }

    // Row 3 of the batch is the same text as the single run, so the batch
    // must not be pooling across rows.
    let c = turbo_conformance::live::cosine(&first[3 * dim..4 * dim], &short);
    assert!(c > 0.9999, "row 3 of the wide batch and the same text alone differ: cosine {c}");
}

/// Every device the provider lists carries the capability set its kind
/// allows, and the `EMBED x TEXT` cell agrees with the device bits. The NPU
/// arm holds whatever OpenVINO lists on the machine under test: this
/// provider lists NPUs so a caller can see them, offers them no task, and
/// reports `caps = 0` for them (`providers/openvino/src/provider.cpp`).
#[test]
fn live_openvino_device_capabilities_follow_the_device_kind() {
    let Some(lib) = std::env::var("TURBO_LIVE_LIB").ok().filter(|v| !v.is_empty()) else {
        println!("not applicable: TURBO_LIVE_LIB is not set");
        return;
    };
    if std::env::var("TURBO_LIVE_PROVIDER").as_deref() != Ok("openvino") {
        println!("not applicable: TURBO_LIVE_PROVIDER is not `openvino`");
        return;
    }
    let rt = turbo::create_runtime(RuntimeDesc { provider_paths: vec![lib], ..Default::default() })
        .unwrap_or_else(|e| panic!("load the openvino provider: {e}"));
    assert!(rt.failures().is_empty(), "provider failures: {:?}", rt.failures());

    let devices = rt.devices();
    let mine: Vec<(u32, &turbo::DeviceEntry)> = devices
        .iter()
        .enumerate()
        .filter(|(_, d)| d.info.provider_id == "openvino")
        .map(|(i, d)| (i as u32, d))
        .collect();
    assert!(!mine.is_empty(), "the openvino provider enumerated no devices");

    let mut kinds = Vec::new();
    for (index, d) in &mine {
        let caps = d.info.caps;
        let cell = rt.capability(*index, Task::Embed, Modality::Text).expect("EMBED x TEXT capability");
        kinds.push(d.info.kind);
        eprintln!("ordinal {} kind {:?} caps {caps:#x} status {:?}", d.info.ordinal, d.info.kind, cell.status);
        match d.info.kind {
            DeviceKind::Npu => {
                assert_eq!(
                    caps, 0,
                    "the NPU device `{}` is listed but not qualified, so it reports no bits",
                    d.info.name
                );
                assert_eq!(
                    cell.status,
                    CapStatus::Unsupported,
                    "the NPU device `{}` offers no task, so its EMBED x TEXT cell is UNSUPPORTED",
                    d.info.name
                );
                assert!(!cell.deterministic, "an unsupported cell claims nothing");
            }
            kind => {
                assert_ne!(
                    cell.status,
                    CapStatus::Unsupported,
                    "the openvino provider offers EMBED x TEXT on every non-NPU device, including `{}`",
                    d.info.name
                );
                assert_eq!(
                    caps & abi::TURBO_CAP_HOST_PTR_IMPORT,
                    abi::TURBO_CAP_HOST_PTR_IMPORT,
                    "host-pointer import is implemented for every device this provider serves, but `{}` does not \
                     claim it",
                    d.info.name
                );
                let device_result = caps & abi::TURBO_CAP_DEVICE_RESULT != 0;
                assert_eq!(
                    device_result,
                    kind != DeviceKind::Cpu,
                    "only a GPU keeps results device-resident; `{}` is {kind:?} and claims DEVICE_RESULT={device_result}",
                    d.info.name
                );
                assert_eq!(
                    caps & abi::TURBO_CAP_DETERMINISTIC != 0,
                    cell.deterministic,
                    "`{}` must report the same determinism in its device bits and in its capability cell",
                    d.info.name
                );
                // Pooling and normalization always follow the bundle
                // contract here, so neither override bit may be set.
                assert_eq!(
                    caps & (abi::TURBO_CAP_OPT_NORMALIZE | abi::TURBO_CAP_OPT_POOLING_OVERRIDE),
                    0,
                    "`{}` follows the bundle contract for pooling and normalization, so neither override bit is set",
                    d.info.name
                );
            }
        }
    }
    // An ordinal past the end of the table is an error, not the last device.
    let past = mine.len() as u32;
    let err = rt
        .select(&DeviceSelector {
            policy: SelectPolicy::Explicit,
            provider_id: "openvino".into(),
            ordinal: past,
            ..Default::default()
        })
        .expect_err("ordinal past the last openvino device must fail");
    eprintln!("ordinal {past}: {err}");
    assert_eq!(err.code(), abi::TURBO_E_DEVICE_NOT_FOUND, "an absent ordinal is not found, never a fallback: {err}");
    assert!(kinds.contains(&DeviceKind::Cpu), "the openvino provider always lists the CPU plugin");
}

/// The limits and the malformed inputs the provider refuses before it hands
/// anything to OpenVINO. Each one names the field or the offending value, so
/// the check is on the code and the message, not merely that something
/// failed.
#[test]
fn live_openvino_limits_and_bad_inputs_are_refused() {
    let Some((live, dir)) = setup() else { return };
    let model = live.ctx.load_model(&dir, &ModelDesc::default()).expect("load MiniLM");
    let info = model.info();
    let (max_batch, max_seq) = (info.max_batch, info.max_seq);
    assert!(max_batch > 0 && max_seq > 0, "the bundle contract states both limits");

    // A session wider or longer than the model allows, naming the field.
    let e = model
        .create_session(&SessionDesc { max_batch: 1, max_seq: max_seq + 1, ..Default::default() })
        .expect_err("a session longer than the model must be refused");
    assert_eq!(e.code(), abi::TURBO_E_CAPACITY, "max_seq over the model's: {e}");
    assert_eq!(e.field(), 3, "max_seq is field 3 of turbo_session_desc: {e}");
    let e = model
        .create_session(&SessionDesc { max_batch: max_batch + 1, max_seq: 32, ..Default::default() })
        .expect_err("a session wider than the model must be refused");
    assert_eq!(e.code(), abi::TURBO_E_CAPACITY, "max_batch over the model's: {e}");
    assert_eq!(e.field(), 2, "max_batch is field 2 of turbo_session_desc: {e}");

    let session =
        model.create_session(&SessionDesc { max_batch: 2, max_seq: 32, ..Default::default() }).expect("session");

    // More rows than the session was created for.
    let e = session
        .write_text(&["a", "b", "c"], &EmbedOptions::default())
        .expect_err("more rows than max_batch must be refused");
    assert_eq!(e.code(), abi::TURBO_E_CAPACITY, "three rows into a max_batch of 2: {e}");

    // A token id outside the tokenizer's vocabulary, either side of it. The
    // provider checks before the embedding gather, so this is an error and
    // not an out-of-bounds read inside OpenVINO.
    let row = |id: i32| -> Vec<i32> {
        let mut v = vec![0i32; 32];
        v[0] = 101;
        v[1] = id;
        v[2] = 102;
        v
    };
    let mut mask = vec![0i32; 32];
    mask[0] = 1;
    mask[1] = 1;
    mask[2] = 1;
    for bad in [i32::MAX, 7_000_000, -5] {
        let ids = row(bad);
        let batch = TokenBatch { batch: 1, seq: 32, row_stride: 32, ids: &ids, mask: &mask, types: None };
        let e = session.write_tokens(&batch).expect_err("a token id outside the vocabulary must be refused");
        assert_eq!(e.code(), abi::TURBO_E_INVALID_ARGUMENT, "token id {bad}: {e}");
        let msg = e.to_string();
        assert!(msg.contains(&bad.to_string()), "the message names the offending id {bad}: {msg}");
        assert!(msg.contains("row 0 column 1"), "the message names where the id was: {msg}");
    }

    // A prompt role the bundle declares no prefix for. The provider reports
    // OPT_PROMPT_ROLE, so the role is honored when the bundle has a prefix
    // and refused naming the field when it does not; it is never dropped.
    assert!(live.has_cap(abi::TURBO_CAP_OPT_PROMPT_ROLE), "the openvino provider reports TURBO_CAP_OPT_PROMPT_ROLE");
    let opts = EmbedOptions { prompt_role: PromptRole::Query, ..Default::default() };
    match session.write_text(&["hello"], &opts) {
        Err(e) => {
            assert_eq!(e.code(), abi::TURBO_E_INVALID_ARGUMENT, "prompt_role with no prefix: {e}");
            assert_eq!(e.field(), 4, "prompt_role is field 4 of turbo_embed_options: {e}");
            assert!(e.to_string().contains("query"), "the message names the role asked for: {e}");
        }
        Ok(()) => {
            // Honored: the bundle declares a query prefix, so the prefixed
            // text must not embed to the same vector as the bare text.
            let with_prefix = read_f32(&session.run(&Default::default()).expect("run"), 0);
            session.write_text(&["hello"], &EmbedOptions::default()).expect("write_text");
            let bare = read_f32(&session.run(&Default::default()).expect("run"), 0);
            assert_ne!(with_prefix, bare, "a honored query prefix changes the vector");
        }
    }
}

/// A second context on the same device runs its own sessions while the first
/// is alive. The suite's `threading` group covers two sessions inside one
/// context; this covers two contexts, which for this provider means two
/// `ov::Core` inference requests against the same compiled model cache.
#[test]
fn live_openvino_two_contexts_on_one_device_agree() {
    let Some((live, dir)) = setup() else { return };
    let text = ["contexts must not interfere"];
    let model_a = live.ctx.load_model(&dir, &ModelDesc::default()).expect("load MiniLM");
    let session_a =
        model_a.create_session(&SessionDesc { max_batch: 1, max_seq: 32, ..Default::default() }).expect("session");

    let ctx_b = Context::create(live.ctx.runtime().clone(), live.ctx.device_index(), &ContextDesc::default())
        .expect("a second context on the same device");
    let model_b = ctx_b.load_model(&dir, &ModelDesc::default()).expect("load MiniLM into the second context");
    let session_b =
        model_b.create_session(&SessionDesc { max_batch: 1, max_seq: 32, ..Default::default() }).expect("session");

    session_a.write_text(&text, &EmbedOptions::default()).expect("write_text");
    session_b.write_text(&text, &EmbedOptions::default()).expect("write_text");
    let a = read_f32(&session_a.run(&Default::default()).expect("run in the first context"), 0);
    let b = read_f32(&session_b.run(&Default::default()).expect("run in the second context"), 0);
    let d = maxdiff(&a, &b);
    eprintln!("two contexts on `{}` differ by {d:e}", live.device.name);
    if live.has_cap(abi::TURBO_CAP_DETERMINISTIC) {
        assert_eq!(a, b, "two contexts on a deterministic device must give the same vector");
    } else {
        assert!(d <= REPRO_TOLERANCE, "two contexts on one device disagree by {d:e}, past reduction order");
    }
}
