//! Group `capability`, C ABI layer: the capability matrix, `turbo_can_run`,
//! and per-option gating with the documented field index.

use std::ptr;

use turbo_abi::*;
use turbo_capi::*;
use turbo_conformance::c::{self, ssz};
use turbo_conformance::{assert_rc, fixtures, BundleKind, Target};

struct Fixture {
    target: Target,
    _ct: c::CTarget,
    ctx: *mut turbo_context,
    model: *mut turbo_model,
    session: *mut turbo_session,
    info: turbo_model_info,
}

impl Fixture {
    fn new(kind: BundleKind) -> Self {
        let target = Target::from_env();
        let ct = c::CTarget::new(&target);
        let ctx = ct.context();
        let model = ct.model(ctx, kind);
        let session = ct.session(model);
        let mut e = c::err();
        let mut info = turbo_model_info { struct_size: ssz::<turbo_model_info>(), ..unsafe { std::mem::zeroed() } };
        // SAFETY: valid model handle and out pointer.
        assert_rc!(unsafe { turbo_model_get_info(model, &mut info, &mut e) }, TURBO_OK, e);
        Self { target, _ct: ct, ctx, model, session, info }
    }

    fn caps(&self) -> u64 {
        self.target.caps()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        // SAFETY: each handle is released exactly once.
        unsafe {
            turbo_session_release(self.session);
            turbo_model_release(self.model);
            turbo_context_release(self.ctx);
        }
    }
}

/// A mutation of the embed options, paired with the bit that gates it.
type EmbedCase = (u64, u32, fn(&mut turbo_embed_options, &turbo_model_info));

/// A mutation of the generate descriptor, paired with the bit that gates it.
type GenCase<'a> = (u64, u32, &'a dyn Fn(&mut turbo_generate_desc));

fn long_text(words: usize) -> String {
    (0..words).map(|i| format!("w{i} ")).collect::<String>().trim_end().to_string()
}

#[test]
fn capability_embed_options_are_honored_or_rejected_with_their_field() {
    let f = Fixture::new(BundleKind::Embedding);
    let mut e = c::err();
    let over = long_text(f.info.max_seq as usize + 8);
    let texts = [c::text(&over)];

    // (capability bit, 1-based field, option mutator)
    let cases: [EmbedCase; 6] = [
        (TURBO_CAP_OPT_TRUNCATE, 2, |o, _| o.truncate = TURBO_TRUNCATE_LEFT),
        (TURBO_CAP_OPT_MAX_TOKENS, 3, |o, i| o.max_tokens = i.max_seq / 2),
        (TURBO_CAP_OPT_PROMPT_ROLE, 4, |o, _| o.prompt_role = TURBO_PROMPT_QUERY),
        (TURBO_CAP_OPT_NORMALIZE, 5, |o, i| {
            o.normalize = if i.normalize == TURBO_NORMALIZE_L2 { TURBO_NORMALIZE_NONE } else { TURBO_NORMALIZE_L2 }
        }),
        (TURBO_CAP_OPT_POOLING_OVERRIDE, 6, |o, i| {
            o.pooling = if i.pooling == TURBO_POOLING_CLS { TURBO_POOLING_MEAN } else { TURBO_POOLING_CLS }
        }),
        (TURBO_CAP_OPT_OUTPUT_DTYPE, 8, |o, _| o.output_dtype = TURBO_OUTPUT_F16),
    ];
    for (bit, field, mutate) in cases {
        let mut o = c::embed_options();
        mutate(&mut o, &f.info);
        // SAFETY: valid session, one readable text, well-formed options.
        let rc = unsafe { turbo_session_write_text(f.session, texts.as_ptr(), 1, &o, &mut e) };
        if f.caps() & bit != 0 {
            assert_eq!(rc, TURBO_OK, "field {field} is advertised but failed: {}", c::message(&e));
        } else {
            assert_eq!(
                rc,
                TURBO_E_UNSUPPORTED_OPTION,
                "field {field} is not advertised and must be refused, got {}",
                c::status_name(rc)
            );
            assert_eq!(e.field, field, "the refusal must name field {field}: {}", c::message(&e));
        }
    }
}

#[test]
fn capability_output_dim_is_honored_or_rejected_with_its_field() {
    let f = Fixture::new(BundleKind::Embedding);
    let mut e = c::err();
    let text = "hello world";
    let texts = [c::text(text)];
    // The bundle's Matryoshka dimensions come from the contract; the suite
    // reads them through the Rust API because the C ABI does not expose them.
    let model = f.target.model(BundleKind::Embedding);
    let dims = model.bundle().contract().truncate_dims.clone();
    let target_dim = dims.first().copied().unwrap_or(f.info.dim);
    let mut o = c::embed_options();
    o.output_dim = target_dim;
    // SAFETY: valid session and options.
    let rc = unsafe { turbo_session_write_text(f.session, texts.as_ptr(), 1, &o, &mut e) };
    if f.caps() & TURBO_CAP_OPT_OUTPUT_DIM != 0 && !dims.is_empty() {
        assert_eq!(rc, TURBO_OK, "{}", c::message(&e));
        let mut r: *mut turbo_result = ptr::null_mut();
        // SAFETY: valid session and out pointer.
        assert_rc!(unsafe { turbo_session_run(f.session, ptr::null(), &mut r, &mut e) }, TURBO_OK, e);
        let ri = c::result_info(r);
        assert_eq!(ri.dim, target_dim, "output_dim must be reflected in turbo_result_info");
        assert_eq!(ri.bytes, (target_dim as u64) * 4);
        // SAFETY: released once.
        unsafe { turbo_result_release(r) };
    } else if dims.is_empty() && target_dim == f.info.dim {
        assert_eq!(rc, TURBO_OK, "the model's own dimension is never gated: {}", c::message(&e));
    } else {
        assert_eq!(rc, TURBO_E_UNSUPPORTED_OPTION, "got {}", c::status_name(rc));
        assert_eq!(e.field, 7);
    }
}

#[test]
fn capability_rerank_top_n_is_honored_or_rejected_with_its_field() {
    let f = Fixture::new(BundleKind::Reranker);
    let mut e = c::err();
    let query = c::text("alpha beta");
    let docs: Vec<&str> = vec!["alpha beta", "gamma", "alpha beta gamma", "delta"];
    let views: Vec<turbo_text> = docs.iter().map(|d| c::text(d)).collect();
    let mut o = c::rerank_options();
    o.top_n = 2;
    o.return_sorted = 1;
    // SAFETY: valid session; `views` holds `docs.len()` readable views.
    let rc = unsafe { turbo_session_write_pairs(f.session, &query, views.as_ptr(), views.len() as u32, &o, &mut e) };
    if f.caps() & TURBO_CAP_OPT_TOP_N != 0 {
        assert_eq!(rc, TURBO_OK, "{}", c::message(&e));
        let mut r: *mut turbo_result = ptr::null_mut();
        // SAFETY: valid session and out pointer.
        assert_rc!(unsafe { turbo_session_run(f.session, ptr::null(), &mut r, &mut e) }, TURBO_OK, e);
        let ri = c::result_info(r);
        assert_eq!(ri.n_outputs, 2, "a sorted request adds a second output");
        let scores = c::read_f32(r, 0, docs.len());
        let mut sorted = vec![0i32; 2];
        let mut written = 0u64;
        // SAFETY: `sorted` holds 8 writable bytes.
        assert_rc!(
            unsafe { turbo_result_read(r, 1, sorted.as_mut_ptr().cast(), 8, &mut written, &mut e) },
            TURBO_OK,
            e
        );
        assert_eq!(written, 8, "top_n = 2 must produce exactly two indices");
        assert!(scores[sorted[0] as usize] >= scores[sorted[1] as usize]);
        // SAFETY: released once.
        unsafe { turbo_result_release(r) };
    } else {
        assert_eq!(rc, TURBO_E_UNSUPPORTED_OPTION, "got {}", c::status_name(rc));
        assert_eq!(e.field, 4);
    }
    // Booleans are 0 or 1, nothing else.
    let mut bad = c::rerank_options();
    bad.return_sorted = 2;
    // SAFETY: as above.
    assert_rc!(
        unsafe { turbo_session_write_pairs(f.session, &query, views.as_ptr(), views.len() as u32, &bad, &mut e) },
        TURBO_E_INVALID_ARGUMENT,
        e,
        field = 5
    );
    let mut bad = c::rerank_options();
    bad.raw_scores = 7;
    // SAFETY: as above.
    assert_rc!(
        unsafe { turbo_session_write_pairs(f.session, &query, views.as_ptr(), views.len() as u32, &bad, &mut e) },
        TURBO_E_INVALID_ARGUMENT,
        e,
        field = 6
    );
}

#[test]
fn capability_aggregation_is_honored_or_rejected_with_its_field() {
    let f = Fixture::new(BundleKind::TokenClassifier);
    let mut e = c::err();
    let text = "Paris Paris Bob";
    let texts = [c::text(text)];
    let mut o = c::classify_options();
    o.aggregation = TURBO_AGGREGATE_NONE;
    // SAFETY: valid session and one readable text.
    let rc = unsafe { turbo_session_write_text_classify(f.session, texts.as_ptr(), 1, &o, &mut e) };
    if f.caps() & TURBO_CAP_OPT_AGGREGATION != 0 {
        assert_eq!(rc, TURBO_OK, "{}", c::message(&e));
        let mut r: *mut turbo_result = ptr::null_mut();
        // SAFETY: valid session and out pointer.
        assert_rc!(unsafe { turbo_session_run(f.session, ptr::null(), &mut r, &mut e) }, TURBO_OK, e);
        let mut count = 0u32;
        // SAFETY: a zero capacity only asks for the count.
        assert_rc!(unsafe { turbo_result_spans(r, ptr::null_mut(), 0, &mut count, &mut e) }, TURBO_OK, e);
        assert!(count > 0, "NONE aggregation must report per-token spans");
        let mut spans = vec![turbo_span::default(); count as usize];
        // SAFETY: `spans` holds `count` writable entries.
        assert_rc!(unsafe { turbo_result_spans(r, spans.as_mut_ptr(), count, &mut count, &mut e) }, TURBO_OK, e);
        for s in &spans {
            assert!(s.byte_start < s.byte_end && (s.byte_end as usize) <= text.len(), "span {s:?}");
        }
        // SAFETY: released once.
        unsafe { turbo_result_release(r) };
    } else {
        assert_eq!(rc, TURBO_E_UNSUPPORTED_OPTION, "got {}", c::status_name(rc));
        assert_eq!(e.field, 4);
    }
    // `reserved` must be zero.
    let mut reserved = c::classify_options();
    reserved.reserved = 1;
    // SAFETY: as above.
    assert_rc!(
        unsafe { turbo_session_write_text_classify(f.session, texts.as_ptr(), 1, &reserved, &mut e) },
        TURBO_E_INVALID_ARGUMENT,
        e,
        field = 6
    );
}

#[test]
fn capability_generation_options_are_honored_or_rejected_with_their_field() {
    let t = Target::from_env();
    let ct = c::CTarget::new(&t);
    let ctx = ct.context();
    let model = ct.model(ctx, BundleKind::Generative);
    let mut e = c::err();
    let schema = "{\"type\":\"object\"}";
    let tool = "{\"name\":\"x\"}";
    let stop_text = [c::text("zzz-never-matches")];
    let tools = [c::text(tool)];
    let bias = [turbo_logit_bias { token: 3, bias: 1.0 }];

    let cases: [GenCase; 9] = [
        (TURBO_CAP_OPT_GEN_N, 4, &|d| d.n_sequences = 2),
        (TURBO_CAP_OPT_GEN_PENALTIES, 9, &|d| d.repeat_penalty = 1.5),
        (TURBO_CAP_OPT_GEN_PENALTIES, 10, &|d| d.presence_penalty = 0.5),
        (TURBO_CAP_OPT_GEN_PENALTIES, 11, &|d| d.frequency_penalty = 0.5),
        (TURBO_CAP_OPT_GEN_SEED, 12, &|d| {
            d.has_seed = 1;
            d.seed = 42;
        }),
        (TURBO_CAP_OPT_GEN_STOP_STRINGS, 14, &|d| {
            d.n_stop = 1;
            d.stop = stop_text.as_ptr();
        }),
        (TURBO_CAP_OPT_GEN_LOGIT_BIAS, 18, &|d| {
            d.n_logit_bias = 1;
            d.logit_bias = bias.as_ptr();
        }),
        (TURBO_CAP_OPT_GEN_LOGPROBS, 19, &|d| d.logprobs = 2),
        (TURBO_CAP_OPT_GEN_STRUCTURED, 21, &|d| {
            d.structured_kind = TURBO_STRUCTURED_JSON_SCHEMA;
            d.structured = c::text(schema);
        }),
    ];
    for (bit, field, mutate) in cases {
        let mut d = c::generate_desc();
        d.max_new_tokens = 2;
        mutate(&mut d);
        let mut g: *mut turbo_generation = ptr::null_mut();
        // SAFETY: valid model handle; every pointer in `d` outlives the call.
        let rc = unsafe { turbo_generation_create(model, &d, &mut g, &mut e) };
        if t.caps() & bit != 0 {
            assert_eq!(rc, TURBO_OK, "field {field} is advertised but failed: {}", c::message(&e));
            // SAFETY: released once.
            unsafe { turbo_generation_release(g) };
        } else {
            assert_eq!(rc, TURBO_E_UNSUPPORTED_OPTION, "field {field}: got {}", c::status_name(rc));
            assert_eq!(e.field, field, "{}", c::message(&e));
            assert!(g.is_null(), "a refused descriptor must not produce a generation");
        }
    }
    // Tools are the one field whose count and array are separate.
    let mut d = c::generate_desc();
    d.n_tools = 1;
    d.tools = tools.as_ptr();
    let mut g: *mut turbo_generation = ptr::null_mut();
    // SAFETY: `tools` holds one readable view.
    let rc = unsafe { turbo_generation_create(model, &d, &mut g, &mut e) };
    if t.caps() & TURBO_CAP_OPT_GEN_TOOLS != 0 {
        assert_eq!(rc, TURBO_OK, "{}", c::message(&e));
        // SAFETY: released once.
        unsafe { turbo_generation_release(g) };
    } else {
        assert_eq!(rc, TURBO_E_UNSUPPORTED_OPTION);
        assert_eq!(e.field, 24);
    }
    // SAFETY: each handle is released once.
    unsafe {
        turbo_model_release(model);
        turbo_context_release(ctx);
    }
    drop(ct);
}

#[test]
fn capability_matrix_reports_every_cell_and_rejects_unknown_axes() {
    let t = Target::from_env();
    let ct = c::CTarget::new(&t);
    let mut e = c::err();
    let mut cell = turbo_capability {
        struct_size: ssz::<turbo_capability>(),
        status: 99,
        dtype: 0,
        reference_dtype: 0,
        cosine_floor: 0.0,
        max_abs_error: 0.0,
        deterministic: 0,
        reserved: 0,
        notes: [0; 128],
    };
    for task in [
        TURBO_TASK_EMBED,
        TURBO_TASK_RERANK,
        TURBO_TASK_CLASSIFY,
        TURBO_TASK_TOKEN_CLASSIFY,
        TURBO_TASK_GENERATE,
        TURBO_TASK_TOKENIZE,
        TURBO_TASK_RUN,
        TURBO_TASK_CHUNK,
    ] {
        for modality in [TURBO_MODALITY_TEXT, TURBO_MODALITY_AUDIO, TURBO_MODALITY_IMAGE, TURBO_MODALITY_VIDEO] {
            // SAFETY: valid runtime handle and out pointer.
            assert_rc!(
                unsafe { turbo_runtime_capability(ct.rt, ct.device, task, modality, &mut cell, &mut e) },
                TURBO_OK,
                e
            );
            assert!(cell.status <= TURBO_CAP_SUPPORTED, "status {} is not a known constant", cell.status);
            assert!(c::is_nul_terminated(&cell.notes), "notes must be NUL-terminated");
            if cell.status == TURBO_CAP_UNSUPPORTED {
                assert_eq!(cell.dtype, 0, "an unsupported cell reports no dtype");
            }
        }
    }
    // SAFETY: valid runtime handle; the axes are deliberately wrong.
    unsafe {
        assert_rc!(
            turbo_runtime_capability(ct.rt, ct.device, 99, TURBO_MODALITY_TEXT, &mut cell, &mut e),
            TURBO_E_INVALID_ENUM,
            e
        );
        assert_rc!(
            turbo_runtime_capability(ct.rt, ct.device, TURBO_TASK_EMBED, 0, &mut cell, &mut e),
            TURBO_E_INVALID_ENUM,
            e
        );
        assert_rc!(
            turbo_runtime_capability(ct.rt, u32::MAX, TURBO_TASK_EMBED, TURBO_MODALITY_TEXT, &mut cell, &mut e),
            TURBO_E_DEVICE_NOT_FOUND,
            e
        );
        assert_rc!(
            turbo_runtime_capability(ct.rt, ct.device, TURBO_TASK_EMBED, TURBO_MODALITY_TEXT, ptr::null_mut(), &mut e),
            TURBO_E_INVALID_ARGUMENT,
            e
        );
    }
}

#[test]
fn capability_can_run_agrees_with_model_load() {
    let t = Target::from_env();
    let ct = c::CTarget::new(&t);
    let ctx = ct.context();
    let mut e = c::err();
    for kind in BundleKind::ALL {
        let path = ct.bundle(*kind);
        if !std::path::Path::new(&path).is_dir() {
            continue;
        }
        let bundle = turbo::bundle::Bundle::open(std::path::Path::new(&path)).expect("valid bundle");
        let task = bundle.task().as_abi();
        let modality = bundle.modality().as_abi();
        // SAFETY: valid runtime handle; `path` outlives the call.
        let can = unsafe { turbo_can_run(ct.rt, ct.device, c::text(&path), task, modality, &mut e) };
        let mut m: *mut turbo_model = ptr::null_mut();
        let mut e2 = c::err();
        // SAFETY: valid context handle; `path` outlives the call.
        let load = unsafe { turbo_model_load(ctx, c::text(&path), ptr::null(), &mut m, &mut e2) };
        assert_eq!(
            can,
            load,
            "turbo_can_run said {} but turbo_model_load said {} for {path}",
            c::status_name(can),
            c::status_name(load)
        );
        if load == TURBO_OK {
            // SAFETY: released once.
            unsafe { turbo_model_release(m) };
        }
    }
    // An empty path is an argument error, not a bundle error.
    // SAFETY: valid runtime handle.
    unsafe {
        assert_rc!(
            turbo_can_run(ct.rt, ct.device, c::text(""), TURBO_TASK_EMBED, TURBO_MODALITY_TEXT, &mut e),
            TURBO_E_INVALID_ARGUMENT,
            e
        );
        assert_rc!(
            turbo_can_run(
                ct.rt,
                ct.device,
                c::text("/definitely/not/here"),
                TURBO_TASK_EMBED,
                TURBO_MODALITY_TEXT,
                &mut e
            ),
            TURBO_E_BUNDLE_NOT_FOUND,
            e
        );
        assert_rc!(
            turbo_can_run(
                ct.rt,
                ct.device,
                c::text(&ct.bundle(BundleKind::Embedding)),
                99,
                TURBO_MODALITY_TEXT,
                &mut e
            ),
            TURBO_E_INVALID_ENUM,
            e
        );
        turbo_context_release(ctx);
    }
    drop(ct);
}

#[test]
fn capability_unsupported_modality_bundle_is_unsupported_task() {
    let t = Target::from_env();
    let ct = c::CTarget::new(&t);
    let ctx = ct.context();
    let mut e = c::err();
    let mut cell = turbo_capability {
        struct_size: ssz::<turbo_capability>(),
        status: 0,
        dtype: 0,
        reference_dtype: 0,
        cosine_floor: 0.0,
        max_abs_error: 0.0,
        deterministic: 0,
        reserved: 0,
        notes: [0; 128],
    };
    // SAFETY: valid runtime handle and out pointer.
    assert_rc!(
        unsafe {
            turbo_runtime_capability(ct.rt, ct.device, TURBO_TASK_EMBED, TURBO_MODALITY_AUDIO, &mut cell, &mut e)
        },
        TURBO_OK,
        e
    );
    if cell.status != TURBO_CAP_UNSUPPORTED {
        return; // the provider really offers audio
    }
    let scratch = fixtures::copy_of(&t.bundle(BundleKind::Embedding));
    scratch.patch_manifest(|m| m["modality"] = serde_json::Value::String("audio".into()));
    let path = scratch.path().to_string_lossy().into_owned();
    let mut m: *mut turbo_model = ptr::null_mut();
    // SAFETY: valid context handle; `path` outlives the call.
    assert_rc!(
        unsafe { turbo_model_load(ctx, c::text(&path), ptr::null(), &mut m, &mut e) },
        TURBO_E_UNSUPPORTED_TASK,
        e
    );
    assert!(m.is_null());
    // SAFETY: released once.
    unsafe { turbo_context_release(ctx) };
    drop(ct);
}
