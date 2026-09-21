//! Group `capability`, Rust layer: honest capabilities (PLAN.md principle 2).
//!
//! Every `TURBO_CAP_OPT_*` bit the device reports must have an observable
//! effect; every bit it does not report must make the option fail with
//! `TURBO_E_UNSUPPORTED_OPTION` naming the documented 1-based field index.
//! Each case runs both branches so the suite is the same on any provider.

use turbo::abi::*;
use turbo::provider::{ClassifyOptions, EmbedOptions, GenerateDesc, Message, RerankOptions, RunOptions, SessionDesc};
use turbo::types::{
    Aggregation, Modality, Normalize, OutputDType, Pooling, PromptRole, StructuredKind, Task, Truncate,
};
use turbo_conformance::{assert_err, fixtures, norm, read_f32, read_i32, BundleKind, Target};

fn long_text(words: usize) -> String {
    (0..words).map(|i| format!("w{i} ")).collect::<String>().trim_end().to_string()
}

fn embed(t: &Target, text: &str, opts: &EmbedOptions) -> Vec<f32> {
    let (_m, session) = t.session(BundleKind::Embedding);
    session.write_text(&[text], opts).expect("write");
    read_f32(&session.run(&RunOptions::default()).expect("run"), 0)
}

#[test]
fn capability_truncate_is_honored_or_rejected() {
    let t = Target::from_env();
    let (model, session) = t.session(BundleKind::Embedding);
    let over = long_text(model.info().max_seq as usize + 8);
    let right = EmbedOptions { truncate: Truncate::Right, ..Default::default() };
    let left = EmbedOptions { truncate: Truncate::Left, ..Default::default() };
    if t.has(TURBO_CAP_OPT_TRUNCATE) {
        session.write_text(&[&over], &right).expect("truncate RIGHT");
        let r = read_f32(&session.run(&RunOptions::default()).expect("run"), 0);
        session.write_text(&[&over], &left).expect("truncate LEFT");
        let l = read_f32(&session.run(&RunOptions::default()).expect("run"), 0);
        assert_ne!(r, l, "LEFT and RIGHT truncation of an over-long input must differ");
        // NONE means the over-long input is an error, never a silent cut.
        let none = EmbedOptions { truncate: Truncate::None, ..Default::default() };
        assert_err!(session.write_text(&[&over], &none), TURBO_E_CAPACITY);
        // A short input is unaffected by the policy.
        session.write_text(&["short input"], &none).expect("NONE on an input that fits");
    } else {
        assert_err!(session.write_text(&[&over], &right), TURBO_E_UNSUPPORTED_OPTION, field = 2);
        assert_err!(session.write_text(&[&over], &left), TURBO_E_UNSUPPORTED_OPTION, field = 2);
    }
}

#[test]
fn capability_max_tokens_is_honored_or_rejected() {
    let t = Target::from_env();
    let (model, session) = t.session(BundleKind::Embedding);
    let max_seq = model.info().max_seq;
    let text = long_text(max_seq as usize);
    let budget = EmbedOptions { max_tokens: max_seq / 2, ..Default::default() };
    if t.has(TURBO_CAP_OPT_MAX_TOKENS) {
        session.write_text(&[&text], &budget).expect("max_tokens");
        let short = read_f32(&session.run(&RunOptions::default()).expect("run"), 0);
        let full = embed(&t, &text, &EmbedOptions::default());
        assert_ne!(short, full, "a smaller token budget must change the vector");
        // A budget above the model's maximum is a capacity error, not a cap.
        let too_big = EmbedOptions { max_tokens: max_seq + 1, ..Default::default() };
        assert_err!(session.write_text(&[&text], &too_big), TURBO_E_CAPACITY, field = 3);
    } else {
        assert_err!(session.write_text(&[&text], &budget), TURBO_E_UNSUPPORTED_OPTION, field = 3);
    }
}

#[test]
fn capability_prompt_role_is_honored_or_rejected() {
    let t = Target::from_env();
    let (model, session) = t.session(BundleKind::Embedding);
    let query = EmbedOptions { prompt_role: PromptRole::Query, ..Default::default() };
    let document = EmbedOptions { prompt_role: PromptRole::Document, ..Default::default() };
    if t.has(TURBO_CAP_OPT_PROMPT_ROLE) {
        let none = embed(&t, "hello world", &EmbedOptions::default());
        session.write_text(&["hello world"], &query).expect("query role");
        let q = read_f32(&session.run(&RunOptions::default()).expect("run"), 0);
        session.write_text(&["hello world"], &document).expect("document role");
        let d = read_f32(&session.run(&RunOptions::default()).expect("run"), 0);
        let info = model.info();
        if !info.prefix_query.is_empty() {
            assert_ne!(q, none, "the query prefix `{}` must change the vector", info.prefix_query);
        }
        if !info.prefix_document.is_empty() {
            assert_ne!(d, none, "the document prefix `{}` must change the vector", info.prefix_document);
        }
        if !info.prefix_query.is_empty() && info.prefix_query != info.prefix_document {
            assert_ne!(q, d, "query and document prefixes differ, so the vectors must differ");
        }
    } else {
        assert_err!(session.write_text(&["hello world"], &query), TURBO_E_UNSUPPORTED_OPTION, field = 4);
        assert_err!(session.write_text(&["hello world"], &document), TURBO_E_UNSUPPORTED_OPTION, field = 4);
    }
}

#[test]
fn capability_normalize_is_honored_or_rejected() {
    let t = Target::from_env();
    let (model, session) = t.session(BundleKind::Embedding);
    let contract = model.info().normalize;
    let off = EmbedOptions { normalize: Normalize::None, ..Default::default() };
    let on = EmbedOptions { normalize: Normalize::L2, ..Default::default() };
    // Asking for exactly what the contract says is never gated.
    let same_as_contract = match contract {
        Some(Normalize::L2) => on,
        _ => off,
    };
    session.write_text(&["hello world"], &same_as_contract).expect("the contract's own normalization");
    let contract_vector = read_f32(&session.run(&RunOptions::default()).expect("run"), 0);
    if contract == Some(Normalize::L2) {
        assert!((norm(&contract_vector) - 1.0).abs() < 1e-5, "L2 output must be unit length");
    }

    let differing = match contract {
        Some(Normalize::L2) => off,
        _ => on,
    };
    if t.has(TURBO_CAP_OPT_NORMALIZE) {
        session.write_text(&["hello world"], &differing).expect("normalize override");
        let v = read_f32(&session.run(&RunOptions::default()).expect("run"), 0);
        let n = norm(&v);
        if differing.normalize == Normalize::None {
            assert!((n - 1.0).abs() > 1e-4, "NONE must not leave a unit vector (norm {n})");
        } else {
            assert!((n - 1.0).abs() < 1e-5, "L2 must produce a unit vector (norm {n})");
        }
        assert_ne!(v, contract_vector, "an honored normalize override must change the output");
    } else {
        assert_err!(session.write_text(&["hello world"], &differing), TURBO_E_UNSUPPORTED_OPTION, field = 5);
    }
}

#[test]
fn capability_pooling_override_is_honored_or_rejected() {
    let t = Target::from_env();
    let (model, session) = t.session(BundleKind::Embedding);
    let contract = model.info().pooling;
    let other = match contract {
        Some(Pooling::Cls) => Pooling::Mean,
        _ => Pooling::Cls,
    };
    let opts = EmbedOptions { pooling: other, ..Default::default() };
    if t.has(TURBO_CAP_OPT_POOLING_OVERRIDE) {
        let base = embed(&t, "hello world again", &EmbedOptions::default());
        session.write_text(&["hello world again"], &opts).expect("pooling override");
        let v = read_f32(&session.run(&RunOptions::default()).expect("run"), 0);
        assert_ne!(v, base, "an honored pooling override must change the output");
    } else {
        assert_err!(session.write_text(&["hello"], &opts), TURBO_E_UNSUPPORTED_OPTION, field = 6);
    }
    // Asking for the contract's own pooling is never gated.
    if let Some(p) = contract {
        let same = EmbedOptions { pooling: p, ..Default::default() };
        session.write_text(&["hello"], &same).expect("the contract's own pooling");
    }
}

#[test]
fn capability_output_dim_is_honored_or_rejected() {
    let t = Target::from_env();
    let (model, session) = t.session(BundleKind::Embedding);
    let dim = model.info().dim;
    let dims = model.bundle().contract().truncate_dims.clone();
    let normalizes = model.info().normalize == Some(Normalize::L2);
    if t.has(TURBO_CAP_OPT_OUTPUT_DIM) {
        assert!(!dims.is_empty(), "a device advertising OPT_OUTPUT_DIM needs a bundle with truncate_dims");
        for d in &dims {
            let opts = EmbedOptions { output_dim: *d, ..Default::default() };
            session.write_text(&["hello world"], &opts).expect("output_dim");
            let result = session.run(&RunOptions::default()).expect("run");
            let shape = result.output(0).expect("output").shape.clone();
            let v = read_f32(&result, 0);
            assert_eq!(shape, vec![1, *d as u64], "output_dim {d} must change the reported shape");
            assert_eq!(v.len(), *d as usize, "output_dim {d} must produce {d} columns");
            if normalizes {
                assert!((norm(&v) - 1.0).abs() < 1e-5, "a truncated L2 vector must be re-normalized");
            }
        }
        // A dimension the bundle did not declare is refused by name.
        let bogus = (1..=dim).find(|d| !dims.contains(d) && *d != dim).unwrap_or(dim + 1);
        let opts = EmbedOptions { output_dim: bogus, ..Default::default() };
        assert_err!(session.write_text(&["hello"], &opts), TURBO_E_INVALID_ARGUMENT, field = 7);
        // The model's own dimension is always allowed.
        let full = EmbedOptions { output_dim: dim, ..Default::default() };
        session.write_text(&["hello"], &full).expect("output_dim equal to the model dimension");
    } else {
        let d = dims.first().copied().unwrap_or(1);
        let opts = EmbedOptions { output_dim: d, ..Default::default() };
        assert_err!(session.write_text(&["hello"], &opts), TURBO_E_UNSUPPORTED_OPTION, field = 7);
    }
}

#[test]
fn capability_output_dtype_is_honored_or_rejected() {
    let t = Target::from_env();
    let (_m, session) = t.session(BundleKind::Embedding);
    for dtype in [OutputDType::F16, OutputDType::I8] {
        let opts = EmbedOptions { output_dtype: dtype, ..Default::default() };
        if t.has(TURBO_CAP_OPT_OUTPUT_DTYPE) {
            session.write_text(&["hello world"], &opts).expect("output_dtype");
            let result = session.run(&RunOptions::default()).expect("run");
            let reported = result.output(0).expect("output").dtype();
            let expected = match dtype {
                OutputDType::F16 => turbo::types::DType::F16,
                _ => turbo::types::DType::I8,
            };
            assert_eq!(reported, expected, "an honored output_dtype must be reflected in the result");
        } else {
            assert_err!(session.write_text(&["hello"], &opts), TURBO_E_UNSUPPORTED_OPTION, field = 8);
        }
    }
    // F32 is the universal output type and is never gated.
    let f32_opts = EmbedOptions { output_dtype: OutputDType::F32, ..Default::default() };
    session.write_text(&["hello"], &f32_opts).expect("F32 output is always available");
}

#[test]
fn capability_top_n_is_honored_or_rejected() {
    let t = Target::from_env();
    let (_m, session) = t.session(BundleKind::Reranker);
    let docs = ["alpha beta", "gamma", "alpha beta gamma", "delta epsilon"];
    let opts = RerankOptions { top_n: 2, return_sorted: true, ..Default::default() };
    if t.has(TURBO_CAP_OPT_TOP_N) {
        session.write_pairs("alpha beta", &docs, &opts).expect("top_n");
        let result = session.run(&RunOptions::default()).expect("run");
        let scores = read_f32(&result, 0);
        assert_eq!(scores.len(), docs.len(), "scores stay in input order, one per document");
        let sorted = read_i32(&result, 1);
        assert_eq!(sorted.len(), 2, "top_n must bound the sorted output");
        for (rank, &i) in sorted.iter().enumerate() {
            assert!((i as usize) < docs.len(), "sorted[{rank}] = {i} is not a document index");
        }
        assert!(sorted[0] != sorted[1], "the sorted output must not repeat a document");
        assert!(
            scores[sorted[0] as usize] >= scores[sorted[1] as usize],
            "the sorted output must be in descending score order: {scores:?} / {sorted:?}"
        );
        for (i, s) in scores.iter().enumerate() {
            if !sorted.contains(&(i as i32)) {
                assert!(*s <= scores[sorted[1] as usize], "document {i} scored {s} but was left out of the top 2");
            }
        }
        // top_n above the document count is a caller error.
        let too_many = RerankOptions { top_n: docs.len() as u32 + 1, ..Default::default() };
        assert_err!(session.write_pairs("q", &docs, &too_many), TURBO_E_INVALID_ARGUMENT, field = 4);
    } else {
        assert_err!(session.write_pairs("q", &docs, &opts), TURBO_E_UNSUPPORTED_OPTION, field = 4);
        let sorted_only = RerankOptions { return_sorted: true, ..Default::default() };
        assert_err!(session.write_pairs("q", &docs, &sorted_only), TURBO_E_UNSUPPORTED_OPTION, field = 4);
    }
}

#[test]
fn capability_aggregation_is_honored_or_rejected() {
    let t = Target::from_env();
    let (model, session) = t.session(BundleKind::TokenClassifier);
    let text = "Paris Paris Bob";
    let contract = model.info().aggregation;
    let none = ClassifyOptions { aggregation: Aggregation::None, ..Default::default() };
    let simple = ClassifyOptions { aggregation: Aggregation::Simple, ..Default::default() };
    if t.has(TURBO_CAP_OPT_AGGREGATION) {
        session.write_text_classify(&[text], &none).expect("aggregation NONE");
        let r = session.run(&RunOptions::default()).expect("run");
        let per_token: Vec<_> = r.spans().to_vec();
        drop(r);
        session.write_text_classify(&[text], &simple).expect("aggregation SIMPLE");
        let r = session.run(&RunOptions::default()).expect("run");
        let merged: Vec<_> = r.spans().to_vec();
        drop(r);

        assert!(!per_token.is_empty(), "NONE must report per-token spans");
        for s in &per_token {
            assert!(
                (s.byte_end as usize) <= text.len() && s.byte_start < s.byte_end,
                "span {s:?} does not lie inside the input"
            );
        }
        // Repeating one word must merge under SIMPLE but not under NONE.
        assert!(
            merged.len() <= per_token.len(),
            "SIMPLE produced more spans ({}) than NONE ({})",
            merged.len(),
            per_token.len()
        );
        if t.is_mock() {
            assert_eq!(per_token.len(), 3, "the mock reports one span per word under NONE");
            assert_eq!(merged.len(), 2, "the two identical words must merge under SIMPLE");
            assert_eq!((merged[0].byte_start, merged[0].byte_end), (0, 11));
        }
        // Every merged span is a union of consecutive per-token spans.
        for m in &merged {
            assert!(
                per_token.iter().any(|p| p.byte_start == m.byte_start && p.row == m.row),
                "merged span {m:?} does not start at a per-token span"
            );
            assert!(
                per_token.iter().any(|p| p.byte_end == m.byte_end && p.row == m.row),
                "merged span {m:?} does not end at a per-token span"
            );
        }
    } else {
        let differing = if contract == Some(Aggregation::None) { simple } else { none };
        assert_err!(session.write_text_classify(&[text], &differing), TURBO_E_UNSUPPORTED_OPTION, field = 4);
    }
    // Aggregation on a plain classifier is a caller error, not a capability one.
    let (_m2, plain) = t.session(BundleKind::Classifier);
    assert_err!(plain.write_text_classify(&["x"], &simple), TURBO_E_INVALID_ARGUMENT, field = 4);
}

#[test]
fn capability_generation_options_are_honored_or_rejected() {
    let t = Target::from_env();
    let model = t.model(BundleKind::Generative);

    // Seed. It only matters when sampling; greedy decoding ignores it.
    let seeded = GenerateDesc { max_new_tokens: 4, seed: Some(12345), ..Default::default() };
    if t.has(TURBO_CAP_OPT_GEN_SEED) {
        let a = collect(&t, &seeded);
        let b = collect(&t, &seeded);
        assert_eq!(a, b, "the same seed must reproduce the same tokens");
        let greedy_other = GenerateDesc { seed: Some(999_999), ..seeded.clone() };
        assert_eq!(a, collect(&t, &greedy_other), "at temperature 0 the seed must not change the tokens");
        if t.has(TURBO_CAP_OPT_GEN_SAMPLING) {
            let sampled = GenerateDesc { temperature: 1.0, ..seeded.clone() };
            let other = GenerateDesc { seed: Some(999_999), ..sampled.clone() };
            assert_ne!(
                collect(&t, &sampled),
                collect(&t, &other),
                "when sampling, a different seed must change the tokens"
            );
        }
    } else {
        assert_err!(model.create_generation(&seeded), TURBO_E_UNSUPPORTED_OPTION, field = 12);
    }

    // Logprobs.
    let with_logprobs = GenerateDesc { max_new_tokens: 2, logprobs: 3, ..Default::default() };
    if t.has(TURBO_CAP_OPT_GEN_LOGPROBS) {
        let generation = model.create_generation(&with_logprobs).expect("logprobs");
        generation.prompt(&[Message { role: "user", content: "hello" }]).expect("prompt");
        let chunk = generation.step().expect("step");
        assert_eq!(chunk.logprobs.len(), chunk.tokens.len() * 3, "logprobs must be n_tokens * logprobs entries");
        assert!(chunk.logprobs.iter().all(|p| *p <= 0.0), "logprobs are log probabilities: {:?}", chunk.logprobs);
    } else {
        assert_err!(model.create_generation(&with_logprobs), TURBO_E_UNSUPPORTED_OPTION, field = 19);
    }

    // Stop strings.
    let stop = GenerateDesc { max_new_tokens: 8, stop: vec!["zzz-never-matches".into()], ..Default::default() };
    if t.has(TURBO_CAP_OPT_GEN_STOP_STRINGS) {
        let generation = model.create_generation(&stop).expect("stop strings");
        generation.prompt(&[Message { role: "user", content: "hello" }]).expect("prompt");
        drop(generation.step().expect("step"));
    } else {
        assert_err!(model.create_generation(&stop), TURBO_E_UNSUPPORTED_OPTION, field = 14);
    }

    // The gated fields that have no honor test beyond being accepted.
    let cases: [(u64, u32, GenerateDesc); 7] = [
        (TURBO_CAP_OPT_GEN_N, 4, GenerateDesc { n_sequences: 2, ..Default::default() }),
        (TURBO_CAP_OPT_GEN_PENALTIES, 9, GenerateDesc { repeat_penalty: 1.5, ..Default::default() }),
        (TURBO_CAP_OPT_GEN_PENALTIES, 10, GenerateDesc { presence_penalty: 0.5, ..Default::default() }),
        (TURBO_CAP_OPT_GEN_PENALTIES, 11, GenerateDesc { frequency_penalty: 0.5, ..Default::default() }),
        (TURBO_CAP_OPT_GEN_LOGIT_BIAS, 18, GenerateDesc { logit_bias: vec![(3, 1.0)], ..Default::default() }),
        (
            TURBO_CAP_OPT_GEN_STRUCTURED,
            21,
            GenerateDesc {
                structured_kind: StructuredKind::JsonSchema,
                structured: "{\"type\":\"object\"}".into(),
                ..Default::default()
            },
        ),
        (TURBO_CAP_OPT_GEN_TOOLS, 24, GenerateDesc { tools: vec!["{\"name\":\"x\"}".into()], ..Default::default() }),
    ];
    for (bit, field, desc) in cases {
        if t.has(bit) {
            model.create_generation(&desc).unwrap_or_else(|e| panic!("field {field} is advertised but failed: {e}"));
        } else {
            assert_err!(model.create_generation(&desc), TURBO_E_UNSUPPORTED_OPTION, field = field);
        }
    }
}

fn collect(t: &Target, desc: &GenerateDesc) -> Vec<i32> {
    let model = t.model(BundleKind::Generative);
    let generation = model.create_generation(desc).expect("generation");
    generation.prompt(&[Message { role: "user", content: "hello there" }]).expect("prompt");
    let mut tokens = Vec::new();
    loop {
        let chunk = generation.step().expect("step");
        tokens.extend_from_slice(&chunk.tokens);
        if chunk.done {
            return tokens;
        }
        assert!(tokens.len() < 1000, "generation did not finish");
    }
}

#[test]
fn capability_unsupported_modality_cell_is_unsupported() {
    let t = Target::from_env();
    for modality in [Modality::Audio, Modality::Image, Modality::Video] {
        let cell = t.runtime.capability(t.device_index, Task::Embed, modality).expect("capability cell");
        if cell.is_offered() {
            continue; // A provider that really offers it is not a violation.
        }
        // A bundle declaring that modality cannot load on this device.
        let scratch = fixtures::copy_of(&t.bundle(BundleKind::Embedding));
        let name = match modality {
            Modality::Audio => "audio",
            Modality::Image => "image",
            _ => "video",
        };
        scratch.patch_manifest(|m| m["modality"] = serde_json::Value::String(name.into()));
        let ctx = t.context();
        assert_err!(ctx.load_model(scratch.path(), &turbo::provider::ModelDesc::default()), TURBO_E_UNSUPPORTED_TASK);
    }
}

#[test]
fn capability_matrix_rejects_unknown_axes() {
    let t = Target::from_env();
    // Every known (task, modality) cell is readable.
    for task in Task::ALL {
        for modality in Modality::ALL {
            t.runtime.capability(t.device_index, *task, *modality).expect("every cell is readable");
        }
    }
    // An out-of-range device index is a device error, not a panic.
    let missing = t.runtime.device_count();
    assert_err!(t.runtime.capability(missing + 100, Task::Embed, Modality::Text), TURBO_E_DEVICE_NOT_FOUND);
}

#[test]
fn capability_can_run_agrees_with_model_load() {
    let t = Target::from_env();
    for kind in BundleKind::ALL {
        let path = t.bundle(*kind);
        if !path.is_dir() {
            continue;
        }
        let bundle = turbo::bundle::Bundle::open(&path).expect("the test bundles are valid");
        let can = t.runtime.can_run(t.device_index, &path, bundle.task(), bundle.modality());
        let ctx = t.context();
        let loaded = ctx.load_model(&path, &turbo::provider::ModelDesc::default());
        match (&can, &loaded) {
            (Ok(()), Ok(_)) => {}
            (Err(a), Err(b)) => assert_eq!(
                a.code(),
                b.code(),
                "turbo_can_run said {} but load said {} for {}",
                a.code_name(),
                b.code_name(),
                path.display()
            ),
            (Ok(()), Err(b)) => panic!("can_run accepted {} but load failed: {b}", path.display()),
            (Err(a), Ok(_)) => panic!("can_run rejected {} with {a} but load succeeded", path.display()),
        }
    }

    // A bundle with no artifact this provider can load is refused by both.
    let scratch = fixtures::copy_of(&t.bundle(BundleKind::Embedding));
    let artifact_body = b"not an artifact this provider knows";
    std::fs::write(scratch.path().join("foreign.bin"), artifact_body).expect("write");
    let sha = turbo::bundle::sha256_bytes(artifact_body);
    scratch.patch_manifest(|m| {
        m["artifacts"] = serde_json::json!({
            "not_a_real_format": { "path": "foreign.bin", "sha256": sha }
        });
    });
    let bundle = turbo::bundle::Bundle::open(scratch.path()).expect("the manifest itself is valid");
    let can = t.runtime.can_run(t.device_index, scratch.path(), bundle.task(), bundle.modality());
    let e = assert_err!(can, TURBO_E_BUNDLE_NO_ARTIFACT);
    assert!(!e.message().is_empty());
    let ctx = t.context();
    assert_err!(ctx.load_model(scratch.path(), &turbo::provider::ModelDesc::default()), TURBO_E_BUNDLE_NO_ARTIFACT);
}

#[test]
fn capability_cells_report_a_dtype_and_determinism() {
    let t = Target::from_env();
    let cell = t.runtime.capability(t.device_index, Task::Embed, Modality::Text).expect("cell");
    if !cell.is_offered() {
        return;
    }
    assert!(cell.dtype.is_some(), "an offered cell must name the compute dtype");
    // A device claiming DETERMINISTIC must say so in its embed cell too.
    if t.has(TURBO_CAP_DETERMINISTIC) {
        assert!(cell.deterministic, "TURBO_CAP_DETERMINISTIC and the capability cell must agree");
    }
    assert!(cell.cosine_floor >= 0.0 && cell.cosine_floor <= 1.0, "cosine floor {}", cell.cosine_floor);
    assert!(cell.max_abs_error >= 0.0, "max abs error {}", cell.max_abs_error);
}

#[test]
fn capability_session_options_are_rejected_when_unknown() {
    let t = Target::from_env();
    let model = t.model(BundleKind::Embedding);
    // Provider knobs are validated, never ignored (PLAN.md section 5).
    let desc = SessionDesc { options: turbo::handles::options_from_pairs([("tensorrt", "1")]), ..Default::default() };
    match model.create_session(&desc) {
        Ok(_) => {} // the provider really offers this knob
        Err(e) => assert_eq!(
            e.code(),
            TURBO_E_INVALID_ARGUMENT,
            "an unknown session option must be TURBO_E_INVALID_ARGUMENT, got {}",
            e.code_name()
        ),
    }
}
