//! Group `contract`, Rust layer: options are honored exactly or refused.
//!
//! These pin the fixes for the option-honesty findings in
//! `docs/reviews/2026-09-21-p0-p2.md` (H7 and the medium items): a budget the
//! session cannot hold is refused rather than clamped, a prompt role the
//! bundle has no prefix for is refused rather than ignored, left truncation
//! keeps the prefix, and every generation option is gated by a capability
//! bit.

use turbo::abi::*;
use turbo::provider::{EmbedOptions, GenerateDesc, Message, RerankOptions, RunOptions, SessionDesc};
use turbo::types::{PromptRole, Truncate};
use turbo_conformance::fixtures::copy_of;
use turbo_conformance::{assert_err, read_rows, BundleKind, Target};

const PROMPT: [Message<'static>; 1] = [Message { role: "user", content: "say something" }];

#[test]
fn honesty_max_tokens_above_the_session_width_is_refused_not_clamped() {
    let t = Target::from_env();
    if !t.has(TURBO_CAP_OPT_MAX_TOKENS) {
        println!("honesty_max_tokens: device does not advertise TURBO_CAP_OPT_MAX_TOKENS");
        return;
    }
    let model = t.model(BundleKind::Embedding);
    let session = model.create_session(&SessionDesc { max_batch: 1, max_seq: 8, ..Default::default() }).unwrap();
    // 12 fits the model (max_seq 16 or more) but not this session.
    let e = assert_err!(
        session.write_text(&["a b c"], &EmbedOptions { max_tokens: 12, ..Default::default() }),
        TURBO_E_CAPACITY
    );
    assert_eq!(e.field(), EmbedOptions::FIELD_MAX_TOKENS, "the field index names max_tokens");
    session.write_text(&["a b c"], &EmbedOptions { max_tokens: 8, ..Default::default() }).expect("8 fits");
}

#[test]
fn honesty_prompt_role_without_a_bundle_prefix_is_refused() {
    let t = Target::from_env();
    if !t.is_mock() {
        println!("honesty_prompt_role: needs a mock bundle to edit");
        return;
    }
    let scratch = copy_of(&t.bundle(BundleKind::Embedding));
    scratch.patch_manifest(|m| {
        m["contract"]["prompts"] = serde_json::json!({ "query": "", "document": "passage: " });
    });
    let model = t.context().load_model(scratch.path(), &Default::default()).expect("load edited bundle");
    let session = model.create_session(&SessionDesc::default()).unwrap();
    let e = assert_err!(
        session.write_text(&["x"], &EmbedOptions { prompt_role: PromptRole::Query, ..Default::default() }),
        TURBO_E_INVALID_ARGUMENT
    );
    assert_eq!(e.field(), EmbedOptions::FIELD_PROMPT_ROLE);
    session
        .write_text(&["x"], &EmbedOptions { prompt_role: PromptRole::Document, ..Default::default() })
        .expect("the document prefix exists");
}

#[test]
fn honesty_left_truncation_keeps_the_prompt_prefix() {
    let t = Target::from_env();
    if !t.has(TURBO_CAP_OPT_TRUNCATE | TURBO_CAP_OPT_PROMPT_ROLE) {
        println!("honesty_left_truncation: device does not advertise truncate and prompt_role");
        return;
    }
    let model = t.model(BundleKind::Embedding);
    let session = model.create_session(&SessionDesc { max_batch: 1, max_seq: 8, ..Default::default() }).unwrap();
    let long = "w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12";
    let embed = |role: PromptRole| {
        session
            .write_text(&[long], &EmbedOptions { truncate: Truncate::Left, prompt_role: role, ..Default::default() })
            .unwrap();
        let r = session.run(&RunOptions::default()).unwrap();
        read_rows(&r, 0, model.info().dim as usize).remove(0)
    };
    let bare = embed(PromptRole::None);
    let query = embed(PromptRole::Query);
    assert_ne!(bare, query, "with LEFT truncation the query prefix must still be applied");
}

#[test]
fn honesty_return_sorted_names_its_own_field() {
    let t = Target::from_env();
    if t.has(TURBO_CAP_OPT_TOP_N) {
        println!("honesty_return_sorted: device honors top_n; the refusal path is not reachable");
        return;
    }
    let (_, session) = t.session(BundleKind::Reranker);
    let e = assert_err!(
        session.write_pairs("q", &["d"], &RerankOptions { return_sorted: true, ..Default::default() }),
        TURBO_E_UNSUPPORTED_OPTION
    );
    assert_eq!(e.field(), RerankOptions::FIELD_RETURN_SORTED);
}

#[test]
fn honesty_sampling_options_change_the_output_only_when_advertised() {
    let t = Target::from_env();
    let model = t.model(BundleKind::Generative);
    let run = |desc: GenerateDesc| -> Result<Vec<i32>, turbo::Error> {
        let g = model.create_generation(&desc)?;
        g.prompt(&PROMPT)?;
        let mut tokens = Vec::new();
        for _ in 0..64 {
            let chunk = g.step()?;
            tokens.extend_from_slice(&chunk.tokens);
            if chunk.done {
                break;
            }
        }
        Ok(tokens)
    };
    if !t.has(TURBO_CAP_OPT_GEN_SAMPLING) {
        let e = assert_err!(run(GenerateDesc { temperature: 0.7, ..Default::default() }), TURBO_E_UNSUPPORTED_OPTION);
        assert_eq!(e.field(), 5, "temperature is field 5");
        let e = assert_err!(run(GenerateDesc { top_k: 4, ..Default::default() }), TURBO_E_UNSUPPORTED_OPTION);
        assert_eq!(e.field(), 6, "top_k is field 6");
        return;
    }
    // Greedy: the seed does not matter.
    let a = run(GenerateDesc { max_new_tokens: 6, seed: Some(1), ..Default::default() }).unwrap();
    let b = run(GenerateDesc { max_new_tokens: 6, seed: Some(2), ..Default::default() }).unwrap();
    assert_eq!(a, b, "temperature 0 is greedy regardless of seed");
    if t.has(TURBO_CAP_OPT_GEN_SEED) {
        // Sampling: the seed matters and top_k bounds the candidate pool.
        let c = run(GenerateDesc { max_new_tokens: 6, seed: Some(1), temperature: 1.0, ..Default::default() }).unwrap();
        let d = run(GenerateDesc { max_new_tokens: 6, seed: Some(2), temperature: 1.0, ..Default::default() }).unwrap();
        assert_ne!(c, d, "temperature > 0 with different seeds must differ");
        let narrow =
            run(GenerateDesc { max_new_tokens: 12, seed: Some(3), temperature: 1.0, top_k: 2, ..Default::default() })
                .unwrap();
        let distinct: std::collections::BTreeSet<i32> = narrow.iter().copied().collect();
        assert!(distinct.len() <= 2, "top_k 2 allows at most two distinct sampled tokens: {narrow:?}");
    }
}

#[test]
fn honesty_ungated_generation_options_are_refused_without_their_bit() {
    let t = Target::from_env();
    let model = t.model(BundleKind::Generative);
    let cases: [(u64, u32, GenerateDesc); 3] = [
        (TURBO_CAP_OPT_GEN_MIN_TOKENS, 3, GenerateDesc { min_new_tokens: 1, ..Default::default() }),
        (TURBO_CAP_OPT_GEN_ECHO, 22, GenerateDesc { echo: true, ..Default::default() }),
        (TURBO_CAP_OPT_GEN_STOP_TOKENS, 15, GenerateDesc { stop_tokens: vec![7], ..Default::default() }),
    ];
    for (bit, field, desc) in cases {
        let r = model.create_generation(&desc);
        if t.has(bit) {
            r.unwrap_or_else(|e| panic!("bit {bit:#x} is advertised, so field {field} must be accepted: {e}"));
        } else {
            let e = assert_err!(r, TURBO_E_UNSUPPORTED_OPTION);
            assert_eq!(e.field(), field);
        }
    }
}
