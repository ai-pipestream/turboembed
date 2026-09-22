//! Group `contract`, Rust layer: options are honored exactly or refused.
//!
//! These pin the fixes for the option-honesty findings in
//! `docs/reviews/2026-09-21-p0-p2.md` (H7 and the medium items): a budget the
//! session cannot hold is refused rather than clamped, a prompt role the
//! bundle has no prefix for is refused rather than ignored, left truncation
//! keeps the prefix, and every generation option is gated by a capability
//! bit.

use turbo::abi::*;
use turbo::handles::Context;
use turbo::provider::{ContextDesc, EmbedOptions, GenerateDesc, Message, RerankOptions, RunOptions, SessionDesc};
use turbo::types::{Modality, PromptRole, Task, Truncate};
use turbo_conformance::fixtures::copy_of;
use turbo_conformance::{assert_err, needs, read_rows, BundleKind, Target};

const PROMPT: [Message<'static>; 1] = [Message { role: "user", content: "say something" }];

#[test]
fn honesty_max_tokens_above_the_session_width_is_refused_not_clamped() {
    let t = Target::from_env();
    needs!(t, Embedding);
    if !t.has(TURBO_CAP_OPT_MAX_TOKENS) {
        // The refusal without the bit is asserted in
        // `capability_max_tokens_is_honored_or_rejected` (field 3).
        println!("not applicable: the device does not advertise TURBO_CAP_OPT_MAX_TOKENS");
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
    needs!(t, Embedding);
    if !t.has(TURBO_CAP_OPT_PROMPT_ROLE) {
        // Without the bit the role is refused before the prefix is looked
        // at, which `capability_prompt_role_is_honored_or_rejected` asserts.
        println!("not applicable: the device does not advertise TURBO_CAP_OPT_PROMPT_ROLE");
        return;
    }
    // Any bundle serves: only the manifest's declared prefixes are edited,
    // so every artifact still hashes as recorded.
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
    needs!(t, Embedding);
    if !t.has(TURBO_CAP_OPT_TRUNCATE | TURBO_CAP_OPT_PROMPT_ROLE) {
        // Each option's refusal without its bit is asserted in
        // `capability_truncate_is_honored_or_rejected` (field 2) and
        // `capability_prompt_role_is_honored_or_rejected` (field 4).
        println!("not applicable: the device does not advertise both TRUNCATE and PROMPT_ROLE");
        return;
    }
    let model = t.model(BundleKind::Embedding);
    let info = model.info();
    let role = if !info.prefix_query.is_empty() {
        PromptRole::Query
    } else if !info.prefix_document.is_empty() {
        PromptRole::Document
    } else {
        println!("not applicable: bundle `{}` declares no prompt prefix to keep", info.model_id);
        return;
    };
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
    let prefixed = embed(role);
    assert_ne!(bare, prefixed, "with LEFT truncation the {role:?} prefix must still be applied");
}

#[test]
fn honesty_return_sorted_names_its_own_field() {
    // `return_sorted` shares `TURBO_CAP_OPT_TOP_N` with `top_n` but has its
    // own field index (5, docs/c-api.md and `RerankOptions::FIELD_RETURN_SORTED`);
    // it was once reported as field 4, and the assertion that pinned it was
    // in a branch nothing reaches, because every device in this tree that
    // offers RERANK also advertises the bit. So the case looks for any
    // enumerated device that offers RERANK with the bit clear, whatever
    // provider it belongs to, and says so when there is none.
    let t = Target::from_env();
    let candidate = (0..t.runtime.device_count()).find_map(|i| {
        let info = t.runtime.device(i).ok()?.info.clone();
        let offers = t.runtime.capability(i, Task::Rerank, Modality::Text).ok()?.is_offered();
        (offers && info.caps & TURBO_CAP_OPT_TOP_N == 0).then_some((i, info))
    });
    let Some((index, info)) = candidate else {
        // The refusal is not reachable on this machine. The `top_n` half of
        // the same bit is asserted for every device under test in
        // `capability_top_n_is_honored_or_rejected`, which also pins field 5
        // for `return_sorted` in its own refusal branch.
        println!(
            "not applicable: no enumerated device offers RERANK with TURBO_CAP_OPT_TOP_N clear, so the refusal cannot be reached"
        );
        return;
    };
    println!("return_sorted refusal on {} ordinal {} ({})", info.provider_id, info.ordinal, info.name);
    let ctx = Context::create(t.runtime.clone(), index, &ContextDesc::default()).expect("context");
    let bundle = if info.provider_id == turbo::mock::MOCK_PROVIDER_ID {
        turbo_conformance::default_bundle_root().join(BundleKind::Reranker.dir_name())
    } else {
        t.bundle(BundleKind::Reranker)
    };
    let model = ctx.load_model(&bundle, &Default::default()).expect("a reranker bundle for that device");
    let session = model.create_session(&SessionDesc::default()).expect("session");
    // Each of the two options the bit gates names its own field, not the
    // other's.
    let e = assert_err!(
        session.write_pairs("q", &["d"], &RerankOptions { return_sorted: true, ..Default::default() }),
        TURBO_E_UNSUPPORTED_OPTION
    );
    assert_eq!(e.field(), RerankOptions::FIELD_RETURN_SORTED, "return_sorted is field 5: {}", e.message());
    let e = assert_err!(
        session.write_pairs("q", &["d"], &RerankOptions { top_n: 1, ..Default::default() }),
        TURBO_E_UNSUPPORTED_OPTION
    );
    assert_eq!(e.field(), RerankOptions::FIELD_TOP_N, "top_n is field 4: {}", e.message());
    // With neither option set the same session reranks, so the refusals are
    // the options being gated and not the device refusing the task.
    session.write_pairs("q", &["d"], &RerankOptions::default()).expect("a rerank with no gated option");
    session.run(&turbo::provider::RunOptions::default()).expect("run");
}

#[test]
fn honesty_sampling_options_change_the_output_only_when_advertised() {
    let t = Target::from_env();
    needs!(t, Generative);
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
        // Sampling: the seed matters (temperature 2 over 16 tokens, so two
        // seeds coinciding on a real model is not a flake rate worth naming).
        let c =
            run(GenerateDesc { max_new_tokens: 16, seed: Some(1), temperature: 2.0, ..Default::default() }).unwrap();
        let d =
            run(GenerateDesc { max_new_tokens: 16, seed: Some(2), temperature: 2.0, ..Default::default() }).unwrap();
        assert_ne!(c, d, "temperature > 0 with different seeds must differ");
        // top_k 1 leaves one candidate per step, so sampling at any
        // temperature must reproduce the greedy tokens exactly. (top_k
        // bounds the candidates per step, not the distinct tokens of a
        // sequence, so a global count is not a property of the option.)
        let narrow =
            run(GenerateDesc { max_new_tokens: 6, seed: Some(3), temperature: 1.0, top_k: 1, ..Default::default() })
                .unwrap();
        assert_eq!(narrow, a, "top_k 1 must reproduce the greedy tokens");
    }
}

#[test]
fn honesty_ungated_generation_options_are_refused_without_their_bit() {
    let t = Target::from_env();
    needs!(t, Generative);
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

/// The tokens a generation produces for `prompt` under `desc`, greedily
/// unless the descriptor says otherwise.
fn generate(
    model: &std::sync::Arc<turbo::handles::Model>,
    prompt: &str,
    desc: &GenerateDesc,
) -> Result<Vec<i32>, turbo::Error> {
    let g = model.create_generation(desc)?;
    g.prompt(&[Message { role: "user", content: prompt }])?;
    let mut tokens = Vec::new();
    loop {
        let chunk = g.step()?;
        tokens.extend_from_slice(&chunk.tokens);
        if chunk.done {
            return Ok(tokens);
        }
        if tokens.len() > 256 {
            panic!("generation did not finish");
        }
    }
}

const REPEAT_PROMPT: &str = "Repeat this exactly, nothing else: cat cat cat cat cat cat cat cat";

#[test]
fn honesty_logit_bias_decides_the_first_token() {
    // A bias is not advice: `logit_bias` adds to a token's logit, so at
    // temperature 0 a large enough positive bias makes that token the one
    // that is generated, and a large enough negative bias makes the token
    // that would have been generated impossible. Anything less than that is
    // an option that was accepted and then ignored, which PLAN.md principle
    // 2 does not allow.
    let t = Target::from_env();
    needs!(t, Generative);
    if !t.has(TURBO_CAP_OPT_GEN_LOGIT_BIAS) {
        // The refusal without the bit (field 18) is asserted in
        // `capability_generation_options_are_honored_or_rejected`.
        println!("not applicable: the device does not advertise TURBO_CAP_OPT_GEN_LOGIT_BIAS");
        return;
    }
    let model = t.model(BundleKind::Generative);
    let greedy = GenerateDesc { max_new_tokens: 4, ..Default::default() };
    let baseline = generate(&model, REPEAT_PROMPT, &greedy).expect("greedy");
    let Some(&first) = baseline.first() else {
        println!("not applicable: the model generates nothing for this prompt, so there is no first token to move");
        return;
    };
    // A token the model itself produced later in the same continuation: a
    // real content token of this vocabulary, not a guess at an id.
    let Some(&other) = baseline.iter().find(|id| **id != first) else {
        println!("not applicable: the greedy continuation {baseline:?} is one repeated token, so it offers no second token to bias toward");
        return;
    };

    let up = GenerateDesc { logit_bias: vec![(other, 50.0)], ..greedy.clone() };
    let biased = generate(&model, REPEAT_PROMPT, &up).expect("logit_bias +50");
    assert_eq!(
        biased.first(),
        Some(&other),
        "a +50 bias on token {other} must make it the first token, but the stream began {biased:?}"
    );

    let down = GenerateDesc { logit_bias: vec![(first, -50.0)], ..greedy.clone() };
    let pushed = generate(&model, REPEAT_PROMPT, &down).expect("logit_bias -50");
    assert_ne!(
        pushed.first(),
        Some(&first),
        "a -50 bias on token {first} must make it impossible, but it was generated anyway"
    );
    // The option is per generation, not sticky: the next one is greedy again.
    assert_eq!(generate(&model, REPEAT_PROMPT, &greedy).expect("greedy again"), baseline);
}

#[test]
fn honesty_penalties_change_a_repeating_continuation() {
    // The three penalty fields share one capability bit and one promise: a
    // continuation that repeats a token is not the continuation the caller
    // gets once a penalty is set. The prompt is chosen so the unpenalized
    // greedy answer repeats; if it does not on the model at hand there is
    // nothing for a penalty to act on and the case says so.
    let t = Target::from_env();
    needs!(t, Generative);
    if !t.has(TURBO_CAP_OPT_GEN_PENALTIES) {
        // The refusals (fields 9, 10, 11) are asserted in
        // `capability_generation_options_are_honored_or_rejected`.
        println!("not applicable: the device does not advertise TURBO_CAP_OPT_GEN_PENALTIES");
        return;
    }
    let model = t.model(BundleKind::Generative);
    let greedy = GenerateDesc { max_new_tokens: 8, ..Default::default() };
    let baseline = generate(&model, REPEAT_PROMPT, &greedy).expect("greedy");
    let repeats = baseline.iter().enumerate().any(|(i, a)| baseline[..i].contains(a));
    if !repeats {
        println!(
            "not applicable: the greedy continuation {baseline:?} of {REPEAT_PROMPT:?} repeats no token, so a penalty has nothing to change"
        );
        return;
    }
    for (name, desc) in [
        ("repeat_penalty", GenerateDesc { repeat_penalty: 2.0, ..greedy.clone() }),
        ("presence_penalty", GenerateDesc { presence_penalty: 2.0, ..greedy.clone() }),
        ("frequency_penalty", GenerateDesc { frequency_penalty: 2.0, ..greedy.clone() }),
    ] {
        let penalized = generate(&model, REPEAT_PROMPT, &desc).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_ne!(penalized, baseline, "{name} 2.0 left the repeating continuation {baseline:?} exactly as it was");
    }
}

#[test]
fn honesty_tools_are_rendered_into_the_prompt() {
    // A tool definition the provider accepts has to reach the model, and the
    // only place it can reach is the prompt: the same messages must tokenize
    // to more prompt tokens with a tool than without one.
    let t = Target::from_env();
    needs!(t, Generative);
    if !t.has(TURBO_CAP_OPT_GEN_TOOLS) {
        // The refusal without the bit (field 24) is asserted in
        // `capability_generation_options_are_honored_or_rejected`.
        println!("not applicable: the device does not advertise TURBO_CAP_OPT_GEN_TOOLS");
        return;
    }
    let model = t.model(BundleKind::Generative);
    let prompt_tokens = |desc: &GenerateDesc| -> u32 {
        let g = model.create_generation(desc).expect("generation");
        g.prompt(&PROMPT).expect("prompt");
        let n = g.step().expect("step").prompt_tokens;
        n
    };
    let bare = GenerateDesc { max_new_tokens: 1, ..Default::default() };
    let tool = "{\"name\":\"get_weather\",\"description\":\"weather for a city\",\"parameters\":{}}";
    let with_tool = GenerateDesc { tools: vec![tool.into()], ..bare.clone() };
    let without = prompt_tokens(&bare);
    let with = prompt_tokens(&with_tool);
    assert!(without > 0, "a prompt is always some tokens");
    assert!(with > without, "the tool definition never reached the prompt: {with} tokens with it, {without} without");
}
