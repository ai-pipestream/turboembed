//! Live generation checks for any provider on a real GGUF bundle.
//!
//! Selection is described in `turbo_conformance::live`; the bundle comes
//! from `TURBO_LIVE_GGUF_BUNDLE` (an instruct model such as
//! Qwen2.5-0.5B-Instruct). The checks are semantic: the pull iterator
//! yields tokens in order, stops on the model's end token or the budget,
//! honors stop strings and cancellation, and reproduces a seeded sample.

use turbo::abi;
use turbo::provider::{GenerateDesc, Message};
use turbo::types::{FinishReason, StructuredKind};
use turbo::{CapStatus, Modality, ModelDesc, Task};
use turbo_conformance::live::{live, Live};

const CHAT: [Message<'static>; 2] = [
    Message { role: "system", content: "You are a terse assistant. Answer in one short sentence." },
    Message { role: "user", content: "What is the capital of France?" },
];

struct Run {
    tokens: Vec<i32>,
    text: String,
    finish: FinishReason,
    logprobs: usize,
}

fn drain(model: &std::sync::Arc<turbo::Model>, desc: &GenerateDesc, limit: usize) -> Run {
    let g = model.create_generation(desc).expect("generation");
    g.prompt(&CHAT).expect("prompt");
    let mut run = Run { tokens: Vec::new(), text: String::new(), finish: FinishReason::None, logprobs: 0 };
    for _ in 0..limit {
        let chunk = g.step().expect("step");
        run.tokens.extend_from_slice(&chunk.tokens);
        run.text.push_str(&chunk.text);
        run.logprobs += chunk.logprobs.len();
        if chunk.done {
            run.finish = chunk.finish_reason;
            assert_ne!(run.finish, FinishReason::None, "a finished chunk must say why");
            return run;
        }
    }
    panic!("the generation did not finish within {limit} steps ({} tokens so far)", run.tokens.len());
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

fn setup() -> Option<(Live, std::sync::Arc<turbo::Model>)> {
    let live = live()?;
    let dir = bundle_for(&live, Task::Generate, "TURBO_LIVE_GGUF_BUNDLE")?;
    let model = live.ctx.load_model(&dir, &ModelDesc::default()).expect("load the GGUF bundle");
    assert_eq!(model.info().kind, turbo::ModelKind::Generative);
    assert!(model.info().vocab_size > 0);
    Some((live, model))
}

#[test]
fn live_generate_answers_and_stops_on_eos() {
    let Some((_live, model)) = setup() else { return };
    let run = drain(&model, &GenerateDesc { max_new_tokens: 64, ..Default::default() }, 100);
    eprintln!("greedy: {:?} ({} tokens, {:?})", run.text, run.tokens.len(), run.finish);
    assert!(!run.tokens.is_empty());
    assert!(run.text.to_lowercase().contains("paris"), "the model should name Paris: {:?}", run.text);
    assert_eq!(run.finish, FinishReason::Eos, "a short answer ends with the model's end token");
}

#[test]
fn live_generate_length_budget_is_exact() {
    let Some((_live, model)) = setup() else { return };
    let run = drain(&model, &GenerateDesc { max_new_tokens: 3, ..Default::default() }, 10);
    assert_eq!(run.tokens.len(), 3);
    assert_eq!(run.finish, FinishReason::Length);
}

#[test]
fn live_generate_greedy_is_deterministic_and_seeds_reproduce_samples() {
    let Some((live, model)) = setup() else { return };
    let greedy = GenerateDesc { max_new_tokens: 12, ..Default::default() };
    let a = drain(&model, &greedy, 20);
    let b = drain(&model, &greedy, 20);
    if live.device.kind == turbo::DeviceKind::Cpu || live.has_cap(abi::TURBO_CAP_DETERMINISTIC) {
        assert_eq!(a.tokens, b.tokens, "greedy decoding on a deterministic device is bit-reproducible");
    }
    if live.has_cap(abi::TURBO_CAP_OPT_GEN_SAMPLING | abi::TURBO_CAP_OPT_GEN_SEED) {
        let sampled =
            GenerateDesc { max_new_tokens: 12, temperature: 0.8, top_k: 40, seed: Some(7), ..Default::default() };
        let c = drain(&model, &sampled, 20);
        let d = drain(&model, &sampled, 20);
        assert_eq!(c.tokens, d.tokens, "the same seed reproduces the same sample");
        // A seed that reproduces itself but ignores its value is not a
        // seed. Temperature 2 over 16 tokens: two seeds drawing the same
        // sequence from a real model has a probability far below any flake
        // rate worth naming (12 tokens at 0.8 coincided on Qwen2.5-0.5B,
        // whose answer here is nearly certain).
        let wide = GenerateDesc { max_new_tokens: 16, temperature: 2.0, seed: Some(7), ..Default::default() };
        let other = GenerateDesc { seed: Some(999_983), ..wide.clone() };
        let one = drain(&model, &wide, 24);
        let two = drain(&model, &other, 24);
        eprintln!("seed 7: {:?}\nseed 999983: {:?}", one.text, two.text);
        assert_ne!(one.tokens, two.tokens, "a different seed must draw a different sample");
    }
}

#[test]
fn live_generate_stop_strings_and_stop_tokens() {
    let Some((_live, model)) = setup() else { return };
    // Stop on the first period: the answer is one sentence, so the text
    // returned must not contain it.
    let run = drain(&model, &GenerateDesc { max_new_tokens: 64, stop: vec![".".into()], ..Default::default() }, 100);
    assert_eq!(run.finish, FinishReason::Stop, "{:?}", run.text);
    assert!(!run.text.contains('.'), "text before the stop string only: {:?}", run.text);
    // Stop on the first generated token id.
    let first = drain(&model, &GenerateDesc { max_new_tokens: 1, ..Default::default() }, 2).tokens[0];
    let run = drain(&model, &GenerateDesc { max_new_tokens: 64, stop_tokens: vec![first], ..Default::default() }, 100);
    assert_eq!(run.tokens, vec![first]);
    assert_eq!(run.finish, FinishReason::Stop);
}

#[test]
fn live_generate_cancel_and_logprobs() {
    let Some((_live, model)) = setup() else { return };
    let g = model.create_generation(&GenerateDesc { max_new_tokens: 64, logprobs: 3, ..Default::default() }).unwrap();
    g.prompt(&CHAT).unwrap();
    let first = g.step().unwrap();
    assert_eq!(first.tokens.len(), 1);
    assert_eq!(first.logprobs.len(), 3, "three logprobs per token");
    assert!(first.logprobs.iter().all(|l| *l <= 0.0), "log probabilities are non-positive");
    assert!(first.logprobs.windows(2).all(|w| w[0] >= w[1]), "descending");
    drop(first);
    g.cancel().unwrap();
    let last = g.step().unwrap();
    assert!(last.done);
    assert_eq!(last.finish_reason, FinishReason::Cancelled);
    assert_eq!(g.step().unwrap_err().code(), abi::TURBO_E_INVALID_STATE);
}

#[test]
fn live_generate_min_new_tokens_suppresses_the_end_token() {
    let Some((_live, model)) = setup() else { return };
    let short = drain(&model, &GenerateDesc { max_new_tokens: 64, ..Default::default() }, 100);
    // There is an end token to suppress: left alone the model stops on it
    // well inside the budget.
    assert_eq!(short.finish, FinishReason::Eos, "the unconstrained answer must end on the model's end token");
    let want = short.tokens.len() as u32 + 8;
    let long = drain(&model, &GenerateDesc { max_new_tokens: 128, min_new_tokens: want, ..Default::default() }, 200);
    assert!(long.tokens.len() as u32 >= want, "{} tokens generated, {want} required", long.tokens.len());
    assert!(
        long.tokens.len() > short.tokens.len(),
        "the floor did not outlast the end token: {} tokens against {}",
        long.tokens.len(),
        short.tokens.len()
    );
}

#[test]
fn live_generate_refuses_what_it_cannot_honor() {
    let Some((live, model)) = setup() else { return };
    let schema = GenerateDesc {
        structured_kind: StructuredKind::JsonSchema,
        structured: "{\"type\":\"object\"}".into(),
        ..Default::default()
    };
    // A JSON schema needs both bits; with either clear the refusal names
    // structured_kind, and there is no third behavior.
    let json_schema = abi::TURBO_CAP_OPT_GEN_STRUCTURED | abi::TURBO_CAP_OPT_GEN_JSON_SCHEMA;
    if live.has_cap(json_schema) {
        model.create_generation(&schema).expect("the device advertises JSON-schema structured output");
    } else {
        let e = model.create_generation(&schema).expect_err("the device does not advertise JSON schema output");
        assert_eq!(e.code(), abi::TURBO_E_UNSUPPORTED_OPTION, "{e}");
        assert_eq!(e.field(), GenerateDesc::FIELD_STRUCTURED_KIND);
    }
    let e = model.create_generation(&GenerateDesc { n_sequences: 2, ..Default::default() }).unwrap_err();
    assert_eq!(e.code(), abi::TURBO_E_UNSUPPORTED_OPTION, "{e}");
    assert_eq!(e.field(), GenerateDesc::FIELD_N_SEQUENCES);
}
