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
use turbo::ModelDesc;
use turbo_conformance::live::{bundle, live, Live};

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
            break;
        }
    }
    run
}

fn setup() -> Option<(Live, std::sync::Arc<turbo::Model>)> {
    let live = live()?;
    let dir = bundle("TURBO_LIVE_GGUF_BUNDLE")?;
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
    if live.device.kind == turbo::DeviceKind::Cpu {
        assert_eq!(a.tokens, b.tokens, "greedy decoding on the CPU is bit-reproducible");
    }
    if live.has_cap(abi::TURBO_CAP_OPT_GEN_SAMPLING | abi::TURBO_CAP_OPT_GEN_SEED) {
        let sampled =
            GenerateDesc { max_new_tokens: 12, temperature: 0.8, top_k: 40, seed: Some(7), ..Default::default() };
        let c = drain(&model, &sampled, 20);
        let d = drain(&model, &sampled, 20);
        assert_eq!(c.tokens, d.tokens, "the same seed reproduces the same sample");
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
    let want = short.tokens.len() as u32 + 8;
    let long = drain(&model, &GenerateDesc { max_new_tokens: 128, min_new_tokens: want, ..Default::default() }, 200);
    assert!(long.tokens.len() as u32 >= want, "{} tokens generated, {want} required", long.tokens.len());
}

#[test]
fn live_generate_refuses_what_it_cannot_honor() {
    let Some((_live, model)) = setup() else { return };
    let e = model
        .create_generation(&GenerateDesc {
            structured_kind: StructuredKind::JsonSchema,
            structured: "{\"type\":\"object\"}".into(),
            ..Default::default()
        })
        .unwrap_err();
    assert_eq!(e.code(), abi::TURBO_E_UNSUPPORTED_OPTION, "{e}");
    assert_eq!(e.field(), GenerateDesc::FIELD_STRUCTURED_KIND);
    let e = model.create_generation(&GenerateDesc { n_sequences: 2, ..Default::default() }).unwrap_err();
    assert_eq!(e.code(), abi::TURBO_E_UNSUPPORTED_OPTION, "{e}");
    assert_eq!(e.field(), GenerateDesc::FIELD_N_SEQUENCES);
}
