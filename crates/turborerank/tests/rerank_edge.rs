//! Rerank edge cases on the live CPU MiniLM-L6 cross-encoder: empty and
//! unicode inputs, truncation, determinism, activation math, and error
//! contracts. Weights live under `models/rerank/` (gitignored; materialized
//! by `make fetch-rerankers`), so weight-dependent tests follow the
//! `cpu_minilm.rs` skip idiom to keep fresh checkouts green.
//! Golden-file conventions mirror `tests/cpu_minilm.rs`.

use std::fs;
use std::path::PathBuf;

use serde::Deserialize;
use turborerank::{
    default_model_dir, weights_present, Activation, Device, Engine, Error, TokenBuffer, Truncation,
};

const QUERY: &str = "How many people live in Berlin?";
const REL: &str = "Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.";
const MID: &str = "Berlin is well known for its museums.";
const IRREL: &str = "New York City is famous for its pizza and bagels.";
const DOCS: &[&str] = &[REL, MID, IRREL];

/// Skips the test (rather than failing) when weights are absent, matching
/// the `cpu_minilm.rs` idiom: checkouts without `make fetch-rerankers`
/// must keep `cargo test --locked --workspace` green.
fn load_cpu() -> Option<Engine> {
    if !weights_present() {
        eprintln!(
            "skipping: weights missing at {} (run `make fetch-rerankers`)",
            default_model_dir().display()
        );
        return None;
    }
    let dir = default_model_dir();
    let engine = Engine::create_with_config(Device::Cpu, Some(&dir)).expect("CPU engine");
    engine
        .load_model("ms-marco-minilm-l6")
        .unwrap_or_else(|e| panic!("load MiniLM CE: {e} ({})", engine.last_error()));
    Some(engine)
}

fn reject_mock_scores(scores: &[f32]) {
    assert!(!scores.is_empty());
    let all_equal = scores.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-8);
    assert!(
        !all_equal,
        "FAKE: all scores equal {scores:?} — constant/mock scorer"
    );
    let only_unit = scores
        .iter()
        .all(|s| (*s == 0.0) || (*s == 1.0) || (*s == 0.5));
    assert!(
        !only_unit,
        "FAKE: scores look like the word-overlap mock {scores:?}"
    );
}

fn assert_finite(scores: &[f32]) {
    assert!(
        scores.iter().all(|s| s.is_finite()),
        "non-finite score in {scores:?}"
    );
}

fn score(
    engine: &Engine,
    query: &str,
    docs: &[&str],
    truncation: Truncation,
    activation: Activation,
    max_length: u32,
) -> Vec<f32> {
    engine
        .score(None, query, docs, truncation, activation, max_length)
        .unwrap_or_else(|e| panic!("score({query:?}, {docs:?}): {e} ({})", engine.last_error()))
}

#[test]
fn empty_query_scores_document() {
    let Some(engine) = load_cpu() else {
        return;
    };
    let scores = score(
        &engine,
        "",
        DOCS,
        Truncation::LongestFirst,
        Activation::Identity,
        512,
    );
    assert_eq!(scores.len(), 3);
    assert_finite(&scores);
    reject_mock_scores(&scores);
    // Relevance signal survives an absent query; ordering must not flip.
    assert!(
        scores[0] > scores[2],
        "expected Berlin > NYC, got {scores:?}"
    );

    // Pack level: empty query is just [CLS] [SEP] before the document.
    let mut buf = TokenBuffer::alloc(Device::Cpu, 1, 64).unwrap();
    engine
        .pack_text(&mut buf, 0, "", REL, Truncation::LongestFirst, 64)
        .unwrap();
    assert_eq!(&buf.input_ids()[..2], &[101, 102]);
    assert_eq!(&buf.token_type_ids()[..2], &[0, 0]);
    // 3 specials ([CLS] [SEP] [SEP]) + 24 document tokens for the Berlin
    // golden document (see input_ids_row0 in the golden file).
    let used: i32 = buf.attention_mask().iter().sum();
    assert_eq!(used, 27, "mask ones {used}");
    assert_eq!(buf.input_ids()[26], 102, "row ends with the document [SEP]");
}

#[test]
fn empty_document_scores() {
    let Some(engine) = load_cpu() else {
        return;
    };
    let scores = score(
        &engine,
        QUERY,
        &[""],
        Truncation::LongestFirst,
        Activation::Identity,
        512,
    );
    assert_finite(&scores);
    // Pack level: [CLS] query… [SEP] [SEP] with an empty document side.
    let mut buf = TokenBuffer::alloc(Device::Cpu, 1, 64).unwrap();
    engine
        .pack_text(&mut buf, 0, QUERY, "", Truncation::LongestFirst, 64)
        .unwrap();
    assert_eq!(
        &buf.input_ids()[..9],
        &[101, 2129, 2116, 2111, 2444, 1999, 4068, 1029, 102]
    );
    assert_eq!(buf.input_ids()[9], 102);
    let used: i32 = buf.attention_mask().iter().sum();
    assert_eq!(used, 10, "7 query tokens + 3 specials, got {used}");
}

#[test]
fn empty_query_and_document_rejected() {
    let Some(engine) = load_cpu() else {
        return;
    };
    let err = engine
        .score(
            None,
            "",
            &[""],
            Truncation::LongestFirst,
            Activation::Identity,
            512,
        )
        .unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(_)), "{err:?}");
    assert!(
        err.to_string().contains("empty"),
        "error should name the empty-input contract: {err}"
    );
    assert!(
        engine
            .last_error()
            .contains("empty query and empty document"),
        "last_error: {}",
        engine.last_error()
    );
}

#[test]
fn zero_documents_rejected() {
    let Some(engine) = load_cpu() else {
        return;
    };
    let err = engine
        .score(
            None,
            QUERY,
            &[],
            Truncation::LongestFirst,
            Activation::Identity,
            512,
        )
        .unwrap_err();
    // Wrapper-level contract: rejected before the native call.
    assert!(matches!(err, Error::InvalidArgument(_)), "{err:?}");
    assert!(
        err.to_string().contains("documents must not be empty"),
        "{err}"
    );
}

#[test]
fn unicode_emoji_and_mixed_scripts_score() {
    let Some(engine) = load_cpu() else {
        return;
    };
    let docs = [
        "Berlin 🐻🎉 has great museums 🍻 and galleries.",
        "柏林是德国的首都，人口超过三百万。",
        "café naïve Zürich — über alles. e\u{0301}tude",
        "שלום ברלין, עיר של תרבות ומוזיאונים",
        "🚀🚀🚀",
    ];
    let scores = score(
        &engine,
        QUERY,
        &docs,
        Truncation::LongestFirst,
        Activation::Identity,
        512,
    );
    assert_eq!(scores.len(), 5);
    assert_finite(&scores);
    // Only the pure-emoji row may plausibly collapse toward UNK soup; the
    // mixed English row must still beat it for the Berlin query.
    assert!(
        scores[0] > scores[4],
        "emoji-only row should rank last: {scores:?}"
    );
}

#[test]
fn embedded_nul_and_control_chars_are_deleted() {
    let Some(engine) = load_cpu() else {
        return;
    };
    // The WordPiece tokenizer drops NUL and control code points without
    // flushing the in-progress word (native/wordpiece/encode.cpp skips them
    // before the whitespace flush), so "berlin\0museum" tokenizes exactly
    // like "berlinmuseum", not like "berlin museum". Pin that deletion
    // contract: scores are bitwise identical to the concatenated text.
    let with_nul = "berlin\u{0}museum district";
    let with_ctl = "berlin\u{0001}museum district";
    let with_sep = "berlin museum district";
    let deleted = "berlinmuseum district";
    let a = score(
        &engine,
        QUERY,
        &[with_nul],
        Truncation::LongestFirst,
        Activation::Identity,
        512,
    );
    let b = score(
        &engine,
        QUERY,
        &[with_ctl],
        Truncation::LongestFirst,
        Activation::Identity,
        512,
    );
    let c = score(
        &engine,
        QUERY,
        &[deleted],
        Truncation::LongestFirst,
        Activation::Identity,
        512,
    );
    let d = score(
        &engine,
        QUERY,
        &[with_sep],
        Truncation::LongestFirst,
        Activation::Identity,
        512,
    );
    assert_eq!(
        a[0].to_bits(),
        c[0].to_bits(),
        "NUL must be deleted, not a separator"
    );
    assert_eq!(
        b[0].to_bits(),
        c[0].to_bits(),
        "control char must be deleted, not a separator"
    );
    assert_ne!(
        a[0].to_bits(),
        d[0].to_bits(),
        "deleted NUL must not tokenize like a whitespace separator"
    );
}

#[test]
fn invalid_utf8_document_rejected() {
    let Some(engine) = load_cpu() else {
        return;
    };
    // Built at runtime so the deny-by-default invalid_from_utf8_unchecked
    // lint does not flag a literal. SAFETY: deliberately invalid UTF-8 to
    // assert the native boundary rejects it loudly instead of
    // mis-tokenizing the bytes.
    let bytes: Vec<u8> = vec![0x61, 0xFF, 0x62, 0xFE];
    let bad = unsafe { std::str::from_utf8_unchecked(&bytes) };
    let err = engine
        .score(
            None,
            QUERY,
            &[bad],
            Truncation::LongestFirst,
            Activation::Identity,
            512,
        )
        .unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(_)), "{err:?}");
    assert!(!engine.last_error().is_empty());
}

#[test]
fn long_document_truncates_and_error_truncation_fails_loud() {
    let Some(engine) = load_cpu() else {
        return;
    };
    let long_doc = "museum ".repeat(1500); // ~1500 tokens, far beyond 512.
    let truncated = score(
        &engine,
        QUERY,
        &[&long_doc],
        Truncation::LongestFirst,
        Activation::Identity,
        512,
    );
    assert_finite(&truncated);

    // Pack level: the row fills exactly max_length with [SEP] at the end.
    let mut buf = TokenBuffer::alloc(Device::Cpu, 1, 512).unwrap();
    engine
        .pack_text(&mut buf, 0, QUERY, &long_doc, Truncation::LongestFirst, 512)
        .unwrap();
    let used: i32 = buf.attention_mask().iter().sum();
    assert_eq!(used, 512, "truncated row must use the full budget");
    assert_eq!(buf.input_ids()[511], 102, "final position is [SEP]");

    // Tiny budget still scores (truncation, not failure).
    let tiny = score(
        &engine,
        QUERY,
        &[&long_doc],
        Truncation::LongestFirst,
        Activation::Identity,
        8,
    );
    assert_finite(&tiny);

    // Truncation::Error must refuse instead of silently truncating.
    let err = engine
        .score(
            None,
            QUERY,
            &[&long_doc],
            Truncation::Error,
            Activation::Identity,
            512,
        )
        .unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(_)), "{err:?}");
    assert!(
        engine.last_error().contains("TRUNC_ERROR"),
        "last_error: {}",
        engine.last_error()
    );

    // max_length below the 3-special minimum is invalid.
    let err = engine
        .score(
            None,
            QUERY,
            &[MID],
            Truncation::LongestFirst,
            Activation::Identity,
            2,
        )
        .unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(_)), "{err:?}");
}

#[test]
fn duplicate_documents_get_identical_scores() {
    let Some(engine) = load_cpu() else {
        return;
    };
    let dups = [REL, REL, REL];
    let scores = score(
        &engine,
        QUERY,
        &dups,
        Truncation::LongestFirst,
        Activation::Identity,
        512,
    );
    assert_eq!(scores.len(), 3);
    assert_eq!(
        scores[0].to_bits(),
        scores[1].to_bits(),
        "dup rows must match"
    );
    assert_eq!(
        scores[1].to_bits(),
        scores[2].to_bits(),
        "dup rows must match"
    );
}

#[test]
fn repeated_runs_are_bitwise_deterministic() {
    let Some(engine) = load_cpu() else {
        return;
    };
    for activation in [Activation::Identity, Activation::Sigmoid] {
        let first = score(
            &engine,
            QUERY,
            DOCS,
            Truncation::LongestFirst,
            activation,
            512,
        );
        for _ in 0..3 {
            let again = score(
                &engine,
                QUERY,
                DOCS,
                Truncation::LongestFirst,
                activation,
                512,
            );
            assert_eq!(
                first.iter().map(|s| s.to_bits()).collect::<Vec<_>>(),
                again.iter().map(|s| s.to_bits()).collect::<Vec<_>>(),
                "CPU CE must be run-to-run deterministic"
            );
        }
    }
}

#[test]
fn sigmoid_matches_identity_pointwise() {
    let Some(engine) = load_cpu() else {
        return;
    };
    let identity = score(
        &engine,
        QUERY,
        DOCS,
        Truncation::LongestFirst,
        Activation::Identity,
        512,
    );
    let sigmoid = score(
        &engine,
        QUERY,
        DOCS,
        Truncation::LongestFirst,
        Activation::Sigmoid,
        512,
    );
    for (i, (s, x)) in sigmoid.iter().zip(identity.iter()).enumerate() {
        let expect = 1.0f64 / (1.0 + (-(*x as f64)).exp());
        let got = *s as f64;
        assert!(
            (got - expect).abs() < 1e-6,
            "sigmoid[{i}] got {got} expected {expect} from logit {x}"
        );
    }
}

#[test]
fn sigmoid_output_in_open_unit_interval() {
    let Some(engine) = load_cpu() else {
        return;
    };
    // Golden texts span sigmoid ≈ 0.99986 down to ≈ 0.0000127.
    let sigmoid = score(
        &engine,
        QUERY,
        DOCS,
        Truncation::LongestFirst,
        Activation::Sigmoid,
        512,
    );
    for (i, s) in sigmoid.iter().enumerate() {
        assert!(*s > 0.0 && *s < 1.0, "sigmoid[{i}] = {s} outside (0,1)");
    }
}

#[test]
fn berlin_relevance_ordering_identity_and_sigmoid() {
    let Some(engine) = load_cpu() else {
        return;
    };
    let logits = score(
        &engine,
        QUERY,
        DOCS,
        Truncation::LongestFirst,
        Activation::Identity,
        512,
    );
    assert!(
        logits[0] > logits[1] && logits[1] > logits[2],
        "expected relevant > mid > irrelevant, got {logits:?}"
    );
    assert!(
        logits[0] - logits[2] > 2.0,
        "CE must separate Berlin pop vs NYC pizza by >2 logits, got {logits:?}"
    );
    let sigmoid = score(
        &engine,
        QUERY,
        DOCS,
        Truncation::LongestFirst,
        Activation::Sigmoid,
        512,
    );
    assert!(
        sigmoid[0] > sigmoid[1] && sigmoid[1] > sigmoid[2],
        "sigmoid must preserve ordering, got {sigmoid:?}"
    );
}

#[derive(Deserialize)]
struct GoldenFile {
    model: String,
    revision: String,
    activation: String,
    texts: GoldenTexts,
    logits: Vec<f32>,
    sigmoid: Vec<f32>,
}

#[derive(Deserialize)]
struct GoldenTexts {
    query: String,
    documents: Vec<String>,
}

#[test]
fn matches_committed_berlin_golden() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/reference_rerank/ms_marco_minilm_l6_berlin.json");
    assert!(
        path.is_file(),
        "committed golden missing at {}",
        path.display()
    );
    let raw = fs::read_to_string(&path).unwrap();
    let g: GoldenFile = serde_json::from_str(&raw).unwrap();
    assert!(g.model.contains("MiniLM"));
    assert_eq!(g.revision.len(), 40);
    assert_eq!(g.texts.query, QUERY);
    assert_eq!(g.activation, "identity+sigmoid");

    let Some(engine) = load_cpu() else {
        return;
    };
    let docs: Vec<&str> = g.texts.documents.iter().map(String::as_str).collect();
    let logits = score(
        &engine,
        &g.texts.query,
        &docs,
        Truncation::LongestFirst,
        Activation::Identity,
        512,
    );
    reject_mock_scores(&logits);
    assert_eq!(logits.len(), g.logits.len());
    // Same convention as tests/cpu_minilm.rs: absolute atol 2e-3.
    for (i, (got, exp)) in logits.iter().zip(g.logits.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 2e-3,
            "logit[{i}] got {got} expected {exp}"
        );
    }
    let sigmoid = score(
        &engine,
        &g.texts.query,
        &docs,
        Truncation::LongestFirst,
        Activation::Sigmoid,
        512,
    );
    for (i, (got, exp)) in sigmoid.iter().zip(g.sigmoid.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 2e-3,
            "sigmoid[{i}] got {got} expected {exp}"
        );
    }
}

#[test]
fn score_without_loaded_model_errors_with_last_error() {
    // No alias on an unloaded engine: loud UNAVAILABLE with detail.
    let unloaded = Engine::create(Device::Cpu).unwrap();
    let err = unloaded
        .score(
            None,
            QUERY,
            DOCS,
            Truncation::LongestFirst,
            Activation::Identity,
            512,
        )
        .unwrap_err();
    assert!(matches!(err, Error::Unavailable(_)), "{err:?}");
    assert!(
        unloaded.last_error().contains("model not loaded"),
        "last_error: {}",
        unloaded.last_error()
    );

    // Unknown alias on an unloaded engine: auto-load is attempted and fails.
    let unloaded = Engine::create(Device::Cpu).unwrap();
    let err = unloaded
        .score(
            Some("not-a-real-ce"),
            QUERY,
            DOCS,
            Truncation::LongestFirst,
            Activation::Identity,
            512,
        )
        .unwrap_err();
    assert!(matches!(err, Error::Unavailable(_)), "{err:?}");
    assert!(
        unloaded.last_error().contains("weights missing for alias"),
        "last_error: {}",
        unloaded.last_error()
    );

    // Known alias on an unloaded engine auto-loads and scores (native
    // turborerank_score loads the alias when engine->ready is false).
    // This case needs real weights; the error-path cases above do not.
    if !weights_present() {
        eprintln!("skipping auto-load case: weights missing (run `make fetch-rerankers`)");
        return;
    }
    let unloaded = Engine::create(Device::Cpu).unwrap();
    let auto = unloaded
        .score(
            Some("ms-marco-minilm-l6"),
            QUERY,
            DOCS,
            Truncation::LongestFirst,
            Activation::Identity,
            512,
        )
        .expect("score(Some(alias)) auto-loads on an unloaded engine");
    let Some(loaded) = load_cpu() else {
        return;
    };
    let direct = score(
        &loaded,
        QUERY,
        DOCS,
        Truncation::LongestFirst,
        Activation::Identity,
        512,
    );
    for (a, d) in auto.iter().zip(direct.iter()) {
        assert!(
            (a - d).abs() < 1e-4,
            "auto-load score {a} diverged from pre-loaded {d}"
        );
    }
}
