//! Extended tests for the optional chunking utility (`turboembed::chunker`).
//!
//! `chunker_contract.rs` pins the core contract on fixed fixtures; this file
//! extends coverage to edge-case inputs (empty and single-word sources, CRLF
//! vs LF line endings, multibyte scripts), adversarial token budgets, custom
//! [`TokenCounter`] implementations (closures, a named struct, and a real
//! `tokenizers` tokenizer), plan determinism under repetition, and offsets
//! relative to the caller's exact input string. No inference runs and no
//! goldens change: chunking only plans byte ranges over caller text.

use serde::Serialize;
use turboembed::chunker::{
    chunk_source, paragraph_spans, ChunkPlan, ChunkerConfig, SourceChunk, TokenCounter,
};

const EXCERPT: &str =
    include_str!("../../../testdata/corpus/fixtures/tiny-shakespeare.excerpt.txt");

/// Deterministic fixture counter: whitespace-separated words. Stands in for a
/// model tokenizer; the contract only requires a deterministic count.
fn words(text: &str) -> usize {
    text.split_whitespace().count()
}

/// Deterministic fixture counter: one token per Unicode scalar value.
fn per_char(text: &str) -> usize {
    text.chars().count()
}

fn config(max_tokens: usize, reserved: usize, overlap: usize) -> ChunkerConfig {
    ChunkerConfig {
        max_tokens,
        reserved_tokens: reserved,
        overlap_tokens: overlap,
        sentence_boundaries: true,
    }
}

fn plan(text: &str, cfg: &ChunkerConfig) -> ChunkPlan {
    chunk_source("fixture", text, cfg, &words).expect("chunking succeeds")
}

/// Invariants every plan must satisfy regardless of input, config, or counter.
fn assert_plan_invariants(
    text: &str,
    cfg: &ChunkerConfig,
    p: &ChunkPlan,
    counter: &dyn TokenCounter,
) {
    let budget = cfg.content_budget();
    let paras = paragraph_spans(text);
    assert_eq!(p.config, *cfg, "plan echoes the config that produced it");
    let mut prev: Option<&SourceChunk> = None;
    for (i, c) in p.chunks.iter().enumerate() {
        assert_eq!(c.index, i, "indices are dense and in source order");
        assert!(c.byte_range.start < c.byte_range.end);
        assert!(c.byte_range.end <= text.len());
        assert!(
            text.is_char_boundary(c.byte_range.start) && text.is_char_boundary(c.byte_range.end),
            "offsets land on char boundaries"
        );
        let slice = c.text(text);
        assert_eq!(
            slice,
            &text[c.byte_range.clone()],
            "slice is the source text"
        );
        assert_eq!(c.token_count, counter.count_tokens(slice));
        assert!(
            c.token_count <= budget,
            "chunk {i} uses {} of {budget} content tokens",
            c.token_count
        );
        let para = &paras[c.paragraph_index];
        assert!(
            c.byte_range.start >= para.start && c.byte_range.end <= para.end,
            "chunk {i} stays inside its paragraph"
        );
        if let Some(prev) = prev {
            assert!(
                c.byte_range.end > prev.byte_range.end,
                "strict forward progress"
            );
            if cfg.overlap_tokens == 0 {
                assert!(
                    c.byte_range.start >= prev.byte_range.end,
                    "zero-overlap plans are disjoint"
                );
            }
        }
        prev = Some(c);
    }
}

#[test]
fn empty_and_blank_inputs_echo_identity_and_ignore_the_counter() {
    let cfg = config(16, 2, 0);
    for text in ["", "   \n\t\r\n  ", "\u{00a0}\u{2000}\u{3000}"] {
        let by_words = chunk_source("empty", text, &cfg, &words).expect("chunking succeeds");
        let by_chars = chunk_source("empty", text, &cfg, &per_char).expect("chunking succeeds");
        assert!(by_words.chunks.is_empty());
        assert_eq!(by_words.source, "empty");
        assert_eq!(by_words.config, cfg);
        assert_eq!(
            by_words, by_chars,
            "empty plans do not depend on the counter"
        );
    }
}

#[test]
fn single_word_text_yields_one_trimmed_chunk() {
    let cfg = config(16, 2, 0);
    let p = plan("famish?", &cfg);
    assert_eq!(p.chunks.len(), 1);
    let c = &p.chunks[0];
    assert_eq!(c.text("famish?"), "famish?");
    assert_eq!(c.byte_range, 0..7);
    assert_eq!(c.token_count, 1);
    // Surrounding whitespace and trailing blank lines do not leak in.
    let spaced = "   famish?\n\n";
    let q = plan(spaced, &cfg);
    assert_eq!(q.chunks.len(), 1);
    assert_eq!(q.chunks[0].text(spaced), "famish?");
    assert_eq!(q.chunks[0].byte_range, 3..10);
}

#[test]
fn crlf_and_lf_blank_lines_produce_identical_chunk_texts() {
    let lf = "alpha beta\n\ngamma delta\n\nepsilon zeta";
    let crlf = "alpha beta\r\n\r\ngamma delta\r\n\r\nepsilon zeta";
    let cfg = config(64, 2, 0);
    let a = plan(lf, &cfg);
    let b = plan(crlf, &cfg);
    assert_eq!(a.chunks.len(), 3);
    assert_eq!(a.chunks.len(), b.chunks.len());
    for (ca, cb) in a.chunks.iter().zip(&b.chunks) {
        assert_eq!(
            ca.text(lf),
            cb.text(crlf),
            "line endings do not change content"
        );
        assert!(
            !cb.text(crlf).contains('\r'),
            "CR is trimmed, never content"
        );
        assert_eq!(cb.token_count, words(cb.text(crlf)));
    }
    assert_eq!(crlf.len(), lf.len() + 4);
    assert!(b.chunks[1].byte_range.start > a.chunks[1].byte_range.start);
    // Paragraph spans trim the carriage return from line endings.
    let spans = paragraph_spans(crlf);
    assert_eq!(spans.len(), 3);
    assert_eq!(&crlf[spans[0].clone()], "alpha beta");
    assert_eq!(&crlf[spans[2].clone()], "epsilon zeta");
}

#[test]
fn mixed_line_endings_keep_interior_crlf_but_trim_edges() {
    let text = "a\r\nb\n\nc\r\n";
    let spans = paragraph_spans(text);
    assert_eq!(spans.len(), 2);
    assert_eq!(
        &text[spans[0].clone()],
        "a\r\nb",
        "interior CRLF is content"
    );
    assert_eq!(&text[spans[1].clone()], "c", "trailing CR is trimmed");
    // A line holding only a carriage return counts as a blank separator.
    let text2 = "x\n\r\ny";
    let spans2 = paragraph_spans(text2);
    assert_eq!(spans2.len(), 2);
    assert_eq!(&text2[spans2[0].clone()], "x");
    assert_eq!(&text2[spans2[1].clone()], "y");
}

#[test]
fn multibyte_mixed_scripts_offsets_are_byte_based_char_aligned_and_cover_source() {
    let text = "Café 😀 日本語のテキスト です。";
    let cfg = config(6, 2, 0);
    let p = chunk_source("multi", text, &cfg, &per_char).expect("chunking succeeds");
    assert_plan_invariants(text, &cfg, &p, &per_char);
    // Zero-overlap packing covers every non-whitespace byte of the source.
    let first = p.chunks.first().expect("nonempty plan");
    let last = p.chunks.last().expect("nonempty plan");
    assert_eq!(first.byte_range.start, text.len() - text.trim_start().len());
    assert_eq!(last.byte_range.end, text.trim_end().len());
    for w in p.chunks.windows(2) {
        let gap = &text[w[0].byte_range.end..w[1].byte_range.start];
        assert!(
            gap.trim().is_empty(),
            "gap between chunks is whitespace only"
        );
    }
    // The 4-byte emoji starts a chunk at its *byte* offset, not its char index.
    let emoji_at = text.find('😀').expect("emoji present");
    let emoji_chunk = p
        .chunks
        .iter()
        .find(|c| c.byte_range.start == emoji_at)
        .expect("emoji starts a chunk");
    assert_eq!(emoji_chunk.text(text), "😀");
    assert_eq!(emoji_at, 6, "é is 2 bytes, so the emoji sits at byte 6");
    assert_eq!(
        text[..emoji_at].chars().count(),
        5,
        "char index differs from the byte offset"
    );
}

#[test]
fn budget_smaller_than_first_word_splits_inside_the_first_word() {
    // Per-char counter, budget 5: the 20-char first word must be char-split.
    let text = "supercalifragilistic expialidocious";
    let cfg = config(7, 2, 0);
    let p = chunk_source("tight", text, &cfg, &per_char).expect("chunking succeeds");
    assert_eq!(p.chunks[0].text(text), "super", "longest 5-char prefix");
    assert_eq!(p.chunks[0].byte_range, 0..5);
    assert_eq!(p.chunks[0].token_count, cfg.content_budget());
    // Byte-counting counter with a multibyte first word: budget 6 bytes < 15.
    let cjk = "日本語テスト later";
    let by_bytes = |t: &str| t.len();
    let byte_cfg = config(8, 2, 0);
    let q = chunk_source("tight-bytes", cjk, &byte_cfg, &by_bytes).expect("chunking succeeds");
    assert_eq!(
        q.chunks[0].text(cjk),
        "日本",
        "longest char prefix within 6 bytes"
    );
    assert_eq!(q.chunks[0].byte_range, 0..6);
    assert_eq!(q.chunks[0].token_count, 6);
    assert_plan_invariants(cjk, &byte_cfg, &q, &by_bytes);
}

#[test]
fn budget_exactly_at_a_token_boundary_fills_the_chunk_precisely() {
    let text = "alpha beta gamma delta";
    let cfg = config(4, 2, 0); // content budget: 2 words
    let p = plan(text, &cfg);
    assert_eq!(p.chunks.len(), 2);
    assert_eq!(p.chunks[0].text(text), "alpha beta");
    assert_eq!(p.chunks[0].byte_range, 0..10);
    assert_eq!(p.chunks[1].text(text), "gamma delta");
    assert_eq!(p.chunks[1].byte_range, 11..22);
    // Budget equal to the whole text is fully usable, not off by one token.
    let exact = config(6, 2, 0); // content budget: 4 words == total
    let q = plan(text, &exact);
    assert_eq!(q.chunks.len(), 1);
    assert_eq!(q.chunks[0].text(text), text);
    assert_eq!(q.chunks[0].token_count, exact.content_budget());
}

#[test]
fn constant_counter_treats_any_nonempty_span_as_one_token() {
    let one = |_t: &str| 1usize;
    let text = "one two three four five\n\nsix seven eight nine\n\nten eleven";
    let generous = config(64, 2, 0);
    let p = chunk_source("const", text, &generous, &one).expect("chunking succeeds");
    assert_eq!(p.chunks.len(), 3, "one chunk per paragraph at any length");
    assert!(p.chunks.iter().all(|c| c.token_count == 1));
    // A content budget of exactly 1 still fits every paragraph.
    let minimal = config(4, 3, 0);
    let q = chunk_source("const", text, &minimal, &one).expect("chunking succeeds");
    assert_eq!(q.chunks.len(), 3);
    assert_plan_invariants(text, &minimal, &q, &one);
}

struct CharCounter;

impl TokenCounter for CharCounter {
    fn count_tokens(&self, text: &str) -> usize {
        text.chars().count()
    }
}

#[test]
fn named_struct_implements_token_counter_without_closures() {
    let text = "Café 日本語 😀";
    let cfg = config(6, 2, 0);
    let counter = CharCounter;
    let p = chunk_source("struct", text, &cfg, &counter).expect("chunking succeeds");
    assert_plan_invariants(text, &cfg, &p, &counter);
    assert!(!p.chunks.is_empty());
}

#[derive(Serialize)]
struct Fingerprint {
    source: String,
    max_tokens: usize,
    reserved_tokens: usize,
    overlap_tokens: usize,
    sentence_boundaries: bool,
    chunks: Vec<[usize; 5]>,
}

fn fingerprint(p: &ChunkPlan) -> String {
    let f = Fingerprint {
        source: p.source.clone(),
        max_tokens: p.config.max_tokens,
        reserved_tokens: p.config.reserved_tokens,
        overlap_tokens: p.config.overlap_tokens,
        sentence_boundaries: p.config.sentence_boundaries,
        chunks: p
            .chunks
            .iter()
            .map(|c| {
                [
                    c.index,
                    c.paragraph_index,
                    c.byte_range.start,
                    c.byte_range.end,
                    c.token_count,
                ]
            })
            .collect(),
    };
    serde_json::to_string(&f).expect("fingerprint serializes")
}

#[test]
fn repeat_runs_produce_identical_plans_and_fingerprints() {
    let text = "😀 Café mûre. 日本語 beta!\n\nGamma delta épsilon zeta?\n\n\tOmega 終わり";
    let cfg = config(6, 2, 1);
    let first = chunk_source("run", text, &cfg, &per_char).expect("chunking succeeds");
    for _ in 0..2 {
        let again = chunk_source("run", text, &cfg, &per_char).expect("chunking succeeds");
        assert_eq!(first, again, "same input, config, counter => same plan");
        assert_eq!(fingerprint(&first), fingerprint(&again));
    }
    assert_plan_invariants(text, &cfg, &first, &per_char);
}

#[test]
fn offsets_index_the_passed_string_not_a_global_coordinate() {
    let prefix = "PAD WORDS\n\n";
    let combined = format!("{prefix}{EXCERPT}");
    let cfg = config(24, 2, 0);
    let base = plan(EXCERPT, &cfg);
    let shifted = plan(&combined, &cfg);
    assert_eq!(shifted.chunks.len(), base.chunks.len() + 1);
    assert_eq!(shifted.chunks[0].text(&combined), "PAD WORDS");
    for (i, (a, b)) in base.chunks.iter().zip(&shifted.chunks[1..]).enumerate() {
        assert_eq!(
            b.byte_range.start,
            a.byte_range.start + prefix.len(),
            "chunk {i} is offset by exactly the prefix length"
        );
        assert_eq!(b.byte_range.end, a.byte_range.end + prefix.len());
        assert_eq!(a.text(EXCERPT), b.text(&combined));
        assert_eq!(b.paragraph_index, a.paragraph_index + 1);
        assert_eq!(b.index, a.index + 1);
        assert_eq!(b.token_count, a.token_count);
    }
}

#[test]
fn every_config_field_variation_preserves_the_contract() {
    for max_tokens in [8usize, 24] {
        for reserved in [0usize, 2] {
            for overlap in [0usize, 1] {
                for sentence_boundaries in [false, true] {
                    let cfg = ChunkerConfig {
                        max_tokens,
                        reserved_tokens: reserved,
                        overlap_tokens: overlap,
                        sentence_boundaries,
                    };
                    let p = plan(EXCERPT, &cfg);
                    assert_plan_invariants(EXCERPT, &cfg, &p, &words);
                }
            }
        }
    }
}

#[test]
fn sentence_boundaries_field_changes_where_paragraphs_split() {
    let text = "aaaa bbbb. cc dd. ee ff.";
    let on = config(5, 2, 0); // content budget: 3 words
    let mut off = on.clone();
    off.sentence_boundaries = false;
    let p = plan(text, &on);
    let q = plan(text, &off);
    let on_texts: Vec<&str> = p.chunks.iter().map(|c| c.text(text)).collect();
    assert_eq!(on_texts, ["aaaa bbbb.", "cc dd.", "ee ff."]);
    let off_texts: Vec<&str> = q.chunks.iter().map(|c| c.text(text)).collect();
    assert_eq!(off_texts, ["aaaa bbbb. cc", "dd. ee ff."]);
    assert_ne!(p, q, "the flag is observable, not advisory");
}

#[test]
fn real_tokenizer_from_tokenizers_crate_plugs_in_as_counter() {
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;
    use tokenizers::Tokenizer;

    // Offline word-level tokenizer whose vocab covers every fixture word.
    // WhitespaceSplit matches split_whitespace, so encode lengths must equal
    // the fixture `words` counter on any span.
    let mut vocab: std::collections::BTreeMap<String, u32> = std::collections::BTreeMap::new();
    vocab.insert("<unk>".to_string(), 0);
    for (i, w) in EXCERPT.split_whitespace().enumerate() {
        vocab.entry(w.to_string()).or_insert(i as u32 + 1);
    }
    let path = std::env::temp_dir().join(format!(
        "turboembed-chunker-extended-wordlevel-{}.json",
        std::process::id()
    ));
    std::fs::write(
        &path,
        serde_json::to_string(&vocab).expect("vocab serializes"),
    )
    .expect("write temp vocab");
    let mut tok = Tokenizer::new(
        WordLevel::from_file(path.to_str().expect("utf-8 temp path"), "<unk>".to_string())
            .expect("wordlevel model builds"),
    );
    std::fs::remove_file(&path).ok();
    tok.with_pre_tokenizer(Some(WhitespaceSplit));

    let real = |t: &str| tok.encode(t, false).expect("encode").get_ids().len();
    for span in [EXCERPT, "First Citizen:", "We know't, we know't."] {
        assert_eq!(
            real(span),
            words(span),
            "tokenizer agrees with word counter"
        );
    }
    let cfg = config(16, 2, 0);
    let via_tok = chunk_source("fixture", EXCERPT, &cfg, &real).expect("chunking succeeds");
    let via_words = plan(EXCERPT, &cfg);
    assert_eq!(via_tok, via_words, "real tokenizer plans identically here");
    for c in &via_tok.chunks {
        assert!(
            real(c.text(EXCERPT)) <= cfg.content_budget(),
            "budget holds under the real tokenizer"
        );
    }
}
