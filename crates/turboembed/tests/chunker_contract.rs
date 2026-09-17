//! Contract tests for the optional chunking utility (`turboembed::chunker`).
//!
//! These tests use fixed fixtures and fixed deterministic token counters.
//! They never change embedding goldens: the chunker only plans byte ranges
//! over caller text and performs no inference.

use turboembed::chunker::{
    chunk_source, paragraph_spans, sentence_spans, ChunkError, ChunkPlan, ChunkerConfig,
    SourceChunk,
};

const EXCERPT: &str =
    include_str!("../../../testdata/corpus/fixtures/tiny-shakespeare.excerpt.txt");

/// Deterministic fixture counter: whitespace-separated words. Stands in for a
/// model tokenizer; the contract only requires a deterministic count.
fn words(text: &str) -> usize {
    text.split_whitespace().count()
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

fn assert_invariants(text: &str, cfg: &ChunkerConfig, plan: &ChunkPlan) {
    let budget = cfg.content_budget();
    let mut prev: Option<&SourceChunk> = None;
    for (i, c) in plan.chunks.iter().enumerate() {
        assert_eq!(c.index, i, "indices are dense and in source order");
        // Offsets are half-open byte offsets into the original input.
        assert!(c.byte_range.start < c.byte_range.end);
        assert!(c.byte_range.end <= text.len());
        let slice = c.text(text);
        assert_eq!(&text[c.byte_range.clone()], slice);
        assert!(!slice.trim().is_empty(), "chunks carry content");
        // Budget holds for the exact chunk text, by the same counter.
        assert_eq!(c.token_count, words(slice));
        assert!(
            c.token_count <= budget,
            "chunk {i} exceeds budget: {} > {budget}",
            c.token_count
        );
        if let Some(p) = prev {
            // Forward progress: start and end strictly advance.
            assert!(c.byte_range.start > p.byte_range.start);
            assert!(c.byte_range.end > p.byte_range.end);
            assert!(c.paragraph_index >= p.paragraph_index);
            if c.paragraph_index == p.paragraph_index {
                let overlap_bytes = p.byte_range.end.saturating_sub(c.byte_range.start);
                if overlap_bytes > 0 {
                    let shared = &text[c.byte_range.start..p.byte_range.end];
                    assert!(
                        words(shared) <= cfg.overlap_tokens,
                        "overlap exceeds configured token bound"
                    );
                } else {
                    assert!(c.byte_range.start >= p.byte_range.end);
                }
            } else {
                assert!(
                    c.byte_range.start >= p.byte_range.end,
                    "chunks never span or overlap across paragraphs"
                );
            }
        }
        prev = Some(c);
    }
}

#[test]
fn fixture_plan_is_deterministic_and_offset_true() {
    let cfg = config(16, 2, 0);
    let a = plan(EXCERPT, &cfg);
    let b = plan(EXCERPT, &cfg);
    assert_eq!(a, b, "same input, config, counter => same plan");
    assert!(!a.chunks.is_empty());
    assert_eq!(a.source, "fixture");
    assert_eq!(a.config, cfg);
    assert_invariants(EXCERPT, &cfg, &a);
    // Every byte of every chunk is original source text, not normalized copy.
    for c in &a.chunks {
        assert!(EXCERPT
            .as_bytes()
            .windows(c.byte_range.len())
            .any(|w| w == c.text(EXCERPT).as_bytes()));
    }
}

#[test]
fn fixture_covers_all_paragraph_content() {
    // A budget large enough for every paragraph gives exactly one chunk per
    // paragraph, equal to the trimmed paragraph span.
    let cfg = config(4096, 2, 0);
    let p = plan(EXCERPT, &cfg);
    let paras = paragraph_spans(EXCERPT);
    assert_eq!(p.chunks.len(), paras.len());
    assert_eq!(paras.len(), 12, "fixture paragraph count is pinned");
    for (c, pr) in p.chunks.iter().zip(&paras) {
        assert_eq!(c.byte_range, *pr);
    }
    assert_invariants(EXCERPT, &cfg, &p);
}

#[test]
fn small_budget_splits_at_sentences_then_words() {
    let cfg = config(8, 2, 0);
    let p = plan(EXCERPT, &cfg);
    assert!(p.chunks.len() > 12, "small budget splits paragraphs");
    assert_invariants(EXCERPT, &cfg, &p);
}

#[test]
fn overlap_is_bounded_and_within_paragraph_only() {
    let text = "one two three four five six seven eight nine ten\n\nalpha beta";
    let cfg = config(6, 2, 2);
    let p = plan(text, &cfg);
    assert_invariants(text, &cfg, &p);
    // The first paragraph needs multiple chunks; consecutive ones overlap.
    let first_para: Vec<_> = p.chunks.iter().filter(|c| c.paragraph_index == 0).collect();
    assert!(first_para.len() >= 2);
    let overlapped = first_para
        .windows(2)
        .any(|w| w[1].byte_range.start < w[0].byte_range.end);
    assert!(
        overlapped,
        "requested overlap is applied within a paragraph"
    );
    // Second paragraph starts fresh.
    let second = p
        .chunks
        .iter()
        .find(|c| c.paragraph_index == 1)
        .expect("second paragraph chunk");
    assert_eq!(second.text(text), "alpha beta");
}

#[test]
fn zero_overlap_produces_disjoint_chunks() {
    let text = "one two three four five six seven eight nine ten";
    let cfg = config(5, 2, 0);
    let p = plan(text, &cfg);
    assert_invariants(text, &cfg, &p);
    for w in p.chunks.windows(2) {
        assert!(w[1].byte_range.start >= w[0].byte_range.end);
    }
}

#[test]
fn long_span_without_punctuation_makes_forward_progress() {
    // No terminators, no whitespace: only char-boundary splitting applies.
    // Budget counts bytes/4 to force many splits.
    let text = "a".repeat(100);
    let by_bytes = |t: &str| t.len().div_ceil(4);
    let cfg = config(6, 2, 0);
    let p = chunk_source("wall", &text, &cfg, &by_bytes).expect("chunking succeeds");
    assert!(!p.chunks.is_empty());
    let mut cursor = 0usize;
    for c in &p.chunks {
        assert_eq!(c.byte_range.start, cursor, "no gaps, no stall");
        assert!(c.byte_range.end > c.byte_range.start);
        assert!(by_bytes(c.text(&text)) <= cfg.content_budget());
        cursor = c.byte_range.end;
    }
    assert_eq!(cursor, text.len(), "entire wall of text is covered");
}

#[test]
fn non_ascii_splits_on_char_boundaries() {
    // Multi-byte characters, no whitespace. Counter is one token per char.
    let text = "日本語のテキストです。これは分割の試験です。";
    let per_char = |t: &str| t.chars().count();
    let cfg = config(7, 2, 0);
    let p = chunk_source("ja", text, &cfg, &per_char).expect("chunking succeeds");
    assert!(!p.chunks.is_empty());
    for c in &p.chunks {
        // Slicing would panic on a non-boundary; also verify explicitly.
        assert!(text.is_char_boundary(c.byte_range.start));
        assert!(text.is_char_boundary(c.byte_range.end));
        assert!(per_char(c.text(text)) <= cfg.content_budget());
    }
}

#[test]
fn embedded_nul_and_crlf_offsets_point_into_original() {
    let text = "first line\r\nsecond\0line\r\n\r\nnext paragraph";
    let cfg = config(64, 2, 0);
    let p = plan(text, &cfg);
    assert_eq!(p.chunks.len(), 2, "CRLF blank line separates paragraphs");
    assert_eq!(p.chunks[0].text(text), "first line\r\nsecond\0line");
    assert_eq!(p.chunks[1].text(text), "next paragraph");
    assert_eq!(&text[p.chunks[1].byte_range.clone()], "next paragraph");
    assert_invariants(text, &cfg, &p);
}

#[test]
fn empty_and_whitespace_only_inputs_produce_empty_plans() {
    let cfg = config(16, 2, 0);
    assert!(plan("", &cfg).chunks.is_empty());
    assert!(plan("   \n\t\r\n  ", &cfg).chunks.is_empty());
}

#[test]
fn invalid_configs_are_rejected() {
    let err = chunk_source("s", "text", &config(2, 2, 0), &words).unwrap_err();
    assert!(matches!(err, ChunkError::InvalidConfig(_)));
    let err = chunk_source("s", "text", &config(4, 2, 2), &words).unwrap_err();
    assert!(matches!(err, ChunkError::InvalidConfig(_)));
}

#[test]
fn oversized_single_character_is_an_explicit_error() {
    // Counter that makes any nonempty text exceed the budget: no boundary can
    // make forward progress, so chunking must fail loudly, not loop or drop.
    let huge = |t: &str| if t.is_empty() { 0 } else { 1000 };
    let err = chunk_source("s", "x", &config(16, 2, 0), &huge).unwrap_err();
    assert_eq!(err, ChunkError::NoProgress { byte_offset: 0 });
}

#[test]
fn sentence_spans_match_promoted_e2e_semantics() {
    let para = "Let us kill him, and we'll have corn at our own price.\nIs't a verdict?";
    let spans = sentence_spans(para);
    assert_eq!(spans.len(), 2);
    assert!(para[spans[0].clone()].starts_with("Let us kill him"));
    assert_eq!(&para[spans[1].clone()], "Is't a verdict?");
    // Abbreviations do not split.
    assert_eq!(
        sentence_spans("Dr. Caius met Mr. Marcius vs. the city.").len(),
        1
    );
    // A block without terminators yields no sentence spans.
    assert!(sentence_spans("Alone without a stop").is_empty());
}

#[test]
fn paragraph_spans_are_offsets_into_unnormalized_input() {
    let text = "  a\n\n\nb  \nc\n";
    let spans = paragraph_spans(text);
    assert_eq!(spans.len(), 2);
    assert_eq!(&text[spans[0].clone()], "a");
    assert_eq!(&text[spans[1].clone()], "b  \nc");
    assert!(paragraph_spans("").is_empty());
}
