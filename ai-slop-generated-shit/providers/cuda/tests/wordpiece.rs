//! Word-span alignment for the provider's native WordPiece encoder.
//!
//! Token classification labels a word by reading the row columns its
//! `WordSpan` names, so a span that names columns holding something else
//! reports the wrong entity for the wrong bytes. This file checks the
//! invariant directly against the encoded row, over every budget from too
//! small to comfortable and both truncation directions, using the committed
//! MiniLM tokenizer fixture (no hardware and no bundle needed).

use std::path::PathBuf;

use turbo_core::types::Truncate;
use turbo_provider_cuda::wordpiece::{encode_row, RowScratch, Vocab};

/// Long compounds split into many sub-tokens, so a budget boundary lands
/// inside a word for most sequence lengths.
const TEXT: &str = "Konstantinopel Bartholomaeus Wolfgangsee Schwarzenegger Donaudampfschifffahrt \
Kraftfahrzeughaftpflichtversicherung Ada Lovelace visited Berlin";

/// Runs the normalizer drops entirely (a zero-width joiner, a soft hyphen,
/// a lone combining acute, a variation selector) sit between real words.
/// They occupy no column and must not be reported as words.
const INVISIBLE: &str = "New \u{200d} York \u{ad} is \u{301} here \u{fe0f} today";

fn vocab() -> Vocab {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/bundles/minilm-tokenizer/tokenizer.json");
    Vocab::load(&path).expect("load the committed MiniLM tokenizer fixture")
}

#[test]
fn word_spans_name_the_columns_that_hold_their_own_tokens() {
    let vocab = vocab();
    for truncate in [Truncate::Right, Truncate::Left] {
        for seq in 4u32..=48 {
            let mut ids = vec![0i32; seq as usize];
            let mut mask = vec![0i32; seq as usize];
            let mut scratch = RowScratch::new(seq);
            let mut words = Vec::new();
            let live = encode_row(
                &vocab,
                0,
                "",
                TEXT,
                truncate,
                seq,
                seq,
                &mut scratch,
                &mut ids,
                &mut mask,
                Some(&mut words),
            )
            .unwrap_or_else(|e| panic!("{truncate:?} seq {seq}: encode_row failed: {e}"));
            // The live region is [CLS] word pieces [SEP]; word columns must
            // fall strictly inside it.
            for w in &words {
                let word = &TEXT[w.start as usize..w.end as usize];
                let first = w.first_token as usize;
                let end = first + w.n_tokens as usize;
                assert!(
                    first >= 1 && end < live as usize,
                    "{truncate:?} seq {seq}: word `{word}` claims columns {first}..{end} but the row holds \
                     [CLS] plus {} content tokens plus [SEP]",
                    live - 2
                );
                let mut want = vec![0i32; 64];
                let n = vocab.tokenize_into(word.as_bytes(), &mut want).expect("tokenize the word alone");
                assert_eq!(
                    &ids[first..end],
                    &want[..n],
                    "{truncate:?} seq {seq}: word `{word}` claims columns {first}..{end}, which hold ids that are \
                     not its own tokens"
                );
            }
            // Spans are in row order and never overlap.
            for w in &words {
                assert!(w.n_tokens > 0, "{truncate:?} seq {seq}: a word span with no tokens: {w:?}");
            }
            for pair in words.windows(2) {
                assert!(
                    pair[0].first_token + pair[0].n_tokens < pair[1].first_token + pair[1].n_tokens,
                    "{truncate:?} seq {seq}: spans {:?} and {:?} overlap or are out of order",
                    pair[0],
                    pair[1]
                );
                assert!(pair[0].end <= pair[1].start, "{truncate:?} seq {seq}: byte ranges are out of order");
            }
        }
    }
}

#[test]
fn right_truncation_keeps_the_head_and_left_truncation_keeps_the_tail() {
    let vocab = vocab();
    let seq = 12;
    let mut ids = vec![0i32; seq as usize];
    let mut mask = vec![0i32; seq as usize];
    let mut scratch = RowScratch::new(seq);
    let mut words = Vec::new();
    let mut spans = |truncate: Truncate| -> Vec<&'static str> {
        encode_row(&vocab, 0, "", TEXT, truncate, seq, seq, &mut scratch, &mut ids, &mut mask, Some(&mut words))
            .expect("encode_row");
        words.iter().map(|w| &TEXT[w.start as usize..w.end as usize]).collect()
    };
    let right = spans(Truncate::Right);
    let left = spans(Truncate::Left);
    assert!(!right.is_empty(), "right truncation keeps the first words");
    assert!(!left.is_empty(), "left truncation keeps the last words");
    assert_eq!(right[0], "Konstantinopel", "right truncation keeps the head: {right:?}");
    assert_eq!(*left.last().expect("a last word"), "Berlin", "left truncation keeps the tail: {left:?}");
    assert!(!right.contains(&"Berlin"), "a 12 token window cannot hold both ends: {right:?}");
    assert!(!left.contains(&"Konstantinopel"), "a 12 token window cannot hold both ends: {left:?}");
}

#[test]
fn runs_the_normalizer_drops_are_not_words() {
    let vocab = vocab();
    let seq = 32u32;
    let mut ids = vec![0i32; seq as usize];
    let mut mask = vec![0i32; seq as usize];
    let mut scratch = RowScratch::new(seq);
    let mut words = Vec::new();
    let live = encode_row(
        &vocab,
        0,
        "",
        INVISIBLE,
        Truncate::Right,
        seq,
        seq,
        &mut scratch,
        &mut ids,
        &mut mask,
        Some(&mut words),
    )
    .expect("encode");
    let names: Vec<&str> = words.iter().map(|w| &INVISIBLE[w.start as usize..w.end as usize]).collect();
    assert_eq!(names, ["New", "York", "is", "here", "today"], "only visible words are reported: {names:?}");
    let mut col = 1;
    for w in &words {
        assert!(w.n_tokens > 0, "{w:?}");
        assert_eq!(w.first_token, col, "words name consecutive columns: {w:?}");
        col += w.n_tokens;
    }
    assert_eq!(col + 1, live, "the words cover every content column");
}

#[test]
fn left_truncation_handles_documents_longer_than_the_staging_row() {
    let vocab = vocab();
    let seq = 16u32;
    let long = "word ".repeat(seq as usize * 40);
    let mut ids = vec![0i32; seq as usize];
    let mut mask = vec![0i32; seq as usize];
    let mut scratch = RowScratch::new(seq);
    let live = encode_row(&vocab, 0, "", &long, Truncate::Left, seq, seq, &mut scratch, &mut ids, &mut mask, None)
        .expect("a long document is truncated, not refused");
    assert_eq!(live, seq, "the row is full");
}
