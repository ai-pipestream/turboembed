use super::*;
use turbo_analysis::{Spans, ViewId};

#[test]
fn whitespace_alone_does_not_change_the_id() {
    let a = ChunkId::of(Level::Chunk, &["Intro"], "Cats  purr.\n\nCats   nap.");
    let b = ChunkId::of(Level::Chunk, &[" Intro "], " Cats purr. Cats nap. ");
    assert_eq!(a, b);
    // A changed character, a changed level and a changed path all do.
    assert_ne!(a, ChunkId::of(Level::Chunk, &["Intro"], "Cats purr. Cats nap!"));
    assert_ne!(a, ChunkId::of(Level::Section, &["Intro"], "Cats purr. Cats nap."));
    assert_ne!(a, ChunkId::of(Level::Chunk, &["Other"], "Cats purr. Cats nap."));
    assert_ne!(a, ChunkId::of(Level::Chunk, &[], "Cats purr. Cats nap."));
}

#[test]
fn field_boundaries_cannot_be_smuggled() {
    // "ab" under heading "c" is not "b" under heading "ca", and a
    // two-entry path is not its concatenation.
    assert_ne!(ChunkId::of(Level::Chunk, &["c"], "ab"), ChunkId::of(Level::Chunk, &["ca"], "b"));
    assert_ne!(ChunkId::of(Level::Chunk, &["a", "b"], "x"), ChunkId::of(Level::Chunk, &["ab"], "x"));
}

#[test]
fn the_digest_is_fnv1a_128() {
    // No bytes fed: the state is the offset basis, little-endian.
    assert_eq!(Digest::new().finish(), 0x6c62_272e_07bb_0142_62b8_2175_6295_c58du128.to_le_bytes());
    let id = ChunkId::of(Level::Document, &[], "");
    assert_eq!(id.to_string().len(), 32);
    assert_eq!(ChunkId::from_bytes(*id.as_bytes()), id);
    assert_eq!(normalized_len("  a\u{2003} b \n"), 3);
    assert_eq!(normalized_len(""), 0);
}

fn structure(blocks: &[(u32, u32, Block, Option<u32>)]) -> Structure {
    let mut spans = Spans::new(ViewId::SOURCE);
    for &(a, b, ..) in blocks {
        spans.push(Span::new(a, b));
    }
    Structure {
        spans,
        kind: blocks.iter().map(|b| b.2).collect(),
        parent: blocks.iter().map(|b| b.3.map(Ix::new)).collect(),
        depth: vec![0; blocks.len()],
    }
}

#[test]
fn heading_paths_follow_the_section_tree() {
    // "Top head. body one. Sub head. body two." with a section in a
    // section, each holding its heading first.
    let text = "Top head. body one. Sub head. body two.";
    let st = structure(&[
        (0, 39, Block::Section, None),
        (0, 9, Block::Heading, Some(0)),
        (10, 19, Block::Paragraph, Some(0)),
        (20, 39, Block::Section, Some(0)),
        (20, 29, Block::Heading, Some(3)),
        (30, 39, Block::Paragraph, Some(3)),
    ]);
    let paths = HeadingPaths::new(text, Some(&st));
    assert_eq!(paths.for_span(Span::new(10, 19)), ["Top head."]);
    assert_eq!(paths.for_span(Span::new(30, 39)), ["Top head.", "Sub head."]);
    assert_eq!(paths.for_section(Ix::new(3)), ["Top head.", "Sub head."]);
    assert!(HeadingPaths::new(text, None).for_span(Span::new(0, 9)).is_empty());
}

#[test]
fn a_span_outside_the_text_is_refused_not_hashed_as_empty() {
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 99));
    let chunks = Chunks {
        spans,
        level: vec![0],
        parent: vec![None],
        sentences: vec![turbo_analysis::IxRange::new(0, 1)],
        cut: vec![turbo_analysis_layers::Cut::DocumentStart],
        cost: vec![0],
        model_tokens: vec![1],
        over_budget: vec![false],
    };
    assert!(ContentHashes::<Chunks>::compute("short", &chunks, None).is_err());
}
