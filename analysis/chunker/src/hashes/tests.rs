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
    let paths = HeadingPaths::new(text, Some(&st)).unwrap();
    fn owned<'x>(path: Vec<&'x [&'x str]>) -> Vec<Vec<&'x str>> {
        path.into_iter().map(<[&str]>::to_vec).collect()
    }
    assert_eq!(owned(paths.for_span(Span::new(10, 19))), [vec!["Top head."]]);
    assert_eq!(owned(paths.for_span(Span::new(30, 39))), [vec!["Top head."], vec!["Sub head."]]);
    assert_eq!(owned(paths.for_section(Ix::new(3))), [vec!["Top head."], vec!["Sub head."]]);
    assert!(HeadingPaths::new(text, None).unwrap().for_span(Span::new(0, 9)).is_empty());
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
        prompt_overhead: 0,
    };
    assert!(ContentHashes::<Chunks>::compute("short", &chunks, None).is_err());
}

fn fragmented_chunks(rows: &[&[Span]]) -> Chunks {
    let mut spans = Spans::new(ViewId::SOURCE);
    for r in rows {
        spans.push_fragments(r);
    }
    let n = rows.len();
    Chunks {
        spans,
        level: vec![0; n],
        parent: vec![None; n],
        sentences: (0..n as u32).map(|i| turbo_analysis::IxRange::new(i, i + 1)).collect(),
        cut: vec![turbo_analysis_layers::Cut::Sentence; n],
        cost: vec![0; n],
        model_tokens: vec![1; n],
        over_budget: vec![false; n],
        prompt_overhead: 0,
    }
}

#[test]
fn discontinuous_chunk_fragments_are_what_is_hashed() {
    // Same envelope 0..4, different selections: "ad" vs "abd".
    let ad = fragmented_chunks(&[&[Span::new(0, 1), Span::new(3, 4)]]);
    let abd = fragmented_chunks(&[&[Span::new(0, 2), Span::new(3, 4)]]);
    ad.check().unwrap();
    abd.check().unwrap();
    let id_ad = ContentHashes::<Chunks>::compute("abcd", &ad, None).unwrap().get(Ix::new(0));
    let id_abd = ContentHashes::<Chunks>::compute("abcd", &abd, None).unwrap().get(Ix::new(0));
    assert_ne!(id_ad, id_abd);
    assert_eq!(id_ad, ChunkId::of_fragments(Level::Chunk, &[], &["a", "d"]));
    assert_eq!(id_abd, ChunkId::of_fragments(Level::Chunk, &[], &["ab", "d"]));

    // Only the excluded gap changed: the ID is the same.
    let gap = fragmented_chunks(&[&[Span::new(0, 1), Span::new(2, 3)]]);
    assert_eq!(
        ContentHashes::<Chunks>::compute("aXb", &gap, None).unwrap().get(Ix::new(0)),
        ContentHashes::<Chunks>::compute("aYb", &gap, None).unwrap().get(Ix::new(0))
    );
}

fn fragmented_levels(chunk_fragments: &[Span]) -> Levels {
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 4));
    spans.push_fragments(chunk_fragments);
    Levels { spans, unit: vec![Unit::Document, Unit::Chunk(Ix::new(0))], parent: vec![None, Some(Ix::new(0))] }
}

#[test]
fn discontinuous_level_fragments_are_what_is_hashed() {
    let ad = fragmented_levels(&[Span::new(0, 1), Span::new(3, 4)]);
    let abd = fragmented_levels(&[Span::new(0, 2), Span::new(3, 4)]);
    ad.check().unwrap();
    abd.check().unwrap();
    assert_ne!(
        ContentHashes::<Levels>::compute("abcd", &ad, None).unwrap().get(Ix::new(1)),
        ContentHashes::<Levels>::compute("abcd", &abd, None).unwrap().get(Ix::new(1))
    );
    // The document row is continuous over the whole text either way.
    assert_eq!(
        ContentHashes::<Levels>::compute("abcd", &ad, None).unwrap().get(Ix::new(0)),
        ChunkId::of(Level::Document, &[], "abcd")
    );
}

#[test]
fn fragmented_headings_hash_their_fragments() {
    // A section whose heading selects "ab" out of "a?b" over a body
    // chunk: changing the heading's excluded gap keeps the chunk
    // ID, changing a selected heading byte does not.
    let build = |text: &str| {
        let mut spans = Spans::new(ViewId::SOURCE);
        spans.push(Span::new(0, text.len() as u32));
        spans.push_fragments(&[Span::new(0, 1), Span::new(2, 3)]);
        let st = Structure {
            spans,
            kind: vec![Block::Section, Block::Heading],
            parent: vec![None, Some(Ix::new(0))],
            depth: vec![0; 2],
        };
        let chunks = fragmented_chunks(&[&[Span::new(4, text.len() as u32)]]);
        ContentHashes::<Chunks>::compute(text, &chunks, Some(&st)).unwrap().get(Ix::new(0))
    };
    assert_eq!(build("aXb body."), build("aYb body."));
    assert_ne!(build("aXb body."), build("aXc body."));
}

#[test]
fn invalid_heading_spans_and_layer_columns_are_refused() {
    let chunks = fragmented_chunks(&[&[Span::new(0, 4)]]);
    let with_heading = |heading: Span| {
        let mut spans = Spans::new(ViewId::SOURCE);
        // The section is as wide as its heading, so the structure
        // itself checks out and only the text slicing can refuse.
        spans.push(Span::new(0, heading.end.max(4)));
        spans.push(heading);
        Structure {
            spans,
            kind: vec![Block::Section, Block::Heading],
            parent: vec![None, Some(Ix::new(0))],
            depth: vec![0; 2],
        }
    };
    // Outside the text, and splitting a codepoint (the first
    // character of the second text is two bytes wide).
    assert!(ContentHashes::<Chunks>::compute("abcd", &chunks, Some(&with_heading(Span::new(0, 99)))).is_err());
    assert!(ContentHashes::<Chunks>::compute("\u{e9}bcd", &chunks, Some(&with_heading(Span::new(0, 1)))).is_err());

    // Columns short of the spans refuse through the layer check,
    // never a panic.
    let mut short = fragmented_chunks(&[&[Span::new(0, 4)]]);
    short.level.clear();
    assert!(ContentHashes::<Chunks>::compute("abcd", &short, None).is_err());
    let mut short_levels = fragmented_levels(&[Span::new(0, 4)]);
    short_levels.unit.clear();
    assert!(ContentHashes::<Levels>::compute("abcd", &short_levels, None).is_err());
}

#[test]
fn section_units_need_their_structure_block() {
    let levels = |block: u32| {
        let mut spans = Spans::new(ViewId::SOURCE);
        spans.push(Span::new(0, 4));
        spans.push(Span::new(0, 4));
        Levels {
            spans,
            unit: vec![Unit::Document, Unit::Section(Ix::new(block))],
            parent: vec![None, Some(Ix::new(0))],
        }
    };
    // No structure at all, a block past the structure, and a block
    // that is no section.
    assert!(ContentHashes::<Levels>::compute("abcd", &levels(0), None).is_err());
    let st = structure(&[(0, 4, Block::Section, None), (0, 2, Block::Paragraph, Some(0))]);
    assert!(ContentHashes::<Levels>::compute("abcd", &levels(7), Some(&st)).is_err());
    assert!(ContentHashes::<Levels>::compute("abcd", &levels(1), Some(&st)).is_err());
    assert!(ContentHashes::<Levels>::compute("abcd", &levels(0), Some(&st)).is_ok());
}
