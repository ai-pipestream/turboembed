use turbo_analysis::Span;
use turbo_formats::masc::ChunkSkip;
use turbo_formats::sample::ChunkSample;
use turbo_formats::sample::chunk::phrase_spans;
use turbo_formats::source::Field;

use super::*;
use crate::card::ChunksCard;

fn sample<'a>(text: &'a str, tokens: &[(u32, u32)], tags: &[&str], chunks: &[&str]) -> ChunkSample<'a> {
    let fields: Vec<Field<'a>> = tokens.iter().map(|&(a, b)| Field::of(text, Span::new(a, b))).collect();
    let made = |values: &[&str]| -> Vec<Field<'a>> {
        values.iter().zip(&fields).map(|(v, f)| Field::made((*v).to_owned(), f.span)).collect()
    };
    let (tags, chunks) = (made(tags), made(chunks));
    ChunkSample::new(fields, tags, chunks, Span::new(0, text.len() as u32)).unwrap()
}

#[test]
fn tags_of_types_not_scored_are_made_o_and_their_phrases_left_out() {
    for (tag, scored) in
        [("B-NP", "B-NP"), ("I-VP", "I-VP"), ("O", "O"), ("B-PP", "O"), ("I-ADJP", "O"), ("B-SBAR", "O")]
    {
        assert_eq!(scored_tag(tag, &SPAN_TYPES), scored, "{tag}");
    }
    // making them O leaves the phrases of the types scored as they were
    let tags = ["B-NP", "I-PP", "I-NP", "B-PP", "B-NP", "I-NP", "I-VP", "I-ADVP", "B-VP"];
    let mapped: Vec<&str> = tags.iter().map(|t| scored_tag(t, &SPAN_TYPES)).collect();
    let kept = |tags: &[&str]| -> Vec<(u32, u32, String)> {
        phrase_spans(tags)
            .unwrap()
            .into_iter()
            .map(|(s, e, cut)| (s, e, tags[s as usize][cut..].to_owned()))
            .filter(|(_, _, k)| SPAN_TYPES.predicted(k))
            .collect()
    };
    assert_eq!(kept(&tags), kept(&mapped));
    assert_eq!(
        kept(&mapped),
        phrase_spans(&mapped)
            .unwrap()
            .into_iter()
            .map(|(s, e, cut)| (s, e, mapped[s as usize][cut..].to_owned()))
            .collect::<Vec<_>>()
    );
}

#[test]
fn malformed_chunk_tags_are_not_silently_made_o() {
    for bad in ["", "B", "B-", "I-", "BAD", "X-PP", "O-NP"] {
        assert_eq!(scored_tag(bad, &SPAN_TYPES), bad, "{bad}");
    }
}

#[test]
fn held_out_phrases_are_renamed_or_left_out_as_the_types_say() {
    let text = "a b c d";
    let s =
        sample(text, &[(0, 1), (2, 3), (4, 5), (6, 7)], &["DT", "NN", "IN", "NN"], &["B-NP", "I-NP", "B-PP", "B-X"]);
    let types = SpanTypes { types: &["NP", "Y"], renamed: &[("X", "Y")] };
    let held = HeldOutChunks::new(vec![s], Skipped::default(), ChunksCard::default(), &types).unwrap();
    let chunks: Vec<&str> = held.samples()[0].chunks().iter().map(Field::as_str).collect();
    assert_eq!(chunks, ["B-NP", "I-NP", "O", "B-Y"]);
    assert_eq!((held.spans(), held.left_out(), held.tokens()), (2, 1, 4));
    // the spans of the fields are their tokens'
    assert_eq!(held.samples()[0].chunks()[3].span, Span::new(6, 7));
}

#[test]
fn held_out_chunks_preserve_phrase_boundary_when_renaming_before_i_started_phrase() {
    let text = "a b c";
    let s = sample(text, &[(0, 1), (2, 3), (4, 5)], &["DT", "NN", "NN"], &["B-X", "I-NP", "I-NP"]);
    let types = SpanTypes { types: &["NP"], renamed: &[("X", "NP")] };
    let held = HeldOutChunks::new(vec![s], Skipped::default(), ChunksCard::default(), &types).unwrap();
    let chunks: Vec<&str> = held.samples()[0].chunks().iter().map(Field::as_str).collect();
    assert_eq!(chunks, ["B-NP", "B-NP", "I-NP"]);
    assert_eq!(held.spans(), 2);
    assert_eq!(phrase_spans(&chunks).unwrap().len() as u64, held.spans());
}

#[test]
fn the_held_out_digest_covers_tokens_tags_chunk_tags_types_and_the_reference_tags() {
    let text = "Dogs ran";
    let one = |chunk: &str| {
        let s = sample(text, &[(0, 4), (5, 8)], &["NNS", "VBD"], &["B-NP", chunk]);
        HeldOutChunks::new(vec![s], Skipped::default(), ChunksCard::default(), &SPAN_TYPES).unwrap()
    };
    let mut bytes = Vec::new();
    let put = |bytes: &mut Vec<u8>, s: &str| {
        bytes.extend_from_slice(&(s.len() as u64).to_le_bytes());
        bytes.extend_from_slice(s.as_bytes());
    };
    bytes.extend_from_slice(&2u64.to_le_bytes());
    for (token, tag, chunk) in [("Dogs", "NNS", "B-NP"), ("ran", "VBD", "B-VP")] {
        put(&mut bytes, token);
        put(&mut bytes, tag);
        put(&mut bytes, chunk);
    }
    bytes.extend_from_slice(&2u64.to_le_bytes());
    put(&mut bytes, "NP");
    put(&mut bytes, "VP");
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&3u64.to_le_bytes());
    for (held_out, reference) in [("-", ":"), ("...", ":"), (";", ":")] {
        put(&mut bytes, held_out);
        put(&mut bytes, reference);
    }
    assert_eq!(one("B-VP").sha256(&SPAN_TYPES), Sha256::of(&bytes).to_string());
    assert_ne!(one("B-VP").sha256(&SPAN_TYPES), one("O").sha256(&SPAN_TYPES));
    let other = SpanTypes { types: &["NP"], renamed: &[] };
    assert_ne!(one("B-VP").sha256(&SPAN_TYPES), one("B-VP").sha256(&other));
}

#[test]
fn only_the_reference_is_given_mapped_tags() {
    for (masc, reference) in [("-", ":"), ("...", ":"), (";", ":"), ("(", "("), (")", ")"), ("''", "''"), ("NN", "NN")]
    {
        assert_eq!(Scored::Reference.tag(masc), reference, "{masc}");
        assert_eq!(Scored::Candidate.tag(masc), masc);
    }
}

#[test]
fn skipped_sentences_are_counted_by_why() {
    let mut s = Skipped::default();
    for reason in [
        ChunkSkip::NoTokens,
        ChunkSkip::OverlappingChunks,
        ChunkSkip::OverlappingChunks,
        ChunkSkip::PartialChunk,
        ChunkSkip::NullTag,
        ChunkSkip::ChunkWithGap,
    ] {
        s.add(reason);
    }
    assert_eq!(s, Skipped { no_tokens: 1, null_tag: 1, partial_chunk: 1, overlapping_chunks: 2, chunk_with_gap: 1 });
    assert_eq!(s.total(), 6);
    assert!(MAX_SKIPPED_SENTENCES.allows(10, 100) && !MAX_SKIPPED_SENTENCES.allows(11, 100));
    use turbo_train::Key;
    let p = training_parameters();
    assert_eq!(p.get_str(Key::ALGORITHM).unwrap(), Some("MAXENT"));
    assert_eq!(p.get_int(Key::ITERATIONS).unwrap(), Some(ITERATIONS));
    assert_eq!(p.get_int(Key::CUTOFF).unwrap(), Some(CUTOFF));
    assert_eq!(p.get_int(Key::THREADS).unwrap(), None);
    assert_eq!(chunk::beam_size(&p).unwrap(), chunk::DEFAULT_BEAM_SIZE);
    assert_eq!((ITERATIONS, CUTOFF, THREADS, chunk::DEFAULT_BEAM_SIZE), (100, 5, 1, 10));
    assert_eq!(CHUNKS.task(), Task::Chunker);
}
