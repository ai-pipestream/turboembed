use std::collections::BTreeMap;

use turbo_analysis::{Span, TagSet};
use turbo_formats::source::Field;

use super::*;

fn sample<'a>(text: &'a str, tokens: &[(u32, u32)], tags: &[&'a str]) -> PosSample<'a> {
    let fields = tokens.iter().map(|&(a, b)| Field::of(text, Span::new(a, b))).collect();
    let tags = tags.iter().zip(tokens).map(|(t, &(a, b))| Field::made(t.to_string(), Span::new(a, b))).collect();
    PosSample::new(fields, tags, Span::new(0, text.len() as u32)).unwrap()
}

#[test]
fn the_held_out_digest_covers_tokens_tags_what_is_scored_and_the_mapping() {
    let text = "Dogs ran.";
    let reference = ReferenceTags::of(&TagSet::new("reference", &["NNS", "VBD", "VBN", "."]), &REFERENCE_MAPPINGS);
    let one = HeldOutTags::new(vec![sample(text, &[(0, 4), (5, 8), (8, 9)], &["NNS", "VBD", "."])], 0);
    let mut bytes = Vec::new();
    let put = |bytes: &mut Vec<u8>, s: &str| {
        bytes.extend_from_slice(&(s.len() as u64).to_le_bytes());
        bytes.extend_from_slice(s.as_bytes());
    };
    bytes.extend_from_slice(&3u64.to_le_bytes());
    for (token, tag) in [("Dogs", "NNS"), ("ran", "VBD"), (".", ".")] {
        put(&mut bytes, token);
        put(&mut bytes, tag);
        bytes.push(1);
    }
    bytes.extend_from_slice(&(REFERENCE_MAPPINGS.len() as u64).to_le_bytes());
    for m in REFERENCE_MAPPINGS {
        put(&mut bytes, m.held_out);
        put(&mut bytes, m.reference);
    }
    assert_eq!(one.sha256(&reference), Sha256::of(&bytes).to_string());
    // Another tag, the same text cut into other tokens, or a reference
    // that leaves another token out, is another set.
    let tag = HeldOutTags::new(vec![sample(text, &[(0, 4), (5, 8), (8, 9)], &["NNS", "VBN", "."])], 0);
    let cut = HeldOutTags::new(vec![sample(text, &[(0, 4), (5, 9)], &["NNS", "VBD"])], 0);
    let no_dot = ReferenceTags::of(&TagSet::new("reference", &["NNS", "VBD", "VBN"]), &REFERENCE_MAPPINGS);
    assert_ne!(one.sha256(&reference), tag.sha256(&reference));
    assert_ne!(one.sha256(&reference), cut.sha256(&reference));
    assert_ne!(one.sha256(&reference), one.sha256(&no_dot));
    // With no mapping, the digest ends with a count of 0 mappings.
    let unmapped = ReferenceTags::of(&TagSet::new("reference", &["NNS", "VBD", "VBN", "."]), &[]);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&3u64.to_le_bytes());
    for (token, tag) in [("Dogs", "NNS"), ("ran", "VBD"), (".", ".")] {
        put(&mut bytes, token);
        put(&mut bytes, tag);
        bytes.push(1);
    }
    bytes.extend_from_slice(&0u64.to_le_bytes());
    assert_eq!(one.sha256(&unmapped), Sha256::of(&bytes).to_string());
}

#[test]
fn only_the_reference_s_tags_are_compared_through_the_mapping() {
    let m = &REFERENCE_MAPPINGS;
    for (held_out, given) in [("(", "-LRB-"), (")", "-RRB-"), ("-", ":"), ("...", ":"), (";", ":"), ("''", "``")] {
        assert!(Scored::Reference.is_right(m, held_out, given), "{held_out} {given}");
        assert!(!Scored::Candidate.is_right(m, held_out, given), "{held_out} {given}");
        // With no mapping, as the UD taggers are gated, the reference's
        // tags are compared as written too.
        assert!(!Scored::Reference.is_right(&[], held_out, given), "{held_out} {given}");
    }
    for same in ["(", ":", "''", "``", "NN"] {
        assert!(Scored::Reference.is_right(m, same, same) && Scored::Candidate.is_right(m, same, same));
        assert!(Scored::Reference.is_right(&[], same, same));
    }
    // The mapping goes one way, from the held-out tag to the reference's.
    for (held_out, given) in [("-LRB-", "("), (":", "-"), (":", "..."), ("``", "''"), ("-", "..."), ("NN", "NNS")] {
        assert!(!Scored::Reference.is_right(m, held_out, given), "{held_out} {given}");
    }
    assert_eq!(MISMATCHES_LISTED, 10);
}

#[test]
fn held_out_tokens_are_scored_when_the_reference_can_give_their_tag_as_written_or_mapped() {
    let tags = ReferenceTags::of(&TagSet::new("reference", &["-LRB-", ":", "NN", "UH", "``"]), &REFERENCE_MAPPINGS);
    for scored in ["(", "-LRB-", "-", "...", ";", ":", "NN", "``", "''"] {
        assert!(tags.scores(scored), "{scored}");
    }
    for left_out in [")", "-RRB-", "PRN", "UH`", "nn", "\""] {
        assert!(!tags.scores(left_out), "{left_out}");
    }
    let text = "( Agh ) they ; Agh";
    let held_out = HeldOutTags::new(
        vec![
            sample(text, &[(0, 1), (2, 5), (6, 7)], &["(", "UH`", ")"]),
            sample(text, &[(8, 12), (13, 14), (15, 18)], &["PRN", ";", "UH`"]),
        ],
        0,
    );
    assert_eq!(held_out.tokens(), 6);
    let count = |tag: &str, tokens| TagCount { tag: tag.to_owned(), tokens };
    // Sorted by the tag as written.
    assert_eq!(tags.left_out(&held_out), [count(")", 1), count("PRN", 1), count("UH`", 2)]);
    // With no mapping, a held-out tag is scored only when the reference
    // gives it as written.
    let unmapped = ReferenceTags::of(&TagSet::new("reference", &["-LRB-", ":", "NN", "UH", "``"]), &[]);
    assert!(unmapped.scores(":") && unmapped.scores("-LRB-") && !unmapped.scores("(") && !unmapped.scores(";"));
    assert_eq!(
        unmapped.left_out(&held_out),
        [count("(", 1), count(")", 1), count(";", 1), count("PRN", 1), count("UH`", 2)]
    );
}

#[test]
fn the_most_frequent_mismatches_come_first_and_are_at_most_ten() {
    let mut counts = BTreeMap::new();
    let held: Vec<String> = (0..12).map(|i| format!("T{i:02}")).collect();
    for (i, tag) in held.iter().enumerate() {
        counts.insert((tag.as_str(), "NN"), (i % 4) as u64 + 1);
    }
    let top = most_frequent(counts);
    assert_eq!(top.len(), MISMATCHES_LISTED);
    let got: Vec<(&str, u64)> = top.iter().map(|m| (m.held_out.as_str(), m.tokens)).collect();
    assert_eq!(
        got,
        [
            ("T03", 4),
            ("T07", 4),
            ("T11", 4),
            ("T02", 3),
            ("T06", 3),
            ("T10", 3),
            ("T01", 2),
            ("T05", 2),
            ("T09", 2),
            ("T00", 1)
        ]
    );
}

#[test]
fn score_handles_empty_held_out_sample_without_nan_panic() {
    let f = factory().unwrap();
    let mut stream = postag::PosEventStream::new(&f).unwrap();
    let mut events = Vec::new();
    let train_sample = sample("Dogs ran.", &[(0, 4), (5, 8), (8, 9)], &["NNS", "VBD", "."]);
    for _ in 0..5 {
        stream.events(&train_sample, &mut events);
    }
    let mut config = TrainingConfiguration::default();
    let params = crate::masc::training_parameters();
    let trained = postag::train_pos_model_from_events(&events, &f, &params, &mut config, Stamp::at(TIMESTAMP_MS))
        .unwrap()
        .package
        .to_bytes();
    let empty = sample("", &[], &[]);
    let held_out = HeldOutTags::new(vec![empty], 0);
    let reference = POS.reference_tags(&trained).unwrap();
    let units = POS.score(&trained, Scored::Candidate, &held_out, &reference).unwrap();
    assert_eq!(units, vec![Counts::accuracy(0, 0)]);
}
