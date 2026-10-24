use std::num::NonZeroU32;

use turbo_analysis::{Groups, Ix, IxRange, Layer, Span, Spans, Tag, TagSet, ViewId};
use turbo_analysis_layers::{Chunks, Cut, Entities, Sentences};

use super::*;
use crate::coref::MentionKind;

/// Sentences over `text`, split after each '.'.
fn sentences(text: &str) -> Sentences {
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut start = 0;
    for (i, c) in text.char_indices() {
        if c == '.' {
            spans.push(Span::new(start as u32, i as u32 + 1));
            start = i + 2;
        }
    }
    Sentences { spans }
}

/// Entities at every occurrence of each `(needle, kind)`, in text order.
fn entities(text: &str, needles: &[(&str, u16)]) -> Entities {
    let mut found: Vec<(u32, u32, u16)> = Vec::new();
    for &(needle, kind) in needles {
        for (at, _) in text.match_indices(needle) {
            found.push((at as u32, (at + needle.len()) as u32, kind));
        }
    }
    found.sort_unstable();
    let mut spans = Spans::new(ViewId::SOURCE);
    for &(a, b, _) in &found {
        spans.push(Span::new(a, b));
    }
    let n = found.len();
    Entities {
        spans,
        tokens: vec![IxRange::new(0, 0); n],
        kind: found.iter().map(|f| Tag(f.2)).collect(),
        kinds: TagSet::new("ner", &["PER", "LOC"]),
        score: vec![1.0; n],
    }
}

fn keys(text: &str, ents: &Entities, keying: Keying, chains: Chains<'_>) -> EntityKeys {
    EntityKeys::build(&sentences(text), Some(ents), text, keying, chains)
}

fn chunks(ranges: &[(u32, u32)], parent: &[Option<u32>], level: &[u8], sents: &Sentences) -> Chunks {
    let mut spans = Spans::new(ViewId::SOURCE);
    for &(a, b) in ranges {
        spans.push(Span::new(sents.spans.get(Ix::new(a)).start, sents.spans.get(Ix::new(b - 1)).end));
    }
    let n = ranges.len();
    Chunks {
        spans,
        level: level.to_vec(),
        parent: parent.iter().map(|p| p.map(Ix::new)).collect(),
        sentences: ranges.iter().map(|&(a, b)| IxRange::new(a, b)).collect(),
        cut: vec![Cut::Sentence; n],
        cost: vec![0; n],
        model_tokens: vec![1; n],
        over_budget: vec![false; n],
        prompt_overhead: 0,
    }
}

#[test]
fn surfaces_fold_case_width_whitespace_and_edge_punctuation() {
    assert_eq!(normalize_surface("Marie  Curie"), "marie curie");
    assert_eq!(normalize_surface(" MARIE\tCURIE, "), "marie curie");
    assert_eq!(normalize_surface("\u{FF29}\u{FF22}\u{FF2D}"), "ibm");
    assert_eq!(normalize_surface("Stra\u{DF}e"), normalize_surface("STRASSE"));
    assert_eq!(normalize_surface("U.S."), "u.s");
    assert_eq!(normalize_surface("..."), "");
}

#[test]
fn same_surface_is_one_key_and_kind_keying_splits_it() {
    let text = "Paris is big. PARIS  is old. Paris Hilton came.";
    let ents = entities(text, &[("Paris", 1), ("PARIS", 1), ("Paris Hilton", 0)]);
    let k = keys(text, &ents, Keying::default(), Chains::default());
    // "Paris" (0), "PARIS" (1), "Paris" in "Paris Hilton" (2) and the
    // overlapping "Paris Hilton" (3).
    assert_eq!(k.key, [Some(0), Some(0), Some(0), Some(1)]);
    assert_eq!(k.occurrences, [vec![(0, 1), (1, 1), (2, 1)], vec![(2, 1)]]);
    assert_eq!(k.first, [Ix::new(0), Ix::new(3)]);
    let mut ents = ents;
    ents.kind[2] = Tag(0);
    let k = keys(text, &ents, Keying { kind: true, chains: true }, Chains::default());
    assert_eq!(k.key, [Some(0), Some(0), Some(1), Some(2)]);
}

#[test]
fn entities_outside_sentences_or_without_letters_have_no_key() {
    let text = "Ann ran. -- Ann sat.";
    let ents = entities(text, &[("Ann", 0), ("--", 0)]);
    let k = keys(text, &ents, Keying::default(), Chains::default());
    // The "--" lies between the two sentences and is punctuation anyway.
    assert_eq!(k.key, [Some(0), None, Some(0)]);
    assert_eq!(k.occurrences, [vec![(0, 1), (1, 1)]]);
}

#[test]
fn duplicate_spans_count_once() {
    let text = "Ann ran.";
    let mut ents = entities(text, &[("Ann", 0)]);
    ents.spans.push(Span::new(0, 3));
    ents.tokens.push(IxRange::new(0, 0));
    ents.kind.push(Tag(1));
    ents.score.push(0.5);
    let k = keys(text, &ents, Keying::default(), Chains::default());
    assert_eq!(k.occurrences, [vec![(0, 1)]]);
}

#[test]
fn coreference_chains_merge_keys_and_add_pronoun_mentions() {
    let text = "Marie Curie won. She was Polish. The physicist died. Paris mourned.";
    let ents = entities(text, &[("Marie Curie", 0), ("The physicist", 0), ("Paris", 1)]);
    let groups = Groups::<Entities>::new([[Ix::new(0), Ix::new(1)]]);
    let chains = Chains { entity_groups: Some(&groups), ..Chains::default() };
    let k = keys(text, &ents, Keying::default(), chains);
    assert_eq!(k.key, [Some(0), Some(0), Some(1)]);
    assert_eq!(k.occurrences[0], [(0, 1), (2, 1)]);

    let mut m = Mentions::new(ViewId::SOURCE);
    let she = text.find("She").unwrap() as u32;
    m.push(Span::new(0, 11), Ix::new(0), IxRange::new(0, 2), Ix::new(1), MentionKind::Proper);
    m.push(Span::new(she, she + 3), Ix::new(1), IxRange::new(4, 5), Ix::new(4), MentionKind::Pronominal);
    let mgroups = Groups::<Mentions>::new([[Ix::new(0), Ix::new(1)]]);
    let chains = Chains { mentions: Some(&m), mention_groups: Some(&mgroups), entity_groups: Some(&groups) };
    let k = keys(text, &ents, Keying::default(), chains);
    assert_eq!(k.occurrences[0], [(0, 1), (1, 1), (2, 1)]);
    // With chains off, only surfaces key.
    let k = keys(text, &ents, Keying { kind: false, chains: false }, chains);
    assert_eq!(k.key, [Some(0), Some(1), Some(2)]);
}

#[test]
fn runs_cost_each_gap_they_span_up_to_the_cap() {
    // Ann in 0, 1, 3 and 6; Bob in 1 and 2.
    let text = "Ann a. Ann Bob b. Bob c. Ann d. e. f. Ann g.";
    let ents = entities(text, &[("Ann", 0), ("Bob", 0)]);
    let k = keys(text, &ents, Keying::default(), Chains::default());
    let b = entity_boundaries(7, &k, EntityContinuity::default());
    // Ann 1 -> 3 is two sentences apart (one run); 3 -> 6 is not.
    assert_eq!(b.cost, [500, 1000, 500, 0, 0, 0, 0]);
    assert_eq!(b.reason, [Cut::Sentence; 7]);
    let tight = EntityContinuity { max_gap: NonZeroU32::new(1).unwrap(), ..EntityContinuity::default() };
    assert_eq!(entity_boundaries(7, &k, tight).cost, [500, 500, 0, 0, 0, 0, 0]);
    let capped = EntityContinuity { cap: 600, ..EntityContinuity::default() };
    assert_eq!(entity_boundaries(7, &k, capped).cost, [500, 600, 500, 0, 0, 0, 0]);
    let wide = EntityContinuity { max_gap: NonZeroU32::new(3).unwrap(), ..EntityContinuity::default() };
    assert_eq!(entity_boundaries(7, &k, wide).cost, [500, 1000, 500, 500, 500, 500, 0]);
}

#[test]
fn no_entities_cost_nothing_and_head_nothing() {
    let text = "One. Two.";
    let s = sentences(text);
    let k = EntityKeys::build(&s, None, text, Keying::default(), Chains::default());
    assert!(k.is_empty());
    assert_eq!(entity_boundaries(2, &k, EntityContinuity::default()).cost, [0, 0]);
    let h = chunk_entities(&chunks(&[(0, 1), (1, 2)], &[None, None], &[0, 0], &s), &k, HeaderPolicy::default());
    assert_eq!(h.row, [0, 0, 0]);
    assert!(h.first.is_empty() && h.check().is_ok());
    assert_eq!(entity_boundaries(0, &k, EntityContinuity::default()).cost, Vec::<i32>::new());
}

#[test]
fn headers_list_mentioned_then_carried_entities_within_the_section() {
    let text = "Ann met Bob. Bob left. Cy came. Ann and Cy sat. Dee won.";
    let s = sentences(text);
    let ents = entities(text, &[("Ann", 0), ("Bob", 0), ("Cy", 0), ("Dee", 0)]);
    let k = keys(text, &ents, Keying::default(), Chains::default());
    // A section row over 0..5 holding chunks 0..2, 2..4 and 4..5.
    let c = chunks(&[(0, 5), (0, 2), (2, 4), (4, 5)], &[None, Some(0), Some(0), Some(0)], &[1, 0, 0, 0], &s);
    let h = chunk_entities(&c, &k, HeaderPolicy::default());
    assert!(h.check().is_ok());
    let entries = |i: u32| h.entries(Ix::new(i)).collect::<Vec<_>>();
    // Section: everything it mentions, nothing carried (no parent).
    assert_eq!(entries(0), [(0, 2), (1, 2), (2, 2), (3, 1)]);
    assert_eq!(entries(1), [(0, 1), (1, 2)]);
    // Cy and Ann mentioned; Bob carried from sentence 1.
    assert_eq!(entries(2), [(2, 2), (0, 1), (1, 0)]);
    // Dee mentioned; carried most recent first: Ann and Cy (sentence 3, key order), then Bob.
    assert_eq!(entries(3), [(3, 1), (0, 0), (2, 0), (1, 0)]);
    assert_eq!(h.header(Ix::new(3), &ents, text).unwrap(), "[Entities: Dee | in scope: Ann; Cy; Bob]");
    assert_eq!(h.entities(Ix::new(1)).collect::<Vec<_>>(), [Ix::new(0), Ix::new(1)]);
    let none = chunk_entities(&c, &k, HeaderPolicy { max_carried: 0 });
    assert_eq!(none.entries(Ix::new(3)).collect::<Vec<_>>(), [(3, 1)]);
    let one = chunk_entities(&c, &k, HeaderPolicy { max_carried: 1 });
    assert_eq!(one.entries(Ix::new(3)).collect::<Vec<_>>(), [(3, 1), (0, 0)]);
}

#[test]
fn header_check_refuses_broken_rows() {
    let ok =
        ChunkEntities { first: vec![Ix::new(0), Ix::new(1)], row: vec![0, 2], key: vec![0, 1], mentions: vec![1, 0] };
    assert!(ok.check().is_ok());
    assert_eq!(ok.len(), 1);
    let twice = ChunkEntities { key: vec![1, 1], ..ok.clone() };
    assert!(twice.check().unwrap_err().contains("twice"));
    let order = ChunkEntities { mentions: vec![0, 1], ..ok.clone() };
    assert!(order.check().unwrap_err().contains("after a carried"));
    let outside = ChunkEntities { key: vec![0, 2], ..ok.clone() };
    assert!(outside.check().is_err());
    let rows = ChunkEntities { row: vec![0, 1], ..ok.clone() };
    assert!(rows.check().is_err());
    let counts = ChunkEntities { mentions: vec![1], ..ok };
    assert!(counts.check().is_err());
}

#[test]
fn config_hashes_cover_every_setting() {
    let base = EntityBoundaries::default().config_hash();
    let mut seen = vec![base];
    for c in [
        EntityContinuity { per_entity: 400, ..EntityContinuity::default() },
        EntityContinuity { max_gap: NonZeroU32::new(1).unwrap(), ..EntityContinuity::default() },
        EntityContinuity { cap: 1000, ..EntityContinuity::default() },
        EntityContinuity { keying: Keying { kind: true, chains: true }, ..EntityContinuity::default() },
        EntityContinuity { keying: Keying { kind: false, chains: false }, ..EntityContinuity::default() },
    ] {
        let h = EntityBoundaries { continuity: c }.config_hash();
        assert!(!seen.contains(&h));
        seen.push(h);
    }
    let h = ChunkEntityHeaders::default().config_hash();
    assert_ne!(h, ChunkEntityHeaders { policy: HeaderPolicy { max_carried: 2 }, ..Default::default() }.config_hash());
    assert_ne!(
        h,
        ChunkEntityHeaders { keying: Keying { kind: true, chains: true }, ..Default::default() }.config_hash()
    );
}
