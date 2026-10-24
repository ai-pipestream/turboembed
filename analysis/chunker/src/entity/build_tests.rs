use turbo_analysis::{Ix, IxRange, Span, Spans, Tag, TagSet, ViewId};
use turbo_analysis_layers::{Chunks, Cut, Entities, Sentences};

use super::{chunk_entities, ops};
use crate::entity::{Chains, EntityKeys, HeaderPolicy, Keying};

fn sentence(text: &str) -> Sentences {
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, text.len() as u32));
    Sentences { spans }
}

fn entities(items: &[&[(u32, u32)]]) -> Entities {
    let mut spans = Spans::new(ViewId::SOURCE);
    for frags in items {
        let frags: Vec<Span> = frags.iter().map(|&(a, b)| Span::new(a, b)).collect();
        spans.push_fragments(&frags);
    }
    let n = items.len();
    Entities {
        spans,
        tokens: vec![IxRange::new(0, 0); n],
        kind: vec![Tag(0); n],
        kinds: TagSet::new("ner", &["PER"]),
        score: vec![1.0; n],
    }
}

/// Level 0 chunks without parents, each with its byte fragments, all
/// over the sentences in `sentences`.
fn chunks(rows: &[&[(u32, u32)]], sentences: (u32, u32)) -> Chunks {
    let mut spans = Spans::new(ViewId::SOURCE);
    for frags in rows {
        let frags: Vec<Span> = frags.iter().map(|&(a, b)| Span::new(a, b)).collect();
        spans.push_fragments(&frags);
    }
    let n = rows.len();
    Chunks {
        spans,
        level: vec![0; n],
        parent: vec![None; n],
        sentences: vec![IxRange::new(sentences.0, sentences.1); n],
        cut: vec![Cut::Clause; n],
        cost: vec![0; n],
        model_tokens: vec![1; n],
        over_budget: vec![false; n],
        prompt_overhead: 0,
    }
}

fn entries(h: &crate::entity::ChunkEntities, c: u32) -> Vec<(u32, u32)> {
    h.entries(Ix::new(c)).collect()
}

#[test]
fn clause_chunks_of_one_sentence_name_only_their_own_entities() {
    let text = "Ann walked out, Bob is up.";
    assert_eq!((text.len(), &text[0..3], &text[16..19]), (26, "Ann", "Bob"));
    let s = sentence(text);
    let e = entities(&[&[(0, 3)], &[(16, 19)]]);
    let k = EntityKeys::build(&s, Some(&e), text, Keying::default(), Chains::default());
    let c = chunks(&[&[(0, 9)], &[(10, 26)]], (0, 1));
    let off = chunk_entities(&c, &k, HeaderPolicy { max_carried: 0 });
    assert_eq!(entries(&off, 0), [(0, 1)]);
    assert_eq!(entries(&off, 1), [(1, 1)]);
    // With context, the earlier clause's Ann is carried into the later
    // clause, never the later Bob into the earlier one.
    let on = chunk_entities(&c, &k, HeaderPolicy::default());
    assert_eq!(entries(&on, 0), [(0, 1)]);
    assert_eq!(entries(&on, 1), [(1, 1), (0, 0)]);
    assert_eq!(on.header(Ix::new(1), &e, text).unwrap(), "[Entities: Bob | in scope: Ann]");
}

#[test]
fn discontinuous_chunks_and_entities_match_by_fragment() {
    let text = "Ann walked out, Bob is up.";
    let s = sentence(text);
    // A chunk over "Ann " and "is up." skips Bob inside its envelope.
    let e = entities(&[&[(0, 3)], &[(16, 19)]]);
    let k = EntityKeys::build(&s, Some(&e), text, Keying::default(), Chains::default());
    let c = chunks(&[&[(0, 4), (20, 26)]], (0, 1));
    assert_eq!(entries(&chunk_entities(&c, &k, HeaderPolicy::default()), 0), [(0, 1)]);
    // An entity in two fragments, "Ann" and "is", belongs to a chunk
    // that holds either fragment, not to one that holds only its gap.
    let e = entities(&[&[(0, 3), (20, 22)]]);
    let k = EntityKeys::build(&s, Some(&e), text, Keying::default(), Chains::default());
    let c = chunks(&[&[(4, 19)], &[(19, 26)]], (0, 1));
    let h = chunk_entities(&c, &k, HeaderPolicy { max_carried: 0 });
    assert_eq!(entries(&h, 0), []);
    assert_eq!(entries(&h, 1), [(0, 1)]);
}

/// The counted steps of building headers for `n` one-sentence chunks,
/// with one entity at the start or none.
fn steps(n: usize, one_entity: bool) -> u64 {
    let text = "Ann x.".repeat(n);
    let mut ss = Spans::new(ViewId::SOURCE);
    let mut cs = Spans::new(ViewId::SOURCE);
    for i in 0..n as u32 {
        ss.push(Span::new(i * 6, i * 6 + 6));
        cs.push(Span::new(i * 6, i * 6 + 6));
    }
    let s = Sentences { spans: ss };
    let e = if one_entity { entities(&[&[(0, 3)]]) } else { entities(&[]) };
    let k = EntityKeys::build(&s, Some(&e), &text, Keying::default(), Chains::default());
    let mut c = chunks(&[], (0, 0));
    c.spans = cs;
    c.level = vec![0; n];
    c.parent = vec![None; n];
    c.sentences = (0..n as u32).map(|i| IxRange::new(i, i + 1)).collect();
    c.cut = vec![Cut::Sentence; n];
    c.cost = vec![0; n];
    c.model_tokens = vec![1; n];
    c.over_budget = vec![false; n];
    let before = ops();
    let h = chunk_entities(&c, &k, HeaderPolicy::default());
    let after = ops();
    if one_entity {
        assert_eq!(entries(&h, 0), [(0, 1)]);
        assert_eq!(entries(&h, n as u32 - 1), [(0, 0)]);
    }
    after - before
}

#[test]
fn building_headers_is_linear_in_chunks() {
    for one_entity in [false, true] {
        let (small, large) = (steps(5_000, one_entity), steps(40_000, one_entity));
        // Eight times the chunks: a linear build takes about eight times
        // the steps, a quadratic one sixty-four.
        assert!(large <= small * 9, "one_entity={one_entity}: {small} steps for 5000, {large} for 40000");
        assert!(large >= small * 7, "one_entity={one_entity}: {small} steps for 5000, {large} for 40000");
    }
}

/// Two chunks and two one-mention keys, consistent until `spoil` runs.
fn spoiled(spoil: impl FnOnce(&mut EntityKeys, &mut Chunks)) -> crate::entity::ChunkEntities {
    let text = "Ann walked out, Bob is up.";
    let s = sentence(text);
    let e = entities(&[&[(0, 3)], &[(16, 19)]]);
    let mut k = EntityKeys::build(&s, Some(&e), text, Keying::default(), Chains::default());
    let mut c = chunks(&[&[(0, 9)], &[(10, 26)]], (0, 1));
    spoil(&mut k, &mut c);
    chunk_entities(&c, &k, HeaderPolicy::default())
}

#[test]
fn inconsistent_inputs_give_empty_headers() {
    let empty = |h: crate::entity::ChunkEntities| {
        assert_eq!(h.row, [0, 0, 0]);
        assert!(h.key.is_empty() && h.mentions.is_empty());
    };
    assert_eq!(spoiled(|_, _| {}).key, [0, 1, 0]);
    empty(spoiled(|k, _| k.mentions[1].key = 2));
    empty(spoiled(|k, _| k.mentions[0].fragments = (1, 0)));
    empty(spoiled(|k, _| k.mentions[1].fragments = (1, 3)));
    empty(spoiled(|k, _| k.mentions.swap(0, 1)));
    empty(spoiled(|_, c| c.parent.pop().map(drop).unwrap_or(())));
    empty(spoiled(|_, c| c.level.push(0)));
    empty(spoiled(|_, c| c.parent[1] = Some(Ix::new(2))));
}

#[test]
fn entity_boundaries_skip_occurrences_out_of_order_or_range() {
    use std::num::NonZeroU32;

    use crate::entity::{EntityContinuity, entity_boundaries};
    let keys = EntityKeys {
        occurrences: vec![vec![(3, 1), (1, 1)], vec![(2, 1), (2, 1)], vec![(1, 1), (9, 1)], vec![(0, 1), (1, 1)]],
        ..EntityKeys::default()
    };
    let cfg = EntityContinuity { max_gap: NonZeroU32::new(20).unwrap(), ..EntityContinuity::default() };
    let b = entity_boundaries(4, &keys, cfg);
    // Only the last key's 0 -> 1 pair counts.
    assert_eq!(b.cost, [cfg.per_entity as i32, 0, 0, 0]);
}
