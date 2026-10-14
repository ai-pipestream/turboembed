//! Tests for `Edges<From, To>` and `Groups<L>`: bounds checking against
//! default and named layer instances, `Layer::check` invariants, pipeline
//! provenance and alignment, and property-tested source/target lookups
//! compared against a linear scan.

use std::sync::Arc;

use turbo_analysis::*;
use turbo_analysis_layers::{Entities, Sentences, Tokens};

fn sample_tokens(n: u32) -> Tokens {
    let mut spans = Spans::new(ViewId::SOURCE);
    for i in 0..n {
        spans.push(Span::new(i * 2, i * 2 + 1));
    }
    Tokens { spans, sentence: None }
}

fn sample_entities(kinds: Arc<TagSet>, token_ranges: &[(u32, u32, u16)]) -> Entities {
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut tokens = Vec::new();
    let mut kind = Vec::new();
    let mut score = Vec::new();
    for &(a, b, k) in token_ranges {
        spans.push(Span::new(a * 2, b * 2 - 1));
        tokens.push(IxRange::new(a, b));
        kind.push(Tag(k));
        score.push(1.0);
    }
    Entities { spans, tokens, kind, kinds, score }
}

#[test]
fn edges_from_entities_to_tokens_are_bounds_checked_on_both_ends() {
    let mut doc = Document::new("a b c d e f");
    let ent_kinds = TagSet::new("ner", &["PER", "ORG"]);
    doc.add(&Key::DEFAULT, sample_tokens(4), Provenance::by("test:tokens")).unwrap();
    doc.add(&Key::DEFAULT, sample_entities(ent_kinds, &[(0, 2, 0), (2, 4, 1)]), Provenance::by("test:ner")).unwrap();

    let rels = TagSet::new("rel", &["head", "mod"]);
    let head = rels.tag("head").unwrap();
    let m = rels.tag("mod").unwrap();

    // Source index 2 is past the 2 entities (0..2).
    let bad_from = Edges::<Entities, Tokens>::new(rels.clone(), [Edge::new(Ix::new(2), Ix::new(1), head, 0.9)]);
    let err = doc.add(&Key::DEFAULT, bad_from, Provenance::by("test:edges")).unwrap_err();
    assert_eq!(
        err.to_string(),
        "turbo:edges<opennlp:entities, opennlp:tokens>: an index past the end of opennlp:entities, which has 2"
    );

    // Target index 4 is past the 4 tokens (0..4).
    let bad_to = Edges::<Entities, Tokens>::new(rels.clone(), [Edge::new(Ix::new(1), Ix::new(4), head, 0.9)]);
    let err = doc.add(&Key::DEFAULT, bad_to, Provenance::by("test:edges")).unwrap_err();
    assert_eq!(
        err.to_string(),
        "turbo:edges<opennlp:entities, opennlp:tokens>: an index past the end of opennlp:tokens, which has 4"
    );

    // Valid edges are accepted, sorted, and record both target layers in `made_on`.
    let edges = Edges::<Entities, Tokens>::new(
        rels,
        [
            Edge::new(Ix::new(1), Ix::new(3), head, 0.95),
            Edge::new(Ix::new(0), Ix::new(1), head, 0.85),
            Edge::new(Ix::new(0), Ix::new(0), m, 0.5),
        ],
    );
    doc.add(&Key::DEFAULT, edges, Provenance::by("test:edges")).unwrap();
    let stored = doc.get::<Edges<Entities, Tokens>>().unwrap();
    assert_eq!(stored.len(), 3);
    assert_eq!(
        doc.made_on(Key::<Edges<Entities, Tokens>>::DEFAULT.id()),
        [Key::<Entities>::DEFAULT.id(), Key::<Tokens>::DEFAULT.id()]
    );
    let e0_out: Vec<_> = stored.outgoing(Ix::new(0)).map(|(_, e)| (e.to.get(), e.kind, e.score)).collect();
    assert_eq!(e0_out, [(0, m, 0.5), (1, head, 0.85)]);
}

#[test]
fn edges_between_named_layer_instances_check_the_named_instances() {
    const GOLD_ENT: Key<Entities> = Key::named("gold");
    const PRED_TOK: Key<Tokens> = Key::named("predicted");

    let mut doc = Document::new("a b c d e f g h");
    let ent_kinds = TagSet::new("ner", &["PER"]);
    // Default layers have different sizes from the named ones, so checking
    // against the wrong instance would either fail or miss an out-of-bounds index.
    doc.add(&Key::DEFAULT, sample_tokens(1), Provenance::by("tok:default")).unwrap();
    doc.add(&PRED_TOK, sample_tokens(3), Provenance::by("tok:pred")).unwrap();
    doc.add(&Key::DEFAULT, sample_entities(ent_kinds.clone(), &[(0, 1, 0)]), Provenance::by("ner:default")).unwrap();
    let mut gold_prov = Provenance::by("ner:gold");
    gold_prov.reads.push(PRED_TOK.id());
    doc.add(&GOLD_ENT, sample_entities(ent_kinds, &[(0, 1, 0), (1, 3, 0)]), gold_prov).unwrap();

    let rels = TagSet::new("rel", &["anchor"]);
    let anchor = rels.tag("anchor").unwrap();

    // Source index 2 is out of bounds for `opennlp:entities@gold` (len 2).
    let bad_source = Edges::between(GOLD_ENT, PRED_TOK, rels.clone(), [Edge::new(Ix::new(2), Ix::new(2), anchor, 1.0)]);
    let err = doc.add(&Key::named("align"), bad_source, Provenance::by("test:align")).unwrap_err();
    assert_eq!(
        err.to_string(),
        "turbo:edges<opennlp:entities, opennlp:tokens>@align: an index past the end of opennlp:entities@gold, which has 2"
    );

    // Target index 3 is out of bounds for `opennlp:tokens@predicted` (len 3).
    let bad_target = Edges::between(GOLD_ENT, PRED_TOK, rels.clone(), [Edge::new(Ix::new(1), Ix::new(3), anchor, 1.0)]);
    let err = doc.add(&Key::named("align"), bad_target, Provenance::by("test:align")).unwrap_err();
    assert_eq!(
        err.to_string(),
        "turbo:edges<opennlp:entities, opennlp:tokens>@align: an index past the end of opennlp:tokens@predicted, which has 3"
    );

    // Within both named instances (entity 1, token 2), though past the default instances (len 1).
    let ok = Edges::between(GOLD_ENT, PRED_TOK, rels, [Edge::new(Ix::new(1), Ix::new(2), anchor, 0.75)]);
    let key = Key::<Edges<Entities, Tokens>>::named("align");
    doc.add(&key, ok, Provenance::by("test:align")).unwrap();
    assert_eq!(doc.made_on(key.id()), [GOLD_ENT.id(), PRED_TOK.id()]);

    // Also test an edge layer between two named instances of the SAME layer type (`Tokens`).
    const TOK_A: Key<Tokens> = Key::named("a");
    const TOK_B: Key<Tokens> = Key::named("b");
    doc.add(&TOK_A, sample_tokens(2), Provenance::by("tok:a")).unwrap();
    doc.add(&TOK_B, sample_tokens(4), Provenance::by("tok:b")).unwrap();
    let tok_rels = TagSet::new("align", &["match"]);
    let m = tok_rels.tag("match").unwrap();
    let cross = Edges::between(TOK_A, TOK_B, tok_rels, [Edge::new(Ix::new(1), Ix::new(3), m, 1.0)]);
    let cross_key = Key::<Edges<Tokens, Tokens>>::named("a_to_b");
    doc.add(&cross_key, cross, Provenance::by("test:cross")).unwrap();
    assert_eq!(doc.made_on(cross_key.id()), [TOK_A.id(), TOK_B.id()]);
}

#[test]
fn edges_check_validates_tags_columns_and_both_sort_orders() {
    let kinds = TagSet::new("rel", &["a", "b"]);
    let ok = Edges::<Entities, Tokens>::new(
        kinds.clone(),
        [
            Edge::new(Ix::new(0), Ix::new(2), Tag(1), 0.5),
            Edge::new(Ix::new(0), Ix::new(2), Tag(0), 0.8),
            Edge::new(Ix::new(1), Ix::new(0), Tag(0), 0.9),
        ],
    );
    assert_eq!(ok.check(), Ok(()));

    let mut bad_tag = ok.clone();
    bad_tag.kind[0] = Tag(2);
    assert_eq!(bad_tag.check(), Err("a kind outside rel".into()));

    let mut bad_col = ok.clone();
    bad_col.score.pop();
    assert_eq!(bad_col.check(), Err("a target, a kind, a score and a target index for every edge".into()));

    let mut unsorted = ok.clone();
    unsorted.from.swap(1, 2);
    assert_eq!(unsorted.check(), Err("edges in (from, to, kind) order".into()));

    let mut out_of_range_by_to = ok.clone();
    out_of_range_by_to.by_to[0] = Ix::new(9);
    assert_eq!(out_of_range_by_to.check(), Err("a target index outside the edges".into()));

    // Duplicate row index in `by_to` violates strict `(to[r], r)` ordering.
    let mut dup_by_to = ok.clone();
    dup_by_to.by_to[2] = dup_by_to.by_to[1];
    assert_eq!(dup_by_to.check(), Err("target index in (to, row) order".into()));

    let mut reversed_by_to = ok;
    reversed_by_to.by_to.swap(0, 1);
    assert_eq!(reversed_by_to.check(), Err("target index in (to, row) order".into()));
}

#[test]
fn lookups_by_source_and_by_target_match_a_linear_scan_across_graphs() {
    let kinds = TagSet::new("rel", &["r0", "r1", "r2", "r3"]);
    // Deterministic PRNG (xorshift64) generating graphs of varying sizes,
    // densities, duplicate endpoints, and isolated vertices.
    let mut state = 0x853c_49e6_748f_ea9b_u64;
    let mut next_u32 = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 32) as u32
    };

    for trial in 0..80 {
        let num_from = (trial % 17) + 1;
        let num_to = ((trial * 3) % 19) + 1;
        let num_edges = (trial * 7) % 90;

        let mut raw = Vec::with_capacity(num_edges);
        for _ in 0..num_edges {
            let f = Ix::<Entities>::new(next_u32() % num_from as u32);
            let t = Ix::<Tokens>::new(next_u32() % num_to as u32);
            let k = Tag((next_u32() % 4) as u16);
            let score = f64::from(next_u32() % 1000) / 1000.0;
            raw.push(Edge::new(f, t, k, score));
        }

        let edges = Edges::<Entities, Tokens>::new(kinds.clone(), raw);
        assert_eq!(edges.check(), Ok(()));
        let all: Vec<_> = edges.iter().collect();
        assert_eq!(all.len(), num_edges);

        // Check every source vertex (plus an out-of-range vertex with no edges).
        for s in 0..=(num_from as u32 + 1) {
            let s_ix = Ix::<Entities>::new(s);
            let expected: Vec<_> = all.iter().copied().filter(|(_, e)| e.from == s_ix).collect();
            let got: Vec<_> = edges.outgoing(s_ix).collect();
            assert_eq!(got, expected, "outgoing mismatch on trial {trial}, source {s}");
            let range_rows: Vec<_> = edges.from_source(s_ix).iter().map(|r| (r, edges.get(r))).collect();
            assert_eq!(range_rows, expected, "from_source mismatch on trial {trial}, source {s}");
        }

        // Check every target vertex (plus an out-of-range vertex with no edges).
        for t in 0..=(num_to as u32 + 1) {
            let t_ix = Ix::<Tokens>::new(t);
            let expected: Vec<_> = all.iter().copied().filter(|(_, e)| e.to == t_ix).collect();
            let got: Vec<_> = edges.incoming(t_ix).collect();
            assert_eq!(got, expected, "incoming mismatch on trial {trial}, target {t}");
            let slice_rows: Vec<_> = edges.to_target(t_ix).iter().copied().map(|r| (r, edges.get(r))).collect();
            assert_eq!(slice_rows, expected, "to_target mismatch on trial {trial}, target {t}");
        }
    }
}

#[test]
fn groups_store_sorted_unique_members_and_check_bounds_and_invariants() {
    const GOLD_TOK: Key<Tokens> = Key::named("gold");
    let mut doc = Document::new("a b c d e f");
    doc.add(&Key::DEFAULT, sample_tokens(3), Provenance::by("tok:default")).unwrap();
    doc.add(&GOLD_TOK, sample_tokens(5), Provenance::by("tok:gold")).unwrap();

    // Group 0 has unsorted and duplicate indices; Group 1 starts with the same
    // index Group 0 ends with (Ix(2)), verifying that deduplication does not
    // cross group boundaries.
    let groups = Groups::<Tokens>::of(
        GOLD_TOK,
        vec![vec![Ix::new(2), Ix::new(0), Ix::new(2)], vec![Ix::new(2), Ix::new(4), Ix::new(1)], vec![]],
    );
    assert_eq!(groups.len(), 3);
    assert_eq!(groups.get(Ix::new(0)), &[Ix::new(0), Ix::new(2)]);
    assert_eq!(groups.get(Ix::new(1)), &[Ix::new(1), Ix::new(2), Ix::new(4)]);
    assert_eq!(groups.get(Ix::new(2)), &[]);
    assert!(groups.contains(Ix::new(1), Ix::new(4)));
    assert!(!groups.contains(Ix::new(0), Ix::new(4)));
    assert_eq!(groups.check(), Ok(()));

    let key = Key::<Groups<Tokens>>::named("coref");
    doc.add(&key, groups, Provenance::by("test:coref")).unwrap();
    assert_eq!(doc.made_on(key.id()), [GOLD_TOK.id()]);

    // Out-of-bounds member against default tokens (len 3).
    let out_of_bounds = Groups::<Tokens>::new([vec![Ix::new(0), Ix::new(3)]]);
    let err = doc.add(&Key::DEFAULT, out_of_bounds, Provenance::by("test:coref")).unwrap_err();
    assert_eq!(err.to_string(), "turbo:groups<opennlp:tokens>: an index past the end of opennlp:tokens, which has 3");

    // Invariant checks on manually constructed `Groups`.
    let bad_empty: Groups<Sentences> = Groups { of_key: Key::DEFAULT, members: vec![Ix::new(0)], end: Vec::new() };
    assert_eq!(bad_empty.check(), Err("members with no groups".into()));

    let bad_end: Groups<Sentences> =
        Groups { of_key: Key::DEFAULT, members: vec![Ix::new(0), Ix::new(1)], end: vec![1] };
    assert_eq!(bad_end.check(), Err("group ends do not match members".into()));

    let backwards_end: Groups<Sentences> =
        Groups { of_key: Key::DEFAULT, members: vec![Ix::new(0), Ix::new(1)], end: vec![2, 1, 2] };
    assert_eq!(backwards_end.check(), Err("group ends in order".into()));

    let dup_member: Groups<Sentences> =
        Groups { of_key: Key::DEFAULT, members: vec![Ix::new(1), Ix::new(1)], end: vec![2] };
    assert_eq!(dup_member.check(), Err("group 0 members in order, each once".into()));
}
