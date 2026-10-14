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
    // The step that made the edges read both named instances.
    let align_prov = || {
        let mut p = Provenance::by("test:align");
        p.reads = vec![GOLD_ENT.id(), PRED_TOK.id()];
        p
    };

    // Naming an instance the step did not read is refused, whatever the bounds.
    let unread = Edges::between(GOLD_ENT, PRED_TOK, rels.clone(), [Edge::new(Ix::new(0), Ix::new(0), anchor, 1.0)]);
    let err = doc.add(&Key::named("align"), unread, Provenance::by("test:align")).unwrap_err();
    assert_eq!(
        err.to_string(),
        "turbo:edges<opennlp:entities, opennlp:tokens>@align: an index into opennlp:entities@gold, which its step neither read nor wrote"
    );
    let mut half = Provenance::by("test:align");
    half.reads.push(GOLD_ENT.id());
    let unread = Edges::between(GOLD_ENT, PRED_TOK, rels.clone(), [Edge::new(Ix::new(0), Ix::new(0), anchor, 1.0)]);
    let err = doc.add(&Key::named("align"), unread, half).unwrap_err();
    assert_eq!(
        err.to_string(),
        "turbo:edges<opennlp:entities, opennlp:tokens>@align: an index into opennlp:tokens@predicted, which its step neither read nor wrote"
    );

    let bad_source = Edges::between(GOLD_ENT, PRED_TOK, rels.clone(), [Edge::new(Ix::new(2), Ix::new(2), anchor, 1.0)]);
    let err = doc.add(&Key::named("align"), bad_source, align_prov()).unwrap_err();
    assert_eq!(
        err.to_string(),
        "turbo:edges<opennlp:entities, opennlp:tokens>@align: an index past the end of opennlp:entities@gold, which has 2"
    );

    // Target index 3 is out of bounds for `opennlp:tokens@predicted` (len 3).
    let bad_target = Edges::between(GOLD_ENT, PRED_TOK, rels.clone(), [Edge::new(Ix::new(1), Ix::new(3), anchor, 1.0)]);
    let err = doc.add(&Key::named("align"), bad_target, align_prov()).unwrap_err();
    assert_eq!(
        err.to_string(),
        "turbo:edges<opennlp:entities, opennlp:tokens>@align: an index past the end of opennlp:tokens@predicted, which has 3"
    );

    // Within both named instances (entity 1, token 2), though past the default instances (len 1).
    let ok = Edges::between(GOLD_ENT, PRED_TOK, rels, [Edge::new(Ix::new(1), Ix::new(2), anchor, 0.75)]);
    let key = Key::<Edges<Entities, Tokens>>::named("align");
    doc.add(&key, ok, align_prov()).unwrap();
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
    let mut cross_prov = Provenance::by("test:cross");
    cross_prov.reads = vec![TOK_A.id(), TOK_B.id()];
    doc.add(&cross_key, cross, cross_prov).unwrap();
    assert_eq!(doc.made_on(cross_key.id()), [TOK_A.id(), TOK_B.id()]);

    // Both ends in the same instance: it is recorded once.
    let within =
        Edges::between(TOK_B, TOK_B, TagSet::new("next", &["next"]), [Edge::new(Ix::new(0), Ix::new(1), Tag(0), 1.0)]);
    let within_key = Key::<Edges<Tokens, Tokens>>::named("b_to_b");
    let mut within_prov = Provenance::by("test:within");
    within_prov.reads.push(TOK_B.id());
    doc.add(&within_key, within, within_prov).unwrap();
    assert_eq!(doc.made_on(within_key.id()), [TOK_B.id()]);
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
    assert_eq!(unsorted.check(), Err("edges in (from, to, kind) order, each once".into()));

    // One relation twice, with two scores: refused by `check`, and so by
    // `Document::add`.
    let repeated = Edges::<Entities, Tokens>::new(
        kinds.clone(),
        [Edge::new(Ix::new(0), Ix::new(2), Tag(0), 0.5), Edge::new(Ix::new(0), Ix::new(2), Tag(0), 0.8)],
    );
    assert_eq!(repeated.check(), Err("edges in (from, to, kind) order, each once".into()));

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

        // Edges in random order, each (from, to, kind) at most once.
        let mut raw = Vec::with_capacity(num_edges);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..num_edges {
            let f = Ix::<Entities>::new(next_u32() % num_from as u32);
            let t = Ix::<Tokens>::new(next_u32() % num_to as u32);
            let k = Tag((next_u32() % 4) as u16);
            let score = f64::from(next_u32() % 1000) / 1000.0;
            if seen.insert((f, t, k)) {
                raw.push(Edge::new(f, t, k, score));
            }
        }
        let distinct = raw.len();

        let edges = Edges::<Entities, Tokens>::new(kinds.clone(), raw.clone());
        assert_eq!(edges.check(), Ok(()));
        let all: Vec<_> = edges.iter().collect();
        assert_eq!(all.len(), distinct);
        // Every input row is stored, with its score.
        for e in &raw {
            assert!(all.iter().any(|(_, s)| s == e), "row {e:?} lost on trial {trial}");
        }

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
        vec![vec![Ix::new(2), Ix::new(0), Ix::new(2)], vec![Ix::new(2), Ix::new(4), Ix::new(1)], vec![Ix::new(3)]],
    );
    assert_eq!(groups.len(), 3);
    assert_eq!(groups.get(Ix::new(0)), &[Ix::new(0), Ix::new(2)]);
    assert_eq!(groups.get(Ix::new(1)), &[Ix::new(1), Ix::new(2), Ix::new(4)]);
    assert_eq!(groups.get(Ix::new(2)), &[Ix::new(3)]);
    assert!(groups.contains(Ix::new(1), Ix::new(4)));
    assert!(!groups.contains(Ix::new(0), Ix::new(4)));
    assert_eq!(groups.check(), Ok(()));

    let key = Key::<Groups<Tokens>>::named("coref");
    // The step must have read the named instance.
    let err = doc.add(&key, groups.clone(), Provenance::by("test:coref")).unwrap_err();
    assert_eq!(
        err.to_string(),
        "turbo:groups<opennlp:tokens>@coref: an index into opennlp:tokens@gold, which its step neither read nor wrote"
    );
    let mut coref_prov = Provenance::by("test:coref");
    coref_prov.reads.push(GOLD_TOK.id());
    doc.add(&key, groups, coref_prov).unwrap();
    assert_eq!(doc.made_on(key.id()), [GOLD_TOK.id()]);

    // The bound is the largest member of any group, not of the last group:
    // member 4 of group 1 is past the 3 default tokens.
    let largest_inside = Groups::<Tokens>::new([vec![Ix::new(4)], vec![Ix::new(0)]]);
    let err = doc.add(&Key::DEFAULT, largest_inside, Provenance::by("test:coref")).unwrap_err();
    assert_eq!(err.to_string(), "turbo:groups<opennlp:tokens>: an index past the end of opennlp:tokens, which has 3");

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
    assert_eq!(backwards_end.check(), Err("group ends in order, every group with a member".into()));

    // An empty group has no meaning: refused, first or later.
    let empty_first = Groups::<Sentences>::new([vec![], vec![Ix::new(0)]]);
    assert_eq!(empty_first.check(), Err("group ends in order, every group with a member".into()));
    let empty_later = Groups::<Sentences>::new([vec![Ix::new(0)], vec![]]);
    assert_eq!(empty_later.check(), Err("group ends in order, every group with a member".into()));
    let mut doc = Document::new("a");
    doc.add(&Key::DEFAULT, Sentences { spans: Spans::new(ViewId::SOURCE) }, Provenance::by("s")).unwrap();
    let err = doc.add(&Key::DEFAULT, Groups::<Sentences>::new([Vec::new()]), Provenance::by("g")).unwrap_err();
    assert_eq!(err.to_string(), "turbo:groups<opennlp:sentences>: group ends in order, every group with a member");

    let dup_member: Groups<Sentences> =
        Groups { of_key: Key::DEFAULT, members: vec![Ix::new(1), Ix::new(1)], end: vec![2] };
    assert_eq!(dup_member.check(), Err("group 0 members in order, each once".into()));
}

#[test]
fn an_empty_edge_layer_still_needs_both_endpoint_layers() {
    let rels = TagSet::new("rel", &["head"]);
    let mut doc = Document::new("a b");
    doc.add(&Key::DEFAULT, sample_tokens(2), Provenance::by("tok")).unwrap();

    // No rows, but the layer is about entities, and there are none here.
    let err =
        doc.add(&Key::DEFAULT, Edges::<Entities, Tokens>::new(rels.clone(), []), Provenance::by("e")).unwrap_err();
    assert_eq!(err.to_string(), "the document has no opennlp:entities layer");

    doc.add(&Key::DEFAULT, sample_entities(TagSet::new("ner", &["PER"]), &[]), Provenance::by("ner")).unwrap();
    doc.add(&Key::DEFAULT, Edges::<Entities, Tokens>::new(rels, []), Provenance::by("e")).unwrap();
    let edges = doc.get::<Edges<Entities, Tokens>>().unwrap();
    assert!(edges.is_empty());
    assert_eq!(edges.outgoing(Ix::new(0)).len(), 0);
    assert_eq!(edges.incoming(Ix::new(0)).len(), 0);
    assert_eq!(
        doc.made_on(Key::<Edges<Entities, Tokens>>::DEFAULT.id()),
        [Key::<Entities>::DEFAULT.id(), Key::<Tokens>::DEFAULT.id()]
    );
}

#[test]
fn nested_generic_layers_have_distinct_names() {
    type EntityHeads = Edges<Entities, Tokens>;
    type TokenLinks = Edges<Tokens, Tokens>;
    assert_eq!(EntityHeads::NAME.to_string(), "turbo:edges<opennlp:entities, opennlp:tokens>");
    assert_eq!(Groups::<EntityHeads>::NAME.to_string(), "turbo:groups<turbo:edges<opennlp:entities, opennlp:tokens>>");
    assert_eq!(Groups::<TokenLinks>::NAME.to_string(), "turbo:groups<turbo:edges<opennlp:tokens, opennlp:tokens>>");
    assert_ne!(Groups::<EntityHeads>::NAME, Groups::<TokenLinks>::NAME);
    assert_eq!(
        Edges::<Groups<Tokens>, Tokens>::NAME.to_string(),
        "turbo:edges<turbo:groups<opennlp:tokens>, opennlp:tokens>"
    );

    // Both live on one document: neither name is taken for the other's.
    let mut doc = Document::new("a b c");
    let rels = TagSet::new("rel", &["r"]);
    doc.add(&Key::DEFAULT, sample_tokens(3), Provenance::by("tok")).unwrap();
    doc.add(&Key::DEFAULT, sample_entities(TagSet::new("ner", &["PER"]), &[(0, 1, 0)]), Provenance::by("ner")).unwrap();
    let heads = EntityHeads::new(rels.clone(), [Edge::new(Ix::new(0), Ix::new(0), Tag(0), 1.0)]);
    doc.add(&Key::DEFAULT, heads, Provenance::by("heads")).unwrap();
    let links = TokenLinks::new(rels, [Edge::new(Ix::new(0), Ix::new(1), Tag(0), 1.0)]);
    doc.add(&Key::DEFAULT, links, Provenance::by("links")).unwrap();
    doc.add(&Key::DEFAULT, Groups::<EntityHeads>::new([[Ix::new(0)]]), Provenance::by("g1")).unwrap();
    doc.add(&Key::DEFAULT, Groups::<TokenLinks>::new([[Ix::new(0)]]), Provenance::by("g2")).unwrap();
    assert_eq!(doc.get::<Groups<EntityHeads>>().unwrap().len(), 1);
    assert_eq!(doc.get::<Groups<TokenLinks>>().unwrap().len(), 1);

    // And in one pipeline.
    let pipeline = Analyzer::builder()
        .given(Key::<Groups<EntityHeads>>::DEFAULT)
        .given(Key::<Groups<TokenLinks>>::DEFAULT)
        .build();
    assert!(pipeline.is_ok());
}

/// Links each token to the next one, with `Edges::new`: the instance both
/// ends are on is the one the step reads.
struct NextToken;

impl Annotator for NextToken {
    const ID: &'static str = "test:next-token";
    type Reads = (Tokens,);
    type Writes = (Edges<Tokens, Tokens>,);

    fn annotate(&self, (tokens,): Refs<'_, Self>) -> Result<(Edges<Tokens, Tokens>,), Error> {
        let n = tokens.spans.len() as u32;
        let rows = (1..n).map(|i| Edge::new(Ix::new(i - 1), Ix::new(i), Tag(0), 1.0));
        Ok((Edges::new(TagSet::new("link", &["next"]), rows),))
    }
}

/// Groups each token with the tokens it links to.
struct Reach;

impl Annotator for Reach {
    const ID: &'static str = "test:reach";
    type Reads = (Edges<Tokens, Tokens>, Tokens);
    type Writes = (Groups<Tokens>,);

    fn annotate(&self, (edges, tokens): Refs<'_, Self>) -> Result<(Groups<Tokens>,), Error> {
        let n = tokens.spans.len() as u32;
        let groups = (0..n).map(|i| {
            let s = Ix::new(i);
            std::iter::once(s).chain(edges.outgoing(s).map(|(_, e)| e.to)).collect::<Vec<_>>()
        });
        Ok((Groups::new(groups),))
    }
}

/// Names an instance it never read.
struct Undeclared;

impl Annotator for Undeclared {
    const ID: &'static str = "test:undeclared";
    type Reads = (Tokens,);
    type Writes = (Edges<Tokens, Tokens>,);

    fn annotate(&self, _: Refs<'_, Self>) -> Result<(Edges<Tokens, Tokens>,), Error> {
        let rows = [Edge::new(Ix::new(0), Ix::new(0), Tag(0), 1.0)];
        Ok((Edges::between(Key::named("gold"), Key::named("gold"), TagSet::new("link", &["self"]), rows),))
    }
}

#[test]
fn edges_written_in_a_pipeline_are_made_on_the_instances_their_step_read() {
    const GOLD: Key<Tokens> = Key::named("gold");
    let mut doc = Document::new("a b c d e f");
    doc.add(&GOLD, sample_tokens(3), Provenance::by("corpus:gold")).unwrap();
    doc.add(&Key::DEFAULT, sample_tokens(2), Provenance::by("corpus:predicted")).unwrap();
    let edges_key = Key::<Edges<Tokens, Tokens>>::DEFAULT.id();

    // Edges made on the gold tokens say so, through the step's reads.
    let reach = |r: Bound<Reach>| {
        Analyzer::builder()
            .given(GOLD)
            .given(Key::<Tokens>::DEFAULT)
            .step(Bound::new(NextToken).read(0, "gold"))
            .step(r)
            .build()
            .unwrap()
    };
    let mut docs = vec![doc.clone()];
    assert!(reach(Bound::new(Reach).read(1, "gold")).run(&mut docs)[0].is_ok());
    assert_eq!(docs[0].made_on(edges_key), [GOLD.id()]);
    assert_eq!(docs[0].provenance(edges_key).unwrap().reads, [GOLD.id()]);
    let groups = docs[0].get::<Groups<Tokens>>().unwrap();
    let got: Vec<Vec<usize>> = groups.iter().map(|(_, g)| g.iter().map(|i| i.get()).collect()).collect();
    assert_eq!(got, [vec![0, 1], vec![1, 2], vec![2]]);

    // Read beside the predicted tokens, they do not line up.
    let mut docs = vec![doc.clone()];
    let e = reach(Bound::new(Reach)).run(&mut docs).remove(0).unwrap_err();
    assert_eq!(
        e.to_string(),
        "turbo:edges<opennlp:tokens, opennlp:tokens> was made on opennlp:tokens@gold, not on the opennlp:tokens the step reads"
    );

    // A step that names an instance it did not read fails on the document.
    let a = Analyzer::builder().given(GOLD).given(Key::<Tokens>::DEFAULT).add(Undeclared).build().unwrap();
    let mut docs = vec![doc];
    let e = a.run(&mut docs).remove(0).unwrap_err();
    assert_eq!(
        e.to_string(),
        "turbo:edges<opennlp:tokens, opennlp:tokens>: an index into opennlp:tokens@gold, which its step neither read nor wrote"
    );
    assert!(docs[0].get::<Edges<Tokens, Tokens>>().is_none());
}
