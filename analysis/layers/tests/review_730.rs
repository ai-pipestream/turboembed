use turbo_analysis::{
    Anchor, Annotator, Bound, ConditionalAnnotator, Document, Error, FilterAnnotator, Key, Layer, Provenance, Span,
    Spans, Tag, TagSet, ViewId,
};
use turbo_analysis_layers::{PosTags, Tokens};

struct PairLayer(#[allow(dead_code)] u32);

impl Layer for PairLayer {
    const ID: &'static str = "test:review-730-pair";

    fn anchor() -> Anchor {
        Anchor::Document
    }

    fn len(&self) -> usize {
        1
    }
}

struct PairWriter;

impl Annotator for PairWriter {
    const ID: &'static str = "test:review-730-pair-writer";
    type Reads = ();
    type Writes = (PairLayer, PairLayer);

    fn annotate(&self, (): ()) -> Result<(PairLayer, PairLayer), Error> {
        Ok((PairLayer(1), PairLayer(2)))
    }
}

#[test]
fn branch_output_bindings_must_affect_config_hash() {
    let a: Key<PairLayer> = Key::named("a");
    let b: Key<PairLayer> = Key::named("b");

    let original = ConditionalAnnotator::if_else(
        "pred",
        |_| false,
        Bound::new(PairWriter).write(0, "a").write(1, "b"),
        Bound::new(PairWriter).write(0, "a").write(1, "b"),
    )
    .unwrap();
    let swapped = ConditionalAnnotator::if_else(
        "pred",
        |_| false,
        Bound::new(PairWriter).write(0, "a").write(1, "b"),
        Bound::new(PairWriter).write(0, "b").write(1, "a"),
    )
    .unwrap();

    let mut first = Document::new("x");
    let mut second = Document::new("x");
    original.annotate_doc(&mut first).unwrap();
    swapped.annotate_doc(&mut second).unwrap();

    assert_ne!(
        original.config_hash(),
        swapped.config_hash(),
        "rebinding branch outputs must change ConditionalAnnotator::config_hash"
    );
    assert_ne!(
        first.provenance(a.id()),
        second.provenance(a.id()),
        "swapping false-branch output bindings must change stored provenance"
    );
    assert_ne!(
        first.provenance(b.id()),
        second.provenance(b.id()),
        "swapping false-branch output bindings must change stored provenance"
    );
}

#[test]
fn additional_predicate_read_must_not_rebind_aligned_output() {
    let mut doc = Document::new("ab cd");

    let original: Key<Tokens> = Key::named("original");
    let other: Key<Tokens> = Key::named("other");
    let mut orig_spans = Spans::new(ViewId::SOURCE);
    orig_spans.push(Span::new(0, 2));
    let mut other_spans = Spans::new(ViewId::SOURCE);
    other_spans.push(Span::new(3, 5));
    doc.add(&original, Tokens { spans: orig_spans, sentence: None }, Provenance::by("test")).unwrap();
    doc.add(&other, Tokens { spans: other_spans, sentence: None }, Provenance::by("test")).unwrap();

    let mut pos_prov = Provenance::by("test");
    pos_prov.reads.push(original.id());
    let tagset = TagSet::new("upos", &["NOUN"]);
    doc.add(&Key::DEFAULT, PosTags { tagset, tags: vec![Tag(0)], prob: None }, pos_prov).unwrap();
    assert_eq!(doc.made_on(Key::<PosTags>::DEFAULT.id()), &[original.id()]);

    let filtered: Key<PosTags> = Key::named("filtered");
    let filter = FilterAnnotator::with_document(Key::DEFAULT, filtered, move |d, _pos, _ix| {
        d.get_key(&other).is_some_and(|t| t.len() == 1)
    })
    .unwrap()
    .require(other)
    .require(original);

    filter.annotate_doc(&mut doc).unwrap();
    assert_eq!(
        doc.made_on(filtered.id()),
        &[original.id()],
        "filtered PosTags must stay anchored to the source PosTags' token layer"
    );
}

#[test]
fn aligned_filter_can_remove_an_item() {
    let mut doc = Document::new("ab cd");
    let mut tok_spans = Spans::new(ViewId::SOURCE);
    tok_spans.push(Span::new(0, 2));
    tok_spans.push(Span::new(3, 5));
    doc.add(&Key::DEFAULT, Tokens { spans: tok_spans, sentence: None }, Provenance::by("test")).unwrap();

    let mut pos_prov = Provenance::by("test");
    pos_prov.reads.push(Key::<Tokens>::DEFAULT.id());
    let tagset = TagSet::new("upos", &["NOUN", "VERB"]);
    doc.add(&Key::DEFAULT, PosTags { tagset, tags: vec![Tag(0), Tag(1)], prob: None }, pos_prov).unwrap();

    let filtered: Key<PosTags> = Key::named("filtered");
    let filter = FilterAnnotator::new(Key::DEFAULT, filtered, |pos, ix| pos.tags[ix.get()] == Tag(0)).unwrap();

    filter.annotate_doc(&mut doc).expect("filtering aligned PosTags down to one item should succeed");
    assert_eq!(doc.get_key(&filtered).unwrap().len(), 1);
    assert_eq!(doc.made_on(filtered.id()), &[Key::<Tokens>::named("filtered").id()]);
}
