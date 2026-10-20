use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use turbo_analysis::{
    Analyzer, Anchor, Annotator, Bound, ConditionalAnnotator, Document, Error, FilterAnnotator, Ix, Key, Layer,
    Provenance, Span, Spans, Tag, TagSet, ViewId,
};
use turbo_analysis_layers::{Language, Lemmas, PosTags, Strs, Tokens};

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

fn two_token_doc() -> Document {
    let mut doc = Document::new("ab cd");
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 2));
    spans.push(Span::new(3, 5));
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("test")).unwrap();
    let mut p = Provenance::by("test");
    p.reads.push(Key::<Tokens>::DEFAULT.id());
    let pos = PosTags { tagset: TagSet::new("upos", &["NOUN", "VERB"]), tags: vec![Tag(0), Tag(1)], prob: None };
    doc.add(&Key::DEFAULT, pos, p.clone()).unwrap();
    let mut lemma = Strs::default();
    lemma.push("ab");
    lemma.push("cd");
    doc.add(&Key::DEFAULT, Lemmas { lemma }, p).unwrap();
    doc
}

fn filter_both(pos_row: usize, lemma_row: usize) -> (Document, Result<(), Error>) {
    let mut doc = two_token_doc();
    let kept: Key<PosTags> = Key::named("kept");
    FilterAnnotator::new(Key::DEFAULT, kept, move |_, ix| ix.get() == pos_row).unwrap().annotate_doc(&mut doc).unwrap();
    let kept: Key<Lemmas> = Key::named("kept");
    let result =
        FilterAnnotator::new(Key::DEFAULT, kept, move |_, ix| ix.get() == lemma_row).unwrap().annotate_doc(&mut doc);
    (doc, result)
}

#[test]
fn a_second_filter_keeping_other_rows_is_refused() {
    let (doc, result) = filter_both(0, 1);
    assert!(matches!(result, Err(Error::Pipeline(_))), "{result:?}");
    assert!(doc.get_key(&Key::<Lemmas>::named("kept")).is_none());
}

#[test]
fn a_second_filter_keeping_the_same_rows_shares_the_anchor() {
    let (doc, result) = filter_both(1, 1);
    result.unwrap();
    let tokens = doc.get_key(&Key::<Tokens>::named("kept")).unwrap();
    assert_eq!(tokens.spans.get(Ix::new(0)), Span::new(3, 5));
    assert_eq!(doc.get_key(&Key::<Lemmas>::named("kept")).unwrap().lemma.get(0), "cd");
    assert_eq!(doc.made_on(Key::<Lemmas>::named("kept").id()), &[Key::<Tokens>::named("kept").id()]);
}

#[test]
fn existing_unrelated_anchor_must_not_rebind_filtered_pos() {
    let mut d = two_token_doc();
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(3, 5));
    d.add(&Key::<Tokens>::named("filtered"), Tokens { spans, sentence: None }, Provenance::by("unrelated")).unwrap();
    let f = FilterAnnotator::new(Key::<PosTags>::DEFAULT, Key::named("filtered"), |_, i| i.get() == 0).unwrap();
    assert!(matches!(f.annotate_doc(&mut d), Err(Error::Pipeline(_))));
    assert!(d.get_key(&Key::<PosTags>::named("filtered")).is_none());
}

struct ReadTokens;

impl Annotator for ReadTokens {
    const ID: &'static str = "test:read-filtered";
    type Reads = (Tokens,);
    type Writes = (Language,);

    fn annotate(&self, (tokens,): (&Tokens,)) -> Result<(Language,), Error> {
        let mut code = Strs::default();
        code.push("en");
        Ok((Language { code, score: vec![tokens.len() as f64] },))
    }
}

#[test]
fn cofiltered_tokens_must_be_a_declared_pipeline_output() {
    let f = FilterAnnotator::new(Key::<PosTags>::DEFAULT, Key::named("filtered"), |_, i| i.get() == 0).unwrap();
    let analyzer = Analyzer::builder()
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<PosTags>::DEFAULT)
        .step(f)
        .step(Bound::new(ReadTokens).read(0, "filtered"))
        .build()
        .expect("consumer should be able to read co-filtered anchor");
    let mut docs = vec![two_token_doc()];
    assert!(analyzer.run(&mut docs)[0].is_ok());
    assert_eq!(docs[0].require(&Key::<Language>::DEFAULT).unwrap().score, vec![1.0]);
}

struct BatchCountWriter(Arc<AtomicUsize>);

impl Annotator for BatchCountWriter {
    const ID: &'static str = "test:batch-count-writer";
    type Reads = ();
    type Writes = (PairLayer,);

    fn annotate(&self, (): ()) -> Result<(PairLayer,), Error> {
        Ok((PairLayer(1),))
    }

    fn annotate_batch<'d>(&self, inputs: Vec<()>) -> Vec<Result<(PairLayer,), Error>> {
        self.0.fetch_add(1, Ordering::SeqCst);
        inputs.into_iter().map(|()| Ok((PairLayer(1),))).collect()
    }
}

#[test]
fn conditional_annotator_batches_documents_per_branch() {
    let true_batches = Arc::new(AtomicUsize::new(0));
    let false_batches = Arc::new(AtomicUsize::new(0));
    let cond = ConditionalAnnotator::if_else_annotators(
        "starts_with_a",
        |d| d.source().starts_with('a'),
        BatchCountWriter(Arc::clone(&true_batches)),
        BatchCountWriter(Arc::clone(&false_batches)),
    )
    .unwrap();
    let analyzer = Analyzer::builder().step(cond).build().unwrap();
    let mut docs = vec![Document::new("a1"), Document::new("b1"), Document::new("a2"), Document::new("b2")];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    assert_eq!(true_batches.load(Ordering::SeqCst), 1);
    assert_eq!(false_batches.load(Ordering::SeqCst), 1);
}
