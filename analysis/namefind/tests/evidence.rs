//! Evidence for a name finder's entities: one record per entity, its
//! spans mapped from the view the finder read back to the source bytes,
//! the tokens that support it, and the graph round trip.

use turbo_analysis::export::{DocumentGraph, export, import};
use turbo_analysis::{
    Analyzer, Bound, Document, Evidence, EvidenceStep, Ix, Key, Provenance, Span, Spans, View, ViewBuilder,
};
use turbo_analysis_layers::{Entities, Sentences, Tokens};
use turbo_maxent::Dictionary;
use turbo_namefind::DictionaryNameFinder;

const SOURCE: &str = "<p>Mail Ann in <b>New York</b> .</p> Call <i>new</i> york today";

/// The source with its tags taken out, as the view "clean", with tokens
/// at its spaces and one sentence.
fn document() -> Document {
    let mut b = ViewBuilder::new(SOURCE);
    let mut at = 0;
    while let Some(open) = SOURCE[at..].find('<') {
        let open = at + open;
        let close = open + SOURCE[open..].find('>').unwrap() + 1;
        b.keep(open);
        b.replace(close, "");
        at = close;
    }
    b.keep(SOURCE.len());
    let (text, alignment) = b.finish();
    let mut doc = Document::new(SOURCE);
    let view = View::derived(doc.get::<View>().unwrap(), text.clone(), alignment);
    doc.add(&Key::named("clean"), view, Provenance::by("test:strip")).unwrap();
    let id = doc.get_key(&Key::<View>::named("clean")).unwrap().id();
    let (mut tokens, mut start) = (Spans::new(id), 0);
    for piece in text.split(' ') {
        tokens.push(Span::new(start as u32, (start + piece.len()) as u32));
        start += piece.len() + 1;
    }
    let n = tokens.len();
    let mut sentences = Spans::new(id);
    sentences.push(Span::new(0, text.len() as u32));
    doc.add(&Key::DEFAULT, Sentences { spans: sentences }, Provenance::by("test")).unwrap();
    let sentence = Some(vec![Ix::<Sentences>::new(0); n]);
    doc.add(&Key::DEFAULT, Tokens { spans: tokens, sentence }, Provenance::by("test")).unwrap();
    doc
}

fn finder() -> DictionaryNameFinder {
    let xml = "<dictionary case_sensitive='false'><entry><token>New</token><token>York</token></entry>\
               <entry><token>Ann</token></entry></dictionary>";
    DictionaryNameFinder::new(Dictionary::read(xml.as_bytes()).unwrap(), "name")
}

fn run(evidence: bool) -> Document {
    let mut b = Analyzer::builder()
        .given(Key::<View>::named("clean"))
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .step(Bound::new(finder()).read(0, "clean"));
    if evidence {
        b = b.step(EvidenceStep::new(Key::<Entities>::DEFAULT));
    }
    let mut docs = vec![document()];
    let results = b.build().unwrap().run(&mut docs);
    assert!(results.iter().all(Result::is_ok), "{results:?}");
    docs.pop().unwrap()
}

#[test]
fn each_entity_has_one_record_mapped_to_the_source() {
    let doc = run(true);
    let entities = doc.get::<Entities>().unwrap();
    let ev = doc.get::<Evidence<Entities>>().unwrap();
    assert_eq!((entities.spans.len(), ev.records()), (3, 3));
    let texts: Vec<&str> = (0..3).map(|i| &SOURCE[ev.spans_of(i)[0].range()]).collect();
    // "New York" spans its closing tag in the source, as the view's
    // alignment widens a cut edit; "new</i> york" keeps the tag inside.
    assert_eq!(texts, ["Ann", "New York", "new</i> york"]);
    for i in 0..3 {
        let p = ev.producer_of(i);
        assert_eq!((p.annotator.as_str(), &p.model), ("turbo-namefind:dictionary", &None));
        // A rule-backed finder has no score, so none is recorded.
        assert_eq!((ev.raw_of(i), ev.calibrated_of(i), ev.abstention_of(i)), (None, None, None));
        let s = ev.support_of(i)[0];
        let t = entities.tokens[i];
        assert_eq!((s.start, s.end), (t.start.get() as u32, t.end.get() as u32));
    }
    assert_eq!(ev.layers[0].layer, "opennlp:tokens");
    assert_eq!(ev.verify(&doc), Ok(()));
}

#[test]
fn evidence_crosses_the_graph_and_documents_without_it_cross_as_before() {
    let codecs = turbo_analysis_layers::export::codecs();
    let doc = run(true);
    let graph = export(&doc, &codecs).unwrap();
    let back = import(&DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &codecs).unwrap();
    let (a, b) = (doc.get::<Evidence<Entities>>().unwrap(), back.get::<Evidence<Entities>>().unwrap());
    assert_eq!((&a.producers, &a.spans, &a.support, &a.layers), (&b.producers, &b.spans, &b.support, &b.layers));
    assert_eq!(export(&back, &codecs).unwrap().to_bytes().unwrap(), graph.to_bytes().unwrap());

    // Without the evidence step the graph holds no evidence record, and
    // the rest of it is what the evidence run exported.
    let plain = export(&run(false), &codecs).unwrap();
    assert!(plain.layers.iter().all(|l| !l.layer.starts_with("turbo:evidence")));
    let mut trimmed = graph.clone();
    trimmed.layers.retain(|l| !l.layer.starts_with("turbo:evidence"));
    assert_eq!(trimmed.to_bytes().unwrap(), plain.to_bytes().unwrap());
    let again = import(&DocumentGraph::from_bytes(&plain.to_bytes().unwrap()).unwrap(), &codecs).unwrap();
    assert_eq!(export(&again, &codecs).unwrap().to_bytes().unwrap(), plain.to_bytes().unwrap());
}

#[test]
fn an_imported_record_outside_the_source_is_refused() {
    let codecs = turbo_analysis_layers::export::codecs();
    let doc = run(true);
    let mut ev = doc.get::<Evidence<Entities>>().unwrap().clone();
    ev.spans[0] = Span::new(0, SOURCE.len() as u32 + 1);
    let mut bad = run(false);
    bad.add(&Key::DEFAULT, ev, Provenance { reads: vec![Key::<Entities>::DEFAULT.id()], ..Provenance::by("x") })
        .unwrap();
    let graph = export(&bad, &codecs).unwrap();
    let Err(err) = import(&graph, &codecs) else { panic!("a span past the source imported") };
    let err = err.to_string();
    assert!(err.contains("outside a source of"), "{err}");
    // The view the spans came from is not the source: the same record
    // checked against a document whose source is shorter is refused.
    let short = Document::new("Mail Ann");
    assert!(doc.get::<Evidence<Entities>>().unwrap().verify(&short).is_err());
}
