//! The part-of-speech tags each numerics of `BilstmPosTagger` writes carry
//! its own configuration hash through a pipeline run, the document graph
//! export and its bytes, and back: a store or a cache keyed by a layer's
//! provenance never takes the packed tags for the JVM parity tags, or
//! the reverse.

mod bilstm_support;

use std::collections::HashMap;

use bilstm_support::{model, model_with};
use turbo_analysis::annotate::Step;
use turbo_analysis::export::{Codecs, DocumentGraph, export, import};
use turbo_analysis::{Analyzer, Annotator, Bound, Document, Ix, Key, Provenance, Span, Spans, View};
use turbo_analysis_layers::{PosTags, Sentences, Tokens};
use turbo_postag::neural::{BilstmPosTagger, Numerics};

const TEXT: &str = "x y z w";

fn tagger(numerics: Numerics) -> BilstmPosTagger {
    BilstmPosTagger::with_numerics(model_with(1.0, 0.0, 0.25, 0.0), numerics).unwrap()
}

/// A document of two sentences, `x y z` and `w`, tagged by `tagger` in an
/// analyzer.
fn tagged(tagger: BilstmPosTagger) -> Document {
    let mut doc = Document::new(TEXT);
    let view = doc.get::<View>().unwrap().id();
    let mut sentences = Spans::new(view);
    sentences.push(Span::new(0, 5));
    sentences.push(Span::new(6, 7));
    doc.add(&Key::<Sentences>::DEFAULT, Sentences { spans: sentences }, Provenance::by("test")).unwrap();
    let mut spans = Spans::new(view);
    for (start, end) in [(0, 1), (2, 3), (4, 5), (6, 7)] {
        spans.push(Span::new(start, end));
    }
    let sentence = [0, 0, 0, 1].map(Ix::new).to_vec();
    doc.add(&Key::<Tokens>::DEFAULT, Tokens { spans, sentence: Some(sentence) }, Provenance::by("test")).unwrap();
    let analyzer =
        Analyzer::builder().given(Key::<Sentences>::DEFAULT).given(Key::<Tokens>::DEFAULT).add(tagger).build().unwrap();
    assert_eq!(analyzer.run(std::slice::from_mut(&mut doc)), [Ok(())]);
    doc
}

fn codecs() -> Codecs {
    let mut codecs = Codecs::new();
    turbo_analysis_layers::export::register_codecs(&mut codecs);
    codecs
}

fn pos_config(doc: &Document) -> u64 {
    let prov = doc.provenance(Key::<PosTags>::DEFAULT.id()).unwrap();
    assert_eq!(prov.annotator, "turbo-postag:bilstm");
    prov.config
}

/// The graph's part-of-speech layer record, by the annotator that wrote it.
fn pos_record_config(graph: &DocumentGraph) -> u64 {
    let pos: Vec<_> = graph.layers.iter().filter(|l| l.provenance.annotator == "turbo-postag:bilstm").collect();
    assert_eq!(pos.len(), 1, "one tag layer in the graph");
    pos[0].provenance.config
}

#[test]
fn each_numerics_keeps_its_config_hash_through_the_graph_round_trip() {
    let codecs = codecs();
    let mut hashes = Vec::new();
    let mut graphs = Vec::new();
    for numerics in [Numerics::Packed, Numerics::JvmParity] {
        let t = tagger(numerics);
        let hash = t.config_hash();
        let digest = t.model().model_ref().sha256.clone();
        assert_eq!(Annotator::model(&t).map(|m| m.sha256), Some(digest.clone()), "{numerics:?}: the model's ref");
        let doc = tagged(t);
        assert_eq!(pos_config(&doc), hash, "{numerics:?}: the layer names its annotator's hash");

        let graph = export(&doc, &codecs).unwrap();
        assert_eq!(pos_record_config(&graph), hash, "{numerics:?}: exported");
        let bytes = graph.to_bytes().unwrap();
        let read = DocumentGraph::from_bytes(&bytes).unwrap();
        assert_eq!(read, graph, "{numerics:?}: the bytes read back to the same graph");
        let back = import(&read, &codecs).unwrap();
        assert_eq!(pos_config(&back), hash, "{numerics:?}: imported");
        let prov = back.provenance(Key::<PosTags>::DEFAULT.id()).unwrap();
        assert_eq!(prov.model.as_ref().unwrap().sha256, digest, "{numerics:?}: the model digest after import");
        let record = read.layers.iter().find(|l| l.provenance.annotator == "turbo-postag:bilstm").unwrap();
        assert_eq!(record.provenance.model.as_ref().unwrap().sha256, digest, "{numerics:?}: the digest in the bytes");
        let (a, b) = (doc.get::<PosTags>().unwrap(), back.get::<PosTags>().unwrap());
        assert_eq!((&a.tags, &a.prob), (&b.tags, &b.prob), "{numerics:?}: the tags themselves");
        // Exported again from the imported document: the same bytes.
        assert_eq!(export(&back, &codecs).unwrap().to_bytes().unwrap(), bytes, "{numerics:?}: re-export");
        hashes.push(hash);
        graphs.push(bytes);
    }
    assert_ne!(hashes[0], hashes[1], "packed and JVM parity tags under one hash");
    // The tags agree on this model; the graphs differ by the hash alone.
    assert_ne!(graphs[0], graphs[1]);

    // Deterministic: a second pair of taggers, a second run, the same bytes.
    for (numerics, bytes) in [Numerics::Packed, Numerics::JvmParity].into_iter().zip(&graphs) {
        let again = export(&tagged(tagger(numerics)), &codecs).unwrap().to_bytes().unwrap();
        assert_eq!(&again, bytes, "{numerics:?}: a second run");
    }
}

#[test]
fn a_cache_keyed_by_step_and_config_hash_holds_each_numerics_apart() {
    let codecs = codecs();
    // A pipeline cache as a coordinator would keep one: a step's id and
    // configuration hash to the graph bytes of the documents it made.
    let mut cache: HashMap<(&'static str, u64), Vec<u8>> = HashMap::new();
    for numerics in [Numerics::Packed, Numerics::JvmParity] {
        let step = Bound::new(tagger(numerics));
        let key = (step.id(), step.config_hash());
        assert!(!cache.contains_key(&key), "{numerics:?}: a hit on the other numerics' entry");
        let bytes = export(&tagged(tagger(numerics)), &codecs).unwrap().to_bytes().unwrap();
        cache.insert(key, bytes);
    }
    assert_eq!(cache.len(), 2);
    for numerics in [Numerics::Packed, Numerics::JvmParity] {
        // A fresh step of the same numerics finds its own entry, and the
        // document in it says it was made by that configuration.
        let step = Bound::new(tagger(numerics));
        let hit = &cache[&(step.id(), step.config_hash())];
        let back = import(&DocumentGraph::from_bytes(hit).unwrap(), &codecs).unwrap();
        assert_eq!(pos_config(&back), step.config_hash(), "{numerics:?}");
    }
    // Another model under the same numerics is another entry too.
    let other = Bound::new(BilstmPosTagger::with_numerics(model(1.0), Numerics::Packed).unwrap());
    assert!(!cache.contains_key(&(other.id(), other.config_hash())));
}
