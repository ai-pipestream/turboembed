use std::sync::Arc;

use turbo::deberta::{Config, Encoder};
use turbo::safetensors::File;
use turbo::sentencepiece::SentencePiece;
use turbo_analysis::{
    Analyzer, Annotator, Document, Evidence, EvidenceStep, IxRange, Key, Layer, ModelRef, Provenance, Span, Spans, Tag,
    TagSet, View,
};
use turbo_analysis_layers::{Entities, Tokens};
use turbo_relation::{Gliner2Relations, GlirelRelations, Relations};

fn tokenizer() -> SentencePiece {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/sentencepiece/unigram-identity.model");
    SentencePiece::parse(&std::fs::read(path).unwrap()).unwrap()
}

fn model_ref() -> ModelRef {
    ModelRef { id: "reduced".into(), revision: "seeded".into(), sha256: "fixture".into() }
}

fn encoder(vocabulary: usize) -> Encoder {
    let config = Config {
        hidden: 32,
        intermediate: 48,
        heads: 2,
        layers: 2,
        vocabulary,
        max_sequence: 32,
        position_buckets: 8,
        max_relative_positions: 32,
        layer_norm_eps: 1e-7,
    };
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    let (model, prefix) =
        if vocabulary == 16 { ("gliner2", "encoder.") } else { ("glirel", "token_rep_layer.bert_layer.model.") };
    let bytes = std::fs::read(format!("{dir}/{model}/reduced.safetensors")).unwrap();
    let canonical = turbo_bundle::deberta::convert(&config, &bytes, prefix).unwrap();
    Encoder::from_safetensors(config, &canonical).unwrap()
}

fn gliner2() -> Gliner2Relations {
    let bytes =
        std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/gliner2/reduced.safetensors")).unwrap();
    let heads = turbo_relation::gliner2::heads::Heads::load(&File::parse("reduced", &bytes).unwrap(), 32, 8).unwrap();
    let specials = turbo_relation::gliner2::prompt::Specials { sep_struct: 1, sep_text: 2, p: 3, r: 4 };
    Gliner2Relations::new(encoder(800), heads, tokenizer(), specials, model_ref()).unwrap()
}

fn glirel() -> GlirelRelations {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/glirel");
    let bytes = std::fs::read(format!("{dir}/reduced.safetensors")).unwrap();
    let config = Config {
        hidden: 32,
        intermediate: 48,
        heads: 2,
        layers: 2,
        vocabulary: 800,
        max_sequence: 32,
        position_buckets: 8,
        max_relative_positions: 32,
        layer_norm_eps: 1e-7,
    };
    let canonical = turbo_bundle::deberta::convert(&config, &bytes, "token_rep_layer.bert_layer.model.").unwrap();
    let encoder = Encoder::from_safetensors(config, &canonical).unwrap();
    let heads = turbo_relation::glirel::heads::Heads::load(&File::parse("reduced", &bytes).unwrap(), 32).unwrap();
    let specials = turbo_relation::glirel::prompt::Specials { cls: 1, sep: 2, rel: 3 };
    GlirelRelations::new(encoder, heads, tokenizer(), specials, model_ref(), 512).unwrap()
}

fn inputs(text: &str) -> (Document, Tokens, Entities) {
    let doc = Document::new(text);
    let mut spans = Spans::new(turbo_analysis::ViewId::SOURCE);
    let mut start = 0;
    for word in text.split(' ') {
        spans.push(Span::new(start, start + word.len() as u32));
        start += word.len() as u32 + 1;
    }
    let mut entity_spans = Spans::new(turbo_analysis::ViewId::SOURCE);
    for (_, span) in spans.iter() {
        entity_spans.push(span);
    }
    let tokens = Tokens { spans, sentence: None };
    let entities = Entities {
        spans: entity_spans,
        tokens: vec![IxRange::new(0, 1), IxRange::new(1, 2)],
        kind: vec![Tag(0), Tag(0)],
        kinds: TagSet::new("test", &["thing"]),
        score: vec![1.0, 1.0],
    };
    (doc, tokens, entities)
}

#[test]
fn gliner2_annotator_builds_entities_and_relation_layers() {
    let (doc, tokens, _) = inputs("a b");
    let mut model = gliner2();
    model.configure(vec!["a".into()], 0.0).unwrap();
    let prompt = model.prompt("a b", &["a"]);
    assert!(prompt.ids.iter().all(|&id| id < 800), "{:?}", prompt.ids);
    let before = model.config_hash();
    let (entities, relations, edges) = model.annotate((doc.get::<View>().unwrap(), &tokens, None)).unwrap();
    entities.check().unwrap();
    relations.check().unwrap();
    edges.check().unwrap();
    assert_eq!(relations.kinds.label(Tag(0)), "a");
    assert_eq!(Annotator::model(&model).unwrap().id, "reduced");
    model.configure(vec!["other".into()], 0.5).unwrap();
    assert_ne!(before, model.config_hash());
}

#[test]
fn glirel_annotator_reuses_existing_entities() {
    let (doc, tokens, existing) = inputs("a b");
    let mut model = glirel();
    model.configure(vec!["related".into()], 0.0).unwrap();
    let (entities, relations, edges) = model.annotate((doc.get::<View>().unwrap(), &tokens, Some(&existing))).unwrap();
    assert_eq!(entities.spans.len(), 2);
    assert_eq!(relations.len(), 2);
    assert_eq!(edges.len(), 2);
    for i in 0..relations.len() {
        assert_ne!(relations.subject[i], relations.object[i]);
    }
    let empty = model.annotate((doc.get::<View>().unwrap(), &tokens, None)).unwrap();
    assert_eq!(empty.1.len(), 0);
    assert!(Arc::ptr_eq(&empty.1.kinds, &empty.2.kinds));
}

#[test]
fn configuration_is_validated() {
    assert!(gliner2().configure(Vec::new(), 0.5).is_err());
    assert!(glirel().configure(vec!["x".into()], f32::NAN).is_err());
    let (doc, tokens, _) = inputs("a b");
    let error = gliner2().annotate((doc.get::<View>().unwrap(), &tokens, None));
    assert!(error.is_err());
}

#[test]
fn analyzer_records_model_provenance_without_existing_entities() {
    let (mut doc, tokens, _) = inputs("a b");
    doc.add(&Key::DEFAULT, tokens, Provenance::by("test")).unwrap();
    let mut model = gliner2();
    model.configure(vec!["a".into()], 0.0).unwrap();
    let analyzer = Analyzer::builder()
        .given(Key::<Tokens>::DEFAULT)
        .add(model)
        .step(EvidenceStep::new(Key::<Relations>::DEFAULT))
        .build()
        .unwrap();
    let mut docs = vec![doc];
    analyzer.run(&mut docs).into_iter().for_each(Result::unwrap);
    let provenance = docs[0].provenance(Key::<Relations>::DEFAULT.id()).unwrap();
    assert_eq!(provenance.model.as_ref(), Some(&model_ref()));
    let evidence = docs[0].get::<Evidence<Relations>>().unwrap();
    assert_eq!(evidence.verify(&docs[0]), Ok(()));
}
