//! Full-width checkpoint parity. Requires the independently generated package;
//! reduced real-weight head parity runs offline in the library unit tests.
use serde_json::Value;
use turbo_gliner::{Decode, ExtractionSchema, package::Manifest};
#[test]
#[ignore = "requires converted Apache-2.0 GLiNER checkpoint; see docs/gliner.md"]
fn full_checkpoint_matches_python() {
    let root = std::env::var("GLINER_PACKAGE").expect("GLINER_PACKAGE");
    let reference = std::env::var("GLINER_REFERENCE").expect("GLINER_REFERENCE");
    let model = std::sync::Arc::new(Manifest::load(std::path::Path::new(&root)).unwrap());
    let data: Value = serde_json::from_slice(&std::fs::read(reference).unwrap()).unwrap();
    let mut maximum = 0f64;
    for case in data["cases"].as_array().unwrap() {
        let labels = case["labels"].as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect();
        let schema = ExtractionSchema::new(labels, 0.5, Decode::Nested).unwrap();
        let words: Vec<_> = case["words"].as_array().unwrap().iter().map(|x| x.as_str().unwrap()).collect();
        let p = model.pack(&words, &schema).unwrap();
        assert_eq!(serde_json::to_value(&p.ids).unwrap(), case["ids"]);
        assert_eq!(serde_json::to_value(&p.prompts).unwrap(), case["prompts"]);
        assert_eq!(serde_json::to_value(&p.words).unwrap(), case["positions"]);
        let scores = model.score_packed(&p, &schema).unwrap();
        assert_eq!(scores.len(), case["logits"].as_array().unwrap().len());
        for (name, mode) in [("flat", Decode::Flat), ("nested", Decode::Nested)] {
            let schema = ExtractionSchema::new(schema.labels().to_vec(), 0.5, mode).unwrap();
            let selected = turbo_gliner::decode(scores.clone(), &schema);
            let expected = case["decoded"][name].as_array().unwrap();
            assert_eq!(selected.len(), expected.len());
            for (a, b) in selected.iter().zip(expected) {
                assert_eq!(
                    (a.start, a.end, a.label),
                    (b[0].as_u64().unwrap() as usize, b[1].as_u64().unwrap() as usize, b[2].as_u64().unwrap() as usize)
                );
                assert!((f64::from(a.score) - b[3].as_f64().unwrap()).abs() < 0.0001);
            }
        }
        for (s, r) in scores.iter().zip(case["logits"].as_array().unwrap()) {
            let error = (f64::from(s.logit) - r.as_f64().unwrap()).abs();
            maximum = maximum.max(error);
            assert!(error < 0.002, "logit {:?}: reference={r}, error={error}", s);
        }
    }
    verify_offsets_and_evidence(model);
    println!("full checkpoint maximum logit absolute error: {maximum}");
}

fn verify_offsets_and_evidence(model: std::sync::Arc<turbo_gliner::Model>) {
    use turbo_analysis::{Analyzer, Bound, Document, Evidence, Key, Layer, Provenance, Span, Spans, View};
    use turbo_analysis_layers::{Entities, Tokens};
    use turbo_gliner::{Decisions, GlinerAnnotator};
    let text = "Zoë visited the University of California .";
    let mut doc = Document::new(text);
    let mut spans = Spans::new(doc.get::<View>().unwrap().id());
    let mut offset = 0;
    for word in text.split(' ') {
        spans.push(Span::new(offset, offset + word.len() as u32));
        offset += word.len() as u32 + 1;
    }
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("fixture:tokens")).unwrap();
    let schema =
        ExtractionSchema::new(vec!["person".into(), "university".into(), "location".into()], 0.0, Decode::Nested)
            .unwrap();
    let ann = GlinerAnnotator::new(model, schema);
    let pipeline = Analyzer::builder()
        .given(Key::<Tokens>::DEFAULT)
        .step(Bound::new(ann).write(0, "ner").write(1, "ner"))
        .build()
        .unwrap();
    let mut docs = vec![doc];
    for r in pipeline.run(&mut docs) {
        r.unwrap();
    }
    let mut doc = docs.pop().unwrap();
    let key = Key::<Entities>::named("ner");
    let entities = doc.require(&key).unwrap();
    let (a, b) = entities.spans.columns();
    assert!(a.iter().zip(b).enumerate().any(|(i, (&s, &e))| {
        a.iter().zip(b).enumerate().any(|(j, (&t, &u))| i != j && s <= t && e >= u && (s != t || e != u))
    }));
    let tokens = doc.get::<Tokens>().unwrap();
    let (st, en) = tokens.spans.columns();
    for i in 0..entities.len() {
        let r = entities.tokens[i];
        assert_eq!(
            &text[a[i] as usize..b[i] as usize],
            &text[st[r.start.get()] as usize..en[r.end.get() - 1] as usize]
        );
    }
    let ev = GlinerAnnotator::evidence(&doc, key, Key::<Decisions>::named("ner")).unwrap();
    assert_eq!(ev.verify(&doc), Ok(()));
    let mut provenance = Provenance::by("fixture:evidence");
    provenance.reads = vec![key.id(), Key::<Decisions>::named("ner").id()];
    doc.add(&Key::<Evidence<Entities>>::named("ner"), ev, provenance).unwrap();
    let mut codecs = turbo_analysis_layers::export::codecs();
    codecs.register::<Decisions>();
    let graph = turbo_analysis::export::export(&doc, &codecs).unwrap();
    let back = turbo_analysis::export::import(&graph, &codecs).unwrap();
    assert_eq!(back.require(&Key::<Evidence<Entities>>::named("ner")).unwrap().verify(&back), Ok(()));
}
