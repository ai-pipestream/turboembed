use super::{boundaries, *};
use crate::window::pack_windows;
use turbo_analysis::{Analyzer, Annotator, Bound, Document, Evidence, Ix, Key, Layer, Provenance, Span, Spans, View};
use turbo_analysis_layers::{Entities, Sentences, Tokens};

fn document(words: &[&str], sentence_size: Option<usize>) -> Document {
    let text = words.join(" ");
    let mut doc = Document::new(text);
    let view = doc.get::<View>().unwrap().id();
    let mut spans = Spans::new(view);
    let mut offset = 0;
    for word in words {
        spans.push(Span::new(offset, offset + word.len() as u32));
        offset += word.len() as u32 + 1;
    }
    let sentence = sentence_size.map(|size| {
        let (a, b) = spans.columns();
        let mut s = Spans::new(view);
        // Empty leading/trailing sentences must not change the windows.
        s.push(Span::new(0, 0));
        for i in (0..a.len()).step_by(size) {
            s.push(Span::new(a[i], b[(i + size).min(b.len()) - 1]));
        }
        let end = b.last().copied().unwrap_or(0);
        s.push(Span::new(end, end));
        doc.add(&Key::DEFAULT, Sentences { spans: s }, Provenance::by("fixture:sentences")).unwrap();
        (0..words.len()).map(|i| Ix::new((i / size + 1) as u32)).collect()
    });
    doc.add(&Key::DEFAULT, Tokens { spans, sentence }, Provenance::by("fixture:tokens")).unwrap();
    doc
}
fn schema(mode: Decode) -> ExtractionSchema {
    ExtractionSchema::new(vec!["person".into(), "place".into()], 0.0, mode).unwrap()
}

#[test]
fn sentence_windows_match_standalone_union_with_named_evidence() {
    let model = boundaries::model();
    let mut words = vec!["Zoë"; 2048];
    words[1025] = "\u{200B}";
    words[1026] = "\u{FEFF}";
    words[1027] = "🙂";
    for mode in [Decode::Flat, Decode::Nested] {
        let schema = schema(mode);
        let doc = document(&words, Some(4));
        let tokens = doc.get::<Tokens>().unwrap();
        let windows =
            pack_windows(&model, &schema, doc.get::<View>().unwrap(), tokens, doc.get::<Sentences>()).unwrap();
        assert!(windows.len() > 1);
        assert!(windows.iter().all(|w| w.tokens.start % 4 == 0 && w.tokens.end % 4 == 0));
        let mut expected = Vec::new();
        for w in &windows {
            let independent = model.pack(&words[w.tokens.clone()], &schema).unwrap();
            assert_eq!(w.packed.ids, independent.ids);
            assert_eq!(w.packed.words, independent.words);
            assert_eq!(w.packed.prompts, independent.prompts);
            for mut c in model.predict(&words[w.tokens.clone()], &schema).unwrap() {
                c.start += w.tokens.start;
                c.end += w.tokens.start;
                expected.push(c);
            }
        }
        for w in &windows {
            assert!(expected.iter().any(|c| c.start == w.tokens.start));
            assert!(expected.iter().any(|c| c.end == w.tokens.end));
        }
        let pipeline = Analyzer::builder()
            .given(Key::<Tokens>::DEFAULT)
            .given(Key::<Sentences>::DEFAULT)
            .step(Bound::new(GlinerAnnotator::new(model.clone(), schema)).write(0, "ner").write(1, "ner"))
            .build()
            .unwrap();
        let mut docs = vec![doc];
        for r in pipeline.run(&mut docs) {
            r.unwrap();
        }
        let mut doc = docs.pop().unwrap();
        let key = Key::<Entities>::named("ner");
        let entities = doc.require(&key).unwrap();
        let decisions = doc.require(&Key::<Decisions>::named("ner")).unwrap();
        assert_eq!(entities.len(), expected.len());
        assert!(!expected.is_empty());
        let (a, b) = entities.spans.columns();
        let text = doc.get::<View>().unwrap().text();
        for (i, c) in expected.iter().enumerate() {
            assert_eq!((entities.tokens[i].start.get(), entities.tokens[i].end.get()), (c.start, c.end));
            assert_eq!(entities.kind[i].0 as usize, c.label);
            assert_eq!(entities.score[i], f64::from(c.score));
            assert_eq!(decisions.logits[i], c.logit);
            assert_eq!(&text[a[i] as usize..b[i] as usize], words[c.start..c.end].join(" "));
            assert!(windows.iter().any(|w| c.start >= w.tokens.start && c.end <= w.tokens.end));
        }
        let ev = GlinerAnnotator::evidence(&doc, key, Key::named("ner")).unwrap();
        ev.verify(&doc).unwrap();
        assert_eq!(ev.raw.len(), entities.len());
        let mut provenance = Provenance::by("fixture:evidence");
        provenance.reads = vec![key.id(), Key::<Decisions>::named("ner").id()];
        doc.add(&Key::<Evidence<Entities>>::named("ner"), ev, provenance).unwrap();
        let mut codecs = turbo_analysis_layers::export::codecs();
        codecs.register::<Decisions>();
        let graph = turbo_analysis::export::export(&doc, &codecs).unwrap();
        let back = turbo_analysis::export::import(&graph, &codecs).unwrap();
        back.require(&Key::<Evidence<Entities>>::named("ner")).unwrap().verify(&back).unwrap();
    }
}

#[test]
fn word_and_subword_limits_include_prompt_and_eos() {
    let schema = schema(Decode::Flat);
    let model = boundaries::model_with_sequence(1024);
    let doc = document(&vec!["a"; 800], None);
    let windows =
        pack_windows(&model, &schema, doc.get::<View>().unwrap(), doc.get::<Tokens>().unwrap(), None).unwrap();
    assert_eq!(windows.iter().map(|w| w.tokens.len()).collect::<Vec<_>>(), vec![384, 384, 32]);
    for w in windows {
        assert!(w.packed.ids.len() <= 1024);
    }
    let model = boundaries::model();
    let doc = document(&vec!["a"; 100], None);
    let windows =
        pack_windows(&model, &schema, doc.get::<View>().unwrap(), doc.get::<Tokens>().unwrap(), None).unwrap();
    assert!(windows.iter().all(|w| w.packed.ids.len() <= 32));
    assert!(windows[0].packed.ids.len() >= 31);
    let small = document(&["Zoë", "🙂"], None);
    let w = pack_windows(&model, &schema, small.get::<View>().unwrap(), small.get::<Tokens>().unwrap(), None).unwrap();
    assert_eq!(w.len(), 1);
    assert_eq!(w[0].packed.ids, model.pack(&["Zoë", "🙂"], &schema).unwrap().ids);
}

#[test]
fn oversized_sentence_names_the_index_and_limit() {
    let model = boundaries::model();
    let schema = schema(Decode::Flat);
    for (count, limit) in [(385, "max_tokens"), (25, "max_sequence")] {
        let doc = document(&vec!["a"; count], Some(count));
        let error = GlinerAnnotator::new(model.clone(), schema.clone())
            .annotate((doc.get::<View>().unwrap(), doc.get::<Tokens>().unwrap(), doc.get::<Sentences>()))
            .err()
            .unwrap();
        let message = error.to_string();
        assert!(message.contains("sentence 1"), "{message}");
        assert!(message.contains(limit), "{message}");
    }
}

#[test]
fn invalid_sentence_ownership_and_utf8_are_rejected() {
    let model = boundaries::model();
    let schema = schema(Decode::Flat);
    let doc = document(&["Zoë", "🙂"], None);
    let view = doc.get::<View>().unwrap();
    let tokens = doc.get::<Tokens>().unwrap();
    let check = |spans| pack_windows(&model, &schema, view, tokens, Some(&Sentences { spans })).err().unwrap();
    for (a, b) in [(1, 9), (0, 3), (0, 4), (0, 10)] {
        let mut spans = Spans::new(view.id());
        spans.push(Span::new(a, b));
        assert!(!check(spans).is_empty());
    }
    assert!(check(Spans::new(turbo_analysis::ViewId(1))).contains("input view"));
    assert!(check(Spans::new(view.id())).contains("outside supplied sentences"));
    let doc = document(&["a", "b"], Some(1));
    let original = doc.get::<Tokens>().unwrap();
    let mut spans = Spans::new(doc.get::<View>().unwrap().id());
    let (a, b) = original.spans.columns();
    for (&a, &b) in a.iter().zip(b) {
        spans.push(Span::new(a, b));
    }
    let bad = Tokens { spans, sentence: Some(vec![Ix::new(0), Ix::new(0)]) };
    assert!(
        pack_windows(&model, &schema, doc.get::<View>().unwrap(), &bad, doc.get::<Sentences>())
            .err()
            .unwrap()
            .contains("sentence index mismatch")
    );
}

#[test]
fn indivisible_token_and_prompt_overflow_are_named_errors() {
    let model = boundaries::model_with_sequence(8);
    let schema = ExtractionSchema::new(vec!["a".into()], 0.0, Decode::Flat).unwrap();
    let doc = document(&["a", "a a"], None);
    let error =
        pack_windows(&model, &schema, doc.get::<View>().unwrap(), doc.get::<Tokens>().unwrap(), None).err().unwrap();
    assert!(error.contains("token 1") && error.contains("max_sequence"), "{error}");
    let crowded = ExtractionSchema::new(vec!["a".into(), "b".into()], 0.0, Decode::Flat).unwrap();
    let error =
        pack_windows(&model, &crowded, doc.get::<View>().unwrap(), doc.get::<Tokens>().unwrap(), None).err().unwrap();
    assert!(error.contains("schema prompt") && error.contains("max_sequence"), "{error}");
}
