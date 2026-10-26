mod support;
use turbo_analysis::{
    Analyzer, Annotator, Document, Ix, Key, Layer, Provenance, Span, Spans, Tag, TagSet, View, ViewId,
};
use turbo_analysis_layers::{
    Dependencies, MwtLanguage, PosTags, RuleMwtExpander, Sentences, Tokens, WordPosTags, Words,
};
use turbo_depparse::DependencyAnnotator;

fn spans<L>(ranges: &[(u32, u32)]) -> Spans<L> {
    let mut out = Spans::new(ViewId::SOURCE);
    for &(start, end) in ranges {
        out.push(Span::new(start, end));
    }
    out
}
fn tokens() -> Tokens {
    Tokens {
        spans: spans(&[(0, 2), (3, 4), (5, 6), (7, 8)]),
        sentence: Some(vec![Ix::new(0), Ix::new(0), Ix::new(1), Ix::new(1)]),
    }
}
fn sentences() -> Sentences {
    Sentences { spans: spans(&[(0, 4), (5, 8)]) }
}
fn pos(n: usize) -> PosTags {
    PosTags { tags: vec![Tag(0); n], tagset: TagSet::new("upos", &["NOUN"]), prob: None }
}
#[test]
fn token_pipeline_offsets_sentence_heads_and_records_provenance() {
    let mut doc = Document::new("é a b c");
    doc.add(&Key::DEFAULT, sentences(), Provenance::by("test")).unwrap();
    doc.add(&Key::DEFAULT, tokens(), Provenance::by("test")).unwrap();
    doc.add(&Key::DEFAULT, pos(4), Provenance::by("test")).unwrap();
    let analyzer = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<PosTags>::DEFAULT)
        .add(DependencyAnnotator::<Tokens>::new(support::parser()))
        .build()
        .unwrap();
    let mut docs = vec![doc];
    analyzer.run(&mut docs)[0].as_ref().unwrap();
    let deps = docs[0].get::<Dependencies>().unwrap();
    assert_eq!(deps.head, vec![None, Some(Ix::new(0)), None, Some(Ix::new(2))]);
    let p = docs[0].provenance(Key::<Dependencies>::DEFAULT.id()).unwrap();
    assert_eq!(p.version, "1");
    assert_eq!(p.model.as_ref().unwrap().sha256.len(), 64);
    assert!(p.reads.contains(&Key::<PosTags>::DEFAULT.id()));
}
#[test]
fn word_pipeline_keeps_multiword_token_granularity() {
    let text = "Voy al mar";
    let tok = Tokens { spans: spans(&[(0, 3), (4, 6), (7, 10)]), sentence: None };
    let words = RuleMwtExpander::new(MwtLanguage::Spanish).expand(text, &tok);
    assert_eq!(words.len(), 4);
    let mut doc = Document::new(text);
    doc.add(&Key::DEFAULT, Sentences { spans: spans(&[(0, 10)]) }, Provenance::by("test")).unwrap();
    doc.add(&Key::DEFAULT, tok, Provenance::by("test")).unwrap();
    doc.add(&Key::DEFAULT, words, Provenance::by("test")).unwrap();
    doc.add(
        &Key::DEFAULT,
        WordPosTags { tags: vec![Tag(0); 4], tagset: TagSet::new("upos", &["NOUN"]), prob: None },
        Provenance::by("test"),
    )
    .unwrap();
    let analyzer = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Words>::DEFAULT)
        .given(Key::<WordPosTags>::DEFAULT)
        .add(DependencyAnnotator::<Words>::new(support::parser()))
        .build()
        .unwrap();
    let mut docs = vec![doc];
    analyzer.run(&mut docs)[0].as_ref().unwrap();
    let deps = docs[0].get::<Dependencies<Words>>().unwrap();
    assert_eq!(deps.len(), 4);
    assert!(docs[0].get::<Dependencies<Tokens>>().is_none());
    assert_eq!(Dependencies::<Words>::NAME.to_string(), "opennlp:dependencies<turbo:words>");
}
#[test]
fn direct_boundary_checks_reject_bad_offsets_views_tags_and_alignment() {
    let doc = Document::new("é a b c");
    let view = doc.get::<View>().unwrap();
    let annotator = DependencyAnnotator::<Tokens>::new(support::parser());
    let tok = tokens();
    let sent = sentences();
    let tags = pos(4);
    assert!(annotator.annotate((view, &sent, &tok, &tok, &pos(3))).is_err());
    let bad = Tokens { spans: spans(&[(0, 1), (3, 4), (5, 6), (7, 8)]), sentence: None };
    assert!(annotator.annotate((view, &sent, &bad, &bad, &tags)).is_err());
    let mut bad = tokens();
    bad.sentence = Some(vec![Ix::new(1); 4]);
    assert!(annotator.annotate((view, &sent, &bad, &bad, &tags)).is_err());
    let badsent = Sentences { spans: spans(&[(0, 3), (5, 8)]) };
    assert!(annotator.annotate((view, &badsent, &tok, &tok, &tags)).is_err());
    let badsent = Sentences { spans: Spans::new(ViewId(1)) };
    assert!(annotator.annotate((view, &badsent, &tok, &tok, &tags)).is_err());
    let mut badtags = pos(4);
    badtags.tags[0] = Tag(1);
    assert!(annotator.annotate((view, &sent, &tok, &tok, &badtags)).is_err());
    let other = Tokens { spans: spans(&[(0, 2), (3, 4), (5, 6)]), sentence: None };
    assert!(annotator.annotate((view, &sent, &tok, &other, &tags)).is_err());
}
#[test]
fn empty_document_produces_empty_forest() {
    let doc = Document::new("");
    let tok = Tokens { spans: spans(&[]), sentence: None };
    let sent = Sentences { spans: spans(&[]) };
    let (out,) = DependencyAnnotator::<Tokens>::new(support::parser())
        .annotate((doc.get::<View>().unwrap(), &sent, &tok, &tok, &pos(0)))
        .unwrap();
    assert!(out.head.is_empty());
    out.check().unwrap();
}

#[test]
fn reversed_surface_order_is_rejected_before_resolving_word_ranges() {
    use turbo_analysis::IxRange;
    use turbo_analysis_layers::WordOrTokenAnchor;
    let doc = Document::new("é a");
    let view = doc.get::<View>().unwrap();
    let tok = Tokens { spans: spans(&[(3, 4), (0, 2)]), sentence: None };
    let mut words = Words::from_tokens(&tok);
    words.tokens = vec![IxRange::new(0, 2)];
    assert!(words.resolved(view, &tok).is_err());
}
#[test]
fn unordered_tokens_cannot_be_silently_parsed_in_storage_order() {
    let doc = Document::new("é a");
    let tok = Tokens { spans: spans(&[(3, 4), (0, 2)]), sentence: None };
    let sent = Sentences { spans: spans(&[(0, 4)]) };
    assert!(
        DependencyAnnotator::<Tokens>::new(support::parser())
            .annotate((doc.get::<View>().unwrap(), &sent, &tok, &tok, &pos(2)))
            .is_err()
    );
}

#[test]
fn real_trained_maxent_and_feedforward_models_parse_through_annotator() {
    use turbo_depparse::{
        DependencyBackend, DependencyModel, DependencyParserME, DependencySample, FeedforwardDependencyModel,
        FeedforwardDependencyParser, FeedforwardDependencyTrainer, FeedforwardSettings,
    };
    use turbo_formats::conllu::{ConlluDependencyReader, Tagset};
    use turbo_train::{Key as TrainKey, TrainingParameters};
    use turbo_unicode::WhitespaceMode;

    let conllu = concat!(
        "# text = El gato negro duerme hoy.\n",
        "1\tEl\tel\tDET\t_\t_\t2\tdet\t_\t_\n",
        "2\tgato\tgato\tNOUN\t_\t_\t4\tnsubj\t_\t_\n",
        "3\tnegro\tnegro\tADJ\t_\t_\t2\tamod\t_\t_\n",
        "4\tduerme\tdormir\tVERB\t_\t_\t0\troot\t_\t_\n",
        "5\thoy\thoy\tADV\t_\t_\t4\tadvmod\t_\tSpaceAfter=No\n",
        "6\t.\t.\tPUNCT\t_\t_\t4\tpunct\t_\t_\n\n",
        "# text = La niña pequeña corre rápido.\n",
        "1\tLa\tel\tDET\t_\t_\t2\tdet\t_\t_\n",
        "2\tniña\tniña\tNOUN\t_\t_\t4\tnsubj\t_\t_\n",
        "3\tpequeña\tpequeño\tADJ\t_\t_\t2\tamod\t_\t_\n",
        "4\tcorre\tcorrer\tVERB\t_\t_\t0\troot\t_\t_\n",
        "5\trápido\trápido\tADV\t_\t_\t4\tadvmod\t_\tSpaceAfter=No\n",
        "6\t.\t.\tPUNCT\t_\t_\t4\tpunct\t_\t_\n\n",
    );
    let mut reader = ConlluDependencyReader::new(conllu, Tagset::U, WhitespaceMode::Unicode).unwrap();
    let mut base = Vec::new();
    for item in &mut reader {
        let conllu_sample = item.unwrap();
        base.push(DependencySample::<Tokens>::from_format(&conllu_sample).unwrap());
    }
    let samples: Vec<_> = (0..6).flat_map(|_| base.iter().cloned()).collect();

    let mut params = TrainingParameters::default();
    params.put_str(TrainKey::ITERATIONS, "50");
    params.put_str(TrainKey::CUTOFF, "1");
    let maxent_model = DependencyModel::train("es", &samples, &params).unwrap();
    let reloaded_maxent = DependencyModel::open(maxent_model.bytes().unwrap().to_vec()).unwrap();

    let ff_settings = FeedforwardSettings {
        epochs: 15,
        batch_size: 4,
        hidden_size: 32,
        dropout: 0.0,
        seed: 42,
        ..Default::default()
    };
    let ff_model = FeedforwardDependencyTrainer::train(&samples, &ff_settings).unwrap();
    let reloaded_ff = FeedforwardDependencyModel::from_bytes(&ff_model.to_bytes().unwrap()).unwrap();

    for backend in [
        DependencyBackend::Linear(DependencyParserME::new(reloaded_maxent)),
        DependencyBackend::Feedforward(FeedforwardDependencyParser::new(reloaded_ff, 1).unwrap()),
    ] {
        let text = "El gato negro duerme hoy.";
        let mut doc = Document::new(text);
        let sent_spans = spans(&[(0, 25)]);
        let tok_spans = spans(&[(0, 2), (3, 7), (8, 13), (14, 20), (21, 24), (24, 25)]);
        doc.add(&Key::DEFAULT, Sentences { spans: sent_spans }, Provenance::by("test")).unwrap();
        doc.add(
            &Key::DEFAULT,
            Tokens { spans: tok_spans, sentence: Some(vec![Ix::new(0); 6]) },
            Provenance::by("test"),
        )
        .unwrap();
        let tagset = TagSet::new("upos", &["DET", "NOUN", "ADJ", "VERB", "ADV", "PUNCT"]);
        let tags = vec![Tag(0), Tag(1), Tag(2), Tag(3), Tag(4), Tag(5)];
        doc.add(&Key::DEFAULT, PosTags { tags, tagset, prob: None }, Provenance::by("test")).unwrap();

        let analyzer = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .given(Key::<Tokens>::DEFAULT)
            .given(Key::<PosTags>::DEFAULT)
            .add(DependencyAnnotator::<Tokens>::new(backend))
            .build()
            .unwrap();
        let mut docs = vec![doc];
        analyzer.run(&mut docs)[0].as_ref().unwrap();
        let deps = docs[0].get::<Dependencies<Tokens>>().unwrap();
        deps.check_forest().unwrap();
        assert_eq!(deps.len(), 6);
        assert_eq!(deps.head[3], None);
        assert_eq!(deps.relations.label(deps.relation[3]), "root");
        assert_eq!(deps.head[1], Some(Ix::new(3)));
        assert_eq!(deps.relations.label(deps.relation[1]), "nsubj");
    }
}
