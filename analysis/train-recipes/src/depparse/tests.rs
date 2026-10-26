use super::*;
use crate::card::{CARD_FILE, GATE_FILE};
use turbo_analysis::{Analyzer, Document, Key, Layer, Provenance, Span, Spans, Tag, TagSet, ViewId};
use turbo_analysis_layers::{Dependencies, PosTags, Sentences, Tokens};
use turbo_depparse::{DependencyAnnotator, DependencyBackend};

fn mini_treebank_text() -> TreebankText {
    let mut train = String::new();
    let mut test = String::new();
    // Include both projective and non-projective sentences in training and test.
    let projective = "\
# text = El gato negro come pescado fresco hoy.
1\tEl\tel\tDET\t_\t_\t2\tdet\t_\t_
2\tgato\tgato\tNOUN\t_\t_\t4\tnsubj\t_\t_
3\tnegro\tnegro\tADJ\t_\t_\t2\tamod\t_\t_
4\tcome\tcomer\tVERB\t_\t_\t0\troot\t_\t_
5\tpescado\tpescado\tNOUN\t_\t_\t4\tobj\t_\t_
6\tfresco\tfresco\tADJ\t_\t_\t5\tamod\t_\t_
7\thoy\thoy\tADV\t_\t_\t4\tadvmod\t_\tSpaceAfter=No
8\t.\t.\tPUNCT\t_\t_\t4\tpunct\t_\t_

";
    // Non-projective sentence: token 1 (Un) depends on token 4 (libro) across root token 2 (leí).
    let nonprojective = "\
# text = Un leí ayer libro interesante.
1\tUn\tun\tDET\t_\t_\t4\tdet\t_\t_
2\tleí\tleer\tVERB\t_\t_\t0\troot\t_\t_
3\tayer\tayer\tADV\t_\t_\t2\tadvmod\t_\t_
4\tlibro\tlibro\tNOUN\t_\t_\t2\tobj\t_\t_
5\tinteresante\tinteresante\tADJ\t_\t_\t4\tamod\t_\tSpaceAfter=No
6\t.\t.\tPUNCT\t_\t_\t2\tpunct\t_\t_

";
    for i in 0..40 {
        if i % 4 == 0 {
            train.push_str(nonprojective);
            test.push_str(nonprojective);
        } else {
            train.push_str(projective);
            test.push_str(projective);
        }
    }
    TreebankText { train, test }
}

#[test]
fn recipes_cover_all_five_treebanks_for_both_parsers() {
    for r in DEPPARSE_RECIPES {
        let t = r.treebank;
        let expected = format!("turbo-{}-ud-{}-2.16-depparse-{}-1", t.language, t.name, r.kind.name());
        assert_eq!(r.model_id, expected);
        let parsed: turbo_zoo::ArtifactId = r.model_id.parse().unwrap();
        assert_eq!(parsed.as_str(), r.model_id);
        assert_eq!(DepparseRecipe::trains(r.model_id), Some(r));
    }
    assert_eq!(DepparseRecipe::trains("turbo-en-ud-ewt-2.16-depparse-maxent-1"), None);
}

#[test]
fn train_gate_card_and_annotator_round_trip_for_both_parsers() {
    let text = mini_treebank_text();
    let dir = std::env::temp_dir().join(format!("turbo-depparse-recipe-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    for recipe in [DEPPARSE_RECIPES[0], DEPPARSE_RECIPES[1]] {
        recipe.check_data(&text).unwrap();
        let outcome = run_depparse(&recipe, &text).unwrap();
        assert_eq!(outcome.trained.samples, 40);
        assert_eq!(outcome.trained.projective_samples, 30);
        assert!(outcome.gate.result.pass, "LAS gate: {}", outcome.gate);
        assert!(outcome.uas_gate.result.pass, "UAS gate: {}", outcome.uas_gate);
        write_depparse_model_dir(&dir, &outcome).unwrap();

        let model_dir = dir.join(recipe.model_id);
        let read_card = DepparseModelCard::read(&model_dir.join(CARD_FILE)).unwrap();
        assert_eq!(read_card, outcome.card);
        let read_las = ModelGateRecord::read(&model_dir.join(GATE_FILE)).unwrap();
        assert_eq!(read_las, outcome.gate);
        let read_uas = ModelGateRecord::read(&model_dir.join(UAS_GATE_FILE)).unwrap();
        assert_eq!(read_uas, outcome.uas_gate);

        let url = format!(
            "https://github.com/ai-pipestream/turboembed/releases/download/models-ud-depparse-2.16-1/{}.bin",
            recipe.model_id
        );
        let artifact = read_card.artifact(&url).unwrap();
        assert_eq!(artifact.component, Some(turbo_zoo::Component::DependencyParser));

        // Verify DependencyAnnotator runs on the trained model bytes and preserves source offsets.
        let backend = match recipe.kind {
            DepparseKind::Maxent => DependencyBackend::Linear(DependencyParserME::new(
                DependencyModel::open(outcome.trained.bytes.clone()).unwrap(),
            )),
            DepparseKind::Feedforward => {
                let model = FeedforwardDependencyModel::from_bytes(&outcome.trained.bytes).unwrap();
                let beam = model.default_beam_size();
                DependencyBackend::Feedforward(FeedforwardDependencyParser::new(model, beam).unwrap())
            }
        };
        let raw = "El gato negro come pescado.";
        let mut doc = Document::new(raw);
        let mut sent_spans = Spans::new(ViewId::SOURCE);
        sent_spans.push(Span::new(0, raw.len() as u32));
        doc.add(&Key::DEFAULT, Sentences { spans: sent_spans }, Provenance::by("test")).unwrap();
        let mut tok_spans = Spans::new(ViewId::SOURCE);
        for (start, end) in [(0, 2), (3, 7), (8, 13), (14, 18), (19, 26), (26, 27)] {
            tok_spans.push(Span::new(start, end));
        }
        doc.add(&Key::DEFAULT, Tokens { spans: tok_spans, sentence: None }, Provenance::by("test")).unwrap();
        let pos = PosTags {
            tags: vec![Tag(0), Tag(1), Tag(2), Tag(3), Tag(1), Tag(4)],
            tagset: TagSet::new("upos", &["DET", "NOUN", "ADJ", "VERB", "PUNCT"]),
            prob: None,
        };
        doc.add(&Key::DEFAULT, pos, Provenance::by("test")).unwrap();
        let analyzer = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .given(Key::<Tokens>::DEFAULT)
            .given(Key::<PosTags>::DEFAULT)
            .add(DependencyAnnotator::<Tokens>::new(backend))
            .build()
            .unwrap();
        let mut docs = vec![doc];
        analyzer.run(&mut docs)[0].as_ref().unwrap();
        let deps = docs[0].get::<Dependencies>().unwrap();
        assert_eq!(deps.len(), 6);
    }

    // Portuguese Porttinari card is refused for publication because text rights are not settled.
    let pt_outcome = run_depparse(&DEPPARSE_RECIPES[4], &text).unwrap();
    let pt_err = pt_outcome
        .card
        .artifact("https://github.com/ai-pipestream/turboembed/releases/download/models-ud-depparse-2.16-1/turbo-pt-ud-porttinari-2.16-depparse-maxent-1.bin")
        .unwrap_err();
    assert!(matches!(pt_err, Error::NotShippable { reason: "the rights to its treebank's text are not settled", .. }));

    let _ = std::fs::remove_dir_all(&dir);
}
