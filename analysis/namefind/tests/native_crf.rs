//! CRF document annotators over the typed numerical entity decoder:
//! entities, candidate alternatives, calibration sidecars, and the input
//! checks both annotators share. Scores are test arithmetic, not a model.

use turbo_analysis::{
    Annotator, Document, Error, Fnv1a, Ix, Key, Layer, ModelRef, Provenance, Span, Spans, SplitMode, Tag, TagSet, View,
    ViewId,
};
use turbo_analysis_layers::{Alternatives, Entities, Sentences, Tokens};
use turbo_calibration::{Calibration, DataIdentity, Policy, Transform, digest};
use turbo_namefind::calibrated::{CalibratedCrfNameAnnotator, crf_identity_with_candidates};
use turbo_namefind::{CrfNameAnnotator, CrfSentenceScores};
use turbo_sequence::crf::{EntityDecoder, EntityEncoding, EntityLabel};

const TEXT: &str = "Zoë met 李明";

/// Builds the schema and returns it with the single-token state of each
/// kind (the begin state in BIO, the unit state in BILOU).
fn decoder(encoding: EntityEncoding, kinds: &[&str]) -> (EntityDecoder, Vec<usize>) {
    let mut labels = vec![EntityLabel::Outside];
    let mut single = Vec::new();
    for (i, _) in kinds.iter().enumerate() {
        let begin = labels.len();
        let tag = Tag(i as u16);
        labels.push(EntityLabel::Begin(tag));
        labels.push(EntityLabel::Inside(tag));
        if encoding == EntityEncoding::Bilou {
            labels.extend([EntityLabel::Last(tag), EntityLabel::Unit(tag)]);
            single.push(begin + 3);
        } else {
            single.push(begin);
        }
    }
    (EntityDecoder::new(encoding, TagSet::new("native-crf", kinds), labels).unwrap(), single)
}

/// Uniform test scores: `highs` are the emission peaks, everything else a
/// low constant, with neutral start, transition and end parameters. The
/// fixed-size vectors stay schema-wide even with no tokens, so an empty
/// sentence still validates.
fn scores(states: usize, tokens: usize, highs: &[(usize, usize)]) -> CrfSentenceScores {
    let mut emissions = vec![vec![-5.0; states]; tokens];
    for &(row, state) in highs {
        emissions[row][state] = 4.0;
    }
    CrfSentenceScores {
        emissions,
        transitions: vec![0.0; states * states],
        start: vec![0.0; states],
        end: vec![0.0; states],
    }
}

fn scores_from(emissions: Vec<Vec<f64>>) -> CrfSentenceScores {
    let states = emissions.first().map_or(0, Vec::len);
    CrfSentenceScores {
        emissions,
        transitions: vec![0.0; states * states],
        start: vec![0.0; states],
        end: vec![0.0; states],
    }
}

fn scores_hash(s: &CrfSentenceScores) -> u64 {
    let bytes: Vec<u8> = s.emissions.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
    Fnv1a::new().bytes(&bytes).finish()
}

fn provenance() -> ModelRef {
    ModelRef { id: "test:native-crf".into(), revision: "1".into(), sha256: digest(b"native crf test parameters") }
}

fn document(text: &str, token_spans: &[(u32, u32)], sentence: Option<(u32, u32)>) -> Document {
    let mut doc = Document::new(text.to_owned());
    let mut sentences = Spans::new(ViewId::SOURCE);
    if let Some((a, b)) = sentence {
        sentences.push(Span::new(a, b));
    }
    doc.add(&Key::DEFAULT, Sentences { spans: sentences }, Provenance::by("test")).unwrap();
    let mut spans = Spans::new(ViewId::SOURCE);
    for &(a, b) in token_spans {
        spans.push(Span::new(a, b));
    }
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("test")).unwrap();
    doc
}

/// The three-token document: "Zoë", "met", "李明" in one sentence.
fn document3() -> Document {
    document(TEXT, &[(0, 4), (5, 8), (9, 15)], Some((0, 15)))
}

/// The two-token document: "Zoë", "met" in one sentence.
fn document2() -> Document {
    document(TEXT, &[(0, 4), (5, 8)], Some((0, 8)))
}

fn run(annotator: &CrfNameAnnotator, doc: &Document) -> (Entities, Alternatives<Entities>) {
    annotator
        .annotate((doc.get::<View>().unwrap(), doc.get::<Sentences>().unwrap(), doc.get::<Tokens>().unwrap()))
        .unwrap()
}

fn run_err(annotator: &CrfNameAnnotator, doc: &Document) -> Error {
    annotator
        .annotate((doc.get::<View>().unwrap(), doc.get::<Sentences>().unwrap(), doc.get::<Tokens>().unwrap()))
        .err()
        .unwrap()
}

fn alternatives_table(a: &Alternatives<Entities>) -> Vec<(u32, u32, u32, u32, bool, f64)> {
    let score = a.score.as_ref().unwrap();
    (0..a.spans.len())
        .map(|i| {
            let span = a.spans.get(Ix::new(i as u32));
            (span.start, span.end, a.position[i], a.position_length[i], a.mode[i].contains(SplitMode::Best), score[i])
        })
        .collect()
}

#[test]
fn top_path_decodes_to_entities_and_unit_backbone_for_bio_and_bilou() {
    for encoding in [EntityEncoding::Bio, EntityEncoding::Bilou] {
        let (decoder, single) = decoder(encoding, &["person", "place"]);
        let fixture = scores(decoder.labels().len(), 3, &[(0, single[0]), (2, single[1])]);
        let hash = scores_hash(&fixture);
        let annotator = CrfNameAnnotator::new(decoder, 1, move |_| Ok(fixture.clone()), hash, provenance()).unwrap();
        assert_eq!(annotator.model(), Some(provenance()));
        let (entities, alternatives) = run(&annotator, &document3());
        assert_eq!(entities.spans.columns(), (&[0, 9][..], &[4, 15][..]));
        assert_eq!(entities.tokens.iter().map(|r| (r.start.get(), r.end.get())).collect::<Vec<_>>(), [(0, 1), (2, 3)]);
        assert_eq!(entities.kind, [Tag(0), Tag(1)]);
        assert_eq!(entities.kinds.label(entities.kind[0]), "person");
        assert_eq!(entities.kinds.label(entities.kind[1]), "place");
        // Marginals over all legal paths, not just the best one; both paths
        // with one entity and paths with both count into each marginal.
        assert!(entities.score.iter().all(|p| (0.0..=1.0).contains(p)));
        // Each token keeps its unit edge, covered or not; the rank-0
        // entities ride on top with the BEST mode set.
        let table = alternatives_table(&alternatives);
        assert_eq!(table.len(), 5);
        assert_eq!(table[0], (0, 4, 0, 1, false, 0.0));
        assert_eq!((table[1].0, table[1].1, table[1].2, table[1].3), (0, 4, 0, 1));
        assert!(table[1].4);
        assert!(table[1].5 > 0.0);
        assert_eq!(table[2], (5, 8, 1, 1, true, 0.0));
        assert_eq!(table[3], (9, 15, 2, 1, false, 0.0));
        assert_eq!((table[4].0, table[4].1, table[4].2, table[4].3), (9, 15, 2, 1));
        assert!(table[4].4);
        assert!(table[4].5 > 0.0);
    }
}

#[test]
fn multi_token_entities_close_at_the_last_token_for_bio_and_bilou() {
    for encoding in [EntityEncoding::Bio, EntityEncoding::Bilou] {
        let (decoder, _) = decoder(encoding, &["person", "place"]);
        let begin = 1;
        let last = if encoding == EntityEncoding::Bio { begin + 1 } else { begin + 2 };
        let fixture = scores(decoder.labels().len(), 2, &[(0, begin), (1, last)]);
        let hash = scores_hash(&fixture);
        let annotator = CrfNameAnnotator::new(decoder, 1, move |_| Ok(fixture.clone()), hash, provenance()).unwrap();
        let (entities, alternatives) = run(&annotator, &document2());
        assert_eq!(entities.spans.columns(), (&[0][..], &[8][..]));
        assert_eq!(entities.tokens.iter().map(|r| (r.start.get(), r.end.get())).collect::<Vec<_>>(), [(0, 2)]);
        assert_eq!(entities.kind, [Tag(0)]);
        // Both tokens are inside the entity: unit edges are not BEST, the
        // two-token edge is.
        let table = alternatives_table(&alternatives);
        assert_eq!(table.len(), 3);
        assert_eq!(table[0], (0, 4, 0, 1, false, 0.0));
        assert_eq!((table[1].0, table[1].1, table[1].2, table[1].3), (0, 8, 0, 2));
        assert!(table[1].4);
        assert!((0.0..=1.0).contains(&table[1].5));
        assert_eq!(table[2], (5, 8, 1, 1, false, 0.0));
    }
}

#[test]
fn second_path_spans_join_the_alternatives_without_the_best_mode() {
    for encoding in [EntityEncoding::Bio, EntityEncoding::Bilou] {
        let (decoder, _) = decoder(encoding, &["person"]);
        let emissions = if encoding == EntityEncoding::Bio {
            // [B, I] wins; [B, B] is the runner-up with a (1, 2) entity.
            vec![vec![0.0, 1.0, -99.0], vec![0.0, 0.5, 0.9]]
        } else {
            // [B, L] wins; [O, U] is the runner-up with a (1, 2) entity.
            vec![vec![0.0, 1.0, -99.0, -99.0, -99.0], vec![0.0, -99.0, -99.0, 0.9, 0.5]]
        };
        assert_eq!(emissions[0].len(), decoder.labels().len());
        let fixture = scores_from(emissions);
        let hash = scores_hash(&fixture);
        let annotator = CrfNameAnnotator::new(decoder, 2, move |_| Ok(fixture.clone()), hash, provenance()).unwrap();
        let (entities, alternatives) = run(&annotator, &document2());
        assert_eq!(entities.tokens.iter().map(|r| (r.start.get(), r.end.get())).collect::<Vec<_>>(), [(0, 2)]);
        let table = alternatives_table(&alternatives);
        // The best path covers both tokens, so the unit edges are not BEST;
        // exactly one entity edge carries BEST, and the runner-up spans
        // join it without BEST. BIO's runner-up also splits off a (0, 1)
        // candidate; BILOU's runner-up is the (1, 2) unit place.
        let best = table.iter().filter(|e| e.4 && e.5 > 0.0).collect::<Vec<_>>();
        assert_eq!(best.len(), 1);
        assert_eq!((best[0].0, best[0].1, best[0].2, best[0].3), (0, 8, 0, 2));
        let runners_up = table.iter().filter(|e| !e.4 && e.5 > 0.0).count();
        assert_eq!(runners_up, if encoding == EntityEncoding::Bio { 2 } else { 1 });
        assert!(table.iter().any(|e| (e.0, e.1, e.2, e.3) == (5, 8, 1, 1) && !e.4 && e.5 > 0.0));
    }
}

#[test]
fn calibrated_annotator_aligns_sidecar_rows_and_validates_identity() {
    let domain = "native-crf-test";
    let policy = Policy { coverage: 0.9, max_set_size: 3, min_confidence: 0.0 };
    let build = |domain: &str, paths: usize| {
        let (decoder, single) = decoder(EntityEncoding::Bio, &["person", "place"]);
        let fixture = scores(decoder.labels().len(), 3, &[(0, single[0]), (2, single[1])]);
        let hash = scores_hash(&fixture);
        let identity = crf_identity_with_candidates(&provenance(), domain, &decoder, hash, paths).unwrap();
        let calibration = Calibration::new(
            identity,
            Transform::Temperature(2.0),
            DataIdentity { split: "fit".into(), sha256: digest(b"native crf fit") },
            DataIdentity { split: "conformal".into(), sha256: digest(b"native crf conformal") },
            vec![0.1; 20],
        )
        .unwrap();
        (decoder, calibration, fixture, hash)
    };
    let (decoder, calibration, fixture, hash) = build(domain, 1);
    let annotator =
        CalibratedCrfNameAnnotator::new(decoder, calibration, domain, policy, move |_| Ok(fixture.clone()), hash)
            .unwrap();
    assert_eq!(annotator.model(), Some(provenance()));
    let doc = document3();
    let (entities, side) = annotator
        .annotate((doc.get::<View>().unwrap(), doc.get::<Sentences>().unwrap(), doc.get::<Tokens>().unwrap()))
        .unwrap();
    assert_eq!(entities.len(), 2);
    assert_eq!(entities.spans.columns(), (&[0, 9][..], &[4, 15][..]));
    assert_eq!(side.len(), 2);
    for i in 0..2 {
        let row = side.row(i);
        let column = 1 + entities.kind[i].0 as usize;
        assert!((row.raw[column] - entities.score[i]).abs() < 1e-12);
        let denominator: f64 = row.raw.iter().map(|p| p.sqrt()).sum();
        for (&actual, raw) in row.calibrated.iter().zip(row.raw) {
            assert!((actual - raw.sqrt() / denominator).abs() < 1e-12);
        }
    }
    // A calibration fitted for another domain or another candidate path
    // count does not bind to this annotator.
    let (decoder, calibration, fixture, hash) = build("another-domain", 1);
    let mismatched =
        CalibratedCrfNameAnnotator::new(decoder, calibration, domain, policy, move |_| Ok(fixture.clone()), hash);
    assert!(mismatched.is_err());
    let (decoder, calibration, fixture, hash) = build(domain, 1);
    let mismatched = CalibratedCrfNameAnnotator::with_candidates(
        decoder,
        calibration,
        domain,
        policy,
        move |_| Ok(fixture.clone()),
        hash,
        2,
    );
    assert!(mismatched.is_err());
}

#[test]
fn calibrated_candidates_cover_unique_top_k_spans() {
    for encoding in [EntityEncoding::Bio, EntityEncoding::Bilou] {
        let (decoder, _) = decoder(encoding, &["person"]);
        let emissions = if encoding == EntityEncoding::Bio {
            vec![vec![0.0, 1.0, -99.0], vec![0.0, 0.5, 0.9]]
        } else {
            vec![vec![0.0, 1.0, -99.0, -99.0, -99.0], vec![0.0, -99.0, -99.0, 0.9, 0.5]]
        };
        let fixture = scores_from(emissions);
        let hash = scores_hash(&fixture);
        let identity = crf_identity_with_candidates(&provenance(), "native-crf-test", &decoder, hash, 2).unwrap();
        let calibration = Calibration::new(
            identity,
            Transform::Temperature(2.0),
            DataIdentity { split: "fit".into(), sha256: digest(b"native crf fit") },
            DataIdentity { split: "conformal".into(), sha256: digest(b"native crf conformal") },
            vec![0.1; 20],
        )
        .unwrap();
        let annotator = CalibratedCrfNameAnnotator::with_candidates(
            decoder,
            calibration,
            "native-crf-test",
            Policy { coverage: 0.9, max_set_size: 3, min_confidence: 0.0 },
            move |_| Ok(fixture.clone()),
            hash,
            2,
        )
        .unwrap();
        let doc = document2();
        let (entities, side) = annotator
            .annotate((doc.get::<View>().unwrap(), doc.get::<Sentences>().unwrap(), doc.get::<Tokens>().unwrap()))
            .unwrap();
        let spans = entities.tokens.iter().map(|r| (r.start.get(), r.end.get())).collect::<Vec<_>>();
        assert!(spans.contains(&(0, 2)), "{spans:?}");
        assert!(spans.contains(&(1, 2)), "{spans:?}");
        assert_eq!(entities.len(), side.len());
        for i in 0..entities.len() {
            assert!((side.row(i).raw[1] - entities.score[i]).abs() < 1e-12);
        }
    }
}

#[test]
fn empty_documents_write_empty_layers() {
    for encoding in [EntityEncoding::Bio, EntityEncoding::Bilou] {
        let (decoder, _) = decoder(encoding, &["person"]);
        let fixture = scores(decoder.labels().len(), 0, &[]);
        let hash = scores_hash(&fixture);
        let annotator = CrfNameAnnotator::new(decoder, 1, move |_| Ok(fixture.clone()), hash, provenance()).unwrap();
        let (entities, alternatives) = run(&annotator, &document(TEXT, &[], None));
        assert_eq!(entities.len(), 0);
        assert_eq!(alternatives.spans.len(), 0);
        let (entities, alternatives) = run(&annotator, &document(TEXT, &[], Some((0, 15))));
        assert_eq!(entities.len(), 0);
        assert_eq!(alternatives.spans.len(), 0);
    }
}

#[test]
fn invalid_scores_spans_and_parameters_are_refused() {
    // Emission rows wider than the schema.
    let (dec, _) = decoder(EntityEncoding::Bio, &["person"]);
    let states = dec.labels().len();
    let bad_width = scores_from(vec![vec![0.0; states + 1]]);
    let annotator = CrfNameAnnotator::new(dec, 1, move |_| Ok(bad_width.clone()), 0, provenance()).unwrap();
    let err = run_err(&annotator, &document2());
    assert!(matches!(err, Error::Annotator { annotator: CrfNameAnnotator::ID, .. }));
    let Error::Annotator { detail, .. } = err else { unreachable!() };
    assert!(detail.contains("score dimensions"), "{detail}");
    // A score source that fails fails the annotation with its message.
    let (dec, _) = decoder(EntityEncoding::Bio, &["person"]);
    let annotator = CrfNameAnnotator::new(dec, 1, |_| Err("no scores".to_string()), 0, provenance()).unwrap();
    let err = run_err(&annotator, &document2());
    let Error::Annotator { detail, .. } = err else { unreachable!() };
    assert_eq!(detail, "no scores");
    // Tokens in another view.
    let (dec, _) = decoder(EntityEncoding::Bio, &["person"]);
    let fixture = scores(dec.labels().len(), 2, &[]);
    let annotator = CrfNameAnnotator::new(dec, 1, move |_| Ok(fixture.clone()), 0, provenance()).unwrap();
    let doc = document2();
    let mut spans = Spans::new(ViewId(99));
    spans.push(Span::new(0, 4));
    spans.push(Span::new(5, 8));
    let foreign = Tokens { spans, sentence: None };
    let err =
        annotator.annotate((doc.get::<View>().unwrap(), doc.get::<Sentences>().unwrap(), &foreign)).err().unwrap();
    let Error::Annotator { detail, .. } = err else { unreachable!() };
    assert_eq!(detail, "input view mismatch");
    // Tokens no sentence covers.
    let err = run_err(&annotator, &document(TEXT, &[(5, 8)], Some((0, 4))));
    let Error::Annotator { detail, .. } = err else { unreachable!() };
    assert_eq!(detail, "tokens outside sentences");
    // Candidate path count bounds.
    for k in [0, 4097] {
        let (dec, _) = decoder(EntityEncoding::Bio, &["person"]);
        let rejected = CrfNameAnnotator::new(dec, k, |_| Err("unused".to_string()), 0, provenance());
        assert!(rejected.is_err());
    }
}
