//! Public storage, transport and annotator regressions using production
//! sentence detection, tokenization and normalization. Parse/mention
//! fixtures are hand-annotated; these are not model accuracy scores.
mod quotation_support;
use quotation_support::*;
use turbo_analysis::export::{DocumentGraph, export, import};
use turbo_analysis::{Analyzer, Annotator, Bound, Groups, Ix, Key, Layer, Provenance, Span, Spans, View};
use turbo_analysis_layers::{Quotations, QuoteKind, Tokens};
use turbo_coref::{Mentions, QuotationAnnotator, QuotationConfig, codecs};
use turbo_normalize::{NormalizeStep, UnicodeForm};

fn valid(dependencies: bool) -> turbo_analysis::Document {
    let mut d = prepared(TEXT);
    assert_eq!(spans(&d), ["Zoë", "said", ",", "“", "Café", "grew", ".", "”"]);
    add_mentions(&mut d, &[(0, 1)]);
    if dependencies {
        add_dependencies(
            &mut d,
            &[
                (Some(1), "nsubj"),
                (None, "root"),
                (Some(1), "punct"),
                (Some(5), "punct"),
                (Some(5), "nsubj"),
                (Some(1), "parataxis"),
                (Some(5), "punct"),
                (Some(5), "punct"),
            ],
        );
    }
    run(&mut d);
    d
}
fn metadata_start(body: &[u8]) -> usize {
    let get = |i| u32::from_le_bytes(body[i..i + 4].try_into().unwrap()) as usize;
    let mut at = 4 + get(0);
    let n = get(at);
    at += 4;
    for _ in 0..n {
        let count = get(at);
        at += 4 + 8 * count;
    }
    at
}
#[test]
fn real_surface_dependency_and_chain_paths_roundtrip() {
    for deps in [false, true] {
        let d = valid(deps);
        let q = quotes(&d).get(0).unwrap();
        assert_eq!(q.kind, QuoteKind::Direct);
        assert_eq!(q.speaker_chain, Some(0));
        assert_eq!(&TEXT[q.content.range()], "Café grew .");
        assert_eq!(&TEXT[q.speaker.unwrap().range()], "Zoë");
        assert_eq!(&TEXT[q.cue_verb.unwrap().range()], "said");
        assert_eq!(q.confidence, if deps { 0.9 } else { 0.75 });
        let graph = export(&d, &codecs()).unwrap();
        let restored = import(&DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &codecs()).unwrap();
        assert_eq!(export(&restored, &codecs()).unwrap(), graph);
        assert_eq!(d.provenance(Key::<Quotations>::DEFAULT.id()), restored.provenance(Key::<Quotations>::DEFAULT.id()));
    }
}
#[test]
fn public_insertion_rejects_invalid_secondary_spans() {
    let source = valid(false);
    let q = quotes(&source).clone();
    let mut accepted = Vec::new();
    for field in 0..2 {
        for case in 0..3 {
            let mut d = prepared(TEXT);
            let mut bad = q.clone();
            let s = match case {
                0 => Span { start: 3, end: 4 },
                1 => Span { start: 0, end: u32::MAX },
                _ => Span { start: 5, end: 1 },
            };
            if field == 0 {
                bad.speaker[0] = Some(s);
            } else {
                bad.cue_verb[0] = Some(s);
            }
            if d.add(&Key::DEFAULT, bad, Provenance::by("review:untrusted-input")).is_ok() {
                let graph = export(&d, &codecs()).unwrap();
                assert!(import(&DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &codecs()).is_err());
                accepted.push((field, case));
            }
        }
    }
    assert!(
        accepted.is_empty(),
        "insertion accepts invalid secondary fields that its own graph import rejects: {accepted:?}"
    );
}
#[test]
fn graph_import_rejects_invalid_secondary_fields_and_tags() {
    let d = valid(false);
    let good = export(&d, &codecs()).unwrap();
    for field in 0..2 {
        for case in 0..3 {
            let mut bad = good.clone();
            let body = bad.layers.iter_mut().find(|l| l.layer == Quotations::ID).unwrap().body.as_mut().unwrap();
            let at = metadata_start(body) + 1 + 9 * field;
            let s = match case {
                0 => (3u32, 4u32),
                1 => (0, u32::MAX),
                _ => (5, 1),
            };
            body[at..at + 4].copy_from_slice(&s.0.to_le_bytes());
            body[at + 4..at + 8].copy_from_slice(&s.1.to_le_bytes());
            assert!(import(&DocumentGraph::from_bytes(&bad.to_bytes().unwrap()).unwrap(), &codecs()).is_err());
        }
    }
    for case in 0..4 {
        let mut bad = good.clone();
        let body = bad.layers.iter_mut().find(|l| l.layer == Quotations::ID).unwrap().body.as_mut().unwrap();
        let m = metadata_start(body);
        match case {
            0 => body[m] = 2,
            1 => body[m + 23] = 255,
            2 => {
                let n = body.len();
                body[n - 4..].copy_from_slice(&f32::NAN.to_bits().to_le_bytes());
            }
            _ => body.push(0),
        }
        assert!(
            import(&DocumentGraph::from_bytes(&bad.to_bytes().unwrap()).unwrap(), &codecs()).is_err(),
            "malformed tag/confidence {case}"
        );
    }
}
#[test]
fn raw_chain_limitation_and_typed_chain_bounds_are_distinct() {
    let mut d = prepared(TEXT);
    let mut q = QuotationAnnotator::new().quotations(&d).unwrap();
    q.speaker_chain[0] = Some(u32::MAX);
    d.add(&Key::DEFAULT, q, Provenance::by("review:documented-raw-chain")).unwrap();
    let graph = export(&d, &codecs()).unwrap();
    let restored = import(&DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &codecs()).unwrap();
    assert_eq!(quotes(&restored).speaker_chain[0], Some(u32::MAX), "raw speaker chains are explicitly unchecked");
    let mut d = prepared(TEXT);
    add_mentions(&mut d, &[(0, 1)]);
    assert!(
        d.add(
            &Key::named("invalid"),
            Groups::<Mentions>::new(vec![vec![Ix::new(99)]]),
            Provenance::by("review:invalid-chain-member")
        )
        .is_err()
    );
    let mut graph = export(&valid(false), &codecs()).unwrap();
    let layer = graph.layers.iter_mut().find(|l| l.layer == "turbo:groups" || l.layer.contains("groups<")).unwrap();
    let body = layer.body.as_mut().unwrap();
    let n = body.len();
    body[n - 4..].copy_from_slice(&99u32.to_le_bytes());
    assert!(import(&DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &codecs()).is_err());
}
#[test]
fn unbounded_cue_distance_does_not_panic() {
    let mut panics = Vec::new();
    for text in [TEXT, "“ Café grew . ” said Zoë"] {
        let d = prepared(text);
        let expected = QuotationAnnotator::new().quotations(&d).unwrap();
        for max_cue_distance in [usize::MAX, usize::MAX - 1] {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                QuotationAnnotator::new()
                    .with_config(QuotationConfig { max_cue_distance, ..Default::default() })
                    .quotations(&d)
            }));
            match result {
                Ok(result) => assert_eq!(result.unwrap(), expected),
                Err(_) => panics.push((text, max_cue_distance)),
            }
        }
    }
    assert!(panics.is_empty(), "unbounded cue windows overflow: {panics:?}");
}
#[test]
fn wrong_view_mentions_are_rejected() {
    let mut d = valid(false);
    let a =
        Analyzer::builder().step(Bound::new(NormalizeStep::new(UnicodeForm::nfc())).write(0, "nfc")).build().unwrap();
    assert_eq!(a.run(std::slice::from_mut(&mut d))[0], Ok(()));
    let nfc = d.get_key(&Key::<View>::named("nfc")).unwrap();
    let original = d.get::<Mentions>().unwrap();
    let mut wrong = original.clone();
    wrong.spans = Spans::new(nfc.id());
    wrong.spans.push(original.spans.get(Ix::new(0)));
    let result = QuotationAnnotator::new().annotate((
        view(&d),
        d.get::<Tokens>().unwrap(),
        None,
        Some(&wrong),
        d.get::<Groups<Mentions>>(),
    ));
    let direct_rejected = result.is_err();
    d.add(&Key::<Mentions>::named("wrong-view"), wrong, Provenance::by("review:nfc-mentions")).unwrap();
    let mut provenance = Provenance::by("review:nfc-chains");
    provenance.reads.push(Key::<Mentions>::named("wrong-view").id());
    d.add(
        &Key::<Groups<Mentions>>::named("wrong-view"),
        Groups::of(Key::<Mentions>::named("wrong-view"), vec![vec![Ix::new(0)]]),
        provenance,
    )
    .unwrap();
    let a = Analyzer::builder()
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Mentions>::named("wrong-view"))
        .given(Key::<Groups<Mentions>>::named("wrong-view"))
        .step(
            Bound::new(QuotationAnnotator::new())
                .read(3, "wrong-view")
                .read(4, "wrong-view")
                .write(0, "wrong-view-result"),
        )
        .build()
        .unwrap();
    let result = a.run(std::slice::from_mut(&mut d));
    let stored = d.get_key(&Key::<Quotations>::named("wrong-view-result")).map(|q| q.speaker_chain.clone());
    assert!(
        direct_rejected && result[0].is_err(),
        "wrong-view mentions are silently dropped: direct_rejected={direct_rejected}, pipeline_result={result:?}, stored_chains={stored:?}"
    );
}
