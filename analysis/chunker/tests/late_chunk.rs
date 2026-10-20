//! Late pooling over hand-computed fixtures: `aa bb, cc dd.`
//! with one 2-dimensional vector per piece, checked component by
//! component, plus the window policies over a two-window seam
//! and the contract refusals (wrong model, uncovered pieces,
//! non-finite vectors).

use turbo_analysis::export::{export, import};
use turbo_analysis::{
    Annotator, Document, Ix, IxRange, Key, Layer, ModelRef, Provenance, Span, Spans, Tag, TagSet, ViewId,
};
use turbo_analysis_layers::{Entities, Sentences, Subwords, Vectors};
use turbo_chunker::export::register_codecs;
use turbo_chunker::{LateChunkAnnotator, Pooling, TokenVectorWindows, WindowPolicy};

const TEXT: &str = "aa bb, cc dd.";

fn model() -> ModelRef {
    ModelRef { id: "toy".into(), revision: "1".into(), sha256: "0".into() }
}

fn spans<L>(pairs: &[(u32, u32)]) -> Spans<L> {
    let mut s = Spans::new(ViewId::SOURCE);
    for &(a, b) in pairs {
        s.push(Span::new(a, b));
    }
    s
}

struct Fixture {
    doc: Document,
    subwords: Subwords,
    vectors: Vectors<Subwords>,
}

fn fixture() -> Fixture {
    let doc = Document::new(TEXT);
    let pieces = spans(&[(0, 2), (3, 5), (5, 6), (7, 9), (10, 12), (12, 13)]);
    let n = pieces.len();
    let subwords = Subwords { spans: pieces, ids: vec![0; n], model: model(), token: None };
    let vectors =
        Vectors::new(2, vec![1.0, 0.0, 3.0, 2.0, 100.0, 100.0, 0.0, 4.0, 2.0, 2.0, 50.0, 50.0], Vec::new(), model());
    Fixture { doc, subwords, vectors }
}

fn windows(pairs: &[(u32, u32)]) -> TokenVectorWindows {
    TokenVectorWindows { spans: spans(pairs) }
}

fn targets() -> Sentences {
    Sentences { spans: spans(&[(0, 6), (6, 7), (7, 12), (12, 13)]) }
}

fn pool<L: Layer>(
    f: &Fixture,
    annotator: &LateChunkAnnotator<L>,
    wins: &TokenVectorWindows,
    target: &L,
) -> Result<Vectors<L>, turbo_analysis::Error> {
    let view = f.doc.get::<turbo_analysis::View>().unwrap();
    Ok(annotator.annotate((view, &f.vectors, &f.subwords, wins, target))?.0)
}

fn err_of<L: Layer>(r: Result<Vectors<L>, turbo_analysis::Error>) -> String {
    match r {
        Err(e) => e.to_string(),
        Ok(_) => panic!("expected an error"),
    }
}

fn close(row: &[f32], expected: &[f64]) {
    assert_eq!(row.len(), expected.len());
    for (a, e) in row.iter().zip(expected) {
        // Rows are stored as f32, so the tolerance is 1e-6
        // relative to the magnitude being checked.
        assert!((f64::from(*a) - e).abs() <= 1e-6 * (1.0 + e.abs()), "row {row:?} vs {expected:?}");
    }
}

#[test]
fn mean_max_and_masked_mean_match_hand_values() {
    let f = fixture();
    let wins = windows(&[(0, 13)]);
    let target = targets();

    let mean = pool(&f, &LateChunkAnnotator::new(Pooling::Mean, WindowPolicy::Reject), &wins, &target).unwrap();
    close(mean.row(Ix::new(0)), &[104.0 / 3.0, 102.0 / 3.0]);
    close(mean.row(Ix::new(2)), &[1.0, 3.0]);
    close(mean.row(Ix::new(3)), &[50.0, 50.0]);
    assert!(mean.is_empty(Ix::new(1)), "the space between pieces selects nothing");
    assert_eq!(mean.len(), 4);

    let max = pool(&f, &LateChunkAnnotator::new(Pooling::Max, WindowPolicy::Reject), &wins, &target).unwrap();
    close(max.row(Ix::new(0)), &[100.0, 100.0]);
    close(max.row(Ix::new(2)), &[2.0, 4.0]);

    let masked = pool(&f, &LateChunkAnnotator::new(Pooling::MaskedMean, WindowPolicy::Reject), &wins, &target).unwrap();
    close(masked.row(Ix::new(0)), &[2.0, 1.0]);
    assert!(masked.is_empty(Ix::new(3)), "an all-punctuation item is empty under the mask");

    let stopped =
        pool(&f, &LateChunkAnnotator::new(Pooling::MaskedMean, WindowPolicy::Reject).stopwords(["BB"]), &wins, &target)
            .unwrap();
    close(stopped.row(Ix::new(0)), &[1.0, 0.0]);

    let unit = pool(&f, &LateChunkAnnotator::new(Pooling::Mean, WindowPolicy::Reject).normalize(true), &wins, &target)
        .unwrap();
    close(unit.row(Ix::new(2)), &[1.0 / 10f64.sqrt(), 3.0 / 10f64.sqrt()]);
    assert!(unit.is_empty(Ix::new(1)));
}

#[test]
fn discontinuous_targets_pool_only_their_fragments() {
    let f = fixture();
    let mut entity_spans = Spans::new(ViewId::SOURCE);
    entity_spans.push_fragments(&[Span::new(0, 2), Span::new(10, 12)]);
    let entities = Entities {
        spans: entity_spans,
        tokens: vec![IxRange::new(0, 1)],
        kind: vec![Tag(0)],
        kinds: TagSet::new("entity", &["thing"]),
        score: vec![1.0],
    };
    let out = pool(&f, &LateChunkAnnotator::<Entities>::default(), &windows(&[(0, 13)]), &entities).unwrap();
    close(out.row(Ix::new(0)), &[1.5, 1.0]);
}

#[test]
fn empty_subword_rows_never_contribute() {
    let mut f = fixture();
    f.vectors = Vectors::new(
        2,
        vec![1.0, 0.0, 0.0, 0.0, 100.0, 100.0, 0.0, 4.0, 2.0, 2.0, 50.0, 50.0],
        vec![Ix::new(1)],
        model(),
    );
    let out = pool(&f, &LateChunkAnnotator::<Sentences>::default(), &windows(&[(0, 13)]), &targets()).unwrap();
    close(out.row(Ix::new(0)), &[50.5, 50.0]);
}

#[test]
fn window_policies_over_a_seam() {
    let f = fixture();
    let wins = windows(&[(0, 6), (6, 13)]);
    let whole = Sentences { spans: spans(&[(0, 13)]) };

    let rejected = pool(&f, &LateChunkAnnotator::new(Pooling::Mean, WindowPolicy::Reject), &wins, &whole);
    let detail = match rejected {
        Err(turbo_analysis::Error::Annotator { detail, .. }) => detail,
        _ => panic!("expected a straddle refusal"),
    };
    assert!(detail.contains("item 0") && detail.contains("0..6") && detail.contains("6..13"), "{detail}");

    let allow = pool(&f, &LateChunkAnnotator::new(Pooling::Mean, WindowPolicy::Allow), &wins, &whole).unwrap();
    close(allow.row(Ix::new(0)), &[26.0, 158.0 / 6.0]);
    let split = pool(&f, &LateChunkAnnotator::new(Pooling::Mean, WindowPolicy::SplitPool), &wins, &whole).unwrap();
    close(split.row(Ix::new(0)), &[26.0, 158.0 / 6.0]);

    let allow_max = pool(&f, &LateChunkAnnotator::new(Pooling::Max, WindowPolicy::Allow), &wins, &whole).unwrap();
    close(allow_max.row(Ix::new(0)), &[100.0, 100.0]);
    let split_max = pool(&f, &LateChunkAnnotator::new(Pooling::Max, WindowPolicy::SplitPool), &wins, &whole).unwrap();
    close(split_max.row(Ix::new(0)), &[75.0, 75.0]);

    // Items inside one window pool the same under every policy.
    let inside = pool(&f, &LateChunkAnnotator::<Sentences>::default(), &wins, &targets()).unwrap();
    close(inside.row(Ix::new(0)), &[104.0 / 3.0, 102.0 / 3.0]);
    close(inside.row(Ix::new(2)), &[1.0, 3.0]);
}

#[test]
fn contract_refusals() {
    let f = fixture();
    let annotator = LateChunkAnnotator::<Sentences>::default();

    // Windows that do not cover every piece.
    let err = err_of(pool(&f, &annotator, &windows(&[(0, 6)]), &targets()));
    assert!(err.contains("subword 3"), "{err}");

    // Vectors of a different model than the pieces.
    let mut other = fixture();
    other.vectors = Vectors::new(
        2,
        vec![0.0; 12],
        Vec::new(),
        ModelRef { id: "table".into(), revision: "9".into(), sha256: "0".into() },
    );
    let err = err_of(pool(&other, &annotator, &windows(&[(0, 13)]), &targets()));
    assert!(err.contains("table") && err.contains("toy"), "{err}");

    // A non-finite vector value.
    let mut bad = fixture();
    bad.vectors.data[3] = f32::NAN;
    assert!(pool(&bad, &annotator, &windows(&[(0, 13)]), &targets()).is_err());

    // Overlapping windows are not a valid layer at all.
    assert!(windows(&[(0, 8), (6, 13)]).check().is_err());
    let err = err_of(pool(&f, &annotator, &windows(&[(0, 8), (6, 13)]), &targets()));
    assert!(!err.is_empty());

    // Settings that change the output change the config hash.
    let base = LateChunkAnnotator::<Sentences>::default().config_hash();
    assert_ne!(base, LateChunkAnnotator::<Sentences>::new(Pooling::Max, WindowPolicy::Reject).config_hash());
    assert_ne!(base, LateChunkAnnotator::<Sentences>::new(Pooling::Mean, WindowPolicy::Allow).config_hash());
    assert_ne!(base, LateChunkAnnotator::<Sentences>::default().normalize(true).config_hash());
    assert_ne!(base, LateChunkAnnotator::<Sentences>::default().stopwords(["bb"]).config_hash());
}

#[test]
fn windows_cross_the_graph_boundary() {
    let mut doc = Document::new(TEXT);
    let by = Provenance::by("test");
    doc.add(&Key::DEFAULT, targets(), by.clone()).unwrap();
    doc.add(&Key::DEFAULT, windows(&[(0, 6), (6, 13)]), by).unwrap();
    let mut registry = turbo_analysis_layers::export::codecs();
    register_codecs(&mut registry);
    let graph = export(&doc, &registry).unwrap();
    let back = import(&graph, &registry).unwrap();
    let wins = back.get::<TokenVectorWindows>().unwrap();
    assert_eq!(wins.len(), 2);
    assert_eq!(export(&back, &registry).unwrap().to_bytes(), graph.to_bytes());
}

/// The retrieval acceptance fixture, with synthetic vectors:
/// ten documents whose second sentence means what it means
/// only through a first-sentence antecedent. The contextual
/// piece vectors give the pronoun the antecedent's direction
/// (what an encoder's attention does); the naive arm pools
/// static per-word vectors per chunk, where the pronoun is a
/// generic direction. Late pooling must bring the second
/// chunk closer to an antecedent-naming query on every
/// document. This proves the pooling carries context that is
/// in the token vectors; it makes no claim about any model.
#[test]
fn late_vectors_beat_naive_on_antecedent_queries() {
    use turbo_analysis::Analyzer;
    const DIM: usize = 16;
    let unit = |d: usize, scale: f32| {
        let mut v = vec![0.0f32; DIM];
        v[d] = scale;
        v
    };
    let cosine = |a: &[f32], b: &[f32]| {
        let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for (x, y) in a.iter().zip(b) {
            dot += f64::from(*x) * f64::from(*y);
            na += f64::from(*x) * f64::from(*x);
            nb += f64::from(*y) * f64::from(*y);
        }
        dot / (na.sqrt() * nb.sqrt())
    };
    let names = ["Acme", "Bolt", "Cobalt", "Delta", "Ember", "Flint", "Garnet", "Harbor", "Ion", "Jasper"];
    let mut wins = 0;
    let mut margin = 0.0;
    for (i, name) in names.iter().enumerate() {
        let text = format!("{name} was founded in 1999. Its board is elected annually.");
        let words: Vec<&str> = text.split(' ').collect();
        let mut pieces = Spans::new(ViewId::SOURCE);
        let mut at = 0u32;
        for w in &words {
            pieces.push(Span::new(at, at + w.len() as u32));
            at += w.len() as u32 + 1;
        }
        let mid = text.find(". ").unwrap() as u32 + 1;
        let sentences = Sentences { spans: spans(&[(0, mid), (mid + 1, text.len() as u32)]) };
        let static_vec = |w: &str| -> Vec<f32> {
            match w.trim_end_matches('.') {
                w if Some(w) == names.get(i).copied() => unit(i, 1.0),
                "Its" => unit(14, 1.0),
                "board" => unit(10, 1.0),
                "elected" => unit(11, 1.0),
                "annually" => unit(12, 1.0),
                "founded" => unit(13, 1.0),
                _ => unit(15, 0.2),
            }
        };
        let mut contextual = Vec::new();
        for w in &words {
            if *w == "Its" {
                contextual.extend(unit(i, 1.0));
            } else {
                contextual.extend(static_vec(w));
            }
        }
        let mut doc = Document::new(text.as_str());
        let by = Provenance::by("test");
        let n = pieces.len();
        doc.add(&Key::DEFAULT, Subwords { spans: pieces, ids: vec![0; n], model: model(), token: None }, by.clone())
            .unwrap();
        doc.add(&Key::DEFAULT, Vectors::<Subwords>::new(DIM as u32, contextual, Vec::new(), model()), by.clone())
            .unwrap();
        doc.add(&Key::DEFAULT, windows(&[(0, text.len() as u32)]), by.clone()).unwrap();
        doc.add(&Key::DEFAULT, sentences, by).unwrap();
        let analyzer = Analyzer::builder()
            .given(Key::<turbo_analysis::View>::DEFAULT)
            .given(Key::<Vectors<Subwords>>::DEFAULT)
            .given(Key::<Subwords>::DEFAULT)
            .given(Key::<TokenVectorWindows>::DEFAULT)
            .given(Key::<Sentences>::DEFAULT)
            .add(LateChunkAnnotator::<Sentences>::default())
            .build()
            .unwrap();
        let mut docs = vec![doc];
        analyzer.run(&mut docs)[0].as_ref().unwrap();
        let late = docs[0].get::<Vectors<Sentences>>().unwrap().row(Ix::new(1)).to_vec();

        let second: Vec<Vec<f32>> = words[5..].iter().map(|w| static_vec(w)).collect();
        let mut naive = vec![0.0f32; DIM];
        for v in &second {
            for (a, b) in naive.iter_mut().zip(v) {
                *a += b / second.len() as f32;
            }
        }
        let mut query = unit(i, 1.0);
        query[10] = 1.0;
        query[11] = 1.0;
        let (late_cos, naive_cos) = (cosine(&late, &query), cosine(&naive, &query));
        margin += late_cos - naive_cos;
        if late_cos > naive_cos {
            wins += 1;
        }
    }
    assert_eq!(wins, 10, "late pooling wins on every antecedent document");
    assert!(margin / 10.0 > 0.1, "mean cosine margin {} is a real gap, not noise", margin / 10.0);
}
