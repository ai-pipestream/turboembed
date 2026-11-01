//! The parser's loss under extreme but finite scores and under
//! non-finite input: a large common offset leaves the loss and its
//! gradients as without it, and a non-finite feature or result is refused
//! with the caller's gradients untouched. The probes are codex's review
//! of #1588.

use turbo::manifest::{Biaffine, Decode, HeadKind, HeadSpec, Scheme, WordPooling};
use turbo_trainer::parser::Work;
use turbo_trainer::{Parser, rng::Rng};

fn spec() -> HeadSpec {
    HeadSpec {
        name: "review".into(),
        layer: "turboembed:test".into(),
        kind: HeadKind::BiaffineParser,
        labels: vec!["a".into(), "b".into()],
        scheme: Scheme::Tags,
        word_pooling: WordPooling::First,
        adapter: None,
        decode: Decode::Mst,
        biaffine: Some(Biaffine { arc: 1, rel: 1 }),
        pos_input: None,
        hash_embed: None,
        layer_mix: None,
        span: None,
        relation: None,
        license_class: None,
        artifact: "weights".into(),
        tensor_prefix: "review".into(),
    }
}

/// A parser of one-wide views with every weight zero.
fn zeroed() -> Parser<f64> {
    let mut p = Parser::<f64>::new(&spec(), 1, &mut Rng::new(1)).unwrap();
    for i in 0..p.params().len() {
        p.param_mut(i).fill(0.0);
    }
    p
}

fn at(p: &Parser<f64>, name: &str) -> usize {
    p.names().iter().position(|n| n == name).unwrap()
}

#[test]
fn equal_large_relation_scores_keep_log_two_loss() {
    for big in [0.0, 1e6, 1e20] {
        let mut p = zeroed();
        let i = at(&p, "rel.weight");
        // [labels, rel + 1, rel + 1]: each label's bias-bias weight.
        p.param_mut(i)[3] = big;
        p.param_mut(i)[7] = big;
        let mut g = p.zero_grads();
        let loss = p.loss(&[0.0], &[0], &[0], &mut g, &mut Work::default()).unwrap();
        assert!((loss - 2f64.ln()).abs() < 1e-12, "offset {big}: loss {loss}");
        assert_eq!([g[i][3], g[i][7]], [-0.5, 0.5], "offset {big}");
    }
}

#[test]
fn equal_large_arc_scores_keep_their_normaliser() {
    // Two words: every node's arc views are 1, so each arc score is
    // U[0] + U[1] whatever the head, and each word has two heads to pick
    // from. With the relations all zero too, the loss is 4 ln 2.
    for big in [0.0, 1e6, 1e20] {
        let mut p = zeroed();
        for b in ["arc_dep.bias", "arc_head.bias"] {
            let i = at(&p, b);
            p.param_mut(i)[0] = 1.0;
        }
        let u = at(&p, "arc.weight");
        p.param_mut(u)[0] = big;
        let mut g = p.zero_grads();
        let loss = p.loss(&[0.0, 0.0], &[0, 1], &[0, 1], &mut g, &mut Work::default()).unwrap();
        assert!((loss - 4.0 * 2f64.ln()).abs() < 1e-9, "offset {big}: loss {loss}");
        assert!(g.iter().flatten().all(|v| v.is_finite()));
    }
}

#[test]
fn nonfinite_parser_features_refuse_without_mutating_gradients() {
    let p = zeroed();
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut g = p.zero_grads();
        g.iter_mut().for_each(|t| t.fill(7.0));
        let before = g.clone();
        let loss = p.loss(&[bad], &[0], &[0], &mut g, &mut Work::default());
        assert!(loss.is_err(), "{bad} returned {loss:?}");
        assert_eq!(g, before);
    }
}

#[test]
fn a_loss_that_overflows_is_refused_without_mutating_gradients() {
    // Finite weights whose scores overflow: the relation scores are
    // infinite, so the loss and its gradients are not finite.
    let mut p = zeroed();
    let i = at(&p, "rel.weight");
    p.param_mut(i)[3] = f64::MAX;
    p.param_mut(i)[7] = -f64::MAX;
    let r = at(&p, "rel_dep.bias");
    p.param_mut(r)[0] = 1e10;
    let mut g = p.zero_grads();
    g.iter_mut().for_each(|t| t.fill(7.0));
    let before = g.clone();
    assert!(p.loss(&[0.0], &[0], &[1], &mut g, &mut Work::default()).is_err());
    assert_eq!(g, before);
}

#[test]
fn nonfinite_hash_tables_are_refused_at_load() {
    use turbo::head::TokenTagger;
    use turbo::head::tagger::tensors;
    use turbo::manifest::HashEmbed;
    let s = HeadSpec {
        kind: HeadKind::TokenTagger,
        decode: Decode::Argmax,
        biaffine: None,
        hash_embed: Some(HashEmbed { dim: 1, rows: [1; 4] }),
        span: None,
        license_class: None,
        ..spec()
    };
    for table in ["hash.norm", "hash.prefix", "hash.suffix", "hash.shape"] {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let values: Vec<(String, Vec<f32>)> = tensors(&s, 1)
                .into_iter()
                .map(|(n, shape)| {
                    let mut data = vec![0.0; shape.iter().product::<u64>() as usize];
                    if n == table {
                        data[0] = bad;
                    }
                    (n, data)
                })
                .collect();
            let get = |n: &str| values.iter().find(|(name, _)| name == n).map(|(_, v)| v.clone());
            let e = TokenTagger::from_tensors(&s, 1, &get).expect_err("a non-finite table is refused");
            assert!(e.message.contains(table) && e.message.contains("not finite"), "{table} {bad}: {e:?}");
        }
    }
}
