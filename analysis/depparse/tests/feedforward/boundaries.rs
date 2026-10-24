use super::support::*;
use turbo_analysis::{Ix, Layer};
use turbo_analysis_layers::Tokens;
use turbo_depparse::*;

#[test]
fn expanded_children_are_directional_ranked_and_include_grandchildren() {
    // 0 -> 1 -> 3; 2 -> 3; 4 -> 5 -> 3; 6 -> 3. Stop before attaching root 3.
    let gold = sample(
        &["a", "b", "c", "d", "e", "f", "g"],
        &["X"; 7],
        &[Some(1), Some(3), Some(3), None, Some(5), Some(3), Some(3)],
        &["dep"; 7],
    );
    let steps = ArcStandardOracle::transitions(gold.graph()).unwrap();
    let mut state = ArcStandardState::<Tokens>::new(7).unwrap();
    for t in &steps[..steps.len() - 1] {
        state.apply(t).unwrap();
    }
    let f = FeedforwardContext::extract(
        &state,
        &["a", "b", "c", "d", "e", "f", "g"],
        &["X"; 7],
        FeatureLayout::ChenManning18,
    )
    .unwrap();
    assert_eq!(
        f.words.iter().map(|s| s.as_deref()).collect::<Vec<_>>(),
        vec![
            Some("d"),
            Some("*ROOT*"),
            None,
            None,
            None,
            None,
            Some("b"),
            Some("g"),
            Some("c"),
            Some("f"),
            None,
            None,
            None,
            None,
            Some("a"),
            None,
            None,
            None
        ]
    );
    assert_eq!(f.labels.iter().filter(|x| x.is_some()).count(), 5);
}
#[test]
fn models_validate_vocabulary_transitions_dimensions_and_finite_parameters() {
    let m = minimal(FeatureLayout::ChenManning18);
    let build = |v, t, w| FeedforwardDependencyModel::new(v, t, m.layout(), m.activation(), w);
    let mut v = m.vocabulary().clone();
    v.pos.insert("bad".into(), 0);
    assert!(build(v, m.transitions().to_vec(), m.weights().clone()).is_err());
    let mut v = m.vocabulary().clone();
    v.words.remove("*UNK*");
    assert!(build(v, m.transitions().to_vec(), m.weights().clone()).is_err());
    let mut w = m.weights().clone();
    w.hidden_weights.pop();
    assert!(build(m.vocabulary().clone(), m.transitions().to_vec(), w).is_err());
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for column in 0..5 {
            let mut w = m.weights().clone();
            [&mut w.embeddings, &mut w.hidden_weights, &mut w.hidden_bias, &mut w.output_weights, &mut w.output_bias]
                [column][0] = value;
            assert!(build(m.vocabulary().clone(), m.transitions().to_vec(), w).is_err());
        }
    }
    for t in [
        vec![Transition::Shift],
        vec![Transition::RightArc("r".into())],
        vec![Transition::Shift, Transition::Shift, Transition::RightArc("r".into())],
        vec![Transition::Shift, Transition::LeftArc("".into()), Transition::RightArc("r".into())],
    ] {
        assert!(build(m.vocabulary().clone(), t, m.weights().clone()).is_err());
    }
    assert!(m.score(&[0]).is_err());
    assert!(m.score(&vec![usize::MAX; m.layout().feature_count()]).is_err());
    assert!(m.to_java_bytes().is_err());
}
#[test]
fn raw_model_rejects_every_truncation_and_absurd_counts_without_allocating() {
    let bytes = include_bytes!("data/plain.bin");
    for end in 0..bytes.len() {
        assert!(FeedforwardDependencyModel::from_java_bytes(&bytes[..end]).is_err(), "offset {end}");
    }
    let mut bytes = bytes.to_vec();
    bytes.push(0);
    assert!(FeedforwardDependencyModel::from_java_bytes(&bytes).is_err());
    let mut bytes = include_bytes!("data/plain.bin").to_vec();
    for n in [i32::MAX, -1, 2_000_001] {
        bytes[13..17].copy_from_slice(&n.to_be_bytes());
        assert!(FeedforwardDependencyModel::from_java_bytes(&bytes).is_err());
    }
}
#[test]
fn zip_roundtrip_and_invalid_configuration() {
    use turbo_maxent::zip::Zip;
    use turbo_train::package::{Manifest, ModelPackage};
    for layout in [FeatureLayout::OpenNlp14, FeatureLayout::ChenManning18] {
        let model = minimal(layout);
        let bytes = model.to_bytes().unwrap();
        let zip = Zip::new(&bytes).unwrap();
        let decoded = FeedforwardDependencyModel::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.to_bytes().unwrap(), bytes);
        for config in ["layout=ChenManning18\nactivation=cube\nactivation=cube\n", "layout=bad\nactivation=cube\n", ""]
        {
            let mut package = ModelPackage::new(Manifest::new("FeedforwardDependencyParser", "und", 0));
            for name in ["format_version", "config.properties", "vocab.bin", "weights.bin"] {
                package.add_entry(
                    name,
                    if name == "config.properties" {
                        config.as_bytes().to_vec()
                    } else {
                        zip.entry(name).unwrap().into_owned()
                    },
                );
            }
            assert!(FeedforwardDependencyModel::from_bytes(&package.to_bytes()).is_err());
        }
    }
}
#[test]
fn learned_expanded_relu_and_cube_models_produce_checked_trees() {
    for activation in [Activation::Cube, Activation::Relu] {
        let settings = FeedforwardSettings {
            layout: FeatureLayout::ChenManning18,
            activation,
            epochs: 60,
            embedding_size: 8,
            hidden_size: 16,
            learning_rate: 0.05,
            ..settings(false)
        };
        let model = FeedforwardDependencyTrainer::train(&samples(), &settings).unwrap();
        for beam in [1, 4] {
            let p = FeedforwardDependencyParser::new(model.clone(), beam).unwrap();
            for s in samples() {
                let words = s.tokens().iter().map(String::as_str).collect::<Vec<_>>();
                let tags = s.tags().iter().map(String::as_str).collect::<Vec<_>>();
                let g: DependencyGraph = p.parse(&words, &tags).unwrap();
                g.check_tree().unwrap();
                for i in 0..g.len() {
                    let a = g.arc(Ix::new(i as u32)).unwrap();
                    let b = s.graph().arc(Ix::new(i as u32)).unwrap();
                    assert_eq!((a.head, a.relation), (b.head, b.relation));
                }
                g.to_layer().check().unwrap();
            }
        }
    }
}
#[test]
fn invalid_settings_and_parser_inputs_fail() {
    for value in [f64::NAN, f64::INFINITY, -1.0] {
        assert!(FeedforwardSettings { learning_rate: value, ..settings(false) }.check().is_err());
        assert!(FeedforwardSettings { dropout: value, ..settings(false) }.check().is_err());
        assert!(FeedforwardSettings { l2: value, ..settings(false) }.check().is_err());
    }
    assert!(FeedforwardSettings { embedding_size: 4096, hidden_size: 65536, ..settings(false) }.check().is_err());
    assert!(FeedforwardDependencyTrainer::train::<Tokens>(&[], &settings(false)).is_err());
    assert!(FeedforwardDependencyParser::new(minimal(FeatureLayout::OpenNlp14), 0).is_err());
    let p = FeedforwardDependencyParser::new(minimal(FeatureLayout::OpenNlp14), 4).unwrap();
    assert!(p.parse::<Tokens>(&[], &[]).is_err());
    assert!(p.parse::<Tokens>(&["a"], &[]).is_err());
}

#[test]
fn a_model_read_from_bytes_caches_its_pos_label_and_special_slots() {
    let java = FeedforwardDependencyModel::from_java_bytes(include_bytes!("data/plain.bin")).unwrap();
    let (n, h) = (java.layout().positions(), java.weights().hidden_bias.len());
    let v = java.vocabulary();
    let expected = (n * 3 + n * v.pos.len() + (n - 6) * v.labels.len()) * h;
    assert_eq!(java.cached_slot_values(), expected);
    let rebuilt = FeedforwardDependencyModel::from_bytes(&java.to_bytes().unwrap()).unwrap();
    assert_eq!(rebuilt.cached_slot_values(), expected);
}
