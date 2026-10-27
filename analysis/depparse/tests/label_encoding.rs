use turbo_analysis::Ix;
use turbo_analysis_layers::Tokens;
use turbo_depparse::*;
use turbo_maxent::zip::Zip;
use turbo_train::{Manifest, ModelPackage, TrainingParameters};

#[test]
fn label_contents_do_not_select_decoding_and_escaped_roundtrips_are_exact() {
    for label in ["root^literal", "root^>literal", r"root\^literal", r"root\\^>literal", "r<oot>", "根^>𝄞"] {
        for heads in [vec![None], vec![Some(Ix::new(2)), None, Some(Ix::new(1))]] {
            let labels =
                if heads.len() == 1 { vec![label.into()] } else { vec!["det".into(), label.into(), "obj".into()] };
            let graph = DependencyGraph::<Tokens>::new(heads, labels).unwrap();
            let plain = graph.deprojectivize().unwrap();
            let decoded = graph.projectivize().unwrap().deprojectivize().unwrap();
            assert_eq!(decoded.to_layer().head, graph.to_layer().head);
            for i in 0..graph.len() {
                assert_eq!(
                    plain.arc(Ix::new(i as u32)).unwrap().relation,
                    graph.arc(Ix::new(i as u32)).unwrap().relation
                );
                assert_eq!(
                    decoded.arc(Ix::new(i as u32)).unwrap().relation,
                    graph.arc(Ix::new(i as u32)).unwrap().relation
                );
            }
        }
    }
}

fn sample() -> DependencySample<Tokens> {
    DependencySample::new(
        vec!["Hello".into()],
        vec!["INTJ".into()],
        DependencyGraph::new(vec![None], vec![r"root\^>literal".into()]).unwrap(),
    )
    .unwrap()
}

#[test]
fn trained_modes_survive_reload_and_custom_predictions() {
    let samples = vec![sample(); 3];
    let mut p = TrainingParameters::new();
    p.put_str("Algorithm", "MAXENT");
    p.put_int("Cutoff", 0);
    p.put_int("Iterations", 10);
    let linear = DependencyModel::train("en", &samples, &p).unwrap();
    let linear = DependencyModel::open(linear.bytes().unwrap().to_vec()).unwrap();
    assert_eq!(linear.label_encoding(), LabelEncoding::EscapedV1);
    let predicted = DependencyParserME::new(linear).parse::<Tokens>(&["Hello"], &["INTJ"]).unwrap();
    assert_eq!(predicted.arc(Ix::new(0)).unwrap().relation, r"root\^>literal");
    let settings = FeedforwardSettings {
        epochs: 2,
        embedding_size: 2,
        hidden_size: 4,
        batch_size: 2,
        dropout: 0.0,
        ..Default::default()
    };
    let neural = FeedforwardDependencyTrainer::train(&samples, &settings).unwrap();
    let neural = FeedforwardDependencyModel::from_bytes(&neural.to_bytes().unwrap()).unwrap();
    assert_eq!(neural.label_encoding(), LabelEncoding::EscapedV1);
    assert!(neural.to_java_bytes().is_err());
    for beam in [1, 3] {
        let predicted = FeedforwardDependencyParser::new(neural.clone(), beam)
            .unwrap()
            .parse::<Tokens>(&["Hello"], &["INTJ"])
            .unwrap();
        assert_eq!(predicted.arc(Ix::new(0)).unwrap().relation, r"root\^>literal");
    }
}

fn repackage(bytes: &[u8], version: &str, config: &str) -> Vec<u8> {
    let zip = Zip::new(bytes).unwrap();
    let mut p = ModelPackage::new(Manifest::new("FeedforwardDependencyParser", "und", 0));
    p.add_entry("format_version", version.as_bytes().to_vec());
    p.add_entry("config.properties", config.as_bytes().to_vec());
    for name in ["vocab.bin", "weights.bin"] {
        p.add_entry(name, zip.entry(name).unwrap().into_owned());
    }
    p.to_bytes()
}

#[test]
fn genuine_old_native_and_java_readers_remain_plain_and_v3_refuses_unknown_modes() {
    let old = FeedforwardDependencyModel::from_bytes(include_bytes!("apache/native-v1.bin")).unwrap();
    let java = FeedforwardDependencyModel::from_bytes(include_bytes!("feedforward/data/plain.bin")).unwrap();
    for model in [old, java] {
        assert_eq!(model.label_encoding(), LabelEncoding::Plain);
        let bytes = model.to_bytes().unwrap();
        let zip = Zip::new(&bytes).unwrap();
        let config = String::from_utf8(zip.entry("config.properties").unwrap().into_owned()).unwrap();
        let v2_config = config.replace("label_encoding=plain\n", "");
        let v2 = FeedforwardDependencyModel::from_bytes(&repackage(&bytes, "2\n", &v2_config)).unwrap();
        assert_eq!(v2.label_encoding(), LabelEncoding::Plain);
        assert_eq!(v2.weights(), model.weights());
        assert_eq!(v2.transitions(), model.transitions());
        for config in [
            v2_config,
            config.replace("label_encoding=plain", "label_encoding=future"),
            format!("{config}label_encoding=plain\n"),
        ] {
            assert!(FeedforwardDependencyModel::from_bytes(&repackage(&bytes, "3\n", &config)).is_err());
        }
    }
}

#[test]
#[ignore = "requires both published Finnish dependency model files"]
fn published_files_use_hash_scoped_compatibility_but_repacked_files_do_not() {
    for (env, neural) in [("TURBO_DEPPARSE_PUBLISHED_MAXENT", false), ("TURBO_DEPPARSE_PUBLISHED_NEURAL", true)] {
        let bytes = std::fs::read(std::env::var(env).expect("published file required")).unwrap();
        let mode = |b: &[u8]| {
            if neural {
                FeedforwardDependencyModel::from_bytes(b).unwrap().label_encoding()
            } else {
                DependencyModel::open(b.to_vec()).unwrap().label_encoding()
            }
        };
        assert_eq!(mode(&bytes), LabelEncoding::LegacyPseudo);
        if neural {
            let model = FeedforwardDependencyModel::from_bytes(&bytes).unwrap();
            let converted = model.to_bytes().unwrap();
            let reloaded = FeedforwardDependencyModel::from_bytes(&converted).unwrap();
            assert_eq!(reloaded.label_encoding(), LabelEncoding::LegacyPseudo);
            assert_eq!(reloaded.to_bytes().unwrap(), converted);
            assert_eq!(reloaded.identity().unwrap(), model.identity().unwrap());
            let before = FeedforwardDependencyParser::new(model, 1)
                .unwrap()
                .parse::<Tokens>(&["I", "see", "birds"], &["PRON", "VERB", "NOUN"])
                .unwrap();
            let after = FeedforwardDependencyParser::new(reloaded, 1)
                .unwrap()
                .parse::<Tokens>(&["I", "see", "birds"], &["PRON", "VERB", "NOUN"])
                .unwrap();
            assert_eq!(before.to_layer().head, after.to_layer().head);
            for i in 0..3 {
                assert_eq!(before.arc(Ix::new(i)).unwrap().relation, after.arc(Ix::new(i)).unwrap().relation);
            }
        }
        let zip = Zip::new(&bytes).unwrap();
        let component = if neural { "FeedforwardDependencyParser" } else { "DependencyParserME" };
        let mut package = ModelPackage::new(Manifest::new(component, "fi", 123));
        for name in zip.names() {
            let name = std::str::from_utf8(name).unwrap();
            if name != "manifest.properties" {
                package.add_entry(name, zip.entry(name).unwrap().into_owned());
            }
        }
        let changed = package.to_bytes();
        assert_ne!(turbo_maxent::sha256(&bytes), turbo_maxent::sha256(&changed));
        assert_eq!(mode(&changed), LabelEncoding::Plain);
    }
}

#[test]
fn legacy_java_custom_labels_are_not_decoded_by_production_parsers() {
    let original = FeedforwardDependencyModel::from_java_bytes(include_bytes!("feedforward/data/plain.bin")).unwrap();
    for prefix in ["custom^>literal", r"custom\^literal"] {
        let transitions = original
            .transitions()
            .iter()
            .map(|t| match t {
                Transition::Shift => Transition::Shift,
                Transition::LeftArc(r) => Transition::LeftArc(format!("{prefix}:{r}")),
                Transition::RightArc(r) => Transition::RightArc(format!("{prefix}:{r}")),
            })
            .collect();
        let model = FeedforwardDependencyModel::new(
            original.vocabulary().clone(),
            transitions,
            original.layout(),
            original.activation(),
            original.weights().clone(),
        )
        .unwrap();
        let bytes = model.to_java_bytes().unwrap();
        let loaded = FeedforwardDependencyModel::from_bytes(&bytes).unwrap();
        for beam in [1, 3] {
            let graph = FeedforwardDependencyParser::new(loaded.clone(), beam)
                .unwrap()
                .parse::<Tokens>(&["Hello"], &["INTJ"])
                .unwrap();
            assert!(graph.arc(Ix::new(0)).unwrap().relation.starts_with(prefix));
        }
    }
}

#[test]
fn old_linear_model_is_plain_and_unknown_manifest_encoding_is_refused() {
    let bytes = include_bytes!("apache/linear.bin");
    let model = DependencyModel::open(bytes.to_vec()).unwrap();
    assert_eq!(model.label_encoding(), LabelEncoding::Plain);
    let zip = Zip::new(bytes).unwrap();
    let mut manifest = Manifest::new("DependencyParserME", "en", 0);
    manifest.set("Turbo-Dependency-Label-Encoding", "future");
    let mut package = ModelPackage::new(manifest);
    package.add_entry("depparse.model", zip.entry("depparse.model").unwrap().into_owned());
    assert!(DependencyModel::open(package.to_bytes()).is_err());
}
