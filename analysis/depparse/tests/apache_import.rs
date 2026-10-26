use turbo_analysis::{Ix, ViewId};
use turbo_analysis_layers::Words;
use turbo_depparse::{DependencySample, FeedforwardDependencyModel, FeedforwardDependencyParser};
use turbo_formats::conllu::{ConlluDependencySample, Tagset};
use turbo_unicode::WhitespaceMode;

#[test]
fn current_apache_trained_models_import_and_match_predictions() {
    for (bytes, expected) in [
        (include_bytes!("apache/apache.bin").as_slice(), include_str!("apache/apache.tsv")),
        (include_bytes!("apache/apache-refined.bin").as_slice(), include_str!("apache/apache-refined.tsv")),
    ] {
        let model = FeedforwardDependencyModel::from_bytes(bytes).unwrap();
        let mut actual = String::new();
        let mut sentence = 0;
        for input in [include_str!("data/ud-en.conllu"), include_str!("data/ud-es.conllu")] {
            for block in input.trim().split("\n\n") {
                let data =
                    ConlluDependencySample::read_sentence(block, Tagset::U, WhitespaceMode::Unicode, ViewId::SOURCE)
                        .unwrap();
                let sample = DependencySample::<Words>::from_conllu(data).unwrap();
                let words: Vec<_> = sample.tokens().iter().map(String::as_str).collect();
                let tags: Vec<_> = sample.tags().iter().map(String::as_str).collect();
                for beam in [1, 4] {
                    let parser = FeedforwardDependencyParser::new(model.clone(), beam).unwrap();
                    let graph = parser.parse::<Words>(&words, &tags).unwrap();
                    for i in 0..graph.len() {
                        let arc = graph.arc(Ix::new(i as u32)).unwrap();
                        let head = arc.head.map(|h| h.get() as i64).unwrap_or(-1);
                        actual.push_str(&format!("{beam}\t{sentence}\t{i}\t{head}\t{}\n", arc.relation));
                    }
                }
                sentence += 1;
            }
        }
        assert_eq!(actual, expected);
    }
}

#[test]
fn native_conversion_preserves_parameters_and_default_beam() {
    for (bytes, beam) in [
        (include_bytes!("apache/apache.bin").as_slice(), 1),
        (include_bytes!("apache/apache-refined.bin").as_slice(), 4),
    ] {
        let model = FeedforwardDependencyModel::from_bytes(bytes).unwrap();
        assert_eq!(model.default_beam_size(), beam);
        let zip = turbo_maxent::zip::Zip::new(bytes).unwrap();
        let raw = FeedforwardDependencyModel::from_bytes(&zip.entry("feedforward.network").unwrap()).unwrap();
        assert_eq!(raw.weights(), model.weights());
        assert_eq!(raw.default_beam_size(), 1);
        let converted = model.to_bytes().unwrap();
        let loaded = FeedforwardDependencyModel::from_bytes(&converted).unwrap();
        assert_eq!(loaded.default_beam_size(), beam);
        assert_eq!(model.weights(), loaded.weights());
        assert_eq!(model.vocabulary(), loaded.vocabulary());
        assert_eq!(model.transitions(), loaded.transitions());
        assert_eq!(converted, loaded.to_bytes().unwrap());
        if beam != 1 {
            assert!(loaded.to_java_bytes().is_err());
        }
    }
}

#[test]
fn original_native_v1_and_legacy_models_remain_loadable() {
    let legacy = include_bytes!("feedforward/data/plain.bin");
    let original = FeedforwardDependencyModel::from_bytes(include_bytes!("apache/native-v1.bin")).unwrap();
    assert_eq!(original.default_beam_size(), 1);
    assert_eq!(original.to_java_bytes().unwrap(), legacy);
    let upgraded = FeedforwardDependencyModel::from_bytes(&original.to_bytes().unwrap()).unwrap();
    assert_eq!(upgraded.to_java_bytes().unwrap(), legacy);
}

fn apache_network() -> Vec<u8> {
    turbo_maxent::zip::Zip::new(include_bytes!("apache/apache.bin").as_slice())
        .unwrap()
        .entry("feedforward.network")
        .unwrap()
        .into_owned()
}

#[test]
fn apache_network_rejects_truncation_version_lengths_and_nonfinite_weights() {
    let network = apache_network();
    for end in (0..network.len()).step_by(17).chain([network.len() - 1]) {
        assert!(FeedforwardDependencyModel::from_apache_bytes(&network[..end]).is_err(), "end={end}");
    }
    for (at, value) in [(4, 2u32), (8, u32::MAX), (12, 10_000_001), (network.len() - 4, 0x7fc00000)] {
        let mut broken = network.clone();
        broken[at..at + 4].copy_from_slice(&value.to_be_bytes());
        assert!(FeedforwardDependencyModel::from_apache_bytes(&broken).is_err(), "at={at}");
    }
    let mut trailing = network.clone();
    trailing.push(0);
    assert!(FeedforwardDependencyModel::from_apache_bytes(&trailing).is_err());
    let archive = include_bytes!("apache/apache.bin");
    for end in [0, 1, 2, archive.len() / 2, archive.len() - 1] {
        assert!(FeedforwardDependencyModel::from_bytes(&archive[..end]).is_err());
    }
}

#[test]
fn apache_package_rejects_wrong_component_invalid_beam_and_mixed_schemas() {
    use turbo_train::{Manifest, ModelPackage};
    for (component, beam, extra) in [
        ("TokenizerME", "1", None),
        ("FeedforwardDependencyParser", "0", None),
        ("FeedforwardDependencyParser", "-1", None),
        ("FeedforwardDependencyParser", "2147483648", None),
        ("FeedforwardDependencyParser", "NaN", None),
        ("FeedforwardDependencyParser", "1", Some("format_version")),
    ] {
        let mut manifest = Manifest::new(component, "und", 0);
        manifest.set("BeamSize", beam);
        let mut package = ModelPackage::new(manifest);
        package.add_entry("feedforward.network", apache_network());
        if let Some(name) = extra {
            package.add_entry(name, b"2\n".to_vec());
        }
        assert!(FeedforwardDependencyModel::from_bytes(&package.to_bytes()).is_err());
    }
}

#[test]
fn native_utf8_v2_keeps_long_vocabulary_keys_and_refuses_legacy_export() {
    let original = FeedforwardDependencyModel::from_bytes(include_bytes!("apache/apache.bin")).unwrap();
    let mut vocabulary = original.vocabulary().clone();
    let key = vocabulary.words.keys().find(|key| !key.starts_with('*')).unwrap().clone();
    let row = vocabulary.words.remove(&key).unwrap();
    vocabulary.words.insert("é".repeat(40_000), row);
    let model = FeedforwardDependencyModel::new(
        vocabulary,
        original.transitions().to_vec(),
        original.layout(),
        original.activation(),
        original.weights().clone(),
    )
    .unwrap();
    let restored = FeedforwardDependencyModel::from_bytes(&model.to_bytes().unwrap()).unwrap();
    assert_eq!(model.vocabulary(), restored.vocabulary());
    assert_eq!(model.weights(), restored.weights());
    assert!(model.to_java_bytes().is_err());
}

#[test]
fn native_reader_refuses_unknown_version_and_missing_beam() {
    use turbo_train::{Manifest, ModelPackage};
    let model = FeedforwardDependencyModel::from_bytes(include_bytes!("apache/apache.bin")).unwrap();
    let bytes = model.to_bytes().unwrap();
    let zip = turbo_maxent::zip::Zip::new(bytes.as_slice()).unwrap();
    for (version, config) in [
        (b"3\n".as_slice(), b"layout=OpenNlp14\nactivation=cube\nbeam_size=1\n".as_slice()),
        (b"2\n", b"layout=OpenNlp14\nactivation=cube\n"),
    ] {
        let mut package = ModelPackage::new(Manifest::new("FeedforwardDependencyParser", "und", 0));
        package.add_entry("format_version", version.to_vec());
        package.add_entry("config.properties", config.to_vec());
        for entry in ["vocab.bin", "weights.bin"] {
            package.add_entry(entry, zip.entry(entry).unwrap().into_owned());
        }
        assert!(FeedforwardDependencyModel::from_bytes(&package.to_bytes()).is_err());
    }
}

#[test]
fn non_ascii_and_non_bmp_keys_survive_each_distinct_wire_encoding() {
    let original = FeedforwardDependencyModel::from_bytes(include_bytes!("apache/apache.bin")).unwrap();
    let mut vocabulary = original.vocabulary().clone();
    let key = vocabulary.words.keys().find(|key| !key.starts_with('*')).unwrap().clone();
    let row = vocabulary.words.remove(&key).unwrap();
    vocabulary.words.insert("é水\u{1f980}".into(), row);
    let model = FeedforwardDependencyModel::new(
        vocabulary,
        original.transitions().to_vec(),
        original.layout(),
        original.activation(),
        original.weights().clone(),
    )
    .unwrap();
    let legacy = model.to_java_bytes().unwrap();
    let native = model.to_bytes().unwrap();
    let zip = turbo_maxent::zip::Zip::new(native.as_slice()).unwrap();
    let mut apache = b"FFDP\0\0\0\x01".to_vec();
    apache.extend_from_slice(&zip.entry("vocab.bin").unwrap());
    apache.extend_from_slice(&zip.entry("weights.bin").unwrap());
    // These are encoding boundary probes using trained weights, not new Java training receipts.
    for restored in [
        FeedforwardDependencyModel::from_java_bytes(&legacy).unwrap(),
        FeedforwardDependencyModel::from_apache_bytes(&apache).unwrap(),
        FeedforwardDependencyModel::from_bytes(&native).unwrap(),
    ] {
        assert_eq!(model.vocabulary(), restored.vocabulary());
        assert_eq!(model.weights(), restored.weights());
    }
    assert!(apache.windows("é水\u{1f980}".len()).any(|s| s == "é水\u{1f980}".as_bytes()));
    assert!(!legacy.windows("é水\u{1f980}".len()).any(|s| s == "é水\u{1f980}".as_bytes()));
}
