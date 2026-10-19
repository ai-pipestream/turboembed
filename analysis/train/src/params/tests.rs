use std::cmp::Ordering;

use super::*;

#[test]
fn keys_are_case_insensitive_and_keep_their_first_spelling() {
    let mut p = TrainingParameters::new();
    p.put("Cutoff", 5);
    p.put("CUTOFF", 7);
    assert_eq!(p.len(), 1);
    assert_eq!(p.get("cutoff"), Some(&Value::Int(7)));
    assert_eq!(p.iter().next().unwrap().0, "Cutoff");
    p.put_if_absent("cutOFF", 9);
    assert_eq!(p.get_int("Cutoff").unwrap(), Some(7));
    p.put_if_absent("Iterations", 3);
    assert_eq!(p.get_int("iterations").unwrap(), Some(3));
}

#[test]
fn entries_iterate_in_the_case_insensitive_order() {
    let mut p = TrainingParameters::new();
    for k in ["b", "A", "_x", "C", "a.b", "ab"] {
        p.put(k, k);
    }
    // Letters compare by their lower case forms, so "_" (0x5F) sorts
    // before every letter, and a prefix before the longer key.
    let keys: Vec<&str> = p.iter().map(|(k, _)| k).collect();
    assert_eq!(keys, ["_x", "A", "a.b", "ab", "b", "C"]);
    assert_eq!(case_insensitive_compare("straße", "STRASSE"), Ordering::Greater);
    assert_eq!(case_insensitive_compare("ǅ", "ǆ"), Ordering::Equal);
    assert_eq!(case_insensitive_compare("abc", "ab"), Ordering::Greater);
}

#[test]
fn getters_cast_then_parse_as_java_does() {
    let mut p = TrainingParameters::new();
    p.put("i", 4);
    p.put("d", 0.5);
    p.put("b", true);
    p.put("s", "12");
    p.put("t", "true");
    p.put("w", "x");
    assert_eq!(p.get_int("i").unwrap(), Some(4));
    assert_eq!(p.get_int("s").unwrap(), Some(12));
    assert_eq!(p.get_double("d").unwrap(), Some(0.5));
    assert_eq!(p.get_double("s").unwrap(), Some(12.0));
    assert_eq!(p.get_bool("b").unwrap(), Some(true));
    assert_eq!(p.get_bool("t").unwrap(), Some(true));
    assert_eq!(p.get_bool("w").unwrap(), Some(false));
    assert_eq!(p.get_str("s").unwrap(), Some("12"));
    assert_eq!(p.get_str("missing").unwrap(), None);
    assert_eq!(p.get_int("missing").unwrap(), None);
    assert_eq!(p.get_str("i"), Err(Error::ParameterType { key: "i".into(), expected: "String", found: "Integer" }));
    assert_eq!(p.get_int("d"), Err(Error::ParameterType { key: "d".into(), expected: "Integer", found: "Double" }));
    assert_eq!(p.get_double("i"), Err(Error::ParameterType { key: "i".into(), expected: "Double", found: "Integer" }));
    assert_eq!(p.get_bool("i"), Err(Error::ParameterType { key: "i".into(), expected: "Boolean", found: "Integer" }));
    assert_eq!(p.get_int("w"), Err(Error::ParameterFormat { key: "w".into(), expected: "Integer", value: "x".into() }));
    assert_eq!(
        p.get_double("w"),
        Err(Error::ParameterFormat { key: "w".into(), expected: "Double", value: "x".into() })
    );
    // parseInt refuses what parseDouble reads.
    p.put("f", "1.5");
    assert!(p.get_int("f").is_err());
    assert_eq!(p.get_double("f").unwrap(), Some(1.5));
}

#[test]
fn namespaces_join_with_a_dot_and_are_matched_with_case() {
    let mut p = TrainingParameters::new();
    p.put_in("ns", "Cutoff", 3);
    p.put("Cutoff", 5);
    p.put("other.x", "y");
    assert_eq!(p.get_int_in("ns", "cutoff").unwrap(), Some(3));
    assert_eq!(p.get_int("cutoff").unwrap(), Some(5));
    let ns = p.parameters(Some("ns"));
    assert_eq!(ns.iter().collect::<Vec<_>>(), [("Cutoff", &Value::Int(3))]);
    // getObjectSettings compares the stored key with case.
    assert!(p.parameters(Some("NS")).is_empty());
    let top = p.parameters(None);
    assert_eq!(top.iter().map(|(k, _)| k).collect::<Vec<_>>(), ["Cutoff"]);
    assert_eq!(p.algorithm_in("ns").unwrap(), None);
}

#[test]
fn defaults_are_default_params() {
    let p = TrainingParameters::defaults();
    assert_eq!(p.algorithm().unwrap(), Some("MAXENT"));
    assert_eq!(p.get_str(Key::TRAINER_TYPE).unwrap(), Some("Event"));
    assert_eq!(p.get(Key::ITERATIONS), Some(&Value::Int(100)));
    assert_eq!(p.get(Key::CUTOFF), Some(&Value::Int(5)));
    assert_eq!(p.len(), 4);
}

#[test]
fn a_properties_file_loads_as_strings() {
    let p =
        TrainingParameters::load(b"# comment\nIterations=10\ncutoff : 2\nAlgorithm=PERCEPTRON\nns.Cutoff=1\n").unwrap();
    assert_eq!(p.get(Key::ITERATIONS), Some(&Value::Str("10".into())));
    assert_eq!(p.get_int(Key::ITERATIONS).unwrap(), Some(10));
    assert_eq!(p.get_int(Key::CUTOFF).unwrap(), Some(2));
    assert_eq!(p.get_int_in("ns", "cutoff").unwrap(), Some(1));
    assert_eq!(trainer_type(&p).unwrap(), Some(TrainerType::Event));
}

#[test]
fn store_writes_sorted_escaped_properties_and_refuses_non_strings() {
    let mut p = TrainingParameters::new();
    p.put("b key", "v=1 #x");
    p.put("a", " lead é\u{10000}");
    p.put("Z", "tab\tend\\");
    let text = p.store("Tue Oct 14 00:00:00 UTC 2026").unwrap();
    assert_eq!(
        text,
        "#Tue Oct 14 00:00:00 UTC 2026\nZ=tab\\tend\\\\\na=\\ lead \\u00E9\\uD800\\uDC00\nb\\ key=v\\=1 \\#x\n"
    );
    p.put("n", 3);
    assert_eq!(p.store("d"), Err(Error::SerializeType { key: "n".into(), found: "Integer" }));
}

#[test]
fn algorithms_and_trainer_types() {
    assert_eq!(Algorithm::from_name("MAXENT").unwrap(), Algorithm::Maxent);
    assert_eq!(Algorithm::from_name("maxent"), Err(Error::UnknownAlgorithm("maxent".into())));
    assert_eq!(Algorithm::Maxent.model_type(), "GIS");
    assert_eq!(Algorithm::from_model_type("Perceptron"), Some(Algorithm::Perceptron));
    assert_eq!(Algorithm::from_model_type("gis"), None);
    assert_eq!(Algorithm::PerceptronSequence.trainer_type(), TrainerType::EventModelSequence);
    assert_eq!(Algorithm::NaiveBayes.trainer_type(), TrainerType::Event);
    for a in Algorithm::ALL {
        assert_eq!(Algorithm::from_name(a.name()).unwrap(), a);
    }
    assert_eq!(TrainerType::EventModelSequence.value(), "EventModelSequence");
}

#[test]
fn trainer_type_and_validity_follow_the_factory() {
    let mut p = TrainingParameters::new();
    assert_eq!(trainer_type(&p).unwrap(), Some(TrainerType::Event));
    assert!(is_valid(&p).unwrap());
    p.put(Key::ALGORITHM, "PERCEPTRON_SEQUENCE");
    assert_eq!(trainer_type(&p).unwrap(), Some(TrainerType::EventModelSequence));
    p.put(Key::ALGORITHM, "com.example.MyTrainer");
    assert_eq!(trainer_type(&p).unwrap(), None);
    assert!(!is_valid(&p).unwrap());
    p.put(Key::ALGORITHM, "MAXENT");
    p.put(Key::CUTOFF, "many");
    assert!(!is_valid(&p).unwrap());
    p.put(Key::CUTOFF, 2.5);
    assert!(is_valid(&p).is_err());
    p.put(Key::CUTOFF, 2);
    for indexer in ["OnePass", "TwoPass", "OnePassRealValue"] {
        p.put(Key::DATA_INDEXER, indexer);
        assert!(is_valid(&p).unwrap(), "{indexer}");
    }
    p.put(Key::DATA_INDEXER, "InvalidIndexer");
    assert!(!is_valid(&p).unwrap());
    p.put(Key::DATA_INDEXER, 1);
    assert!(is_valid(&p).is_err());
    p.put(Key::ALGORITHM, 1);
    assert!(trainer_type(&p).is_err());
}

#[test]
fn values_print_as_java_strings() {
    assert_eq!(Value::Double(0.0001).java_string(), "1.0E-4");
    assert_eq!(Value::Int(-3).to_string(), "-3");
    assert_eq!(Value::Bool(false).to_string(), "false");
    assert_eq!(Value::from("x").java_class(), "String");
}
