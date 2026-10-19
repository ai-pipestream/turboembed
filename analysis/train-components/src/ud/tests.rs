use super::compare::{ulps_apart, update_parameter_distance};
use super::*;
use turbo_maxent::JavaString;
use turbo_zoo::LicenceClass;

#[test]
fn ulps_count_the_numbers_between_two_values_across_zero() {
    assert_eq!(ulps_apart(1.0, 1.0), 0);
    assert_eq!(ulps_apart(1.0, f64::from_bits(1.0f64.to_bits() + 3)), 3);
    assert_eq!(ulps_apart(0.0, -0.0), 0);
    let tiny = f64::from_bits(2);
    assert_eq!(ulps_apart(tiny, -tiny), 4);
    assert_eq!(ulps_apart(-1.0, -f64::from_bits(1.0f64.to_bits() + 1)), 1);
}

fn comparison_with(parameters: Option<std::result::Result<ParameterDistance, String>>) -> Comparison {
    let digest = EntryDigest { len: 1, sha256: String::new() };
    Comparison {
        model: String::new(),
        events: 0,
        event_hash: Agreement::Same(1),
        model_entry: Agreement::Differ { published: digest.clone(), trained: digest },
        first_difference: Some(0),
        parameters,
        manifest: Agreement::Same(String::new()),
        entry_names: Agreement::Same(Vec::new()),
    }
}

#[test]
fn a_nan_distance_is_no_agreement_and_the_report_shows_every_manifest_line() {
    let close = ParameterDistance { total: 2, differing: 1, max_ulps: 1, max_abs: 1e-15 };
    assert!(comparison_with(Some(Ok(close.clone()))).agrees_within(1e-12));
    let nan = ParameterDistance { max_abs: f64::NAN, ..close.clone() };
    assert!(!comparison_with(Some(Ok(nan))).agrees_within(1e-12));
    assert!(!comparison_with(Some(Err("shapes".to_owned()))).agrees_within(1e-12));
    assert!(!comparison_with(None).agrees_within(1e-12));
    let mut c = comparison_with(Some(Ok(close)));
    c.manifest = Agreement::Differ { published: "a=1\nb=2\nc=3".to_owned(), trained: "a=1\nb=x".to_owned() };
    let report = c.report();
    assert!(report.contains("    published: b=2\n    trained:   b=x\n"), "{report}");
    assert!(report.contains("    published: c=3\n    trained:   (no line)\n"), "{report}");
}

#[test]
fn agrees_within_rejects_negative_and_nan_tolerances_even_when_model_entries_match() {
    let digest = EntryDigest { len: 1, sha256: String::new() };
    let identical = Comparison {
        model: String::new(),
        events: 0,
        event_hash: Agreement::Same(1),
        model_entry: Agreement::Same(digest),
        first_difference: None,
        parameters: None,
        manifest: Agreement::Same(String::new()),
        entry_names: Agreement::Same(Vec::new()),
    };
    assert!(identical.agrees_within(0.0));
    assert!(!identical.agrees_within(-1.0));
    assert!(!identical.agrees_within(f64::NAN));
}

#[test]
fn equal_bit_nan_parameters_record_nan_distance_and_do_not_agree() {
    let mut d = ParameterDistance { total: 0, differing: 0, max_ulps: 0, max_abs: 0.0 };
    update_parameter_distance(&mut d, f64::NAN, f64::NAN);
    assert_eq!(d.total, 1);
    assert_eq!(d.differing, 1);
    assert!(d.max_abs.is_nan());
    assert!(!comparison_with(Some(Ok(d))).agrees_within(1e-12));
}

/// Trains the English sentence recipe on a few CoNLL-U sentences with
/// `iterations` iterations.
fn small_training(iterations: i32) -> Trained {
    let conllu = "# text = Dr. Who left.\n1\tDr.\t_\t_\t_\t_\t_\t_\t_\t_\n2\tWho\t_\t_\t_\t_\t_\t_\t_\t_\n\
                  3\tleft.\t_\t_\t_\t_\t_\t_\t_\t_\n\n\
                  # text = Two? Yes!\n1\tTwo?\t_\t_\t_\t_\t_\t_\t_\t_\n2\tYes!\t_\t_\t_\t_\t_\t_\t_\t_\n\n\
                  # text = Mr. Smith came.\n1\tMr.\t_\t_\t_\t_\t_\t_\t_\t_\n2\tSmith\t_\t_\t_\t_\t_\t_\t_\t_\n\
                  3\tcame.\t_\t_\t_\t_\t_\t_\t_\t_\n\n";
    let recipe = RECIPES[0];
    let factory = recipe.sentence_factory().unwrap();
    let mut events = Vec::new();
    let mut stream = sentdetect::SdEventStream::new(&factory, WHITESPACE);
    for sample in ConlluSentenceReader::new(conllu, SENTENCES_PER_SAMPLE).unwrap() {
        stream.events(&sample.unwrap(), &mut events);
    }
    let mut params = training_parameters();
    params.put(Key::ITERATIONS, iterations);
    params.put(Key::CUTOFF, 0);
    params.put(Key::THREADS, 1);
    let stamp = Stamp { timestamp_ms: 1_749_698_375_454, version: OPENNLP_VERSION, zone: ZONE };
    let t = sentdetect::train_sentence_model_from_events(
        &events,
        &factory,
        &params,
        &mut TrainingConfiguration::default(),
        stamp,
    )
    .unwrap();
    Trained { events: events.len(), model: t.model, report: t.report, package: t.package }
}

#[test]
fn a_model_compared_with_itself_is_reproduced_and_with_another_run_is_not() {
    let recipe = RECIPES[0];
    let ten = small_training(10);
    assert_eq!(ten.events, 6);
    let published = ModelFile::open(ten.package.to_bytes()).unwrap();
    let same = Comparison::of(&recipe, &ten, &published).unwrap();
    assert!(same.reproduced() && same.agrees_within(0.0), "{}", same.report());
    assert!(same.parameters.is_none() && same.first_difference.is_none());
    assert_eq!(same.model, "opennlp-en-ud-ewt-sentence-1.3-2.5.4");

    let twenty = small_training(20);
    let other = Comparison::of(&recipe, &twenty, &published).unwrap();
    assert!(!other.reproduced() && !other.agrees_within(1e-12), "{}", other.report());
    assert!(other.event_hash.is_same() && other.manifest.is_same() && other.entry_names.is_same());
    let distance = other.parameters.as_ref().unwrap().as_ref().unwrap();
    assert_eq!(distance.total, ten.model.all_parameters().len());
    assert!(distance.differing > 0 && distance.max_ulps > 0 && distance.max_abs > 1e-12);
    assert!(other.report().contains("same predicates and patterns;"), "{}", other.report());

    // The published model's predicates, patterns and parameters are read
    // back as the trained model holds them.
    let theirs = published.maxent(sentdetect::MODEL_ENTRY).unwrap();
    for (p, label) in ten.model.pred_labels().iter().enumerate() {
        let q = theirs.predicate_index(label.as_bytes()).unwrap();
        let ours: Vec<&JavaString> =
            ten.model.pattern(p).iter().map(|&o| &ten.model.outcome_labels()[o as usize]).collect();
        let published: Vec<&JavaString> = theirs.pattern(q).iter().map(|&o| theirs.outcome(o as usize)).collect();
        assert_eq!(ours, published);
        assert_eq!(ten.model.parameters(p), theirs.parameters(q));
    }
}

#[test]
fn the_datasets_make_a_valid_share_alike_index() {
    let index = datasets();
    assert_eq!(index.list().len(), 5);
    let ewt = index.get("ud-english-ewt-r2.16-train").unwrap();
    assert_eq!(ewt.class, LicenceClass::ShareAlike);
    for a in index.list() {
        assert_eq!(a.class, a.computed_class(), "{}", a.id);
    }
    // The class follows the licence: a CC BY 4.0 treebank is permissive.
    let permissive = Treebank { licence: Licence::CcBy40, ..ENGLISH_EWT }.dataset();
    assert_eq!(permissive.class, LicenceClass::Permissive);
    assert_eq!(ewt.files[0].size, 14203025);
    assert_eq!(
        ewt.files[0].url.as_str(),
        "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/r2.16/en_ewt-ud-train.conllu"
    );
    assert_eq!(
        ewt.license_terms.as_str(),
        "https://github.com/UniversalDependencies/UD_English-EWT/blob/r2.16/README.md"
    );
    // Fetching one without an acceptance is refused before any transfer.
    let zoo = Zoo::new(datasets(), PathBuf::from("/nonexistent"), Policy::default(), &turbo_zoo::Offline);
    let user: UserId = "someone".parse().unwrap();
    assert!(zoo.permit("ud-english-ewt-r2.16-train", &user, &Acceptances::default()).is_err());
}

#[test]
fn the_recipes_name_the_published_files_and_the_training_settings() {
    let files: Vec<String> = RECIPES.iter().map(Recipe::model_file).collect();
    assert_eq!(
        files,
        [
            "opennlp-en-ud-ewt-sentence-1.3-2.5.4.bin",
            "opennlp-en-ud-ewt-tokens-1.3-2.5.4.bin",
            "opennlp-en-ud-ewt-pos-1.3-2.5.4.bin",
            "opennlp-en-ud-ewt-lemmas-1.3-2.5.4.bin",
            "opennlp-de-ud-gsd-sentence-1.3-2.5.4.bin",
            "opennlp-de-ud-gsd-tokens-1.3-2.5.4.bin",
            "opennlp-de-ud-gsd-pos-1.3-2.5.4.bin",
            "opennlp-de-ud-gsd-lemmas-1.3-2.5.4.bin",
            "opennlp-pt-ud-gsd-sentence-1.3-2.5.4.bin",
            "opennlp-pt-ud-gsd-tokens-1.3-2.5.4.bin",
            "opennlp-pt-ud-gsd-pos-1.3-2.5.4.bin",
            "opennlp-pt-ud-gsd-lemmas-1.3-2.5.4.bin",
            "opennlp-es-ud-gsd-tokens-1.3-2.5.4.bin",
            "opennlp-es-ud-gsd-pos-1.3-2.5.4.bin",
            "opennlp-es-ud-gsd-lemmas-1.3-2.5.4.bin",
            "opennlp-fr-ud-gsd-tokens-1.3-2.5.4.bin",
            "opennlp-fr-ud-gsd-pos-1.3-2.5.4.bin",
            "opennlp-fr-ud-gsd-lemmas-1.3-2.5.4.bin",
        ]
    );
    // Every model is in the zoo's index under its id.
    let index = Index::builtin();
    for r in &RECIPES {
        assert!(index.get(&r.model_id()).is_ok(), "{}", r.model_id());
    }
    let p = training_parameters();
    assert_eq!(p.get_int(Key::ITERATIONS).unwrap(), Some(300));
    assert_eq!(CONF_ITERATIONS, 150);
    assert_eq!(p.get_int(Key::CUTOFF).unwrap(), Some(5));
    assert_eq!(p.get_int(Key::THREADS).unwrap(), Some(4));
    assert_eq!(p.algorithm().unwrap(), Some("MAXENT"));
    let pt = RECIPES[8].sentence_factory().unwrap();
    assert_eq!(pt.eos_string(), ".?!;:()\u{AB}\u{BB}'\"");
    let de = RECIPES[5].tokenizer_factory().unwrap();
    assert!(!de.use_alpha_numeric_optimization());
    assert_eq!(de.alpha_numeric_pattern(), turbo_tokenize::me::alnum::language_pattern("de"));
    let pos = RECIPES[2].pos_factory().unwrap();
    assert_eq!(pos.descriptor(), turbo_sequence::postag::DEFAULT_DESCRIPTOR.as_bytes());
    assert!(pos.tag_dictionary().is_none());
    assert_eq!(RECIPES[3].lemmatizer_factory().language(), "en");
    assert_eq!(RECIPES[6].component.model_entry(), "pos.model");
    assert_eq!(RECIPES[7].component.model_entry(), "lemmatizer.model");
}

#[test]
fn the_tagger_and_lemmatizer_recipes_train_from_conllu_and_compare_with_themselves() {
    let two = "# text = Dogs ran.\n\
        1\tDogs\tdog\tNOUN\tNNS\t_\t2\tnsubj\t_\t_\n\
        2\tran\trun\tVERB\tVBD\t_\t0\troot\t_\tSpaceAfter=No\n\
        3\t.\t.\tPUNCT\t.\t_\t2\tpunct\t_\t_\n\n\
        # text = A dog runs.\n\
        1\tA\ta\tDET\tDT\t_\t2\tdet\t_\t_\n\
        2\tdog\tdog\tNOUN\tNN\t_\t3\tnsubj\t_\t_\n\
        3\truns\trun\tVERB\tVBZ\t_\t0\troot\t_\tSpaceAfter=No\n\
        4\t.\t.\tPUNCT\t.\t_\t3\tpunct\t_\t_\n\n";
    // Five times, so every feature passes the recipe's cutoff of 5.
    let conllu = two.repeat(5);
    let mut config = TrainingConfiguration::default();
    let pos = RECIPES[2].train(&conllu, 1_749_698_398_335, &mut config).unwrap();
    assert_eq!(pos.events, 35);
    assert_eq!(pos.model.outcome_labels(), ["NOUN", "VERB", "PUNCT", "DET"].map(JavaString::new));
    assert_eq!(pos.package.entry_names(), ["manifest.properties", "generator.featuregen", "pos.model"]);
    let manifest = pos.package.manifest().to_properties();
    assert!(manifest.contains("Component-Name=POSTaggerME\n"), "{manifest}");
    assert!(manifest.contains("BeamSize=3\n") && manifest.contains("Language=en\n"), "{manifest}");
    let published = ModelFile::open(pos.package.to_bytes()).unwrap();
    let same = Comparison::of(&RECIPES[2], &pos, &published).unwrap();
    assert!(same.reproduced(), "{}", same.report());

    let lemmas = RECIPES[3].train(&conllu, 1_749_698_430_433, &mut config).unwrap();
    assert_eq!(lemmas.events, 35);
    assert_eq!(lemmas.model.outcome_labels(), ["D0s", "R1au", "O"].map(JavaString::new));
    assert_eq!(lemmas.package.entry_names(), ["manifest.properties", "lemmatizer.model"]);
    assert_eq!(lemmas.package.manifest().component_name(), "StatisticalLemmatizer");
    let published = ModelFile::open(lemmas.package.to_bytes()).unwrap();
    let same = Comparison::of(&RECIPES[3], &lemmas, &published).unwrap();
    assert!(same.reproduced(), "{}", same.report());

    // A model with another outcome is not comparable, before any
    // predicate is looked at.
    let more = format!(
        "{conllu}# text = Big dogs ran.\n\
         1\tBig\tbig\tADJ\tJJ\t_\t2\tamod\t_\t_\n\
         2\tdogs\tdog\tNOUN\tNNS\t_\t3\tnsubj\t_\t_\n\
         3\tran\trun\tVERB\tVBD\t_\t0\troot\t_\tSpaceAfter=No\n\
         4\t.\t.\tPUNCT\t.\t_\t3\tpunct\t_\t_\n\n"
    )
    .repeat(5);
    let five = RECIPES[2].train(&more, 1_749_698_398_335, &mut config).unwrap();
    assert_eq!(five.model.outcome_labels().len(), 5);
    let published = ModelFile::open(pos.package.to_bytes()).unwrap();
    let other = Comparison::of(&RECIPES[2], &five, &published).unwrap();
    let why = other.parameters.as_ref().unwrap().as_ref().unwrap_err();
    assert!(why.starts_with("the outcomes are [\"NOUN\", \"VERB\", \"PUNCT\", \"DET\", \"ADJ\"] trained"), "{why}");
    assert!(!other.agrees_within(1e-12));
    assert!(other.report().contains("parameters not comparable: the outcomes are"), "{}", other.report());
}
