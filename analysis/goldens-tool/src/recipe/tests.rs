use super::*;

fn is_lower_hex(s: &str, len: usize) -> bool {
    s.len() == len && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

#[test]
fn the_commit_is_forty_hex_digits() {
    assert!(is_lower_hex(OPENNLP_COMMIT, 40), "{OPENNLP_COMMIT}");
    assert!(is_lower_hex(ADDONS_COMMIT, 40), "{ADDONS_COMMIT}");
    assert!(is_lower_hex(STRING_PATTERN_COMMIT, 40), "{STRING_PATTERN_COMMIT}");
    assert!(is_lower_hex(PLACE_PROFILES_COMMIT, 40), "{PLACE_PROFILES_COMMIT}");
    assert!(is_lower_hex(LIBLINEAR_ADDON_COMMIT, 40), "{LIBLINEAR_ADDON_COMMIT}");
    assert!(is_lower_hex(NEURAL_POSTAG_COMMIT, 40), "{NEURAL_POSTAG_COMMIT}");
    assert!(is_lower_hex(HUNSPELL_COMMIT, 40), "{HUNSPELL_COMMIT}");
    assert!(is_lower_hex(CONLLU_DEPENDENCY_COMMIT, 40), "{CONLLU_DEPENDENCY_COMMIT}");
    assert!(is_lower_hex(COREF_ADDONS_COMMIT, 40), "{COREF_ADDONS_COMMIT}");
    assert!(is_lower_hex(HUNSPELL_DICTIONARIES_COMMIT, 40), "{HUNSPELL_DICTIONARIES_COMMIT}");
    assert!(IMAGE_TAG.contains(&format!("opennlp-{}", &OPENNLP_COMMIT[..12])));
    assert!(IMAGE_TAG.ends_with(&format!("addons-{}", &ADDONS_COMMIT[..12])));
}

#[test]
fn the_base_image_is_pinned_by_a_sha256_digest() {
    let hex = BASE_IMAGE_DIGEST.strip_prefix("sha256:").expect("a sha256: digest");
    assert!(is_lower_hex(hex, 64), "{BASE_IMAGE_DIGEST}");
}

#[test]
fn every_model_has_a_file_an_https_url_and_a_sha512() {
    for m in MODELS.iter().chain(&LOCAL_MODELS) {
        assert!(is_lower_hex(m.sha512, 128), "{}: {}", m.file, m.sha512);
        assert!(m.url.starts_with("https://"), "{}", m.url);
        assert!(m.url.ends_with(&format!("/{}", m.file)), "{} does not end in {}", m.url, m.file);
        assert!(m.file.ends_with(".bin"));
    }
}

#[test]
fn the_dockerfile_pins_the_same_base_image_and_commit() {
    let dockerfile = include_str!("../../container/Dockerfile");
    assert!(dockerfile.contains(&format!("\nFROM {BASE_IMAGE}@{BASE_IMAGE_DIGEST}\n")));
    assert!(dockerfile.contains(&format!("\nARG OPENNLP_COMMIT={OPENNLP_COMMIT}\n")));
    assert!(dockerfile.contains(&format!("\nARG OPENNLP_REPO={OPENNLP_REPO}\n")));
    assert!(dockerfile.contains(&format!("\nARG ADDONS_COMMIT={ADDONS_COMMIT}\n")));
    assert!(dockerfile.contains(&format!("\nARG ADDONS_REPO={ADDONS_REPO}\n")));
    assert!(dockerfile.contains(&format!("\nARG STRING_PATTERN_COMMIT={STRING_PATTERN_COMMIT}\n")));
    let patterns = include_str!("../../java/src/main/java/ai/pipestream/turbo/goldens/StringPatternGolden.java");
    assert!(patterns.contains(&format!("BRANCH_COMMIT = \"{STRING_PATTERN_COMMIT}\";")));
    assert!(dockerfile.contains(&format!("\nARG HUNSPELL_COMMIT={HUNSPELL_COMMIT}\n")));
    let hunspell = include_str!("../../java/src/main/java/ai/pipestream/turbo/goldens/HunspellGolden.java");
    assert!(hunspell.contains(&format!("BRANCH_COMMIT = \"{HUNSPELL_COMMIT}\";")));
    assert!(dockerfile.contains(&format!("\nARG PLACE_PROFILES_COMMIT={PLACE_PROFILES_COMMIT}\n")));
    let profiles = include_str!("../../java/src/main/java/ai/pipestream/turbo/goldens/PlaceProfilesGolden.java");
    assert!(profiles.contains(&format!("PROFILES_COMMIT = \"{PLACE_PROFILES_COMMIT}\";")));
    assert!(dockerfile.contains(&format!("\nARG LIBLINEAR_ADDON_COMMIT={LIBLINEAR_ADDON_COMMIT}\n")));
    let liblinear = include_str!("../../java/src/main/java/ai/pipestream/turbo/goldens/LiblinearTrainGolden.java");
    assert!(liblinear.contains(&format!("ADDON_COMMIT = \"{LIBLINEAR_ADDON_COMMIT}\";")));
    assert!(liblinear.contains("ANNOTATOR = \"train-liblinear\""));
    assert!(dockerfile.contains(&format!("\nARG CONLLU_DEPENDENCY_COMMIT={CONLLU_DEPENDENCY_COMMIT}\n")));
    let dependency = include_str!("../../java/src/main/java/ai/pipestream/turbo/goldens/ConlluDependencyGolden.java");
    assert!(dependency.contains(&format!("BRANCH_COMMIT = \"{CONLLU_DEPENDENCY_COMMIT}\";")));
    assert!(dockerfile.contains(&format!("\nARG COREF_ADDONS_COMMIT={COREF_ADDONS_COMMIT}\n")));
    let coref = include_str!("../../java/src/main/java/ai/pipestream/turbo/goldens/CorefUdGolden.java");
    assert!(coref.contains(&format!("ADDONS_COMMIT = \"{COREF_ADDONS_COMMIT}\";")));
    assert!(dockerfile.contains(&format!("\nARG NEURAL_POSTAG_COMMIT={NEURAL_POSTAG_COMMIT}\n")));
    let neural = include_str!("../../java/src/main/java/ai/pipestream/turbo/goldens/PostagNeuralGolden.java");
    assert!(neural.contains(&format!("NEURAL_POSTAG_COMMIT = \"{NEURAL_POSTAG_COMMIT}\";")));
    assert!(neural.contains("ANNOTATOR = \"postag-neural\""));
}

#[test]
fn every_hunspell_file_names_the_commit_and_a_sha512() {
    for f in HUNSPELL_FILES {
        assert!(is_lower_hex(f.sha512, 128), "{}: {}", f.file, f.sha512);
        let name = f.file.strip_prefix("hunspell/").expect("a file of the hunspell directory");
        assert!(name.ends_with(".aff") || name.ends_with(".dic"), "{}", f.file);
        let prefix =
            format!("https://raw.githubusercontent.com/LibreOffice/dictionaries/{HUNSPELL_DICTIONARIES_COMMIT}/");
        assert!(f.url.starts_with(&prefix), "{}", f.url);
        assert!(f.url.ends_with(&format!("/{name}")), "{} does not end in {name}", f.url);
        assert!(!f.license.is_empty());
    }
    let golden = include_str!("../../java/src/main/java/ai/pipestream/turbo/goldens/HunspellGolden.java");
    for f in HUNSPELL_FILES.iter().filter(|f| f.file.ends_with(".aff")) {
        let name = &f.file["hunspell/".len()..f.file.len() - ".aff".len()];
        assert!(golden.contains(&format!("\"{name}\"")), "HunspellGolden.java does not name {name}");
    }
}

#[test]
fn every_fixture_has_a_file_and_a_sha512() {
    for f in FIXTURES {
        assert!(is_lower_hex(f.sha512, 128), "{}: {}", f.file, f.sha512);
        match f.file.split_once('/') {
            None => assert!(
                f.file.ends_with(".bin")
                    || f.file.ends_with(".xml")
                    || f.file.ends_with(".txt")
                    || f.file.ends_with(".tsv")
                    || f.file.ends_with(".tagdict"),
                "{}",
                f.file
            ),
            Some((dir, file)) => {
                assert!(!file.contains('/'), "{}", f.file);
                let extensions: &[&str] = match dir {
                    "featuregen" => &[".xml", ".cluster", ".dictionary"],
                    "namefind" => &[".xml", ".txt"],
                    "hunspell" => &[".aff", ".dic"],
                    _ => {
                        assert!(dir.starts_with("mecab-"), "{}", f.file);
                        &[".csv", ".def"]
                    }
                };
                assert!(extensions.iter().any(|e| file.ends_with(e)), "{}", f.file);
            }
        }
    }
}

#[test]
fn every_dictionary_has_an_https_url_and_a_sha512() {
    for d in DICTIONARIES {
        assert!(is_lower_hex(d.sha512, 128), "{}: {}", d.archive, d.sha512);
        assert!(d.url.starts_with("https://"), "{}", d.url);
        assert!(d.url.ends_with(&format!("/{}", d.archive)), "{} does not end in {}", d.url, d.archive);
        assert_eq!(d.archive, format!("{}.tar.gz", d.dir));
    }
}

/// The name of a fixture as `Runs` opens it: the directory of a file in
/// a subdirectory, the file itself otherwise.
fn opened(file: &str) -> &str {
    file.split_once('/').map_or(file, |(dir, _)| dir)
}

#[test]
fn the_generator_opens_the_same_model_and_fixture_files() {
    let runs = include_str!("../../java/src/main/java/ai/pipestream/turbo/goldens/Runs.java");
    let dictionaries = DICTIONARIES.iter().map(|d| d.dir);
    let copies: Vec<String> = DICTIONARIES
        .iter()
        .filter(|d| d.encoding == DictionaryEncoding::EucJp)
        .map(|d| format!("{}{UTF8_COPY_SUFFIX}", d.dir))
        .collect();
    for file in MODELS
        .iter()
        .chain(&LOCAL_MODELS)
        .map(|m| m.file)
        .chain(FIXTURES.iter().map(|f| opened(f.file)))
        .chain(dictionaries)
        .chain(copies.iter().map(String::as_str))
    {
        assert!(runs.contains(&format!("\"{file}\"")), "Runs.java does not name {file}");
    }
}

#[test]
fn the_generator_names_the_same_input_sets() {
    let runs = include_str!("../../java/src/main/java/ai/pipestream/turbo/goldens/Runs.java");
    for set in INPUT_SETS.iter().filter(|s| {
        ![
            NORMALIZER_INPUT_SET,
            NUMBER_INPUT_SET,
            MONEY_INPUT_SET,
            QUANTITY_INPUT_SET,
            TEMPORAL_INPUT_SET,
            SVM_DOCCAT_INPUT_SET,
            SVM_DOCCAT_TRAIN_SET,
            SENTIMENT_INPUT_SET,
            SENTIMENT_TRAIN_SET,
            DOCCAT_INPUT_SET,
            DOCCAT_TRAIN_SET,
            PII_INPUT_SET,
        ]
        .contains(s)
    }) {
        assert!(runs.contains(&format!("= \"{set}\";")), "Runs.java does not name the input set {set}");
    }
}

#[test]
fn the_generator_and_the_entrypoint_take_the_generator_args() {
    let main = include_str!("../../java/src/main/java/ai/pipestream/turbo/goldens/Main.java");
    for arg in GENERATOR_ARGS {
        assert!(main.contains(&format!("\"{arg}\"")), "Main.java does not take {arg}");
    }
    let dockerfile = include_str!("../../container/Dockerfile");
    assert!(
        dockerfile.contains("/inputs /models /fixtures /out \"$(cat /opennlp.commit)\" \"$@\""),
        "the entrypoint does not pass the mounts in the generator's order, or drops its arguments"
    );
}

#[test]
fn the_dumped_annotators_and_the_normalizer_set_are_in_the_recipe() {
    assert!(DUMPED_ANNOTATORS.iter().all(|a| ANNOTATORS.contains(a)));
    assert!(INPUT_SETS.contains(&NORMALIZER_INPUT_SET));
    let normalizer = include_str!("../../java/src/main/java/ai/pipestream/turbo/goldens/NormalizerGolden.java");
    assert!(normalizer.contains(&format!("INPUT_SET = \"{NORMALIZER_INPUT_SET}\"")));
}

#[test]
fn the_number_scan_has_its_own_input_set_and_no_dumps() {
    assert!(INPUT_SETS.contains(&NUMBER_INPUT_SET));
    assert!(ANNOTATORS.contains(&"numbers"));
    assert!(!DUMPED_ANNOTATORS.contains(&"numbers"));
    let numbers = include_str!("../../java/src/main/java/ai/pipestream/turbo/goldens/NumberGolden.java");
    assert!(numbers.contains(&format!("INPUT_SET = \"{NUMBER_INPUT_SET}\"")));
}

#[test]
fn the_money_annotators_have_their_own_input_set_and_no_dumps() {
    assert!(INPUT_SETS.contains(&MONEY_INPUT_SET));
    assert!(ANNOTATORS.contains(&"money"));
    assert!(ANNOTATORS.contains(&"money-convert"));
    assert!(!DUMPED_ANNOTATORS.contains(&"money"));
    assert!(!DUMPED_ANNOTATORS.contains(&"money-convert"));
    let money = include_str!("../../java/src/main/java/ai/pipestream/turbo/goldens/MoneyGolden.java");
    assert!(money.contains(&format!("INPUT_SET = \"{MONEY_INPUT_SET}\"")));
    assert!(money.contains(&format!("SNAPSHOT_DIR = \"{RATES_DIR}\"")));
    assert!(money.contains(&format!("SNAPSHOT_FILE = \"{RATES_FILE}\"")));
}

#[test]
fn the_quantity_extractor_has_its_own_input_set_and_no_dumps() {
    assert!(INPUT_SETS.contains(&QUANTITY_INPUT_SET));
    assert!(ANNOTATORS.contains(&"quantity"));
    assert!(!DUMPED_ANNOTATORS.contains(&"quantity"));
    let quantity = include_str!("../../java/src/main/java/ai/pipestream/turbo/goldens/QuantityGolden.java");
    assert!(quantity.contains(&format!("INPUT_SET = \"{QUANTITY_INPUT_SET}\"")));
}

#[test]
fn the_temporal_extractor_has_its_own_input_set_and_no_dumps() {
    assert!(INPUT_SETS.contains(&TEMPORAL_INPUT_SET));
    assert!(ANNOTATORS.contains(&"temporal"));
    assert!(!DUMPED_ANNOTATORS.contains(&"temporal"));
    let temporal = include_str!("../../java/src/main/java/ai/pipestream/turbo/goldens/TemporalGolden.java");
    assert!(temporal.contains(&format!("INPUT_SET = \"{TEMPORAL_INPUT_SET}\"")));
    assert!(temporal.contains("ANNOTATOR = \"temporal\""));
}

#[test]
fn the_pii_packs_and_rewrites_have_their_own_input_set_and_no_dumps() {
    assert!(INPUT_SETS.contains(&PII_INPUT_SET));
    for annotator in ["pii", "pii-rewrite"] {
        assert!(ANNOTATORS.contains(&annotator));
        assert!(!DUMPED_ANNOTATORS.contains(&annotator));
    }
    let pii = include_str!("../../java/src/main/java/ai/pipestream/turbo/goldens/PiiGolden.java");
    assert!(pii.contains(&format!("INPUT_SET = \"{PII_INPUT_SET}\"")));
    assert!(pii.contains("ANNOTATOR = \"pii\""));
    let rewrite = include_str!("../../java/src/main/java/ai/pipestream/turbo/goldens/PiiRewriteGolden.java");
    assert!(rewrite.contains("ANNOTATOR = \"pii-rewrite\""));
    assert!(rewrite.contains("Set.of(PiiGolden.INPUT_SET)"));
    let runs = include_str!("../../java/src/main/java/ai/pipestream/turbo/goldens/Runs.java");
    assert!(runs.contains("annotators.addAll(PiiGolden.all());"));
    assert!(runs.contains("annotators.addAll(PiiRewriteGolden.all());"));
}

#[test]
fn the_svm_categorizer_has_its_own_input_sets_and_no_dumps() {
    assert!(INPUT_SETS.contains(&SVM_DOCCAT_INPUT_SET));
    assert!(INPUT_SETS.contains(&SVM_DOCCAT_TRAIN_SET));
    assert!(ANNOTATORS.contains(&"svm-doccat"));
    assert!(!DUMPED_ANNOTATORS.contains(&"svm-doccat"));
    let svm = include_str!("../../java/src/main/java/ai/pipestream/turbo/goldens/SvmDoccatGolden.java");
    assert!(svm.contains(&format!("INPUT_SET = \"{SVM_DOCCAT_INPUT_SET}\"")));
    assert!(svm.contains(&format!("TRAIN_SET = \"{SVM_DOCCAT_TRAIN_SET}\"")));
    assert!(svm.contains("ANNOTATOR = \"svm-doccat\""));
}

#[test]
fn the_sentiment_detector_has_its_own_input_sets_and_no_dumps() {
    assert!(INPUT_SETS.contains(&SENTIMENT_INPUT_SET));
    assert!(INPUT_SETS.contains(&SENTIMENT_TRAIN_SET));
    assert!(ANNOTATORS.contains(&"sentiment"));
    assert!(!DUMPED_ANNOTATORS.contains(&"sentiment"));
    let golden = include_str!("../../java/src/main/java/ai/pipestream/turbo/goldens/SentimentGolden.java");
    assert!(golden.contains(&format!("INPUT_SET = \"{SENTIMENT_INPUT_SET}\"")));
    assert!(golden.contains(&format!("TRAIN_SET = \"{SENTIMENT_TRAIN_SET}\"")));
    assert!(golden.contains("ANNOTATOR = \"sentiment\""));
    let runs = include_str!("../../java/src/main/java/ai/pipestream/turbo/goldens/Runs.java");
    assert!(runs.contains("SentimentGolden.all(inputs)"));
}

#[test]
fn the_treebank_streams_have_their_own_input_sets_and_no_dumps() {
    for (set, class) in [(ONTONOTES_INPUT_SET, "OntoNotesGolden"), (FRENCH_TREEBANK_INPUT_SET, "FrenchTreebankGolden")]
    {
        assert!(INPUT_SETS.contains(&set));
        assert!(ANNOTATORS.contains(&set));
        assert!(!DUMPED_ANNOTATORS.contains(&set));
        let runs = include_str!("../../java/src/main/java/ai/pipestream/turbo/goldens/Runs.java");
        assert!(runs.contains(&format!("= \"{set}\";")));
        assert!(runs.contains(&format!("annotators.addAll({class}.all());")));
    }
    let onto = include_str!("../../java/src/main/java/ai/pipestream/turbo/goldens/OntoNotesGolden.java");
    assert!(onto.contains("SET = Runs.ONTONOTES;") && onto.contains("return \"ontonotes\";"));
    let ftb = include_str!("../../java/src/main/java/ai/pipestream/turbo/goldens/FrenchTreebankGolden.java");
    assert!(ftb.contains("SET = Runs.FRENCH_TREEBANK;") && ftb.contains("return \"frenchtreebank\";"));
}

#[test]
fn the_ad_streams_have_their_own_input_set_and_no_dumps() {
    assert!(INPUT_SETS.contains(&AD_INPUT_SET));
    assert!(ANNOTATORS.contains(&"ad"));
    assert!(!DUMPED_ANNOTATORS.contains(&"ad"));
    let golden = include_str!("../../java/src/main/java/ai/pipestream/turbo/goldens/AdGolden.java");
    assert!(golden.contains("SET = Runs.AD;"));
    assert!(golden.contains("return \"ad\";"));
    let runs = include_str!("../../java/src/main/java/ai/pipestream/turbo/goldens/Runs.java");
    assert!(runs.contains(&format!("AD = \"{AD_INPUT_SET}\";")));
    assert!(runs.contains("annotators.addAll(AdGolden.all("));
}

#[test]
fn the_maxent_categorizer_has_its_own_input_sets_and_no_dumps() {
    assert!(INPUT_SETS.contains(&DOCCAT_INPUT_SET));
    assert!(INPUT_SETS.contains(&DOCCAT_TRAIN_SET));
    assert!(ANNOTATORS.contains(&"doccat"));
    assert!(!DUMPED_ANNOTATORS.contains(&"doccat"));
    let golden = include_str!("../../java/src/main/java/ai/pipestream/turbo/goldens/DoccatGolden.java");
    assert!(golden.contains(&format!("INPUT_SET = \"{DOCCAT_INPUT_SET}\"")));
    assert!(golden.contains(&format!("TRAIN_SET = \"{DOCCAT_TRAIN_SET}\"")));
    assert!(golden.contains("ANNOTATOR = \"doccat\""));
    let runs = include_str!("../../java/src/main/java/ai/pipestream/turbo/goldens/Runs.java");
    assert!(runs.contains("DoccatGolden.all(inputs, models)"));
}

#[test]
fn every_corpus_has_an_https_url_a_sha512_and_a_licence() {
    let runs = include_str!("../../java/src/main/java/ai/pipestream/turbo/goldens/Runs.java");
    for c in CORPORA {
        assert!(is_lower_hex(c.sha512, 128), "{}: {}", c.file, c.sha512);
        assert!(c.url.starts_with("https://") && c.source.starts_with("https://"), "{}", c.url);
        assert!(c.file.starts_with("corpora/"), "{}", c.file);
        assert!(!c.license.is_empty());
        assert!(runs.contains(&format!("\"{}\"", c.file)), "Runs.java does not name {}", c.file);
    }
}

#[test]
fn the_rate_snapshot_is_present() {
    let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../goldens/data");
    assert!(data.join(RATES_DIR).join(RATES_FILE).is_file());
}

#[test]
fn every_input_set_and_fixture_is_present() {
    let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../goldens/data");
    for set in INPUT_SETS {
        assert!(data.join("inputs").join(set).is_dir(), "{}", data.join(set).display());
    }
    for f in FIXTURES {
        assert!(data.join("fixtures").join(f.file).is_file(), "{}", f.file);
    }
}
