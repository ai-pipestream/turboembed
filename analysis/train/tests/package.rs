use turbo_maxent::{Method, ModelFile};
use turbo_train::gis::TrainingReport;
use turbo_train::package::{
    MANIFEST_ENTRY, Manifest, ModelPackage, TimeZone, java_date_string, java_date_string_in, sentence_model_package,
};
use turbo_train::params::TrainerType;

/// The instant in the fixture model en-plain.bin's manifest.
const FIXTURE_MS: i64 = 1791786313615;

#[test]
fn dates_print_as_java_prints_them() {
    assert_eq!(java_date_string(FIXTURE_MS), "Mon Oct 12 06:25:13 UTC 2026");
    assert_eq!(java_date_string(0), "Thu Jan 01 00:00:00 UTC 1970");
    assert_eq!(java_date_string(-1), "Wed Dec 31 23:59:59 UTC 1969");
    assert_eq!(java_date_string(951782400000), "Tue Feb 29 00:00:00 UTC 2000");
    assert_eq!(java_date_string(2147483647000), "Tue Jan 19 03:14:07 UTC 2038");
    let _ = java_date_string_in(i64::MAX, TimeZone::CEST);
    let _ = java_date_string_in(i64::MIN, TimeZone { offset_seconds: -7200, name: "WEST" });
}

#[test]
fn the_manifest_is_the_fixture_models_manifest() {
    let report = TrainingReport { event_hash: Some(4204752117), trainer_type: TrainerType::Event };
    let package = sentence_model_package("en", FIXTURE_MS, &report, false, Some(".!?"), Vec::new());
    let expected = "#\n#Mon Oct 12 06:25:13 UTC 2026\nComponent-Name=SentenceDetectorME\nLanguage=en\n\
        Manifest-Version=1.0\nOpenNLP-Version=3.0.0-SNAPSHOT\nTimestamp=1791786313615\nTrainerType=Event\n\
        Training-Eventhash=4204752117\neosCharacters=.\\!?\nfactory=opennlp.tools.sentdetect.SentenceDetectorFactory\n\
        useTokenEnd=false\n";
    assert_eq!(package.manifest().to_properties(), expected);
    let fixture = ModelFile::read(turbo_goldens::data_dir().join("fixtures/en-plain.bin")).unwrap();
    let fixture_manifest = String::from_utf8(fixture.entry(MANIFEST_ENTRY).unwrap().into_owned()).unwrap();
    assert_eq!(fixture_manifest, expected);
}

#[test]
fn set_replaces_and_get_reads_base_entries() {
    let mut m = Manifest::new("TokenizerME", "de", 5);
    m.set("useAlphaNumericOptimization", "true");
    m.set("useAlphaNumericOptimization", "false");
    assert_eq!(m.get("useAlphaNumericOptimization"), Some("false"));
    assert_eq!(m.component_name(), "TokenizerME");
    assert_eq!(m.get("Component-Name"), Some("TokenizerME"));
    assert_eq!(m.language(), "de");
    assert_eq!(m.get("Language"), Some("de"));
    assert_eq!(m.get("OpenNLP-Version"), Some("3.0.0-SNAPSHOT"));
    assert_eq!(m.get("Manifest-Version"), Some("1.0"));
    assert_eq!(m.get("Timestamp"), Some("5"));
    assert_eq!(m.timestamp_ms(), 5);
    assert_eq!(m.get("missing"), None);
    assert_eq!(m.to_properties().matches("useAlphaNumericOptimization").count(), 1);

    m.set("Language", "fr");
    m.set("Component-Name", "POSTaggerME");
    m.set("OpenNLP-Version", "2.5.4");
    m.set("Timestamp", "999");
    m.set("Manifest-Version", "2.0");
    assert_eq!(m.language(), "fr");
    assert_eq!(m.get("Language"), Some("fr"));
    assert_eq!(m.component_name(), "POSTaggerME");
    assert_eq!(m.get("Component-Name"), Some("POSTaggerME"));
    assert_eq!(m.version(), "2.5.4");
    assert_eq!(m.get("OpenNLP-Version"), Some("2.5.4"));
    assert_eq!(m.get("Timestamp"), Some("999"));
    assert_eq!(m.get("Manifest-Version"), Some("2.0"));
    let props = m.to_properties();
    assert_eq!(props.matches("Language=").count(), 1);
    assert_eq!(props.matches("Component-Name=").count(), 1);
    assert_eq!(props.matches("OpenNLP-Version=").count(), 1);
    assert_eq!(props.matches("Timestamp=").count(), 1);
    assert!(props.contains("Timestamp=999\n"));
    assert_eq!(props.matches("Manifest-Version=").count(), 1);
    assert!(props.contains("Manifest-Version=2.0\n"));
}

#[test]
fn the_archive_opens_as_a_model_file_with_stored_entries() {
    let mut manifest = Manifest::new("SentenceDetectorME", "en", FIXTURE_MS);
    manifest.set("useTokenEnd", "true");
    let mut package = ModelPackage::new(manifest);
    package.add_entry("sent.model", vec![1, 2, 3]);
    package.add_entry("abbreviations.dictionary", b"<dictionary/>".to_vec());
    let bytes = package.to_bytes();
    let file = ModelFile::open(bytes).unwrap();
    assert_eq!(file.component_name(), "SentenceDetectorME");
    assert_eq!(file.manifest().get("useTokenEnd"), Some("true"));
    assert_eq!(&*file.entry("sent.model").unwrap(), &[1, 2, 3]);
    assert_eq!(&*file.entry("abbreviations.dictionary").unwrap(), b"<dictionary/>");
    assert_eq!(file.archive().len(), 3);
    assert_eq!(file.archive().method("sent.model"), Some(Method::Stored));
}

#[test]
fn the_manifest_is_written_where_it_is_placed() {
    let mut package = ModelPackage::new(Manifest::new("POSTaggerME", "en", FIXTURE_MS));
    package.add_entry("tags.tagdict", b"<dictionary/>".to_vec());
    package.add_entry("pos.model", vec![1]);
    assert_eq!(package.entry_names(), ["manifest.properties", "tags.tagdict", "pos.model"]);
    package.place_manifest(1);
    assert_eq!(package.entry_names(), ["tags.tagdict", "manifest.properties", "pos.model"]);
    let file = ModelFile::open(package.to_bytes()).unwrap();
    let names: Vec<String> = file.archive().names().map(|n| String::from_utf8(n.to_vec()).unwrap()).collect();
    assert_eq!(names, ["tags.tagdict", "manifest.properties", "pos.model"]);
    assert_eq!(file.component_name(), "POSTaggerME");
    // A count beyond the entries puts it last.
    package.place_manifest(9);
    assert_eq!(package.entry_names(), ["tags.tagdict", "pos.model", "manifest.properties"]);
    assert_eq!(package.entries().count(), 2);
}

#[test]
fn another_version_and_zone_print_as_that_jvm_printed_them() {
    // The published model opennlp-en-ud-ewt-sentence-1.3-2.5.4.bin.
    let ms = 1749698375454;
    assert_eq!(java_date_string_in(ms, TimeZone::CEST), "Thu Jun 12 05:19:35 CEST 2025");
    assert_eq!(java_date_string_in(ms, TimeZone::UTC), "Thu Jun 12 03:19:35 UTC 2025");
    let manifest = Manifest::new("SentenceDetectorME", "en", ms).with_version("2.5.4").with_zone(TimeZone::CEST);
    assert_eq!(manifest.version(), "2.5.4");
    assert_eq!(manifest.zone(), TimeZone::CEST);
    let text = manifest.to_properties();
    assert!(text.starts_with("#\n#Thu Jun 12 05:19:35 CEST 2025\n"), "{text}");
    assert!(text.contains("\nOpenNLP-Version=2.5.4\n"), "{text}");
}
