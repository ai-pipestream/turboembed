//! Opening sentence models: what OpenNLP checks when it loads one, the
//! settings read from the manifest, and the model hash. The broken models
//! are built here as stored zip archives from the entries of a real one,
//! with the manifest changed.

mod common;

use std::path::Path;

use turbo_analysis::Annotator;
use turbo_maxent::{ModelFile, crc32};
use turbo_sentdetect::{EndOfSentence, ModelError, SentenceDetector, SentenceModel, WhitespaceMode};

/// A zip archive of stored entries.
fn zip(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let mut z = Vec::new();
    let mut central = Vec::new();
    for (name, data) in entries {
        let offset = z.len() as u32;
        let head = |sig: u32, out: &mut Vec<u8>, central: bool| {
            out.extend_from_slice(&sig.to_le_bytes());
            if central {
                out.extend_from_slice(&20u16.to_le_bytes());
            }
            out.extend_from_slice(&[20, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
            out.extend_from_slice(&crc32(data).to_le_bytes());
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            out.extend_from_slice(&(name.len() as u16).to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes());
        };
        head(0x0403_4b50, &mut z, false);
        z.extend_from_slice(name.as_bytes());
        z.extend_from_slice(data);
        head(0x0201_4b50, &mut central, true);
        central.extend_from_slice(&[0; 10]);
        central.extend_from_slice(&offset.to_le_bytes());
        central.extend_from_slice(name.as_bytes());
    }
    let dir = z.len() as u32;
    z.extend_from_slice(&central);
    z.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
    let n = entries.len() as u16;
    z.extend_from_slice(&[0, 0, 0, 0]);
    z.extend_from_slice(&n.to_le_bytes());
    z.extend_from_slice(&n.to_le_bytes());
    z.extend_from_slice(&(central.len() as u32).to_le_bytes());
    z.extend_from_slice(&dir.to_le_bytes());
    z.extend_from_slice(&[0, 0]);
    z
}

fn fixture(name: &str) -> ModelFile {
    ModelFile::read(common::fixtures_dir().join(name)).unwrap()
}

/// The fixture model with abbreviations, with a different manifest and,
/// optionally, a different classifier.
fn with_manifest(manifest: &str, classifier: Option<&[u8]>) -> Result<SentenceModel, ModelError> {
    let base = fixture("en-abbrev.bin");
    let model = base.entry("sent.model").unwrap();
    let dict = base.entry("abbreviations.dictionary").unwrap();
    let bytes = zip(&[
        ("manifest.properties", manifest.as_bytes()),
        ("sent.model", classifier.unwrap_or(&model)),
        ("abbreviations.dictionary", &dict),
    ]);
    SentenceModel::open(ModelFile::open(bytes).unwrap())
}

const MANIFEST: &str = "Component-Name=SentenceDetectorME\nLanguage=en\nOpenNLP-Version=3.0.0-SNAPSHOT\n\
    factory=opennlp.tools.sentdetect.SentenceDetectorFactory\nuseTokenEnd=true\neosCharacters=.\\!?\n";

#[test]
fn fixture_models_settings() {
    let m = SentenceModel::open(fixture("en-abbrev.bin")).unwrap();
    assert_eq!(m.language(), "en");
    assert!(m.use_token_end());
    assert_eq!(m.eos(), &EndOfSentence::default());
    assert_eq!(m.maxent().outcome(m.split_outcome()), "s");
    let dict = m.abbreviations().expect("a dictionary");
    assert!(!dict.is_case_sensitive());
    assert_eq!(dict.len(), 13);
    let r = m.model_ref();
    assert_eq!(r.id, "opennlp:SentenceDetectorME:en");
    assert_eq!(r.revision, "3.0.0-SNAPSHOT");
    // The file the goldens were made with, by the SHA-512 they record,
    // and the SHA-256 of the same bytes in the model reference.
    let bytes = std::fs::read(common::fixtures_dir().join("en-abbrev.bin")).unwrap();
    let golden = common::run("en-abbrev").into_iter().find(|c| c.golden.header.input.id == "lone-period").unwrap();
    assert_eq!(golden.golden.header.models[0].sha512, turbo_goldens::sha512(&bytes));
    assert_eq!(r.sha256, turbo_goldens::hex::to_hex(&turbo_goldens::sha256(&bytes)));

    let m = SentenceModel::open(fixture("en-plain.bin")).unwrap();
    assert!(!m.use_token_end());
    assert!(m.abbreviations().is_none());
}

#[test]
fn published_models_settings() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/models");
    let pt =
        SentenceModel::open(ModelFile::read(dir.join("opennlp-pt-ud-gsd-sentence-1.3-2.5.4.bin")).unwrap()).unwrap();
    assert_eq!(pt.eos(), &EndOfSentence::from_chars(".?!;:()\u{AB}\u{BB}'\"").unwrap());
    assert_eq!(pt.model_ref().sha256, "a711463f146555de26d69d806a9f1a88a66ed0c02f3fe2020b4a475ccea047ff");
    assert_eq!(pt.model_ref().revision, "2.5.4");
    let detector = SentenceDetector::new(pt, WhitespaceMode::Unicode);
    assert_eq!(detector.model().unwrap().id, "opennlp:SentenceDetectorME:pt");
    assert_ne!(
        detector.config_hash(),
        SentenceDetector::new(SentenceModel::open(fixture("en-plain.bin")).unwrap(), WhitespaceMode::Legacy)
            .config_hash()
    );
}

#[test]
fn manifest_settings() {
    assert!(with_manifest(MANIFEST, None).is_ok());
    // Boolean.valueOf: any case of "true" is true, anything else false.
    assert!(with_manifest(&MANIFEST.replace("useTokenEnd=true", "useTokenEnd=TRUE"), None).unwrap().use_token_end());
    assert!(!with_manifest(&MANIFEST.replace("useTokenEnd=true", "useTokenEnd=yes"), None).unwrap().use_token_end());
    // Without end characters, or with an empty value, the default set
    // applies whatever the language: the JVM does the same (checked with a
    // Portuguese and a Thai manifest stripped of the property).
    for language in ["pt", "th"] {
        let manifest =
            MANIFEST.replace("eosCharacters=.\\!?\n", "").replace("Language=en", &format!("Language={language}"));
        let m = with_manifest(&manifest, None).unwrap();
        assert_eq!(m.eos(), &EndOfSentence::default(), "{language}");
        assert_eq!(m.language(), language);
    }
    let m = with_manifest(&MANIFEST.replace("eosCharacters=.\\!?", "eosCharacters="), None).unwrap();
    assert_eq!(m.eos(), &EndOfSentence::default());
    // A custom set is taken as given.
    let m = with_manifest(&MANIFEST.replace("eosCharacters=.\\!?", "eosCharacters=\\u3002;"), None).unwrap();
    assert_eq!(m.eos(), &EndOfSentence::from_chars("\u{3002};").unwrap());
    // No factory line is the default factory.
    assert!(
        with_manifest(&MANIFEST.replace("factory=opennlp.tools.sentdetect.SentenceDetectorFactory\n", ""), None)
            .is_ok()
    );
}

#[test]
fn refusals() {
    let e = with_manifest(&MANIFEST.replace("SentenceDetectorME", "TokenizerME"), None).unwrap_err();
    assert!(matches!(&e, ModelError::Component(c) if c == "TokenizerME"), "{e}");
    let e = with_manifest(&MANIFEST.replace("SentenceDetectorFactory", "MyFactory"), None).unwrap_err();
    assert!(matches!(&e, ModelError::Factory(c) if c.ends_with("MyFactory")), "{e}");
    let e = with_manifest(&MANIFEST.replace("useTokenEnd=true\n", ""), None).unwrap_err();
    assert!(matches!(e, ModelError::MissingUseTokenEnd), "{e}");
    let e = with_manifest(&MANIFEST.replace("eosCharacters=.\\!?", "eosCharacters=.\\uD83D\\uDE00"), None).unwrap_err();
    assert!(matches!(e, ModelError::EosCharacters(_)), "{e}");
    assert!(e.to_string().contains("U+D83D"), "{e}");
    // A tokenizer's classifier has the outcomes T and F.
    let tokenizer =
        ModelFile::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("../maxent/tests/fixtures/models/tokenizer.bin"))
            .unwrap();
    let classifier = tokenizer.entry("token.model").unwrap();
    let e = with_manifest(MANIFEST, Some(&classifier)).unwrap_err();
    assert!(matches!(&e, ModelError::Outcomes(o) if o.len() == 2 && o.iter().all(|x| x == "T" || x == "F")), "{e}");
    // A missing classifier is the file error, naming the entry.
    let bytes = zip(&[("manifest.properties", MANIFEST.as_bytes())]);
    let e = SentenceModel::open(ModelFile::open(bytes).unwrap()).unwrap_err();
    assert!(matches!(e, ModelError::File(_)) && e.to_string().contains("sent.model"), "{e}");
}
