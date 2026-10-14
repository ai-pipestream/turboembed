//! The events this crate makes from the golden inputs are the JVM's: the
//! `sentdetect` and `tokenize` runs of `analysis/goldens/data/train/`
//! (`TrainGolden`: a `SentenceDetectorFactory("en", false, null, null)`
//! and a `TokenizerFactory("en", null, false, null)` over the input read
//! as sentence or token samples) and every run of
//! `analysis/goldens/data/train-events/` (`EventGolden`: the factories
//! with a dictionary, Portuguese and Thai end characters, the German
//! alphanumeric pattern with the optimization on, and CoNLL-U input).
//! The `train-events` goldens also hold what the sample stream threw,
//! the manifest and the zip entries of a model file built from the
//! factory, and `Dictionary.serialize` of the dictionaries the generator
//! builds; those are compared with this crate's packaging.

mod common;

use std::path::Path;

use turbo_formats::FormatError;
use turbo_formats::conllu::{ConlluSentenceReader, ConlluTokenReader};
use turbo_formats::native::{SentenceReader, TokenReader};
use turbo_formats::{SentenceSample, TokenSample};
use turbo_goldens::{Golden, data_dir};
use turbo_maxent::Properties;
use turbo_train::Event;
use turbo_train::event::checksum;
use turbo_train::gis::TrainingReport;
use turbo_train::package::MANIFEST_ENTRY;
use turbo_train::params::TrainerType;
use turbo_train_segmentation::sentdetect::package_sentence_model;
use turbo_train_segmentation::tokenize::package_tokenizer_model;
use turbo_train_segmentation::{
    AbbreviationDictionary, SentenceDetectorFactory, Stamp, TokenizerFactory, sd_events, tok_events,
};
use turbo_unicode::WhitespaceMode;

/// The whitespace definition of the OpenNLP the goldens were made with.
const MODE: WhitespaceMode = WhitespaceMode::Unicode;

fn sentence_samples(text: &str) -> Vec<SentenceSample<'_>> {
    SentenceReader::new(text).unwrap().map(|s| s.unwrap()).collect()
}

fn token_samples(text: &str) -> Vec<TokenSample<'_>> {
    TokenReader::new(text, MODE).unwrap().map(|s| s.unwrap()).collect()
}

fn check(path: &Path, g: &Golden, events: &[Event]) {
    let expected = common::events(g);
    assert_eq!(expected, events, "{}: {}", path.display(), common::describe_difference(&expected, events));
    if let Some(l) = common::layer_opt(g, "turbo:event-checksum") {
        assert_eq!(l.u32s("crc32c").unwrap()[0], checksum(events), "{}: checksum", path.display());
    }
}

/// One golden of a `train` run: its source decides the stream.
fn check_train_golden(path: &Path, source: &str) {
    let g = common::read(path);
    assert_eq!(common::config(&g, "source"), Some(source));
    let text = g.input_text(data_dir()).unwrap();
    let events = match source {
        "sentdetect" => {
            let f = SentenceDetectorFactory::new("en", false, None, None).unwrap();
            sd_events(&sentence_samples(&text), &f, MODE)
        }
        "tokenize" => {
            let f = TokenizerFactory::new("en", None, false, None).unwrap();
            tok_events(&token_samples(&text), &f, MODE).unwrap()
        }
        other => panic!("{other}"),
    };
    check(path, &g, &events);
}

#[test]
fn sentence_detector_events_are_the_jvms_over_every_train_golden() {
    let mut n = 0;
    for run in std::fs::read_dir(data_dir().join("train")).unwrap() {
        let run = run.unwrap().path();
        if !run.file_name().unwrap().to_str().unwrap().starts_with("sentdetect-") {
            continue;
        }
        for path in common::golden_files(&run) {
            check_train_golden(&path, "sentdetect");
            n += 1;
        }
    }
    assert!(n >= 100, "{n} goldens");
}

#[test]
fn tokenizer_events_are_the_jvms_over_every_train_golden() {
    let mut n = 0;
    for run in std::fs::read_dir(data_dir().join("train")).unwrap() {
        let run = run.unwrap().path();
        if !run.file_name().unwrap().to_str().unwrap().starts_with("tokenize-") {
            continue;
        }
        for path in common::golden_files(&run) {
            check_train_golden(&path, "tokenize");
            n += 1;
        }
    }
    assert!(n >= 50, "{n} goldens");
}

/// The dictionary a `train-events` run names, from the fixtures.
fn dictionary(g: &Golden) -> Option<AbbreviationDictionary> {
    common::config(g, "abbreviations").map(|file| {
        let xml = std::fs::read(data_dir().join("fixtures").join(file)).unwrap();
        AbbreviationDictionary::read(&xml).unwrap()
    })
}

/// What the JVM's sample stream threw while the events were read: the
/// events made before the exception, its class and its message.
#[derive(Debug, PartialEq, Eq)]
struct StreamError {
    after: u32,
    class: String,
    message: String,
}

impl StreamError {
    fn of(after: usize, e: &FormatError) -> StreamError {
        let class = e.java_class().unwrap_or_else(|| panic!("{e} has no Java class")).to_owned();
        StreamError { after: u32::try_from(after).unwrap(), class, message: e.to_string() }
    }
}

/// The samples a reader gives, and the errors it gives in their place.
/// The generator reads on after an exception, so this collects the error
/// and goes on to the next sample. The events made before the error are
/// counted by the caller, which knows how many events each sample makes,
/// so the errors here carry the sample index and are fixed up after.
fn collect<'a, S>(
    reader: impl Iterator<Item = Result<S, FormatError>>,
    samples: &mut Vec<S>,
    errors: &mut Vec<(usize, FormatError)>,
) where
    S: 'a,
{
    for item in reader {
        match item {
            Ok(s) => samples.push(s),
            Err(e) => errors.push((samples.len(), e)),
        }
    }
}

/// The events of a `train-events` golden from its config records, with
/// the stream errors.
fn event_golden_events(g: &Golden, text: &str) -> (Vec<Event>, Vec<StreamError>) {
    let language = common::config(g, "language").unwrap();
    let conllu = common::config(g, "format") == Some("conllu");
    let mut errors: Vec<(usize, FormatError)> = Vec::new();
    // The events each sample made, so that an error after sample `i` is
    // after `counts[i]` events.
    let mut counts: Vec<usize> = vec![0];
    let events = match common::config(g, "source").unwrap() {
        "sentdetect" => {
            let use_token_end: bool = common::config(g, "useTokenEnd").unwrap().parse().unwrap();
            let eos = common::config(g, "eosCharacters").map(common::lossless_text);
            let f = SentenceDetectorFactory::new(language, use_token_end, dictionary(g), eos.as_deref()).unwrap();
            let mut samples: Vec<SentenceSample<'_>> = Vec::new();
            if conllu {
                let per_sample: usize = common::config(g, "sentencesPerSample").unwrap().parse().unwrap();
                collect(ConlluSentenceReader::new(text, per_sample).unwrap(), &mut samples, &mut errors);
            } else {
                collect(SentenceReader::new(text).unwrap(), &mut samples, &mut errors);
            }
            let mut out = Vec::new();
            for s in &samples {
                out.extend(sd_events([s], &f, MODE));
                counts.push(out.len());
            }
            out
        }
        "tokenize" => {
            let optimization: bool = common::config(g, "useAlphaNumericOptimization").unwrap().parse().unwrap();
            let pattern = common::config(g, "alphaNumericPattern");
            let f = TokenizerFactory::new(language, dictionary(g), optimization, pattern).unwrap();
            let mut samples: Vec<TokenSample<'_>> = Vec::new();
            if conllu {
                collect(ConlluTokenReader::new(text, MODE).unwrap(), &mut samples, &mut errors);
            } else {
                collect(TokenReader::new(text, MODE).unwrap(), &mut samples, &mut errors);
            }
            let mut out = Vec::new();
            for s in &samples {
                out.extend(tok_events([s], &f, MODE).unwrap());
                counts.push(out.len());
            }
            out
        }
        other => panic!("{other}"),
    };
    let errors = errors.iter().map(|(i, e)| StreamError::of(counts[*i], e)).collect();
    (events, errors)
}

fn stream_errors(g: &Golden) -> Vec<StreamError> {
    let l = common::layer(g, "turbo:stream-errors");
    let (after, class, message) = (l.u32s("after").unwrap(), l.strs("class").unwrap(), l.strs("message").unwrap());
    (0..l.len())
        .map(|i| StreamError {
            after: after[i],
            class: class.get(i).to_owned(),
            message: common::lossless_text(message.get(i)),
        })
        .collect()
}

/// The report the generator's placeholder model file carries.
const PLACEHOLDER_REPORT: TrainingReport = TrainingReport { event_hash: Some(1), trainer_type: TrainerType::Event };

/// The model file this crate packages for the golden's factory, with no
/// model bytes: the manifest, the entry order and the dictionary entry
/// are the factory's, not the model's.
fn package(g: &Golden) -> turbo_train::ModelPackage {
    let language = common::config(g, "language").unwrap();
    let stamp = Stamp::at(1_700_000_000_000);
    match common::config(g, "source").unwrap() {
        "sentdetect" => {
            let use_token_end: bool = common::config(g, "useTokenEnd").unwrap().parse().unwrap();
            let eos = common::config(g, "eosCharacters").map(common::lossless_text);
            let f = SentenceDetectorFactory::new(language, use_token_end, dictionary(g), eos.as_deref()).unwrap();
            package_sentence_model(&f, &PLACEHOLDER_REPORT, Vec::new(), stamp).unwrap()
        }
        "tokenize" => {
            let optimization: bool = common::config(g, "useAlphaNumericOptimization").unwrap().parse().unwrap();
            let pattern = common::config(g, "alphaNumericPattern");
            let f = TokenizerFactory::new(language, dictionary(g), optimization, pattern).unwrap();
            package_tokenizer_model(&f, &PLACEHOLDER_REPORT, Vec::new(), stamp).unwrap()
        }
        other => panic!("{other}"),
    }
}

/// The manifest entries of the package, sorted by key, without
/// `Timestamp`, as the golden lists them.
fn manifest_entries(package: &turbo_train::ModelPackage) -> Vec<(String, String)> {
    let text = package.manifest().to_properties();
    let properties = Properties::load(text.as_bytes()).unwrap();
    let mut entries: Vec<(String, String)> =
        properties.iter().filter(|(k, _)| *k != "Timestamp").map(|(k, v)| (k.to_owned(), v.to_owned())).collect();
    entries.sort();
    entries
}

fn check_package(path: &Path, g: &Golden) {
    let package = package(g);
    let l = common::layer(g, "turbo:manifest");
    let (keys, values) = (l.strs("key").unwrap(), l.strs("value").unwrap());
    let expected: Vec<(String, String)> =
        (0..l.len()).map(|i| (keys.get(i).to_owned(), common::lossless_text(values.get(i)))).collect();
    assert_eq!(expected, manifest_entries(&package), "{}: manifest", path.display());

    // The entry names in order, and the lengths of the manifest (the
    // generator's JVM dates in UTC, as `Stamp::at` does, and both
    // timestamps have thirteen digits) and of the dictionary. The model
    // entry's length is the placeholder model's.
    let l = common::layer(g, "turbo:package-entries");
    let (names, lengths) = (l.strs("name").unwrap(), l.u32s("length").unwrap());
    let expected_names: Vec<&str> = (0..l.len()).map(|i| names.get(i)).collect();
    let mut actual_names = vec![MANIFEST_ENTRY];
    actual_names.extend(package.entries().map(|(n, _)| n));
    assert_eq!(expected_names, actual_names, "{}: entry order", path.display());
    let manifest_len = package.manifest().to_properties().len();
    assert_eq!(lengths[0] as usize, manifest_len, "{}: manifest length", path.display());
    for (i, name) in expected_names.iter().enumerate() {
        if *name == "abbreviations.dictionary" {
            let (_, bytes) = package.entries().find(|(n, _)| n == name).unwrap();
            assert_eq!(lengths[i] as usize, bytes.len(), "{}: dictionary length", path.display());
        }
    }

    let dictionary = common::layer_opt(g, "turbo:dictionary");
    let entry = package.entries().find(|(n, _)| *n == "abbreviations.dictionary");
    match (dictionary, entry) {
        (None, None) => {}
        (Some(l), Some((_, bytes))) => {
            let expected = common::lossless_text(l.strs("xml").unwrap().get(0));
            assert_eq!(expected, String::from_utf8(bytes.to_vec()).unwrap(), "{}: dictionary", path.display());
        }
        (l, e) => panic!("{}: dictionary layer {} and entry {}", path.display(), l.is_some(), e.is_some()),
    }
}

fn single(tokens: &[&str]) -> Vec<Vec<String>> {
    tokens.iter().map(|t| vec![t.to_string()]).collect()
}

/// The dictionaries `EventGolden.dictionaryCases` builds, by name, built
/// here the same way.
fn dictionary_case(name: &str, g: &Golden) -> AbbreviationDictionary {
    match name {
        "english" => dictionary(g).unwrap(),
        "special-characters" => {
            AbbreviationDictionary::new(false, single(&["a&b<c>d\"e'f", "cr\rhere", "tab\there", "\u{1F600}."]))
                .unwrap()
        }
        "empty" => AbbreviationDictionary::new(false, Vec::new()).unwrap(),
        "case-sensitive" => AbbreviationDictionary::new(true, single(&["Dr.", "MR.", "u.s."])).unwrap(),
        "forty" => {
            AbbreviationDictionary::new(false, (1..=40).map(|i| vec![format!("w{i}.")]).collect::<Vec<_>>()).unwrap()
        }
        "multi-token" => AbbreviationDictionary::new(
            false,
            vec![
                vec!["et".to_string(), "al.".to_string()],
                vec!["e.".to_string(), "g.".to_string()],
                vec!["a".to_string(), "b".to_string(), "c".to_string()],
                vec!["Dr.".to_string()],
            ],
        )
        .unwrap(),
        "collapsed" => AbbreviationDictionary::new(false, single(&["Dr.", "DR.", "dr.", "Mr."])).unwrap(),
        other => panic!("{other}"),
    }
}

fn check_dictionary_cases(path: &Path, g: &Golden) -> usize {
    let Some(l) = common::layer_opt(g, "turbo:dictionary-cases") else {
        return 0;
    };
    let (names, xmls) = (l.strs("name").unwrap(), l.strs("xml").unwrap());
    for i in 0..l.len() {
        let name = names.get(i);
        let expected = common::lossless_text(xmls.get(i));
        let actual = String::from_utf8(dictionary_case(name, g).serialize().unwrap()).unwrap();
        assert_eq!(expected, actual, "{}: dictionary case {name}", path.display());
    }
    l.len()
}

#[test]
fn events_with_every_factory_option_are_the_jvms_over_every_train_events_golden() {
    let dir = data_dir().join("train-events");
    let files = common::golden_files(&dir);
    assert!(files.len() >= 100, "{} goldens under {}", files.len(), dir.display());
    let mut runs = std::collections::BTreeSet::new();
    let (mut with_errors, mut cases) = (0, 0);
    for path in &files {
        let g = common::read(path);
        runs.insert(common::config(&g, "run").unwrap().to_owned());
        assert!(g.header.error.is_none(), "{}: the generator failed", path.display());
        let text = g.input_text(data_dir()).unwrap();
        let (events, errors) = event_golden_events(&g, &text);
        check(path, &g, &events);
        let expected = stream_errors(&g);
        assert_eq!(expected, errors, "{}: stream errors", path.display());
        with_errors += usize::from(!errors.is_empty());
        check_package(path, &g);
        cases += check_dictionary_cases(path, &g);
    }
    let expected: std::collections::BTreeSet<String> = [
        "sentdetect-dictionary-pt",
        "sentdetect-th",
        "sentdetect-conllu",
        "tokenize-dictionary-alnum-de",
        "tokenize-conllu",
    ]
    .into_iter()
    .map(String::from)
    .collect();
    assert_eq!(runs, expected);
    assert_eq!(with_errors, 3, "goldens with stream errors");
    assert_eq!(cases, 7, "dictionary cases");
}
