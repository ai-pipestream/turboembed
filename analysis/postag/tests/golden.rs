//! The tagger against `POSTaggerME` on the JVM (analysis/goldens/data/postag).
//!
//! Each golden holds the sentences and tokens the JVM's own sentence
//! detector and tokenizer cut, and the tag and probability `tag` and
//! `probs` gave each token. The tokens are loaded as the golden layer and
//! the tagger runs on them, so what is compared is the tagging alone:
//! tags exactly and probabilities bit for bit, since the goldens hold the
//! bits of Java's doubles and the classifier's arithmetic matches the
//! JVM's.
//!
//! The runs are the published English model as it is, asked for Penn
//! tags, and rebuilt with beams of 1 and 10; the published German model;
//! and the fixture model with a case-insensitive tag dictionary. For
//! every run the dumps are checked too: every context the beam search
//! asked the model about, feature by feature in the order the JVM asked,
//! with the probability of every outcome.
//!
//! One golden per English set, `en/emoji-split`, has a token that is half
//! of a surrogate pair, which the JVM tagged as such. Those tokens cannot
//! be loaded onto the Rust text (no layer type loads the `turbo:utf16`
//! layer), so that golden is counted and skipped.

mod common;

use std::path::Path;
use std::sync::Arc;

use common::{Case, Setting};
use turbo_analysis::{Analyzer, Bound, Key};
use turbo_analysis_layers::{PosTags, Tokens};
use turbo_goldens::{GOLDEN_KEY, Tolerance, Unit, golden_key, load_document, matches_golden_key};
use turbo_postag::{PosModel, PosTagFormat, PosTagger};
use turbo_sequence::{BeamSearch, ContextGenerator, Features};
use turbo_unicode::modified_utf8::push_modified_utf8;
use turbo_unicode::utf16::byte_to_utf16;

const ME: &str = "opennlp.tools.postag.POSTaggerME";
const UTF16_TOKENS: &str = "turbo:utf16<opennlp:tokens>";

fn open(path: &Path, setting: Setting) -> Arc<PosModel> {
    let model = PosModel::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let model = match setting {
        Setting::Default => model,
        Setting::Format(f) => model.with_format(f),
        Setting::BeamSize(n) => model.with_beam_size(n).unwrap(),
    };
    Arc::new(model)
}

/// What one run's goldens and dumps held.
#[derive(Debug, Default, PartialEq)]
struct Counts {
    /// Goldens tagged, per input set.
    files: Vec<(String, usize)>,
    /// Goldens with a token cutting a surrogate pair, not tagged.
    skipped: usize,
    /// Tokens tagged.
    tokens: usize,
    /// Contexts the search asked the model about.
    points: usize,
    /// Tokens the tag dictionary lists, whose tag had to be one of its.
    restricted: usize,
}

/// Passes contexts through to the model's generator and keeps each with
/// the index of the word it was about.
struct Recorder<C> {
    inner: C,
    calls: Vec<(usize, Vec<Vec<u8>>)>,
}

impl<T, C: ContextGenerator<T>> ContextGenerator<T> for Recorder<C> {
    fn start_sequence(&mut self) {
        self.inner.start_sequence();
    }

    fn context(&mut self, index: usize, sequence: &[T], prior: &[&str], out: &mut Features) {
        self.inner.context(index, sequence, prior, out);
        self.calls.push((index, out.iter().map(<[u8]>::to_vec).collect()));
    }
}

/// Runs the tagger over every golden of `postag/<name>` and checks every
/// dump against what the search asked the model.
fn check_run(name: &str, model: &Arc<PosModel>, model_file: &Path, setting: Setting) -> Counts {
    let mut counts = Counts::default();
    for case in common::run(name) {
        let what = case.path.display().to_string();
        case.assert_annotator(ME);
        case.assert_models(model_file);
        assert_eq!(case.setting(), setting, "{what}");
        if case.golden.layer(UTF16_TOKENS).is_some() {
            assert_eq!((case.set(), case.id()), ("en", "emoji-split"), "{what}");
            counts.skipped += 1;
            continue;
        }
        let mut doc = load_document(&case.golden, &case.text).unwrap_or_else(|e| panic!("{what}: {e}"));
        let tagger = PosTagger::new(model.clone());
        let analyzer = Analyzer::builder()
            .given(golden_key::<Tokens>())
            .step(Bound::new(tagger).read(1, GOLDEN_KEY))
            .build()
            .unwrap();
        assert_eq!(analyzer.run(std::slice::from_mut(&mut doc)), [Ok(())], "{what}");
        let tags = Key::<PosTags>::DEFAULT;
        if let Err(m) = matches_golden_key(&doc, &tags, &case.golden, Tolerance::exact()) {
            panic!("{what}: {m}");
        }
        let got = doc.get_key(&tags).unwrap();
        assert_eq!(got.tagset.name, model.tagset().name, "{what}");
        counts.tokens += got.tags.len();
        counts.restricted += check_dictionary(&case, &doc, model);
        counts.points += check_dump(&case, &doc, model);
        match counts.files.iter_mut().find(|(set, _)| set == case.set()) {
            Some((_, k)) => *k += 1,
            None => counts.files.push((case.set().to_string(), 1)),
        }
    }
    eprintln!("postag/{name}: {counts:?}");
    counts
}

/// The sentences of the golden tokens as word slices with each token's
/// byte start, in text order.
fn sentences(doc: &turbo_analysis::Document) -> Vec<Vec<(&str, u32)>> {
    let tokens = doc.get_key(&golden_key::<Tokens>()).unwrap();
    let text = doc.text(tokens.spans.view());
    let (starts, ends) = tokens.spans.columns();
    let sentence = tokens.sentence.as_ref().expect("tokens loaded with their sentences");
    let mut out: Vec<Vec<(&str, u32)>> = Vec::new();
    for i in 0..starts.len() {
        if i == 0 || sentence[i] != sentence[i - 1] {
            out.push(Vec::new());
        }
        out.last_mut().unwrap().push((&text[starts[i] as usize..ends[i] as usize], starts[i]));
    }
    out
}

/// Every token the tag dictionary lists got one of the dictionary's
/// tags, as `POSTaggerME`'s validator requires; returns how many.
fn check_dictionary(case: &Case, doc: &turbo_analysis::Document, model: &PosModel) -> usize {
    let Some(dictionary) = model.tag_dictionary() else {
        return 0;
    };
    let tags = doc.get_key(&Key::<PosTags>::DEFAULT).unwrap();
    let mut restricted = 0;
    let mut i = 0;
    for sentence in sentences(doc) {
        for (word, _) in sentence {
            if let Some(allowed) = dictionary.tags(word) {
                let got = tags.tagset.label(tags.tags[i]);
                assert!(
                    allowed.iter().any(|t| model.label_of(t) == got),
                    "{}: {word:?} tagged {got}",
                    case.path.display()
                );
                restricted += 1;
            }
            i += 1;
        }
    }
    restricted
}

/// Every point of the dump against the contexts the search asked for
/// here, and returns their number.
fn check_dump(case: &Case, doc: &turbo_analysis::Document, model: &PosModel) -> usize {
    let dump = case.dump();
    let what = case.path.display().to_string();
    let maxent = model.maxent();
    let labels: Vec<&str> = (0..maxent.num_outcomes()).map(|i| maxent.outcome(i)).collect();
    let mut recorder = Recorder { inner: model.new_context().unwrap(), calls: Vec::new() };
    let validator = model.validator();
    let mut seen: Vec<(u32, Vec<Vec<u8>>)> = Vec::new();
    for sentence in sentences(doc) {
        let words: Vec<&str> = sentence.iter().map(|&(w, _)| w).collect();
        let search = BeamSearch::new(model.beam_size(), maxent);
        assert!(search.best_sequence(&words, &mut recorder, &validator).is_some(), "{what}");
        for (index, features) in recorder.calls.drain(..) {
            seen.push((sentence[index].1, features));
        }
    }
    assert_eq!(seen.len(), dump.points.len(), "{what}: contexts asked for");
    let mut probs = vec![0.0; labels.len()];
    for ((start, features), p) in seen.iter().zip(&dump.points) {
        assert_eq!((p.position, p.unit), (*start, Unit::Start), "{what}");
        assert!(byte_to_utf16(&case.text, *start as usize).is_some(), "{what}");
        let want: Vec<Vec<u8>> = p
            .contexts
            .iter()
            .map(|f| {
                let mut bytes = Vec::new();
                for &u in f {
                    push_modified_utf8(&mut bytes, u32::from(u));
                }
                bytes
            })
            .collect();
        let show = |f: &[Vec<u8>]| f.iter().map(|x| String::from_utf8_lossy(x).into_owned()).collect::<Vec<_>>();
        assert_eq!(features, &want, "{what} at {start}: {:?} against {:?}", show(features), show(&want));
        assert!(p.values.is_none() && p.acceptable.is_none(), "{what} at {start}");
        let got: Vec<&str> = p.outcomes.iter().map(|o| o.label.as_str()).collect();
        assert_eq!(got, labels, "{what} at {start}: outcomes");
        maxent.eval_features(features.iter().map(Vec::as_slice), &mut probs);
        for (o, &got) in p.outcomes.iter().zip(&probs) {
            assert_eq!(got.to_bits(), o.prob.to_bits(), "{what} at {start}: P({}) {got} against {}", o.label, o.prob);
        }
    }
    seen.len()
}

/// Runs a published model's goldens, or returns `None` when the model is
/// not downloaded (`common::published_model`).
fn check_published(language: &str, model_file: &str, setting: Setting) -> Option<Counts> {
    let path = common::published_model(model_file)?;
    let model = open(&path, setting);
    assert_eq!(model.language(), language);
    assert!(model.tag_dictionary().is_none(), "no published model has a tag dictionary");
    let base = model_file.strip_suffix(".bin").unwrap();
    let counts = check_run(&format!("{base}{}", setting.suffix()), &model, &path, setting);
    assert_eq!(counts.restricted, 0);
    Some(counts)
}

const EN: &str = "opennlp-en-ud-ewt-pos-1.3-2.5.4.bin";
const DE: &str = "opennlp-de-ud-gsd-pos-1.3-2.5.4.bin";

#[test]
fn english() {
    let Some(counts) = check_published("en", EN, Setting::Default) else { return };
    assert_eq!(counts.files, [("edge".into(), 34), ("en".into(), 10), ("multi".into(), 11)]);
    assert_eq!(counts.skipped, 1);
    assert_eq!((counts.tokens, counts.points), (1304, 3584), "{counts:?}");
}

#[test]
fn english_penn() {
    let Some(counts) = check_published("en", EN, Setting::Format(PosTagFormat::Penn)) else { return };
    assert_eq!(counts.files, [("en".into(), 10)]);
    assert_eq!(counts.skipped, 1);
    // The same searches as in the UD run over the same set: only the
    // labels are converted.
    assert_eq!((counts.tokens, counts.points), (496, 1380), "{counts:?}");
}

#[test]
fn english_beam_1() {
    let Some(counts) = check_published("en", EN, Setting::BeamSize(1)) else { return };
    assert_eq!(counts.files, [("en".into(), 10)]);
    // A beam of one asks once per token.
    assert_eq!((counts.tokens, counts.points), (496, 496), "{counts:?}");
}

#[test]
fn english_beam_10() {
    let Some(counts) = check_published("en", EN, Setting::BeamSize(10)) else { return };
    assert_eq!(counts.files, [("en".into(), 10)]);
    // More paths kept than the beam of three asks about (1380 over the
    // same set), fewer than ten per token: a sentence's first token is
    // asked about once, and a short sentence has fewer sequences than the
    // beam holds.
    assert_eq!((counts.tokens, counts.points), (496, 4474), "{counts:?}");
}

#[test]
fn german() {
    let Some(counts) = check_published("de", DE, Setting::Default) else { return };
    assert_eq!(counts.files, [("edge".into(), 34)]);
    assert_eq!(counts.skipped, 0);
    assert_eq!((counts.tokens, counts.points), (465, 1271), "{counts:?}");
}

#[test]
fn fixture_with_tag_dictionary() {
    let path = common::fixtures_dir().join("pos-fixture.bin");
    let model = open(&path, Setting::Default);
    assert!(model.tag_dictionary().is_some());
    assert_eq!(model.guessed_format(), PosTagFormat::Penn);
    // UD is asked for, so the Penn outcomes are converted.
    assert_eq!(model.tag_format(), PosTagFormat::Ud);
    let counts = check_run("pos-fixture", &model, &path, Setting::Default);
    assert_eq!(counts.files, [("edge".into(), 34), ("en".into(), 10)]);
    assert_eq!(counts.skipped, 1);
    assert_eq!((counts.tokens, counts.points, counts.restricted), (972, 2648, 42), "{counts:?}");
}

/// The model the Rust trainer wrote from `inputs/tagged/wordtag.txt` with
/// the same tag dictionary (`analysis/train-components/tests/fixtures.rs`),
/// which the JVM loaded and tagged with like the fixture above.
#[test]
fn rust_fixture_with_tag_dictionary() {
    let path = common::fixtures_dir().join("pos-fixture-rust.bin");
    let model = open(&path, Setting::Default);
    assert!(model.tag_dictionary().is_some());
    assert_eq!(model.guessed_format(), PosTagFormat::Penn);
    assert_eq!(model.tag_format(), PosTagFormat::Ud);
    let counts = check_run("pos-fixture-rust", &model, &path, Setting::Default);
    assert_eq!(counts.files, [("edge".into(), 34), ("en".into(), 10)]);
    assert_eq!(counts.skipped, 1);
    assert_eq!((counts.tokens, counts.points, counts.restricted), (972, 2648, 42), "{counts:?}");
}
