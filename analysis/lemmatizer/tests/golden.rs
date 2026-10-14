//! The lemmatizers against OpenNLP on the JVM
//! (analysis/goldens/data/lemmatize and lemmatize-dictionary).
//!
//! Each golden holds the sentences, tokens and part-of-speech tags the
//! JVM's own sentence detector, tokenizer and tagger made, and the lemma
//! each lemmatizer gave each token (with, for the maxent lemmatizer, the
//! probability `probs` reports). The tokens and tags are loaded as golden
//! layers and the lemmatizer runs on them, so what is compared is the
//! lemmatization alone: lemmas exactly and probabilities bit for bit.
//!
//! The runs are the published English and German maxent lemmatizers and
//! the dictionary lemmatizer with the fixture dictionary over the English
//! pipeline. For the maxent runs the dumps are checked too: every context
//! the beam search asked the model about, feature by feature in the JVM's
//! order, with the probability of every lemma class.
//!
//! `en/emoji-split` has a token that is half of a surrogate pair, which
//! the JVM lemmatized as such; it cannot be loaded onto the Rust text, so
//! that golden is counted and skipped.

mod common;

use std::path::Path;
use std::sync::Arc;

use common::Case;
use turbo_analysis::{Analyzer, Bound, Document, Key};
use turbo_analysis_layers::{Confidence, Lemmas, PosTags, Tokens};
use turbo_goldens::{GOLDEN_KEY, Tolerance, Unit, golden_key, load_document, matches_golden_key};
use turbo_lemmatizer::{DictionaryLemmatizer, LemmaContext, LemmatizerModel, MaxentLemmatizer, Tagged};
use turbo_sequence::{AcceptAll, BeamSearch, ContextGenerator, Features};
use turbo_unicode::modified_utf8::push_modified_utf8;

const ME: &str = "opennlp.tools.lemmatizer.LemmatizerME";
const DICTIONARY: &str = "opennlp.tools.lemmatizer.DictionaryLemmatizer";
const UTF16_TOKENS: &str = "turbo:utf16<opennlp:tokens>";

/// What one run's goldens and dumps held.
#[derive(Debug, Default, PartialEq)]
struct Counts {
    /// Goldens lemmatized, per input set.
    files: Vec<(String, usize)>,
    /// Goldens with a token cutting a surrogate pair, not lemmatized:
    /// their lemma layer holds half of the pair and does not read.
    skipped: usize,
    /// Tokens lemmatized.
    tokens: usize,
    /// Contexts the search asked the model about.
    points: usize,
    /// Tokens whose lemma differs from the lowercased token (a class
    /// other than `O`), or, for the dictionary, tokens it listed.
    changed: usize,
}

/// Passes contexts through to the generator and keeps each with the
/// index of the token it was about.
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

/// The golden's sentences as tagged words with each token's byte start.
fn sentences(doc: &Document) -> Vec<Vec<(Tagged<'_>, u32)>> {
    let tokens = doc.get_key(&golden_key::<Tokens>()).unwrap();
    let tags = doc.get_key(&golden_key::<PosTags>()).unwrap();
    let text = doc.text(tokens.spans.view());
    let (starts, ends) = tokens.spans.columns();
    assert_eq!(tags.tags.len(), starts.len());
    let sentence = tokens.sentence.as_ref().expect("tokens loaded with their sentences");
    let mut out: Vec<Vec<(Tagged<'_>, u32)>> = Vec::new();
    for i in 0..starts.len() {
        if i == 0 || sentence[i] != sentence[i - 1] {
            out.push(Vec::new());
        }
        let word = &text[starts[i] as usize..ends[i] as usize];
        out.last_mut().unwrap().push(((word, tags.tagset.label(tags.tags[i])), starts[i]));
    }
    out
}

/// Loads the golden's layers and runs `lemmatizer` on its tokens and tags,
/// comparing the default `Lemmas` (and, when `confidence`, the default
/// `Confidence<Lemmas>`) with the golden's. Returns the document.
fn lemmatize<A>(case: &Case, lemmatizer: A, confidence: bool) -> Document
where
    A: turbo_analysis::Annotator<Reads = (turbo_analysis::View, Tokens, PosTags)> + 'static,
{
    let what = case.path.display().to_string();
    let mut doc = load_document(&case.golden, &case.text).unwrap_or_else(|e| panic!("{what}: {e}"));
    let analyzer = Analyzer::builder()
        .given(golden_key::<Tokens>())
        .given(golden_key::<PosTags>())
        .step(Bound::new(lemmatizer).read(1, GOLDEN_KEY).read(2, GOLDEN_KEY))
        .build()
        .unwrap();
    assert_eq!(analyzer.run(std::slice::from_mut(&mut doc)), [Ok(())], "{what}");
    if let Err(m) = matches_golden_key(&doc, &Key::<Lemmas>::DEFAULT, &case.golden, Tolerance::exact()) {
        panic!("{what}: {m}");
    }
    if confidence
        && let Err(m) = matches_golden_key(&doc, &Key::<Confidence<Lemmas>>::DEFAULT, &case.golden, Tolerance::exact())
    {
        panic!("{what}: {m}");
    }
    doc
}

fn count(counts: &mut Counts, case: &Case) {
    match counts.files.iter_mut().find(|(set, _)| set == case.set()) {
        Some((_, k)) => *k += 1,
        None => counts.files.push((case.set().to_string(), 1)),
    }
}

/// Runs the maxent lemmatizer over every golden of `lemmatize/<name>` and
/// checks every dump against what the search asked the model.
fn check_maxent_run(name: &str, model: &Arc<LemmatizerModel>, model_file: &Path) -> Counts {
    let (cases, unreadable) = common::run("lemmatize", name);
    let mut counts = Counts { skipped: unreadable, ..Counts::default() };
    for case in cases {
        let what = case.path.display().to_string();
        case.assert_annotator(ME);
        case.assert_models(model_file);
        case.assert_no_config();
        assert!(case.golden.layer(UTF16_TOKENS).is_none(), "{what}: a token cut inside a character");
        let doc = lemmatize(&case, MaxentLemmatizer::new(model.clone()), true);
        let lemmas = doc.get_key(&Key::<Lemmas>::DEFAULT).unwrap();
        counts.tokens += lemmas.lemma.len();
        let (points, changed) = check_dump(&case, &doc, model);
        counts.points += points;
        counts.changed += changed;
        count(&mut counts, &case);
    }
    eprintln!("lemmatize/{name}: {counts:?}");
    counts
}

/// Every point of the dump against the contexts the search asked for
/// here; returns their number and how many tokens got a class other than
/// `O`.
fn check_dump(case: &Case, doc: &Document, model: &LemmatizerModel) -> (usize, usize) {
    let dump = case.dump();
    let what = case.path.display().to_string();
    let maxent = model.maxent();
    let labels: Vec<&str> = (0..maxent.num_outcomes()).map(|i| maxent.outcome(i)).collect();
    let mut recorder = Recorder { inner: LemmaContext, calls: Vec::new() };
    let mut seen: Vec<(u32, Vec<Vec<u8>>)> = Vec::new();
    let mut changed = 0;
    for sentence in sentences(doc) {
        let words: Vec<Tagged<'_>> = sentence.iter().map(|&(w, _)| w).collect();
        let search = BeamSearch::new(model.beam_size(), maxent);
        let best = search.best_sequence(&words, &mut recorder, &AcceptAll).unwrap_or_else(|| panic!("{what}"));
        changed += best.outcomes().iter().filter(|&&o| o != "O").count();
        for (index, features) in recorder.calls.drain(..) {
            seen.push((sentence[index].1, features));
        }
    }
    assert_eq!(seen.len(), dump.points.len(), "{what}: contexts asked for");
    let mut probs = vec![0.0; labels.len()];
    for ((start, features), p) in seen.iter().zip(&dump.points) {
        assert_eq!((p.position, p.unit), (*start, Unit::Start), "{what}");
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
    (seen.len(), changed)
}

#[test]
fn english() {
    let Some(path) = common::published_model("opennlp-en-ud-ewt-lemmas-1.3-2.5.4.bin") else { return };
    let model = Arc::new(LemmatizerModel::read(&path).unwrap());
    assert_eq!(model.language(), "en");
    let counts = check_maxent_run("opennlp-en-ud-ewt-lemmas-1.3-2.5.4", &model, &path);
    assert_eq!(counts.files, [("edge".into(), 34), ("en".into(), 10), ("multi".into(), 11)]);
    assert_eq!(counts.skipped, 1);
    // The same searches as the tagger's over these tokens: one per path
    // of a beam of three.
    assert_eq!((counts.tokens, counts.points, counts.changed), (1304, 3584, 181), "{counts:?}");
}

/// The model the Rust trainer wrote from `inputs/tagged/lemmas.txt`
/// (`analysis/train-components/tests/fixtures.rs`), which the JVM loaded
/// and lemmatized with over the English pipeline's tags.
#[test]
fn rust_fixture() {
    let path = common::fixtures_dir().join("lemma-fixture-rust.bin");
    let model = Arc::new(LemmatizerModel::read(&path).unwrap());
    assert_eq!(model.language(), "en");
    let counts = check_maxent_run("lemma-fixture-rust", &model, &path);
    assert_eq!(counts.files, [("edge".into(), 34), ("en".into(), 10)]);
    assert_eq!(counts.skipped, 1);
    assert_eq!((counts.tokens, counts.points, counts.changed), (972, 2672, 7), "{counts:?}");
}

#[test]
fn german() {
    let Some(path) = common::published_model("opennlp-de-ud-gsd-lemmas-1.3-2.5.4.bin") else { return };
    let model = Arc::new(LemmatizerModel::read(&path).unwrap());
    assert_eq!(model.language(), "de");
    let counts = check_maxent_run("opennlp-de-ud-gsd-lemmas-1.3-2.5.4", &model, &path);
    assert_eq!(counts.files, [("edge".into(), 34)]);
    assert_eq!(counts.skipped, 0);
    assert_eq!((counts.tokens, counts.points, counts.changed), (465, 1271, 45), "{counts:?}");
}

#[test]
fn dictionary() {
    let path = common::fixtures_dir().join("lemmatizer-dictionary.txt");
    let dictionary = Arc::new(DictionaryLemmatizer::read(&path, "fixture").unwrap());
    assert_eq!(dictionary.len(), 47);
    let (cases, unreadable) = common::run("lemmatize-dictionary", "lemmatizer-dictionary");
    // The dictionary has no entry for half an emoji, so its lemma is O and
    // the golden reads.
    assert_eq!(unreadable, 0);
    let mut counts = Counts::default();
    for case in cases {
        let what = case.path.display().to_string();
        case.assert_annotator(DICTIONARY);
        case.assert_models(&path);
        case.assert_no_config();
        assert!(case.golden.layer("turbo:confidence<opennlp:lemmas>").is_none(), "{what}");
        if case.golden.layer(UTF16_TOKENS).is_some() {
            assert_eq!((case.set(), case.id()), ("en", "emoji-split"), "{what}");
            counts.skipped += 1;
            continue;
        }
        let doc = lemmatize(&case, Shared(dictionary.clone()), false);
        let lemmas = doc.get_key(&Key::<Lemmas>::DEFAULT).unwrap();
        counts.tokens += lemmas.lemma.len();
        counts.changed += lemmas.lemma.iter().filter(|&l| l != "O").count();
        count(&mut counts, &case);
    }
    eprintln!("lemmatize-dictionary: {counts:?}");
    assert_eq!(counts.files, [("edge".into(), 34), ("en".into(), 10), ("multi".into(), 11)]);
    assert_eq!(counts.skipped, 1);
    assert_eq!((counts.tokens, counts.changed), (1304, 130), "{counts:?}");
}

/// A dictionary lemmatizer shared between runs, as an annotator.
struct Shared(Arc<DictionaryLemmatizer>);

impl turbo_analysis::Annotator for Shared {
    const ID: &'static str = DictionaryLemmatizer::ID;
    type Reads = (turbo_analysis::View, Tokens, PosTags);
    type Writes = (Lemmas,);

    fn annotate(&self, refs: turbo_analysis::Refs<'_, Self>) -> Result<(Lemmas,), turbo_analysis::Error> {
        self.0.annotate(refs)
    }

    fn model(&self) -> Option<turbo_analysis::ModelRef> {
        turbo_analysis::Annotator::model(&*self.0)
    }
}
