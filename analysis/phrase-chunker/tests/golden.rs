//! The chunker against OpenNLP on the JVM (analysis/goldens/data/chunk).
//!
//! Each golden holds the sentences, tokens and Penn Treebank part-of-speech
//! tags the JVM's own sentence detector, tokenizer and tagger made, the
//! chunk tag `ChunkerME.chunk` gave each token with the probability `probs`
//! reports, and the phrases `chunkAsSpans` makes of them. The tokens and
//! tags are loaded as golden layers and the chunker runs on them, so what
//! is compared is the chunking alone: tags and phrases exactly and
//! probabilities bit for bit.
//!
//! The runs are the fixture chunker model as it is and rebuilt with a beam
//! of one. The dumps are checked too: every context the beam search asked
//! the model about, feature by feature in the JVM's order, with the
//! probability of every chunk tag.
//!
//! `en/emoji-split` has a token that is half of a surrogate pair, which
//! cannot be loaded onto the Rust text; that golden is counted and skipped.

mod common;

use std::path::Path;
use std::sync::Arc;

use common::Case;
use turbo_analysis::{Analyzer, Bound, Document, Key};
use turbo_analysis_layers::{ChunkTags, PhraseChunks, PosTags, Tokens};
use turbo_goldens::{GOLDEN_KEY, Tolerance, Unit, golden_key, load_document, matches_golden_key};
use turbo_phrase_chunker::{ChunkerModel, PhraseChunker, TokenTag};
use turbo_sequence::chunker::{ChunkerValidator, DefaultChunkerContext};
use turbo_sequence::{BeamSearch, ContextGenerator, Features};
use turbo_unicode::modified_utf8::push_modified_utf8;

const UTF16_TOKENS: &str = "turbo:utf16<opennlp:tokens>";

/// What one run's goldens and dumps held.
#[derive(Debug, Default, PartialEq)]
struct Counts {
    /// Goldens chunked, per input set.
    files: Vec<(String, usize)>,
    /// Goldens with a token cutting a surrogate pair, not chunked.
    skipped: usize,
    /// Tokens chunked.
    tokens: usize,
    /// Contexts the search asked the model about.
    points: usize,
    /// Phrases found.
    phrases: usize,
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
fn sentences(doc: &Document) -> Vec<Vec<(TokenTag<'_>, u32)>> {
    let tokens = doc.get_key(&golden_key::<Tokens>()).unwrap();
    let tags = doc.get_key(&golden_key::<PosTags>()).unwrap();
    let text = doc.text(tokens.spans.view());
    let (starts, ends) = tokens.spans.columns();
    assert_eq!(tags.tags.len(), starts.len());
    let sentence = tokens.sentence.as_ref().expect("tokens loaded with their sentences");
    let mut out: Vec<Vec<(TokenTag<'_>, u32)>> = Vec::new();
    for i in 0..starts.len() {
        if i == 0 || sentence[i] != sentence[i - 1] {
            out.push(Vec::new());
        }
        let token = &text[starts[i] as usize..ends[i] as usize];
        out.last_mut().unwrap().push((TokenTag { token, tag: tags.tagset.label(tags.tags[i]) }, starts[i]));
    }
    out
}

/// Loads the golden's layers and runs the chunker on its tokens and tags,
/// comparing the default `ChunkTags` and `PhraseChunks` with the golden's.
/// Returns the document.
fn chunk(case: &Case, model: &Arc<ChunkerModel>) -> Document {
    let what = case.path.display().to_string();
    let mut doc = load_document(&case.golden, &case.text).unwrap_or_else(|e| panic!("{what}: {e}"));
    let analyzer = Analyzer::builder()
        .given(golden_key::<Tokens>())
        .given(golden_key::<PosTags>())
        .step(Bound::new(PhraseChunker::new(model.clone())).read(1, GOLDEN_KEY).read(2, GOLDEN_KEY))
        .build()
        .unwrap();
    assert_eq!(analyzer.run(std::slice::from_mut(&mut doc)), [Ok(())], "{what}");
    if let Err(m) = matches_golden_key(&doc, &Key::<ChunkTags>::DEFAULT, &case.golden, Tolerance::exact()) {
        panic!("{what}: {m}");
    }
    if let Err(m) = matches_golden_key(&doc, &Key::<PhraseChunks>::DEFAULT, &case.golden, Tolerance::exact()) {
        panic!("{what}: {m}");
    }
    doc
}

/// Runs the chunker over every golden of `chunk/<name>` and checks every
/// dump against what the search asked the model.
fn check_run(name: &str, model: &Arc<ChunkerModel>, model_file: &Path, beam: Option<usize>) -> Counts {
    let mut counts = Counts::default();
    for case in common::run(name) {
        let what = case.path.display().to_string();
        case.assert_annotator();
        case.assert_models(model_file);
        assert_eq!(case.beam_size(), beam, "{what}");
        // The tags are Penn Treebank tags, what the fixture was trained on.
        let pos = case.golden.layer("opennlp:pos").unwrap();
        let labels = pos.strs("tag").unwrap();
        assert!(labels.iter().all(|l| l != "NOUN" && l != "VERB"), "{what}: UD tags");
        if case.golden.layer(UTF16_TOKENS).is_some() {
            assert_eq!((case.set(), case.id()), ("en", "emoji-split"), "{what}");
            assert!(case.golden.layer("turbo:utf16<opennlp:chunks>").is_some(), "{what}");
            counts.skipped += 1;
            continue;
        }
        assert!(case.golden.layer("turbo:utf16<opennlp:chunks>").is_none(), "{what}");
        let doc = chunk(&case, model);
        let tags = doc.get_key(&Key::<ChunkTags>::DEFAULT).unwrap();
        counts.tokens += tags.tags.len();
        counts.phrases += doc.get_key(&Key::<PhraseChunks>::DEFAULT).unwrap().spans.len();
        counts.points += check_dump(&case, &doc, model);
        match counts.files.iter_mut().find(|(set, _)| set == case.set()) {
            Some((_, k)) => *k += 1,
            None => counts.files.push((case.set().to_string(), 1)),
        }
    }
    eprintln!("chunk/{name}: {counts:?}");
    counts
}

/// Every point of the dump against the contexts the search asked for
/// here; returns their number.
fn check_dump(case: &Case, doc: &Document, model: &ChunkerModel) -> usize {
    let dump = case.dump();
    let what = case.path.display().to_string();
    let maxent = model.maxent();
    let labels: Vec<&str> = (0..maxent.num_outcomes()).map(|i| maxent.outcome(i)).collect();
    let mut recorder = Recorder { inner: DefaultChunkerContext, calls: Vec::new() };
    let mut seen: Vec<(u32, Vec<Vec<u8>>)> = Vec::new();
    for sentence in sentences(doc) {
        let words: Vec<TokenTag<'_>> = sentence.iter().map(|&(w, _)| w).collect();
        let search = BeamSearch::new(model.beam_size(), maxent);
        search.best_sequence(&words, &mut recorder, &ChunkerValidator).unwrap_or_else(|| panic!("{what}"));
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
        assert_eq!(features.len(), 41, "{what} at {start}");
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

#[test]
fn fixture() {
    let path = common::fixtures_dir().join("chunker-fixture.bin");
    let model = Arc::new(ChunkerModel::read(&path).unwrap());
    assert_eq!((model.language(), model.beam_size()), ("en", 10));
    let counts = check_run("chunker-fixture", &model, &path, None);
    assert_eq!(counts.files, [("edge".into(), 34), ("en".into(), 10), ("multi".into(), 11)]);
    assert_eq!(counts.skipped, 1);
    // The same tokens as the tagger's goldens; a context per token per
    // path of a beam of ten, fewer where a sentence is shorter than the
    // beam or the validator cut paths.
    assert_eq!((counts.tokens, counts.points, counts.phrases), (1304, 10936, 768), "{counts:?}");
}

#[test]
fn fixture_with_a_beam_of_one() {
    let path = common::fixtures_dir().join("chunker-fixture.bin");
    let model = Arc::new(ChunkerModel::read(&path).unwrap().with_beam_size(1).unwrap());
    let counts = check_run("chunker-fixture+beam1", &model, &path, Some(1));
    assert_eq!(counts.files, [("en".into(), 10)]);
    assert_eq!(counts.skipped, 1);
    // One context per token: the beam keeps one path.
    assert_eq!(counts.tokens, counts.points, "{counts:?}");
    assert_eq!((counts.tokens, counts.phrases), (496, 292), "{counts:?}");
}
