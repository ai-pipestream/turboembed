//! The beam search of the part-of-speech tagger, the chunker and the name
//! finder against the JVM (`golden/src/BeamGolden.java`): every context
//! the search asks for in the recorded sentences, and for every sentence
//! the sequences of the final beam with their outcomes, probabilities and
//! scores, compared bit for bit, and the names the codec decodes.
//!
//! The fixture models in `golden/models` are always run. The published
//! English UD model comes from `turbo-goldens fetch-models`
//! (`turbo_goldens::published_model`). The OpenNLP 1.5 models are
//! local-only: set `TURBO_TEST_OPENNLP_MODELS` to a directory holding
//! them (see the crate README) to run those tests.

mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use common::{body, escape, escaped, field, unescape_list};
use turbo_maxent::{Dictionary, ModelFile};
use turbo_sequence::chunker::{ChunkerValidator, DefaultChunkerContext, TokenTag};
use turbo_sequence::namefind::{DefaultNameContext, NameCodec};
use turbo_sequence::postag::{DefaultPosContext, PosContext, PosValidator, TagDictionary};
use turbo_sequence::{
    BeamSearch, ContextGenerator, Features, PatternMode, Sequence, SequenceValidator, chunker, manifest_beam_size,
    namefind, postag,
};

/// One context the search asked for.
#[derive(Debug, PartialEq)]
struct Call {
    index: usize,
    prior: Vec<String>,
    features: Vec<String>,
}

/// One sequence of the final beam.
#[derive(Debug, PartialEq)]
struct Seq {
    score: String,
    outcomes: Vec<String>,
    probs: Vec<String>,
}

struct Sentence {
    tokens: Vec<String>,
    tags: Vec<String>,
    calls: Vec<Call>,
    sequences: Vec<Seq>,
    spans: Option<String>,
}

struct Golden {
    model: String,
    sha256: String,
    kind: String,
    beam: usize,
    codec: Option<String>,
    sentences: Vec<Sentence>,
}

fn read_golden(name: &str) -> Golden {
    let text = common::read(&format!("expected/beam-{name}.txt"));
    let mut lines = body(&text, "beam").peekable();
    let model = field(lines.next().unwrap(), "model").to_string();
    let sha256 = field(lines.next().unwrap(), "sha256").to_string();
    let kind = field(lines.next().unwrap(), "kind").to_string();
    let beam = field(lines.next().unwrap(), "beam").parse().unwrap();
    let codec = lines.next_if(|l| l.starts_with("codec:")).map(|l| field(l, "codec").to_string());
    let mut sentences = Vec::new();
    while let Some(line) = lines.next() {
        field(line, "sentence");
        let tokens = unescape_list(field(lines.next().unwrap(), "tokens"));
        let tags =
            lines.next_if(|l| l.starts_with("tags:")).map(|l| unescape_list(field(l, "tags"))).unwrap_or_default();
        let mut calls = Vec::new();
        while let Some(call) = lines.next_if(|l| l.starts_with("call:")) {
            let call = field(call, "call");
            let (index, prior) = call.split_once('\t').unwrap_or((call, ""));
            let features = field(lines.next().unwrap(), "features");
            calls.push(Call {
                index: index.parse().unwrap(),
                prior: unescape_list(prior),
                features: if features.is_empty() {
                    Vec::new()
                } else {
                    features.split('\t').map(str::to_string).collect()
                },
            });
        }
        let mut sequences = Vec::new();
        while let Some(seq) = lines.next_if(|l| l.starts_with("sequence:")) {
            let score = field(seq, "sequence").to_string();
            let outcomes = unescape_list(field(lines.next().unwrap(), "outcomes"));
            let probs = field(lines.next().unwrap(), "probs");
            let probs = if probs.is_empty() { Vec::new() } else { probs.split('\t').map(str::to_string).collect() };
            sequences.push(Seq { score, outcomes, probs });
        }
        let spans = lines.next_if(|l| l.starts_with("spans:")).map(|l| field(l, "spans").to_string());
        sentences.push(Sentence { tokens, tags, calls, sequences, spans });
    }
    Golden { model, sha256, kind, beam, codec, sentences }
}

/// Passes contexts through, keeping them while `calls` is set.
struct Recorder<'c, C> {
    inner: &'c mut C,
    calls: Option<Vec<Call>>,
}

impl<T, C: ContextGenerator<T>> ContextGenerator<T> for Recorder<'_, C> {
    fn start_sequence(&mut self) {
        self.inner.start_sequence();
    }

    fn context(&mut self, index: usize, sequence: &[T], prior: &[&str], out: &mut Features) {
        self.inner.context(index, sequence, prior, out);
        if let Some(calls) = &mut self.calls {
            calls.push(Call { index, prior: prior.iter().map(|s| s.to_string()).collect(), features: escaped(out) });
        }
    }
}

fn bits(d: f64) -> String {
    format!("{:016x}", d.to_bits())
}

fn sequences(found: &[Sequence<'_>]) -> Vec<Seq> {
    found
        .iter()
        .map(|s| Seq {
            score: bits(s.score()),
            outcomes: s.outcomes().iter().map(|o| o.to_string()).collect(),
            probs: s.probs().iter().map(|&p| bits(p)).collect(),
        })
        .collect()
}

/// Runs the search on every sentence of the golden file and compares.
/// `after` gets each sentence's best outcomes and returns the spans line
/// for a name finder.
fn run<'g, T, C, V>(
    golden: &'g Golden,
    search: &BeamSearch<'_>,
    items: impl Fn(&'g Sentence) -> Vec<T>,
    context: &mut C,
    validator: &V,
    mut after: impl FnMut(&mut C, &Sentence, &[&str]) -> Option<String>,
) where
    C: ContextGenerator<T>,
    V: SequenceValidator<T>,
{
    let mut failures = Vec::new();
    for (k, s) in golden.sentences.iter().enumerate() {
        let items = items(s);
        let mut recorder = Recorder { inner: context, calls: (!s.calls.is_empty()).then(Vec::new) };
        let found =
            search.best_sequences(golden.beam, &items, BeamSearch::<str>::DEFAULT_MIN_SCORE, &mut recorder, validator);
        let calls = recorder.calls.take().unwrap_or_default();
        if calls != s.calls {
            let at = calls.iter().zip(&s.calls).position(|(a, b)| a != b).unwrap_or(calls.len().min(s.calls.len()));
            failures.push(format!(
                "sentence {k}: {} calls, expected {}; first difference at call {at}:\n  expected {:?}\n  actual   {:?}",
                calls.len(),
                s.calls.len(),
                s.calls.get(at),
                calls.get(at)
            ));
        }
        let actual = sequences(&found);
        if actual != s.sequences {
            failures.push(format!("sentence {k}:\n  expected {:?}\n  actual   {:?}", s.sequences, actual));
        }
        let best: Vec<&str> = found.first().map(|f| f.outcomes().to_vec()).unwrap_or_default();
        let spans = after(context, s, &best);
        if spans != s.spans {
            failures.push(format!("sentence {k} spans: expected {:?}, got {spans:?}", s.spans));
        }
    }
    assert!(failures.is_empty(), "{}: {} differences:\n{}", golden.model, failures.len(), failures.join("\n"));
}

fn open(dir: &Path, golden: &Golden) -> ModelFile {
    let path = dir.join(&golden.model);
    let file = ModelFile::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(
        file.sha256().to_hex(),
        golden.sha256,
        "{} is not the model the golden file was made with",
        path.display()
    );
    file
}

fn tokens(s: &Sentence) -> Vec<&str> {
    s.tokens.iter().map(String::as_str).collect()
}

fn token_tags(s: &Sentence) -> Vec<TokenTag<'_>> {
    s.tokens.iter().zip(&s.tags).map(|(token, tag)| TokenTag { token, tag }).collect()
}

fn check_pos(dir: &Path, name: &str) {
    let golden = read_golden(name);
    assert_eq!(golden.kind, "pos");
    let file = open(dir, &golden);
    let model = file.maxent(postag::MODEL_ENTRY_NAME).unwrap();
    let beam = manifest_beam_size(&file, postag::DEFAULT_BEAM_SIZE).unwrap();
    assert_eq!(beam, golden.beam);
    let search = BeamSearch::new(beam, &model).unwrap();
    let mut context = PosContext::for_model(&file, beam, PatternMode::Utf16Units).unwrap();
    let validator = PosValidator::new(TagDictionary::for_model(&file).unwrap().map(Arc::new));
    run(&golden, &search, tokens, &mut context, &validator, |_, _, _| None);
}

fn check_chunker(dir: &Path, name: &str) {
    let golden = read_golden(name);
    assert_eq!(golden.kind, "chunk");
    let file = open(dir, &golden);
    let model = file.maxent(chunker::MODEL_ENTRY_NAME).unwrap();
    let beam = manifest_beam_size(&file, chunker::DEFAULT_BEAM_SIZE).unwrap();
    assert_eq!(beam, golden.beam);
    let search = BeamSearch::new(beam, &model).unwrap();
    run(&golden, &search, token_tags, &mut DefaultChunkerContext, &ChunkerValidator, |_, _, _| None);
}

fn check_names(dir: &Path, name: &str) {
    check_names_kind(dir, name, "names");
}

fn check_names_kind(dir: &Path, name: &str, kind: &str) {
    let golden = read_golden(name);
    assert_eq!(golden.kind, kind);
    let file = open(dir, &golden);
    let model = file.maxent(namefind::MODEL_ENTRY_NAME).unwrap();
    let beam = manifest_beam_size(&file, namefind::DEFAULT_BEAM_SIZE).unwrap();
    assert_eq!(beam, golden.beam);
    let search = BeamSearch::new(beam, &model).unwrap();
    let codec = NameCodec::for_model(&file);
    let expected_codec = match codec {
        NameCodec::Bio => "opennlp.tools.namefind.BioCodec",
        NameCodec::Bilou => "opennlp.tools.namefind.BilouCodec",
    };
    assert_eq!(golden.codec.as_deref(), Some(expected_codec));
    let mut context = DefaultNameContext::for_model(&file, PatternMode::Utf16Units).unwrap();
    context.set_additional_context(&[]);
    run(&golden, &search, tokens, &mut context, &codec, |context, s, best| {
        context.update_adaptive_data(&tokens(s), best);
        let spans: Vec<String> = codec
            .decode(best)
            .iter()
            .map(|span| format!("{} {} {}", span.start, span.end, escape(span.kind.as_deref().unwrap_or("null"))))
            .collect();
        Some(spans.join("\t"))
    });
}

fn fixtures() -> PathBuf {
    common::golden().join("models")
}

#[test]
fn pos_fixture() {
    check_pos(&fixtures(), "pos-fixture");
}

/// `DefaultPOSContextGenerator` with an n-gram dictionary, as a model
/// from before OpenNLP 1.8 has, run over the fixture's classifier.
#[test]
fn pos_fixture_default_context() {
    let golden = read_golden("pos-fixture-default-context");
    assert_eq!(golden.kind, "pos-default-context");
    let file = open(&fixtures(), &golden);
    let model = file.maxent(postag::MODEL_ENTRY_NAME).unwrap();
    let search = BeamSearch::new(golden.beam, &model).unwrap();
    let xml = std::fs::read(common::golden().join("resources/names.dictionary")).unwrap();
    let mut context = DefaultPosContext::new(Some(Arc::new(Dictionary::read(&xml).unwrap())));
    let validator = PosValidator::new(TagDictionary::for_model(&file).unwrap().map(Arc::new));
    run(&golden, &search, tokens, &mut context, &validator, |_, _, _| None);
}

#[test]
fn chunker_fixture() {
    check_chunker(&fixtures(), "chunker-fixture");
}

/// QN maxent, every factory with resources stored in the model, BIO.
#[test]
fn ner_bio_fixture() {
    check_names(&fixtures(), "ner-bio-fixture");
}

/// GIS maxent, nested and cached generators, BILOU.
#[test]
fn ner_bilou_fixture() {
    check_names(&fixtures(), "ner-bilou-fixture");
}

/// QN maxent with the default descriptor, BIO, run in Java over a model
/// that evaluates every context into a new array (see the next test).
#[test]
fn ner_qn_fixture_fresh_sums() {
    check_names_kind(&fixtures(), "ner-qn-fixture-fresh-sums", "names-fresh-sums");
}

/// The same model as OpenNLP runs it. At commit cdf33c6
/// `QNModel.eval(context, probs)` adds its sums to what `probs` holds, and
/// `BeamSearch` passes one array to every evaluation, so each evaluation
/// of a QN model in a beam search starts from the previous one's
/// probabilities. Reproducing that needs an evaluation in turbo-maxent
/// that starts from the caller's sums, which it does not have yet.
#[test]
#[ignore = "QN sums carried between evaluations are not reproduced yet"]
fn ner_qn_fixture() {
    check_names(&fixtures(), "ner-qn-fixture");
}

/// The directory of the OpenNLP 1.5 English models (local-only: their
/// training corpora are LDC or undocumented, so CI does not fetch them),
/// or `None` after saying the test is skipped.
fn models_15(test: &str) -> Option<PathBuf> {
    let Some(dir) = std::env::var_os("TURBO_TEST_OPENNLP_MODELS") else {
        println!("{test}: skipped: TURBO_TEST_OPENNLP_MODELS is not set");
        return None;
    };
    Some(PathBuf::from(dir))
}

#[test]
fn en_ud_ewt_pos() {
    // The file name comes from the golden, so the lookup and `open` read
    // the same file.
    let golden = read_golden("en-ud-ewt-pos");
    if let Some(path) = turbo_goldens::published_model(&golden.model) {
        check_pos(path.parent().unwrap(), "en-ud-ewt-pos");
    }
}

#[test]
fn en_pos_maxent() {
    if let Some(dir) = models_15("en_pos_maxent") {
        check_pos(&dir, "en-pos-maxent");
    }
}

#[test]
fn en_pos_perceptron() {
    if let Some(dir) = models_15("en_pos_perceptron") {
        check_pos(&dir, "en-pos-perceptron");
    }
}

#[test]
fn en_chunker() {
    if let Some(dir) = models_15("en_chunker") {
        check_chunker(&dir, "en-chunker");
    }
}

#[test]
fn en_ner_person() {
    if let Some(dir) = models_15("en_ner_person") {
        check_names(&dir, "en-ner-person");
    }
}

#[test]
fn en_ner_location() {
    if let Some(dir) = models_15("en_ner_location") {
        check_names(&dir, "en-ner-location");
    }
}
