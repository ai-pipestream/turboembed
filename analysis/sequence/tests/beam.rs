//! The beam search of the part-of-speech tagger, the chunker and the name
//! finder against the JVM (the goldens of `BeamGolden.java` in
//! `analysis/goldens-tool`): every context the search asks for in the
//! recorded sentences, and for every sentence the sequences of the final
//! beam with their outcomes, probabilities and scores, compared bit for
//! bit, and the names the codec decodes.
//!
//! The fixture models in `analysis/goldens/data/fixtures` are always run.
//! The published English UD model comes from `turbo-goldens fetch-models`
//! (`turbo_goldens::published_model`). The OpenNLP 1.5 models are
//! local-only: set `TURBO_TEST_OPENNLP_MODELS` to a directory holding
//! them (`turbo-goldens fetch-models <dir> --local` fetches them) to run
//! those tests.

mod common;
#[path = "beam/qn.rs"]
mod qn;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use common::{escape, escaped, f64s, featuregen, grouped, layer, lossless_escaped, strs, u32s};
use turbo_maxent::{Dictionary, ModelFile};
use turbo_sequence::chunker::{ChunkerValidator, DefaultChunkerContext, TokenTag};
use turbo_sequence::namefind::{DefaultNameContext, NameCodec, NameFinder};
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

/// One sequence of the final beam, its score and probabilities as the
/// bits of the `f64`s.
#[derive(Debug, PartialEq)]
struct Seq {
    score: u64,
    outcomes: Vec<String>,
    probs: Vec<u64>,
}

struct Sentence {
    tokens: Vec<String>,
    tags: Vec<String>,
    calls: Vec<Call>,
    sequences: Vec<Seq>,
    /// The names decoded from the best sequence, `start end type`, for a
    /// name finder.
    spans: Option<Vec<String>>,
}

struct Golden {
    /// The run, the golden's directory.
    run: String,
    model: turbo_goldens::format::Model,
    kind: String,
    beam: usize,
    codec: Option<String>,
    sentences: Vec<Sentence>,
}

fn span_text(start: usize, end: usize, kind: Option<&str>) -> String {
    format!("{start} {end} {}", escape(kind.unwrap_or("null")))
}

fn read_golden(run: &str) -> Golden {
    let g = common::read("beam", run, "sequence", "sentences");
    let model = g.header.models[0].clone();
    let kind = g.header.config.iter().find(|c| c.key == "kind").expect("a kind").value.clone();
    let config = layer(&g, "turbo:beam-config");
    let beam = u32s(config, "beam-size")[0] as usize;
    let codec = strs(config, "codec").remove(0);
    let mut sentences: Vec<Sentence> = common::sentences(&g)
        .into_iter()
        .map(|s| Sentence { tokens: s.tokens, tags: Vec::new(), calls: Vec::new(), sequences: Vec::new(), spans: None })
        .collect();
    if let Some(tags) = g.layer("turbo:sentence-tags") {
        let mut tags = strs(tags, "tag").into_iter();
        for s in &mut sentences {
            s.tags = tags.by_ref().take(s.tokens.len()).collect();
        }
    }
    let calls = layer(&g, "turbo:beam-calls");
    let priors = layer(&g, "turbo:beam-call-priors");
    let prior_outcomes = strs(priors, "outcome");
    let prior_rows = grouped(priors, "call", calls.len());
    let features = layer(&g, "turbo:beam-call-features");
    let feature_strings = strs(features, "feature");
    let feature_rows = grouped(features, "call", calls.len());
    let (sentence, index) = (u32s(calls, "sentence"), u32s(calls, "index"));
    for c in 0..calls.len() {
        sentences[sentence[c] as usize].calls.push(Call {
            index: index[c] as usize,
            prior: prior_rows[c].iter().map(|&r| prior_outcomes[r].clone()).collect(),
            features: feature_rows[c].iter().map(|&r| lossless_escaped(&feature_strings[r])).collect(),
        });
    }
    let seqs = layer(&g, "turbo:beam-sequences");
    let outcomes = layer(&g, "turbo:beam-outcomes");
    let (outcome, probability) = (strs(outcomes, "outcome"), f64s(outcomes, "probability"));
    let outcome_rows = grouped(outcomes, "sequence", seqs.len());
    let (sentence, score) = (u32s(seqs, "sentence"), f64s(seqs, "score"));
    for q in 0..seqs.len() {
        sentences[sentence[q] as usize].sequences.push(Seq {
            score: score[q].to_bits(),
            outcomes: outcome_rows[q].iter().map(|&r| outcome[r].clone()).collect(),
            probs: outcome_rows[q].iter().map(|&r| probability[r].to_bits()).collect(),
        });
    }
    if let Some(names) = g.layer("turbo:beam-names") {
        for s in &mut sentences {
            s.spans = Some(Vec::new());
        }
        let (sentence, start, end, typed) =
            (u32s(names, "sentence"), u32s(names, "start"), u32s(names, "end"), u32s(names, "typed"));
        let kind = strs(names, "type");
        for r in 0..names.len() {
            let text = span_text(start[r] as usize, end[r] as usize, (typed[r] == 1).then_some(kind[r].as_str()));
            sentences[sentence[r] as usize].spans.as_mut().unwrap().push(text);
        }
    }
    Golden { run: run.to_string(), model, kind, beam, codec: (!codec.is_empty()).then_some(codec), sentences }
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

fn sequences(found: &[Sequence<'_>]) -> Vec<Seq> {
    found
        .iter()
        .map(|s| Seq {
            score: s.score().to_bits(),
            outcomes: s.outcomes().iter().map(|o| o.to_string()).collect(),
            probs: s.probs().iter().map(|p| p.to_bits()).collect(),
        })
        .collect()
}

/// Runs the search on every sentence of the golden and compares.
/// `after` gets each sentence's best outcomes and returns the names for a
/// name finder.
fn run<'g, T, C, V>(
    golden: &'g Golden,
    search: &BeamSearch<'_>,
    items: impl Fn(&'g Sentence) -> Vec<T>,
    context: &mut C,
    validator: &V,
    mut after: impl FnMut(&mut C, &Sentence, &[&str]) -> Option<Vec<String>>,
) where
    C: ContextGenerator<T>,
    V: SequenceValidator<T>,
{
    let mut failures = Vec::new();
    let mut compatible = search.opennlp();
    for (k, s) in golden.sentences.iter().enumerate() {
        let items = items(s);
        let mut recorder = Recorder { inner: context, calls: (!s.calls.is_empty()).then(Vec::new) };
        let found = if golden.kind == "names-fresh-sums" {
            search.best_sequences(golden.beam, &items, BeamSearch::<str>::DEFAULT_MIN_SCORE, &mut recorder, validator)
        } else {
            compatible.best_sequences(
                golden.beam,
                &items,
                BeamSearch::<str>::DEFAULT_MIN_SCORE,
                &mut recorder,
                validator,
            )
        };
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
    assert!(failures.is_empty(), "{}: {} differences:\n{}", golden.run, failures.len(), failures.join("\n"));
}

fn open(dir: &Path, golden: &Golden) -> ModelFile {
    let path = dir.join(&golden.model.file);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(
        turbo_goldens::sha512(&bytes),
        golden.model.sha512,
        "{} is not the model the golden was made with",
        path.display()
    );
    ModelFile::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
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
        Some(codec.decode(best).iter().map(|span| span_text(span.start, span.end, span.kind.as_deref())).collect())
    });
}

fn fixtures() -> PathBuf {
    common::fixtures()
}

#[test]
fn pos_fixture() {
    check_pos(&fixtures(), "pos-fixture");
}

/// `DefaultPOSContextGenerator` with an n-gram dictionary, as a model
/// from before OpenNLP 1.8 has, run over the fixture's classifier.
#[test]
fn pos_fixture_default_context() {
    let golden = read_golden("pos-fixture+default-context");
    assert_eq!(golden.kind, "pos-default-context");
    let file = open(&fixtures(), &golden);
    let model = file.maxent(postag::MODEL_ENTRY_NAME).unwrap();
    let search = BeamSearch::new(golden.beam, &model).unwrap();
    let xml = std::fs::read(featuregen().join("names.dictionary")).unwrap();
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

/// `NameFinder::find`, sentence after sentence with the adaptive data
/// carried over, gives the spans `NameFinderME.find` gave, over the GIS
/// fixture (BILOU) and the published 1.5 models when present.
fn check_name_finder(dir: &Path, name: &str) {
    let golden = read_golden(name);
    let file = open(dir, &golden);
    let model = file.maxent(namefind::MODEL_ENTRY_NAME).unwrap();
    let mut finder = NameFinder::new(&file, &model, PatternMode::Utf16Units).unwrap();
    for (k, s) in golden.sentences.iter().enumerate() {
        let found = finder.find(&tokens(s));
        let spans: Vec<String> =
            found.iter().map(|f| span_text(f.span.start, f.span.end, f.span.kind.as_deref())).collect();
        assert_eq!(Some(spans), s.spans, "{name}: sentence {k}");
        // The span's probability is the mean of its outcomes'.
        if let (Some(f), Some(best)) = (found.first(), s.sequences.first()) {
            let probs: Vec<f64> = best.probs.iter().map(|&p| f64::from_bits(p)).collect();
            let mean = probs[f.span.start..f.span.end].iter().sum::<f64>() / f.span.len() as f64;
            assert_eq!(f.prob.to_bits(), mean.to_bits(), "{name}: sentence {k}");
        }
    }
}

#[test]
fn the_name_finder_finds_what_the_jvm_found_with_the_bilou_fixture() {
    check_name_finder(&fixtures(), "ner-bilou-fixture");
}

/// QN maxent with the default descriptor, BIO, run in Java over a model
/// that evaluates every context into a new array (see the next test).
#[test]
fn ner_qn_fixture_fresh_sums() {
    check_names_kind(&fixtures(), "ner-qn-fixture+fresh-sums", "names-fresh-sums");
}

/// The same model as OpenNLP runs it. At commit cdf33c6
/// `QNModel.eval(context, probs)` adds its sums to what `probs` holds, and
/// `BeamSearch` passes one array to every evaluation, so each evaluation
/// of a QN model in a beam search starts from the previous one's
/// probabilities, including across sentences in one session.
#[test]
fn ner_qn_fixture() {
    check_names(&fixtures(), "ner-qn-fixture");
}

/// The directory of the OpenNLP 1.5 English models (local-only: their
/// training corpora are LDC corpora or undocumented, so CI does not fetch
/// them), or `None` after saying the test is skipped.
fn models_15(test: &str) -> Option<PathBuf> {
    let Some(dir) = std::env::var_os("TURBO_TEST_OPENNLP_MODELS") else {
        println!("{test}: skipped: TURBO_TEST_OPENNLP_MODELS is not set");
        return None;
    };
    Some(PathBuf::from(dir))
}

const RUN_EN_UD_EWT_POS: &str = "opennlp-en-ud-ewt-pos-1.3-2.5.4";

#[test]
fn en_ud_ewt_pos() {
    // The file name comes from the golden, so the lookup and `open` read
    // the same file.
    let golden = read_golden(RUN_EN_UD_EWT_POS);
    if let Some(path) = turbo_goldens::published_model(&golden.model.file) {
        check_pos(path.parent().unwrap(), RUN_EN_UD_EWT_POS);
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
        check_name_finder(&dir, "en-ner-person");
    }
}

#[test]
fn en_ner_location() {
    if let Some(dir) = models_15("en_ner_location") {
        check_names(&dir, "en-ner-location");
        check_name_finder(&dir, "en-ner-location");
    }
}

/// Gives every word the feature `w=` and the word.
struct Words;

impl ContextGenerator<&str, turbo_maxent::JavaString> for Words {
    fn context(&mut self, index: usize, sequence: &[&str], _: &[&turbo_maxent::JavaString], out: &mut Features) {
        out.add(&format!("w={}", sequence[index]));
    }
}

#[test]
fn a_search_over_text_refuses_an_outcome_holding_half_a_surrogate_pair_and_one_over_java_strings_runs() {
    use turbo_maxent::{JavaString, MaxentModel};
    use turbo_train::{Event, Key, TrainingConfiguration, TrainingParameters};
    // "R1" and a lone low surrogate: the class of a word losing the low
    // half of its last character.
    let lone = JavaString::from_utf16(&[0x52, 0x31, 0xDE00]);
    let events = [Event::from_strs(lone.clone(), &["w=a"]), Event::from_strs("O", &["w=b"])];
    let mut params = TrainingParameters::new();
    params.put(Key::ITERATIONS, 10);
    params.put(Key::CUTOFF, 0);
    let (trained, _) = turbo_train::gis::train_events(&events, &params, &mut TrainingConfiguration::default()).unwrap();
    let model = MaxentModel::read(&turbo_train::write_gis(&trained).unwrap()).unwrap();

    let refused = BeamSearch::<str>::new(3, &model).err().unwrap();
    assert!(matches!(&refused, turbo_sequence::Error::OutcomeNotText(o) if *o == lone), "{refused:?}");

    let Ok(search) = BeamSearch::<JavaString>::new(3, &model);
    let best = search.best_sequence(&["a", "b"], &mut Words, &turbo_sequence::AcceptAll).unwrap();
    assert_eq!(best.outcomes(), [&lone, &JavaString::new("O")]);
}

#[test]
fn usize_max_beam_size_handles_empty_and_single_token_inputs_in_ordinary_and_scored_search() {
    let golden = read_golden("pos-fixture");
    let file = open(&fixtures(), &golden);
    let model = file.maxent(postag::MODEL_ENTRY_NAME).unwrap();
    let search = BeamSearch::<str>::new(usize::MAX, &model).unwrap();
    let mut context = PosContext::for_model(&file, 3, PatternMode::Utf16Units).unwrap();
    let validator = PosValidator::new(TagDictionary::for_model(&file).unwrap().map(Arc::new));

    let empty = search.best_sequence::<&str>(&[], &mut context, &validator).unwrap();
    assert!(empty.outcomes().is_empty());
    let (empty_scored, empty_rows) = search.best_sequence_scored::<&str>(&[], &mut context, &validator).unwrap();
    assert!(empty_scored.outcomes().is_empty());
    assert!(empty_rows.is_empty());

    let single = search.best_sequence(&["The"], &mut context, &validator).unwrap();
    assert_eq!(single.outcomes().len(), 1);
    let (single_scored, single_rows) = search.best_sequence_scored(&["The"], &mut context, &validator).unwrap();
    assert_eq!(single_scored.outcomes(), single.outcomes());
    assert_eq!(single_rows.len(), 1);
}
