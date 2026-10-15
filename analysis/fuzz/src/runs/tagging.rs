//! The runs over tokens: the part-of-speech taggers, the lemmatizers and
//! the phrase chunker. With a golden, they run on the JVM's own sentences,
//! tokens and tags loaded from it (`turbo_goldens::load_document`), as the
//! crates' golden tests do, so a difference is the run's own. Without one
//! they run on the simple tokenizer's tokens and the fixture tagger's
//! tags, which is enough for the offset invariants.

use std::sync::Arc;

use turbo_analysis::{Analyzer, Bound, Document};
use turbo_analysis_layers::{ChunkTags, Confidence, Lemmas, PhraseChunks, PosTags, Tokens};
use turbo_goldens::{GOLDEN_KEY, Golden, Tolerance, golden_key, load_document, matches_golden};
use turbo_lemmatizer::{DictionaryLemmatizer, LemmatizerModel, MaxentLemmatizer};
use turbo_phrase_chunker::{ChunkerModel, PhraseChunker};
use turbo_postag::{PosModel, PosTagFormat, PosTagger};
use turbo_tokenize::{SimpleTokenizer, TokenizerOptions};

use super::{Outcome, Run, RustError, analyze, analyze_document, has_cut, outcome};
use crate::resources::{Resources, Unavailable};

const TOKENS: &str = "opennlp:tokens";

/// The POS models that are checked-in fixtures rather than downloads: the
/// JVM's and the one the Rust trainer wrote.
const FIXTURE_TAGGERS: [&str; 2] = ["pos-fixture", "pos-fixture-rust"];

/// The chunker models, both checked-in fixtures: the JVM's and the one the
/// Rust trainer wrote.
pub(crate) const FIXTURE_CHUNKERS: [&str; 2] = ["chunker-fixture", "chunker-fixture-rust"];

fn broken(what: impl std::fmt::Display) -> Unavailable {
    Unavailable::Broken(what.to_string())
}

/// The fixture tagger the plain pipelines tag with.
fn fixture_tagger(res: &Resources, format: PosTagFormat) -> Result<PosTagger, Unavailable> {
    let path = res.fixture("pos-fixture.bin")?;
    let model = PosModel::read(&path).map_err(|e| broken(format!("{}: {e}", path.display())))?;
    Ok(PosTagger::new(model.with_format(format)))
}

/// A document of `text` with the golden's layers, or of the text alone.
fn document(text: &str, golden: Option<&Golden>) -> Result<Document, RustError> {
    match golden {
        Some(g) => load_document(g, text).map_err(|e| RustError { message: e.to_string(), offset: None }),
        None => Ok(Document::new(text)),
    }
}

/// The golden's tokens cut a surrogate pair, so they cannot be loaded.
fn cut(golden: &Golden) -> Option<Outcome> {
    has_cut(golden, TOKENS).then(|| Outcome::Skipped("a JVM token starts or ends inside a surrogate pair".into()))
}

/// `POSTaggerME` with a model, as it is or with a tag format or beam size.
pub(crate) struct Tagger {
    on_golden: Analyzer,
    plain: Analyzer,
}

impl Tagger {
    pub(crate) fn open(model: &str, options: &[&str], res: &Resources) -> Result<Box<dyn Run>, Unavailable> {
        let path = if FIXTURE_TAGGERS.contains(&model) {
            res.fixture(&format!("{model}.bin"))?
        } else {
            res.model(&format!("{model}.bin"))?
        };
        let mut m = PosModel::read(&path).map_err(|e| broken(format!("{}: {e}", path.display())))?;
        match options {
            [] => {}
            ["penn"] => m = m.with_format(PosTagFormat::Penn),
            [beam] if beam.starts_with("beam") => {
                let n: usize =
                    beam[4..].parse().map_err(|_| Unavailable::NotCovered(format!("postag/{model}+{beam}")))?;
                m = m.with_beam_size(n).map_err(broken)?;
            }
            other => return Err(Unavailable::NotCovered(format!("postag/{model} with {other:?}"))),
        }
        let m = Arc::new(m);
        let on_golden = Analyzer::builder()
            .given(golden_key::<Tokens>())
            .step(Bound::new(PosTagger::new(m.clone())).read(1, GOLDEN_KEY))
            .build()
            .map_err(broken)?;
        let plain = Analyzer::builder()
            .add(SimpleTokenizer::new(TokenizerOptions::default()))
            .add(PosTagger::new(m))
            .build()
            .map_err(broken)?;
        Ok(Box::new(Tagger { on_golden, plain }))
    }
}

impl Run for Tagger {
    fn annotate(&self, text: &str, golden: Option<&Golden>) -> Result<Document, RustError> {
        match golden {
            Some(_) => analyze_document(&self.on_golden, document(text, golden)?),
            None => analyze(&self.plain, text),
        }
    }

    fn compare(&self, doc: &Document, golden: &Golden) -> Outcome {
        cut(golden).unwrap_or_else(|| outcome(matches_golden::<PosTags>(doc, golden, Tolerance::exact())))
    }

    fn check(&self, text: &str, golden: &Golden) -> super::Checked {
        if let Some(skipped) = cut(golden) {
            return super::Checked { doc: None, outcome: skipped };
        }
        default_check(self, text, golden)
    }
}

/// The default `Run::check`, for the runs that override it only to skip
/// goldens they cannot load.
fn default_check(run: &dyn Run, text: &str, golden: &Golden) -> super::Checked {
    match run.annotate(text, Some(golden)) {
        Ok(doc) => {
            let outcome = super::jvm_threw(golden).unwrap_or_else(|| run.compare(&doc, golden));
            super::Checked { doc: Some(doc), outcome }
        }
        Err(e) => super::Checked { doc: None, outcome: Outcome::Divergence(format!("the port failed: {e}")) },
    }
}

/// `LemmatizerME` with a published model, or `DictionaryLemmatizer` with
/// the fixture dictionary.
pub(crate) struct Lemmatizer {
    on_golden: Analyzer,
    plain: Analyzer,
    confidence: bool,
}

impl Lemmatizer {
    pub(crate) fn open_maxent(model: &str, res: &Resources) -> Result<Box<dyn Run>, Unavailable> {
        let path = if model == "lemma-fixture-rust" {
            res.fixture(&format!("{model}.bin"))?
        } else {
            res.model(&format!("{model}.bin"))?
        };
        let m = Arc::new(LemmatizerModel::read(&path).map_err(|e| broken(format!("{}: {e}", path.display())))?);
        Lemmatizer::build(|| MaxentLemmatizer::new(m.clone()), true, res)
    }

    pub(crate) fn open_dictionary(res: &Resources) -> Result<Box<dyn Run>, Unavailable> {
        let path = res.fixture("lemmatizer-dictionary.txt")?;
        let read =
            || DictionaryLemmatizer::read(&path, "fixture").map_err(|e| broken(format!("{}: {e}", path.display())));
        read()?;
        Lemmatizer::build(|| read().expect("the dictionary read once already"), false, res)
    }

    fn build<A>(make: impl Fn() -> A, confidence: bool, res: &Resources) -> Result<Box<dyn Run>, Unavailable>
    where
        A: turbo_analysis::Annotator<Reads = (turbo_analysis::View, Tokens, PosTags)> + 'static,
    {
        let on_golden = Analyzer::builder()
            .given(golden_key::<Tokens>())
            .given(golden_key::<PosTags>())
            .step(Bound::new(make()).read(1, GOLDEN_KEY).read(2, GOLDEN_KEY))
            .build()
            .map_err(broken)?;
        let plain = Analyzer::builder()
            .add(SimpleTokenizer::new(TokenizerOptions::default()))
            .add(fixture_tagger(res, PosTagFormat::Ud)?)
            .add(make())
            .build()
            .map_err(broken)?;
        Ok(Box::new(Lemmatizer { on_golden, plain, confidence }))
    }
}

impl Run for Lemmatizer {
    fn annotate(&self, text: &str, golden: Option<&Golden>) -> Result<Document, RustError> {
        match golden {
            Some(_) => analyze_document(&self.on_golden, document(text, golden)?),
            None => analyze(&self.plain, text),
        }
    }

    fn compare(&self, doc: &Document, golden: &Golden) -> Outcome {
        if let Some(skipped) = cut(golden) {
            return skipped;
        }
        let lemmas = outcome(matches_golden::<Lemmas>(doc, golden, Tolerance::exact()));
        if lemmas != Outcome::Match || !self.confidence {
            return lemmas;
        }
        outcome(matches_golden::<Confidence<Lemmas>>(doc, golden, Tolerance::exact()))
    }

    fn check(&self, text: &str, golden: &Golden) -> super::Checked {
        if let Some(skipped) = cut(golden) {
            return super::Checked { doc: None, outcome: skipped };
        }
        default_check(self, text, golden)
    }
}

/// `ChunkerME` with a fixture model ([`FIXTURE_CHUNKERS`]), as it is or,
/// for the JVM's, with a beam of one, on Penn Treebank tags.
pub(crate) struct Chunker {
    on_golden: Analyzer,
    plain: Analyzer,
}

impl Chunker {
    pub(crate) fn open(model: &str, options: &[&str], res: &Resources) -> Result<Box<dyn Run>, Unavailable> {
        let path = res.fixture(&format!("{model}.bin"))?;
        let mut m = ChunkerModel::read(&path).map_err(|e| broken(format!("{}: {e}", path.display())))?;
        match (model, options) {
            (_, []) => {}
            ("chunker-fixture", ["beam1"]) => m = m.with_beam_size(1).map_err(broken)?,
            (_, other) => return Err(Unavailable::NotCovered(format!("chunk/{model} with {other:?}"))),
        }
        let m = Arc::new(m);
        let on_golden = Analyzer::builder()
            .given(golden_key::<Tokens>())
            .given(golden_key::<PosTags>())
            .step(Bound::new(PhraseChunker::new(m.clone())).read(1, GOLDEN_KEY).read(2, GOLDEN_KEY))
            .build()
            .map_err(broken)?;
        let plain = Analyzer::builder()
            .add(SimpleTokenizer::new(TokenizerOptions::default()))
            .add(fixture_tagger(res, PosTagFormat::Penn)?)
            .add(PhraseChunker::new(m))
            .build()
            .map_err(broken)?;
        Ok(Box::new(Chunker { on_golden, plain }))
    }
}

impl Run for Chunker {
    fn annotate(&self, text: &str, golden: Option<&Golden>) -> Result<Document, RustError> {
        match golden {
            Some(_) => analyze_document(&self.on_golden, document(text, golden)?),
            None => analyze(&self.plain, text),
        }
    }

    fn compare(&self, doc: &Document, golden: &Golden) -> Outcome {
        if let Some(skipped) = cut(golden) {
            return skipped;
        }
        let tags = outcome(matches_golden::<ChunkTags>(doc, golden, Tolerance::exact()));
        if tags != Outcome::Match {
            return tags;
        }
        outcome(matches_golden::<PhraseChunks>(doc, golden, Tolerance::exact()))
    }

    fn check(&self, text: &str, golden: &Golden) -> super::Checked {
        if let Some(skipped) = cut(golden) {
            return super::Checked { doc: None, outcome: skipped };
        }
        default_check(self, text, golden)
    }
}
