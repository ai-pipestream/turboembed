//! The runs over tokens: the part-of-speech taggers and dictionary
//! lemmatizer. With a golden, they run on the JVM's own sentences, tokens and
//! tags loaded from it (`turbo_goldens::load_document`), as the crates'
//! golden tests do, so a difference is the run's own. Without one they run
//! on the simple tokenizer's tokens and the fixture tagger's tags, which
//! is enough for the offset invariants.

use std::sync::Arc;

use turbo_analysis::{Analyzer, Annotator, Bound, Document, Layer, Refs, Tag, TagSet, View};
use turbo_analysis_layers::{Lemmas, PosTags, Tokens};
use turbo_goldens::{GOLDEN_KEY, Golden, Tolerance, golden_key, load_document, matches_golden};
use turbo_lemmatizer::DictionaryLemmatizer;
use turbo_postag::{PosModel, PosTagFormat, PosTagger};
use turbo_tokenize::{SimpleTokenizer, TokenizerOptions};

use super::{Outcome, Run, RustError, analyze, analyze_document, has_cut, outcome};
use crate::resources::{Resources, Unavailable};

const TOKENS: &str = "opennlp:tokens";

/// The POS models that are checked-in fixtures rather than downloads: the
/// JVM's and the one the Rust trainer wrote.
const FIXTURE_TAGGERS: [&str; 2] = ["pos-fixture", "pos-fixture-rust"];

fn broken(what: impl std::fmt::Display) -> Unavailable {
    Unavailable::Broken(what.to_string())
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

/// A deterministic majority-tag baseline used instead of a model file.
struct BaselinePos {
    tagset: Arc<TagSet>,
}

impl BaselinePos {
    fn new() -> Self {
        Self { tagset: TagSet::new("upos", &["NOUN"]) }
    }
}

impl Annotator for BaselinePos {
    const ID: &'static str = "turbo-fuzz:baseline-pos";
    type Reads = (View, Tokens);
    type Writes = (PosTags,);

    fn annotate(&self, (_, tokens): Refs<'_, Self>) -> Result<(PosTags,), turbo_analysis::Error> {
        Ok((PosTags { tagset: self.tagset.clone(), tags: vec![Tag(0); tokens.len()], prob: None },))
    }
}

/// The no-model dictionary lemmatizer with the checked-in fixture.
pub(crate) struct DictionaryRun {
    on_golden: Analyzer,
    plain: Analyzer,
}

impl DictionaryRun {
    pub(crate) fn open(res: &Resources) -> Result<Box<dyn Run>, Unavailable> {
        let path = res.fixture("lemmatizer-dictionary.txt")?;
        let read =
            || DictionaryLemmatizer::read(&path, "fixture").map_err(|e| broken(format!("{}: {e}", path.display())));
        read()?;
        let on_golden = Analyzer::builder()
            .given(golden_key::<Tokens>())
            .given(golden_key::<PosTags>())
            .step(Bound::new(read().expect("the dictionary read once already")).read(1, GOLDEN_KEY).read(2, GOLDEN_KEY))
            .build()
            .map_err(broken)?;
        let plain = Analyzer::builder()
            .add(SimpleTokenizer::new(TokenizerOptions::default()))
            .add(BaselinePos::new())
            .add(read().expect("the dictionary read once already"))
            .build()
            .map_err(broken)?;
        Ok(Box::new(DictionaryRun { on_golden, plain }))
    }
}

impl Run for DictionaryRun {
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
        outcome(matches_golden::<Lemmas>(doc, golden, Tolerance::exact()))
    }

    fn check(&self, text: &str, golden: &Golden) -> super::Checked {
        if let Some(skipped) = cut(golden) {
            return super::Checked { doc: None, outcome: skipped };
        }
        default_check(self, text, golden)
    }
}
