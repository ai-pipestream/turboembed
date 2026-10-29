//! The `SentimentModel` file and the factories its manifest can name.

use std::fmt;

use turbo_analysis::ModelRef;
use turbo_maxent::{JavaString, MaxentModel, ModelFile};

/// `SentimentModel.COMPONENT_NAME`.
pub const COMPONENT: &str = "SentimentME";
/// `SentimentModel.SENTIMENT_MODEL_ENTRY_NAME`, the classifier's entry.
pub const MODEL_ENTRY: &str = "sentiment.model";
/// The class name of `SentimentFactory`, as `BaseModel` records it.
pub const TOKEN_FACTORY: &str = "opennlp.tools.sentiment.SentimentFactory";
/// The class name of `EmojiSentimentFactory`.
pub const EMOJI_FACTORY: &str = "opennlp.tools.sentiment.EmojiSentimentFactory";

/// The factory of a sentiment model, which decides its context generator.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SentimentFactory {
    /// `SentimentFactory`: the context is the tokens.
    Tokens,
    /// `EmojiSentimentFactory`: the tokens, then each token's emoji
    /// features.
    Emoji,
}

impl SentimentFactory {
    /// The factory a manifest's `factory` entry names.
    pub fn from_class(class: &str) -> Option<SentimentFactory> {
        match class {
            TOKEN_FACTORY => Some(SentimentFactory::Tokens),
            EMOJI_FACTORY => Some(SentimentFactory::Emoji),
            _ => None,
        }
    }

    /// The class name `BaseModel` records under `factory`.
    pub fn class(self) -> &'static str {
        match self {
            SentimentFactory::Tokens => TOKEN_FACTORY,
            SentimentFactory::Emoji => EMOJI_FACTORY,
        }
    }

    /// The features `getContext(tokens)` appends after the tokens
    /// themselves, written into `out` (cleared first): nothing for
    /// `SentimentFactory`, and for `EmojiSentimentFactory` the emoji
    /// features of every token, token by token.
    pub fn extra_features(self, tokens: &[&str], out: &mut Vec<String>) {
        out.clear();
        if self == SentimentFactory::Emoji {
            for t in tokens {
                turbo_normalize::emoji::features(t, out);
            }
        }
    }

    /// The whole context `getContext(tokens)` returns: the tokens, then
    /// [`extra_features`](Self::extra_features), which are built in
    /// `extra`.
    pub fn context<'a>(self, tokens: &[&'a str], extra: &'a mut Vec<String>) -> Vec<&'a str> {
        self.extra_features(tokens, extra);
        let mut out = Vec::with_capacity(tokens.len() + extra.len());
        out.extend_from_slice(tokens);
        out.extend(extra.iter().map(String::as_str));
        out
    }
}

/// Why a sentiment model could not be opened.
#[derive(Debug)]
pub enum SentimentModelError {
    /// The file or its classifier entry could not be read.
    File(turbo_maxent::Error),
    /// The manifest's `Component-Name` is not `SentimentME`.
    Component(String),
    /// The manifest names no factory. `SentimentModel` has no default
    /// factory, so OpenNLP loads such a file and then fails in the
    /// `SentimentME` constructor.
    NoFactory,
    /// The manifest names a factory other than the two this port has.
    Factory(String),
    /// A classifier outcome holds half of a surrogate pair. The JVM runs
    /// such a model; this port hands out labels as `&str`.
    OutcomeNotText(JavaString),
}

impl fmt::Display for SentimentModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SentimentModelError::File(e) => write!(f, "sentiment model: {e}"),
            SentimentModelError::Component(c) => write!(f, "sentiment model: component {c:?} is not {COMPONENT}"),
            SentimentModelError::NoFactory => write!(f, "sentiment model: the manifest names no factory"),
            SentimentModelError::Factory(c) => {
                write!(f, "sentiment model: factory {c:?} is neither {TOKEN_FACTORY} nor {EMOJI_FACTORY}")
            }
            SentimentModelError::OutcomeNotText(o) => {
                write!(f, "sentiment model: the outcome {o:?} holds half of a surrogate pair")
            }
        }
    }
}

impl std::error::Error for SentimentModelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SentimentModelError::File(e) => Some(e),
            _ => None,
        }
    }
}

impl From<turbo_maxent::Error> for SentimentModelError {
    fn from(e: turbo_maxent::Error) -> SentimentModelError {
        SentimentModelError::File(e)
    }
}

/// An opened `SentimentModel`: the classifier and its factory.
///
/// The manifest's `sentiment.tokenizer` entry names the tokenizer
/// `predict(String)` splits a sentence with. This port takes tokens, so
/// the entry is not read.
#[derive(Debug, Clone)]
pub struct SentimentModel {
    maxent: MaxentModel,
    factory: SentimentFactory,
    model_ref: ModelRef,
}

impl SentimentModel {
    /// Reads the model from an opened model file, checking the component
    /// name, the factory and the classifier entry.
    pub fn open(file: ModelFile) -> Result<SentimentModel, SentimentModelError> {
        if file.component_name() != COMPONENT {
            return Err(SentimentModelError::Component(file.component_name().into()));
        }
        let class = file.manifest().get("factory").ok_or(SentimentModelError::NoFactory)?;
        let factory = SentimentFactory::from_class(class).ok_or_else(|| SentimentModelError::Factory(class.into()))?;
        let maxent = file.maxent(MODEL_ENTRY)?;
        if let Err(outcome) = maxent.outcome_texts() {
            return Err(SentimentModelError::OutcomeNotText(outcome.clone()));
        }
        let model_ref = ModelRef {
            id: format!("opennlp:{COMPONENT}:{}", file.language()),
            revision: file.opennlp_version().into(),
            sha256: file.sha256().to_hex(),
        };
        Ok(SentimentModel { maxent, factory, model_ref })
    }

    /// The classifier.
    pub fn maxent(&self) -> &MaxentModel {
        &self.maxent
    }

    /// The factory, which decides the context.
    pub fn factory(&self) -> SentimentFactory {
        self.factory
    }

    /// The label of outcome `i`; `open` refused any model whose outcomes
    /// are not text.
    pub fn label(&self, i: usize) -> &str {
        self.maxent.outcome_str(i).expect("the outcomes were checked to be text when the model was opened")
    }

    /// What layers record as their model: `opennlp:SentimentME:` and the
    /// manifest's language, the OpenNLP version that wrote it, and the
    /// SHA-256 of the model file.
    pub fn model_ref(&self) -> &ModelRef {
        &self.model_ref
    }
}
