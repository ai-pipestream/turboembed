use std::fmt;

use turbo_tokenize::me::PatternError;

/// Why a factory, an event stream or a reproduction step was refused.
#[derive(Debug)]
pub enum Error {
    /// The trainer, the indexer or a parameter.
    Train(turbo_train::Error),
    /// A sample could not be read.
    Format(turbo_formats::FormatError),
    /// Dictionary XML could not be read.
    DictionaryXml(turbo_maxent::Error),
    /// `new StringList()` with no tokens: `IllegalArgumentException("tokens
    /// must not be empty")`.
    EmptyEntry,
    /// A case-sensitive dictionary built with `put` was given two entries
    /// that differ only in case. The JVM keeps both and finds each by an
    /// exact lookup; the lookup dictionary this crate hands the context
    /// generators keeps one, so the pair is refused rather than looked up
    /// differently.
    CaseVariants { first: String, second: String },
    /// A `java.util.HashMap` bucket that the JVM turns into a tree (nine
    /// entries in a table of 64 or more) holds two entries with one hash
    /// code, which the JVM orders by `System.identityHashCode`, differently
    /// from run to run; the serialized dictionary's entry order would be a
    /// guess, so it is refused.
    TreeBucketTie,
    /// A dictionary token holds a character XML 1.0 cannot carry (a control
    /// character other than tab, line feed and carriage return, or U+FFFE
    /// or U+FFFF): the JDK's XML serializer throws `SAXException` and
    /// `Dictionary.serialize` wraps it in an `IOException`, so such a
    /// dictionary cannot be written to a model file.
    InvalidXmlCharacter(u32),
    /// The tokenizer's alphanumeric pattern is one `turbo-tokenize` cannot
    /// evaluate.
    Pattern(PatternError),
    /// An end-of-sentence character given as a surrogate code unit.
    SurrogateEos(u16),
    /// An empty end-of-sentence character set. The JVM factory then falls
    /// back to the language's scanner and context generator (for Thai the
    /// `SentenceContextGenerator`) while the manifest records
    /// `eosCharacters=` empty, so the model file does not describe the
    /// model; the set is refused rather than reproduced.
    EmptyEos,
    /// A token sample whose tokens are not in text order (a token starting
    /// or ending before the one before it), which the port's
    /// `TokSpanEventStream` refuses (`TokSpanEventStream::events` says what
    /// the Java does with one). `token` is the index of the token out of
    /// order.
    TokensOutOfOrder { token: usize },
    /// A part-of-speech descriptor whose generators cannot be built.
    Sequence(turbo_sequence::Error),
    /// A tag dictionary entry `POSDictionary.create` refuses: more than one
    /// token, or no `tags` attribute.
    TagDictionaryEntry(String),
    /// A tag dictionary lists tags the trained model has no outcome for;
    /// `POSModel`'s constructor throws an `IllegalArgumentException`
    /// wrapping the `InvalidFormatException` of `validatePOSDictionary`,
    /// and writes no model.
    UnknownDictionaryTags(Vec<String>),
    /// A name the codec would write outside the sentence: with BIO an
    /// empty name after the last token, with BILOU an empty name at the
    /// first token. Java's codec throws `ArrayIndexOutOfBoundsException`
    /// for outcome `index` of a sentence of `length` tokens, and the
    /// sample gives no events.
    UnencodableName { index: i64, length: usize },
    /// The trained name finder model has outcomes its codec cannot decode
    /// (with BIO, no `start` outcome at all): `TokenNameFinderModel`'s
    /// constructor throws `IllegalArgumentException` and no model is
    /// written.
    IncompatibleOutcomes,
    /// A name finder trained with a trainer that does not train from
    /// events (`PERCEPTRON_SEQUENCE`), which `NameFinderME.train` feeds
    /// with `NameSampleSequenceStream`; not ported.
    NameSequenceTraining(turbo_train::TrainerType),
    /// `ChunkerME.train` given parameters that name no event trainer:
    /// `IllegalArgumentException("Trainer type is not supported: " +
    /// trainerType)`, the type `TrainerFactory.getTrainerType` gave, `null`
    /// for an algorithm it cannot resolve.
    UnsupportedTrainerType(Option<turbo_train::TrainerType>),
    /// The treebank or a published model could not be fetched or verified.
    Zoo(turbo_zoo::ZooError),
    /// A published model file could not be read.
    Model(turbo_maxent::Error),
    /// A published model's manifest lacks an entry the comparison needs.
    MissingManifestEntry(&'static str),
    /// A file could not be read.
    Io { path: std::path::PathBuf, message: String },
    /// A constituency parser model written here could not be read back,
    /// or its parts do not fit its parser type.
    ParserModel(turbo_parser::model::ParserModelError),
    /// A parse sample a parser's event stream or `ParserEvaluator` fails
    /// on where OpenNLP throws, as `Parse.pruneParse` throws on a node
    /// whose only child has its type.
    ParseSample(turbo_parser::ParseError),
    /// The tree insertion event stream failed where OpenNLP throws or
    /// exits.
    TreeInsert(turbo_parser::treeinsert::TreeInsertError),
    /// A cross-validation fold's prediction the evaluator refuses where
    /// the JVM's evaluator throws.
    Evaluate(turbo_formats::evaluate::EvalError),
    /// A language detector model trained for a cross-validation fold
    /// could not be read back.
    LanguageModel(turbo_langdetect::ModelError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Train(e) => write!(f, "training: {e}"),
            Error::Format(e) => write!(f, "samples: {e}"),
            Error::DictionaryXml(e) => write!(f, "dictionary XML: {e}"),
            Error::EmptyEntry => f.write_str("tokens must not be empty"),
            Error::CaseVariants { first, second } => {
                write!(f, "a case-sensitive dictionary with {first:?} and {second:?}, which differ only in case")
            }
            Error::TreeBucketTie => {
                f.write_str("a HashMap bucket that would be a tree holds two entries with one hash code")
            }
            Error::InvalidXmlCharacter(c) => {
                write!(f, "An invalid XML character (Unicode: 0x{c:x}) was found in the node's character data content.")
            }
            Error::Pattern(e) => write!(f, "alphanumeric pattern: {e}"),
            Error::SurrogateEos(u) => write!(f, "end of sentence character U+{u:04X} is a surrogate"),
            Error::EmptyEos => f.write_str("the end of sentence character set is empty"),
            Error::TokensOutOfOrder { token } => write!(f, "token {token} of the sample is out of text order"),
            Error::Sequence(e) => write!(f, "feature generators: {e}"),
            Error::TagDictionaryEntry(m) => f.write_str(m),
            Error::UnknownDictionaryTags(tags) => {
                write!(f, "Tag dictionary contains tags which are unknown by the model! The unknown tags are: ")?;
                for t in tags {
                    write!(f, "{t} ")?;
                }
                Ok(())
            }
            Error::UnencodableName { index, length } => {
                write!(f, "Index {index} out of bounds for length {length}")
            }
            Error::IncompatibleOutcomes => f.write_str(crate::namefind::INCOMPATIBLE_MESSAGE),
            Error::NameSequenceTraining(t) => {
                write!(f, "training a name finder with a {} trainer is not ported", t.value())
            }
            Error::UnsupportedTrainerType(t) => {
                let name = match t {
                    Some(turbo_train::TrainerType::Event) => "EVENT_MODEL_TRAINER",
                    Some(turbo_train::TrainerType::EventModelSequence) => "EVENT_MODEL_SEQUENCE_TRAINER",
                    Some(turbo_train::TrainerType::Sequence) => "SEQUENCE_TRAINER",
                    None => "null",
                };
                write!(f, "Trainer type is not supported: {name}")
            }
            Error::Zoo(e) => write!(f, "zoo: {e}"),
            Error::Model(e) => write!(f, "model file: {e}"),
            Error::MissingManifestEntry(k) => write!(f, "the published manifest has no {k}"),
            Error::Io { path, message } => write!(f, "{}: {message}", path.display()),
            Error::ParserModel(e) => write!(f, "{e}"),
            Error::ParseSample(e) => write!(f, "parser sample: {e}"),
            Error::TreeInsert(e) => write!(f, "parser training events: {e}"),
            Error::Evaluate(e) => write!(f, "evaluation: {e}"),
            Error::LanguageModel(e) => write!(f, "language detector model: {e}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<turbo_train::Error> for Error {
    fn from(e: turbo_train::Error) -> Error {
        Error::Train(e)
    }
}

impl From<turbo_formats::FormatError> for Error {
    fn from(e: turbo_formats::FormatError) -> Error {
        Error::Format(e)
    }
}

impl From<PatternError> for Error {
    fn from(e: PatternError) -> Error {
        Error::Pattern(e)
    }
}

impl From<turbo_sequence::Error> for Error {
    fn from(e: turbo_sequence::Error) -> Error {
        Error::Sequence(e)
    }
}

impl From<turbo_zoo::ZooError> for Error {
    fn from(e: turbo_zoo::ZooError) -> Error {
        Error::Zoo(e)
    }
}

impl From<turbo_parser::model::ParserModelError> for Error {
    fn from(e: turbo_parser::model::ParserModelError) -> Error {
        Error::ParserModel(e)
    }
}

impl From<turbo_parser::ParseError> for Error {
    fn from(e: turbo_parser::ParseError) -> Error {
        Error::ParseSample(e)
    }
}

impl From<turbo_parser::treeinsert::TreeInsertError> for Error {
    fn from(e: turbo_parser::treeinsert::TreeInsertError) -> Error {
        Error::TreeInsert(e)
    }
}

impl From<turbo_formats::evaluate::EvalError> for Error {
    fn from(e: turbo_formats::evaluate::EvalError) -> Error {
        Error::Evaluate(e)
    }
}

impl From<turbo_langdetect::ModelError> for Error {
    fn from(e: turbo_langdetect::ModelError) -> Error {
        Error::LanguageModel(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
