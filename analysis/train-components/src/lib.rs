//! Training OpenNLP's components: the event streams, the factories that
//! set them up, the model files the `train` methods of `SentenceDetectorME`,
//! `TokenizerME`, `POSTaggerME` and `LemmatizerME` return, and the
//! reproduction of the published UD 1.3 models.
//!
//! - [`sentdetect`][]: `SentenceDetectorFactory`, the events of
//!   `SDEventStream`, and the `SentenceModel` file.
//! - [`tokenize`][]: `TokenizerFactory`, the events of `TokSpanEventStream`,
//!   and the `TokenizerModel` file.
//! - [`postag`][]: `POSTaggerFactory`, the events of `POSSampleEventStream`,
//!   and the `POSModel` file with its feature generator descriptor and tag
//!   dictionary.
//! - [`lemmatize`][]: `LemmatizerFactory`, the events of
//!   `LemmaSampleEventStream`, and the `LemmatizerModel` file.
//! - [`dictionary`][]: the abbreviation dictionary as `Dictionary.serialize`
//!   writes it, entries in Java's hash set order.
//! - [`tagdict`][]: the tag dictionary as `POSDictionary.serialize` writes
//!   it, entries in Java's hash map order.
//! - [`java_hash`][]: `String.hashCode` and the order a `java.util.HashMap`
//!   iterates in, which those orders and the zip entry order depend on.
//! - [`package`][]: the factory and serializer manifest entries and the
//!   zip entry order of `BaseModel.serialize`.
//! - [`ud`][]: what Apache OpenNLP trained the UD 1.3 models from and
//!   with, as typed recipes, and the comparison with the published files.
//!
//! The samples come from `turbo-formats`, the context generators from
//! `turbo-sentdetect`, `turbo-tokenize`, `turbo-sequence` and
//! `turbo-lemmatizer`, and the trainer, writer and packaging from
//! `turbo-train`; this crate connects them as the `train` methods do, so
//! training and inference share one implementation of each context
//! generator. It is a crate of its own because `turbo-formats` depends on
//! `turbo-tokenize`, so a training module inside the annotator crates
//! would close a cycle.
//!
//! OpenNLP behaviour a reader may not expect, all kept and tested:
//!
//! - A sentence with no end character gives no event, and a sentence's
//!   last end character is its split whatever follows it.
//! - The factories resolve a missing end character set or alphanumeric
//!   pattern to the language's before anything reads them, so a trained
//!   model's manifest always has `eosCharacters` and `alphaNumericPattern`.
//!   An empty end character array is not resolved: the JVM then trains
//!   with the language's scanner and context generator (Thai's
//!   `SentenceContextGenerator`) but records `eosCharacters=` empty, so
//!   the model file does not describe the model; this crate refuses it.
//! - The tokenizer's events are at UTF-16 positions, so a character
//!   outside the Basic Multilingual Plane gives an event between its two
//!   halves, with predicates that hold half a character.
//! - A whitespace unit of one code unit, and with the optimization an
//!   alphanumeric one, gives no event at all, so a model never sees a
//!   split next to such a unit.
//! - A sample token that overlaps a whitespace unit without lying inside
//!   it is skipped (the JVM logs a warning); the search for tokens never
//!   goes back, so tokens must be in text order.
//! - `Dictionary.serialize` writes the entries in the order of a
//!   `HashSet<StringList>`, by the hash of the lowercased tokens joined
//!   with commas, and `BaseModel.serialize` writes the zip entries in the
//!   order of a `HashMap<String, Object>`; both are simulated.
//! - `new Dictionary(InputStream)` collapses entries that differ only in
//!   case before it reads the `case_sensitive` attribute.
//! - `BaseModel.serialize` adds `serializer-class-abbreviations.dictionary`
//!   to the manifest only when a dictionary is present.
//! - The JDK's XML serializer writes a dictionary token's character outside
//!   the Basic Multilingual Plane, U+007F and the C1 controls U+0080 to
//!   U+009F as decimal character references, and throws on a C0 control
//!   character other than tab, line feed and carriage return, so a
//!   dictionary holding one cannot be written into a model file at all
//!   ([`Error::InvalidXmlCharacter`]).
//! - An empty training token (CoNLL-U with an empty word form gives one)
//!   makes a split event at its position, at index 0 of the unit when it
//!   is first there, where the context has `p1=bok` and no `p2` or `p1f1`.
//! - `POSTaggerFactory` with no descriptor loads `pos-default-features.xml`
//!   when its context generator is first made, so every trained `POSModel`
//!   carries a `generator.featuregen` entry; training uses no feature
//!   cache, so each token's features are computed afresh.
//! - `POSDictionary.create` keeps the entries in a `HashMap`, and when the
//!   dictionary is case insensitive copies them lowercased into a second
//!   one in the first one's iteration order; `serialize` writes that order,
//!   so the `tags.tagdict` of a trained model is not the file it was read
//!   from. A later entry for a word already present replaces its tags.
//! - `POSModel` refuses a tag dictionary with a tag the model has no
//!   outcome for ([`Error::UnknownDictionaryTags`]).
//! - The tag dictionary entry is written before the manifest when both are
//!   in the model file, because the `HashMap` of artifacts orders it so.
//! - The lemmatizer's features name a missing suffix or prefix `suf=null`
//!   and `pre=null`, Java's string of a null reference.
//! - A lemma class is the shortest edit script over UTF-16 units; a token
//!   or lemma outside the Basic Multilingual Plane can give a class that
//!   holds half of a surrogate pair. Outcomes are `JavaString`s, so such a
//!   class is trained on and written as the JVM does it, and the features
//!   that name the previous class hold it as the JVM's modified UTF-8.
//! - The published UD 1.3 models were trained with 300 iterations, not
//!   the 150 their training configuration file says ([`ud`][]).

pub mod dictionary;
pub mod error;
pub mod java_hash;
pub mod lemmatize;
pub mod package;
pub mod postag;
pub mod sentdetect;
pub mod tagdict;
pub mod tokenize;
pub mod ud;

pub use dictionary::AbbreviationDictionary;
pub use error::{Error, Result};
pub use lemmatize::{LemmaEventStream, LemmatizerFactory, lemma_events, train_lemmatizer_model};
pub use package::Stamp;
pub use postag::{PosEventStream, PosTaggerFactory, pos_events, train_pos_model};
pub use sentdetect::{SdEventStream, SentenceDetectorFactory, sd_events, train_sentence_model};
pub use tagdict::PosDictionary;
pub use tokenize::{TokSpanEventStream, TokenizerFactory, tok_events, train_tokenizer_model};
