//! Training OpenNLP's sentence detector and tokenizer: the event streams
//! `SDEventStream` and `TokSpanEventStream`, the factories that set them
//! up, the model files `SentenceDetectorME.train` and `TokenizerME.train`
//! return, and the reproduction of the published UD 1.3 models.
//!
//! - [`sentdetect`][]: `SentenceDetectorFactory`, the events, and the
//!   `SentenceModel` file.
//! - [`tokenize`][]: `TokenizerFactory`, the events, and the
//!   `TokenizerModel` file.
//! - [`dictionary`][]: the abbreviation dictionary as `Dictionary.serialize`
//!   writes it, entries in Java's hash set order.
//! - [`java_hash`][]: `String.hashCode` and the order a `java.util.HashMap`
//!   iterates in, which that order and the zip entry order depend on.
//! - [`package`][]: the factory and serializer manifest entries and the
//!   zip entry order of `BaseModel.serialize`.
//! - [`ud`][]: what Apache OpenNLP trained the UD 1.3 sentence and token
//!   models from and with, as typed recipes, and the comparison with the
//!   published files.
//!
//! The samples come from `turbo-formats`, the context generators from
//! `turbo-sentdetect` and `turbo-tokenize`, and the trainer, writer and
//! packaging from `turbo-train`; this crate connects them as the two
//! `train` methods do. It is a crate of its own because `turbo-formats`
//! depends on `turbo-tokenize`, so a training module inside the
//! annotator crates would close a cycle.
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
//! - The published UD 1.3 models were trained with 300 iterations, not
//!   the 150 their training configuration file says ([`ud`][]).

pub mod dictionary;
pub mod error;
pub mod java_hash;
pub mod package;
pub mod sentdetect;
pub mod tokenize;
pub mod ud;

pub use dictionary::AbbreviationDictionary;
pub use error::{Error, Result};
pub use package::Stamp;
pub use sentdetect::{SdEventStream, SentenceDetectorFactory, sd_events, train_sentence_model};
pub use tokenize::{TokSpanEventStream, TokenizerFactory, tok_events, train_tokenizer_model};
