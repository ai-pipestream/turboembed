//! OpenNLP's Hunspell stemmer: `HunspellDictionary` (here [`Dictionary`]),
//! `HunspellStemmer` ([`Stemmer`]), `HunspellStemmerFactory` (a shared
//! [`Dictionary`] and [`Stemmer::new`]) and, through `StemmerAnnotator`,
//! an analysis step that reads `opennlp:tokens` and writes `turbo:stems`
//! ([`HunspellAnnotator`]).
//!
//! The behaviour follows the classes of the `OPENNLP-1927-hunspell-compatibility`
//! branch of OpenNLP at commit `fb299cf32ccf6f5dbc81b39448d2d7bea1cf66dc`
//! (with `AffixCondition`, `HunspellAffixTable`, `HunspellCompoundRule` and
//! `HunspellConversion`), and the tests compare it with what those classes
//! return for LibreOffice's `en_US`, `de_DE_frami` and `hu_HU` dictionaries
//! and for small hand-written ones (`README.md`).
//!
//! No dictionary ships with the crate: the published ones are under the
//! GPL, LGPL or MPL, or carry their own notices, and are downloaded where
//! they are used.
//!
//! # Modules
//!
//! - `read`: the reader of the two files, from the bytes on: `SET`
//!   encodings, the directives read and the ones refused, flags, `AF` and
//!   `AM` aliases, the tables (`ICONV`, `OCONV`, `REP`, `BREAK`,
//!   `COMPOUNDRULE`, `CHECKCOMPOUNDPATTERN`) and the affix blocks.
//! - `dictionary`: the loaded words and rules, and what the flags allow.
//! - `stemmer`: the search for readings: affixes, case variants, compounds
//!   and word breaks.
//! - `morphology`: the stem and analysis a reading gives from its fields.
//!
//! # Where the port differs
//!
//! - Java decodes any encoding the JDK knows. The port decodes UTF-8 and
//!   the single byte encodings of the JDK listed in `charset/tables.rs`
//!   (the ISO 8859 parts, KOI8, the Windows code pages, TIS-620), and
//!   refuses another name the JDK knows with
//!   [`LoadError::EncodingNotPorted`].
//! - [`Stemmer::stems`] and [`Stemmer::stem`] return nothing for a word
//!   the dictionary does not read, where OpenNLP returns the word itself.
//!   The annotator writes the word itself, as OpenNLP does.
//! - OpenNLP's `HunspellDictionaryInstaller` copies dictionaries named in
//!   an application's catalog; the crate has no catalog and no download.
//! - Where two words have the same capitalized form, OpenNLP orders their
//!   hidden entries by the iteration order of a `java.util.HashMap`, which
//!   the port reproduces except for a bucket the JDK turns into a tree
//!   (eight or more colliding words in one bucket of a large table).
//! - A compound judges case on the input as written only where its
//!   characters stand where the case variant's do (`stemmer/compound.rs`);
//!   OpenNLP asks only for the same UTF-16 length. The two agree on every
//!   variant the stemmer makes.
//! - Affix rules are looked up by the text they add rather than scanned by
//!   their boundary character; the order of the rules tried is OpenNLP's.

mod affix;
mod annotate;
mod charset;
mod compound_rule;
mod condition;
mod conversion;
mod dictionary;
mod error;
mod flag;
mod hash;
mod morphology;
mod read;
mod settings;
mod stemmer;
mod text;

pub use affix::{Affix, AffixKind};
pub use annotate::HunspellAnnotator;
pub use condition::Condition;
pub use dictionary::Dictionary;
pub use error::{LoadError, Stream};
pub use flag::{Flag, FlagSet};
pub use read::MAX_FILE_BYTES;
pub use read::bytes::UnsupportedDirective;
pub use stemmer::Stemmer;
pub use turbo_stem_layers::Stems;

/// What a load does with a directive the reader does not support:
/// OpenNLP's `LoadMode`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum LoadMode {
    /// Fail with [`LoadError::UnsupportedDirective`].
    #[default]
    Strict,
    /// Skip it and list it in [`Dictionary::unsupported_directives`].
    AllowPartial,
}
