//! OpenNLP's name finders that need no model: `DictionaryNameFinder`,
//! `RegexNameFinder` and the patterns of `RegexNameFinderFactory`, and the
//! merge of several finders' names.
//!
//! Each finder reads a text view and its `opennlp:tokens` and writes
//! `opennlp:entities`: token ranges, UTF-8 byte spans of the view, entity
//! types and scores. [`MergeEntities`] reads several named instances of
//! `opennlp:entities` and writes one without overlaps, as OpenNLP's
//! `TokenNameFinderTool` combines its finders.
//!
//! The behaviour follows OpenNLP at commit
//! `cdf33c653d5d61d5af7d37ffee189a83bd8392ea` and `java.util.regex` in JDK
//! 21, and the tests compare it with what they return (the goldens under
//! `analysis/goldens/data/namefind`, see `README.md`). On the token level
//! the finders are [`TokenNameFinder`]s, which take a sentence's tokens as
//! strings, as the Java `find` does.
//!
//! # Modules
//!
//! - [`dictionary`]: [`DictionaryNameFinder`] over a `turbo_maxent`
//!   [`Dictionary`](turbo_maxent::Dictionary) read from OpenNLP's XML.
//! - [`regex_finder`]: [`RegexNameFinder`], its factory functions and
//!   [`Pattern`].
//! - [`builtin`]: [`BuiltIn`], the factory's five patterns, matched as
//!   Java matches them.
//! - [`java_regex`]: [`JavaRegex`], a user's Java pattern compiled to a
//!   direct bytecode matcher, or refused.
//! - [`merge`](mod@merge): `dropOverlappingSpans` and [`MergeEntities`].
//! - [`annotate`]: the two finders as analysis steps.
//!
//! # Regular expressions
//!
//! This crate uses no external `regex` engine. The five built-in patterns
//! are matched directly over the joined UTF-8 bytes by hand-written
//! scanners with no regex compilation or stand-in buffer ([`builtin`]
//! explains how, including `EMAIL`'s lookbehind and `URL`'s word
//! boundaries). A user's pattern goes through [`JavaRegex`], which parses
//! `java.util.regex.Pattern` syntax into a compact bytecode program and
//! runs it with a direct cursor matcher that preserves Java's ASCII
//! character classes, ASCII case folding, and backtracking priority while
//! refusing unsupported features with a typed error; [`java_regex`] lists
//! what is refused, and the three constructs that refuse some texts when
//! matched.
//!
//! # OpenNLP behaviour worth knowing
//!
//! - `RegexNameFinder` returns names type by type, in the iteration order
//!   of its map, not in text order. The factory's map is a `HashMap`, so
//!   the order of its types follows `String.hashCode` and the map's table
//!   size, which depends on how the map was filled: the factory adds the
//!   defaults one map at a time, which starts the table at two buckets
//!   and doubles it as it fills.
//! - A regex match counts only when it starts and ends at a token
//!   boundary of the space-joined tokens. A match that starts with the
//!   separating space starts at the next token; one that ends with it ends
//!   at the token before. A match dropped because it ends inside a token
//!   still consumes its text, so a shorter match inside it is not found.
//! - `URL` starts with `\b`, so its `/` and `~/` forms need a word
//!   character just before the slash or tilde: `~/a.com` at the start of
//!   a sentence or after a space is never found, and in `x/a.com` the
//!   match starts inside the token and is dropped.
//! - The classes of `DEGREES_MIN_SEC_LAT_LON` are written like
//!   `[N|n|S|s]`, so they also match `|`.
//! - In `java.util.regex`, a nonspacing mark after a letter or digit is a
//!   word character for `\b` but not for `\w`. A mark outside the Basic
//!   Multilingual Plane counts so on its right side only, since Java reads
//!   the character before a position as its last UTF-16 unit.
//! - `DictionaryNameFinder` keeps the longest entry from each token and
//!   resumes after it, so of two overlapping entries only the one that
//!   starts first is found, even when the other is longer.
//! - `Dictionary(InputStream)` stores every entry before it reads its
//!   case setting, so of two entries that differ only in case the first is
//!   kept even in a case-sensitive dictionary (`turbo_maxent` reproduces
//!   it).
//! - `dropOverlappingSpans` counts an empty span at the end of a kept
//!   span as inside it, and drops it.
//! - Names from these finders carry probability 0, which becomes the
//!   entity's score.
//!
//! # Not reproduced
//!
//! - `RegexNameFinder.find(String)`, over the characters of a text rather
//!   than tokens.
//! - Characters assigned after Unicode 15.0 are unassigned in JDK 21; the
//!   tables here are Unicode 18.0, so such a character can count as a
//!   letter or a nonspacing mark here and not there, which matters for
//!   `\b` only. U+1171E, a spacing mark since Unicode 16.0, is treated as
//!   the nonspacing mark JDK 21 knows.

pub mod annotate;
pub mod crf;
pub use crf::{CrfNameAnnotator, CrfNameFinder, CrfNameSequence, CrfSequenceModel};
pub mod builtin;
pub mod dictionary;
pub mod error;
pub mod java_regex;
pub mod merge;
pub mod name;
#[cfg(test)]
mod parity;
pub mod regex_finder;

pub use builtin::BuiltIn;
pub use dictionary::DictionaryNameFinder;
pub use error::{Error, Feature, TextLimit};
pub use java_regex::{Flags, JavaRegex};
pub use merge::{FilterEntities, MergeEntities, drop_overlapping, merge};
pub use name::{Name, TokenNameFinder};
pub use regex_finder::{Pattern, RegexNameFinder};
pub use turbo_analysis::FilterRule;
