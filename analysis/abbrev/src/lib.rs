//! Abbreviation and acronym definitions, found with the algorithm of
//! Schwartz and Hearst (2003), "A simple algorithm for identifying
//! abbreviation definitions in biomedical text", and the later uses of
//! each short form, linked to its long form.
//!
//! - [`definitions_in`] finds the definitions in one sentence: "World
//!   Health Organization (WHO)" and the reverse, "WHO (World Health
//!   Organization)", with byte offsets of both forms.
//! - [`detect`] runs it over a view's sentences and finds the mentions.
//! - [`AbbreviationAnnotator`] is the step: it reads a text view, its
//!   `opennlp:sentences` and, when the document has them, its
//!   `opennlp:tokens`, and writes `turbo:abbreviations`
//!   ([`Abbreviations`]) and the links between its items,
//!   `turbo:edges<turbo:abbreviations, turbo:abbreviations>`.
//! - [`gold`] reads Ab3P's labelled MED1250 set and scores pairs against
//!   it; `examples/med1250.rs` runs the measurement.
//!
//! There is no OpenNLP original. The rules follow the paper and the
//! authors' reference implementation, `ExtractAbbrev.java`, as described
//! in [`definitions_in`]; scispaCy's `AbbreviationDetector` (Apache-2.0)
//! supplied edge cases for the tests. Characters are compared as
//! `Character.isLetterOrDigit` and `Character.toLowerCase` see them, read
//! from `turbo-unicode`'s tables. Unlike the Java, which cuts a line at
//! ". " and ", " and needs a space before a parenthesis, any whitespace
//! counts here, and the sentences come from the document.
//!
//! # What may surprise
//!
//! - **A short form may start with a digit.** The paper allows a letter or
//!   a digit first, so "(3-i)" after "data:" can define a short form when
//!   the long form candidate happens to fit; scispaCy refuses a short form
//!   that does not start with a letter.
//! - **The long form runs to the parenthesis.** Only its start is
//!   chosen, so words after the last matched letter stay in it: "World
//!   Health Organization in Geneva (WHO)" gives the long form "World
//!   Health Organization in Geneva", within the word limit.
//! - **Mentions are exact.** "WHO" mentions "WHO" only: not "who", not
//!   "WHOs". A mention inside another word is not one ("WHOLE"), and with
//!   tokens, a mention must be whole tokens.
//! - **A short form opening with `(` still anchors at a word start.** When
//!   a short form begins with an unclosed `(` (such as `"(AB"` from
//!   `((AB)`), its first letter or digit after the leading `(` must match
//!   the start of a word in the long form, whereas `ExtractAbbrev.java`
//!   tests `sIndex == 0` only and leaves the first letter unanchored when
//!   index 0 is punctuation.

mod annotate;
mod detect;
pub mod export;
pub mod gold;

pub use annotate::{AbbreviationAnnotator, detect};
pub use detect::{Definition, best_long_form, definitions_in};
pub use turbo_analysis_layers::{AbbreviationForm, Abbreviations, DEFINITION, MENTION, abbreviation_links};
