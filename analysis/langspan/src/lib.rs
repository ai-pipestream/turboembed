//! Language by span (code switching): the language and script of each
//! run of a text, for documents that change language part way.
//!
//! # Layers
//!
//! `opennlp:language` stays OpenNLP's document-level layer, one ranking
//! for the whole text, so a pipeline written against OpenNLP reads what it
//! expects. The spans are a new layer, `turbo:language-spans`
//! ([`LanguageSpans`](turbo_analysis_layers::LanguageSpans)): changing the
//! anchor of `opennlp:language` from the document to spans would change
//! what every reader of it gets. The step here writes both, and derives
//! the document layer from the spans ([`annotate::summary`]): each
//! language with its share of the letters, largest first. For OpenNLP's
//! own document ranking (every language with its probability) run
//! `turbo_langdetect::LanguageDetector` instead; a document has one
//! writer of `opennlp:language`.
//!
//! # How it works
//!
//! 1. **Words** ([`units`]): the letters of the text cut into words, each
//!    in one script group (Han and kana are one group). Inside the tokens
//!    of a token layer when the document has one. Words OpenNLP's
//!    normalizers would remove (URLs, e-mail addresses, hashtags,
//!    mentions) are skipped.
//! 2. **Script first** ([`scripts`]): consecutive words of one script
//!    group make a segment, and a segment can only be a language written
//!    in that script. A script with one such language in the model (Greek,
//!    Hebrew, Thai, Hangul and others for `langdetect-183.bin`) is that
//!    language with no scoring; a script with none is `und`.
//! 3. **Shortlist**: OpenNLP's detector ([`turbo_langdetect`]) scores the
//!    whole segment and each clause (words between punctuation or
//!    sentence breaks) of at least [`Settings::clause_letters`] letters;
//!    the best language of each, among the script's languages, makes the
//!    shortlist. A segment with a shortlist of one is that language.
//! 4. **Labelling** ([`decode`]): each word is scored alone (with a space
//!    either side) against the shortlist, and a Viterbi search finds the
//!    labelling with the best total score less a cost for every switch.
//!    A switch costs least at a sentence boundary or line break, more at
//!    punctuation, most between two words with only a space between
//!    ([`Settings::switch`]). With a sentence layer, its boundaries are
//!    the sentence boundaries; without one, sentence-final punctuation is.
//! 5. **Runs**: each stretch of one label is scored again as a whole and
//!    given its best shortlisted language, and neighbours that end up
//!    with the same language merge. A run's confidence is the model's
//!    probability of its language for its whole text, over the languages
//!    of its script.
//! 6. **Spans**: runs touch. A run starts at the start of a sentence that
//!    begins between the word before it and its first word (snapped to
//!    the sentence layer), or else at its first word less any opening
//!    punctuation attached to it, moved back to the start of the token it
//!    falls inside (snapped to the token layer). The last run ends after
//!    its last word and any closing punctuation attached to it.
//!
//! The approach follows lingua's `detect_multiple_languages_of` (a
//! candidate set from the whole text and its parts, then a label per
//! word, merged into runs) and whatlang and CLD3 (script before
//! n-grams); no code or data of theirs is used. The settings were chosen
//! on `tests/data/mixed-dev.txt` and the detector is measured on the
//! held-out `tests/data/mixed-test.txt`; docs/langdetect.md has the
//! numbers.

pub mod annotate;
pub mod decode;
pub mod detector;
pub mod gold;
pub mod scripts;
pub mod units;

pub use annotate::summary;
pub use detector::{Run, Settings, SpanDetector, SpanScratch};
