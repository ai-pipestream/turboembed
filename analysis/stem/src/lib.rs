//! OpenNLP's stemmers: `PorterStemmer`, the Snowball stemmers of
//! `opennlp.tools.stemmer.snowball`, the light and minimal stemmers of
//! `opennlp.tools.stemmer.light`, `CachingStemmer`, and `StemmerAnnotator`
//! (here [`StemAnnotator`]) as an analysis step that reads `opennlp:tokens`
//! and writes `turbo:stems`.
//!
//! The behaviour follows OpenNLP at commit
//! `cdf33c653d5d61d5af7d37ffee189a83bd8392ea` (the light and minimal
//! stemmers there are the same as on the `ready/light-stemmers` branch of
//! opennlp-addons), and the tests compare it with what the Java classes
//! return (`golden/`, see `README.md`).
//!
//! # Modules
//!
//! - [`stemmer`]: the [`Stemmer`] trait, on UTF-16 code units.
//! - [`porter`]: [`PorterStemmer`], Porter's own Java implementation.
//! - [`snowball`]: [`SnowballStemmer`] and [`SnowballAlgorithm`]. The 21
//!   algorithms are generated from the Snowball sources (`snowball/` in
//!   this crate says how).
//! - [`light`]: [`LightStemmer`] and [`NorwegianStandard`].
//! - [`algorithm`]: [`Algorithm`], any of the above as one value.
//! - [`cache`]: [`CachingStemmer`], a cache of recent stems.
//! - [`annotate`]: [`StemAnnotator`], the analysis step.
//!
//! # UTF-16
//!
//! The Java stemmers work on `char`s, UTF-16 code units, and so do these:
//! a character outside the Basic Multilingual Plane is two units to every
//! length test and every step that moves by one character (so it can move
//! where a Snowball region starts), and neither unit is a letter or vowel
//! to any rule. [`Stemmer::stem_utf16`] takes and gives units as Java
//! does, lone surrogates included; [`Stemmer::stem`] and the annotator
//! hold UTF-8 and would put U+FFFD in place of a lone surrogate in a stem.
//! None of the stemmers returned a lone surrogate for a well-formed word in
//! the golden files or in a run over two million random words mixing
//! letters and supplementary characters.
//!
//! # OpenNLP behaviour worth knowing
//!
//! - No stemmer folds case. The Snowball stemmers, `PorterStemmer` and the
//!   light stemmers all expect lower-case words: an upper-case letter is
//!   no vowel (or no letter) to them, so `RUNNING` keeps its ending while
//!   `Running` stems to `Run`. The light stemmers also expect NFC, so an
//!   accent written as a combining mark is not folded or matched.
//! - `PorterStemmer` is Porter's own Java implementation of the 1980
//!   algorithm, which maps -bli and -logi as the paper does not (see
//!   [`porter`]) and returns words of one or two units as they are.
//!   Snowball's `porter` algorithm ([`SnowballAlgorithm::Porter`]) follows
//!   the paper there, and also stems `s` to the empty string, so the two
//!   give different stems for some words.
//! - `StemmerAnnotatorProvider` builds `PorterStemmer` for the option
//!   `porter` in any case, so Snowball's `PORTER` cannot be chosen through
//!   it, and none of the light stemmers can.
//! - `new SnowballStemmer(algorithm, repeat)` sets the word once and runs
//!   the algorithm `repeat` times on the state the last run left, cursor
//!   and limits included, so the second run is not always the stem of the
//!   stem: English `ugly` gives `ugli` after two runs, though `ugli` stems
//!   to `ug` ([`SnowballStemmer`]).
//! - The Snowball classes are generated from Snowball at commit
//!   `9a22f0d3f44cda36677829328fe2642750114d57` (January 2025, after
//!   release 2.2.0 and before 3.0.0). Later Snowball releases changed
//!   several algorithms, so OpenNLP's stems differ from current Snowball's
//!   for some words; `README.md` lists how many per language.
//! - The French light and minimal stemmers drop the second of two equal
//!   letters with `Character.isLetter` on a code unit; every other rule of
//!   the light stemmers compares fixed letters.
//! - `NorwegianLightStemmer` and `NorwegianMinimalStemmer` take a
//!   non-empty set of varieties; [`NorwegianStandard`] has its three
//!   values.
//!
//! # Not reproduced
//!
//! - The Hunspell stemmer of `opennlp.tools.stemmer.hunspell`, which needs
//!   a dictionary, and `Stemmer.stemAll`, which only it overrides.
//! - `SharingStemmer`, `DelegatingStemmer` and `clearThreadLocalState`:
//!   the stemmers here hold no state between words, so a value is shared
//!   by threads as it is.
//! - `StemmerAnnotatorProvider` and its string options; [`Algorithm`] is
//!   the typed choice.
//! - Characters assigned after Unicode 15.0 are unassigned in JDK 21; the
//!   tables here are Unicode 18.0, so `Character.isLetter` in the French
//!   light and minimal stemmers can answer differently for such a letter.

/// A string literal as a `&'static [u16]` of its UTF-16 code units, made at
/// compile time. The generated Snowball code writes every literal with it,
/// and the light stemmers their suffixes.
macro_rules! u16s {
    ($s:literal) => {{
        const S: &str = $s;
        const UNITS: [u16; $crate::snowball::runtime::utf16_len(S)] = $crate::snowball::runtime::utf16(S);
        &UNITS as &[u16]
    }};
}

pub mod algorithm;
pub mod annotate;
pub mod cache;
pub mod light;
pub mod porter;
pub mod snowball;
pub mod stemmer;

pub use algorithm::Algorithm;
pub use annotate::StemAnnotator;
pub use cache::CachingStemmer;
pub use light::{LightStemmer, NorwegianStandard};
pub use porter::PorterStemmer;
pub use snowball::{SnowballAlgorithm, SnowballStemmer};
pub use stemmer::Stemmer;
pub use turbo_stem_layers::Stems;
