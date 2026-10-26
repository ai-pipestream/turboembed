//! Frequency lexicons from the model zoo (#935): the unigram and bigram
//! lists [`FrequencyDictionary`] reads, handed out by zoo id as verified
//! local files, and the engine and language model built from them.
//!
//! [`Lexicon::fetch`] asks a [`Zoo`] for an artifact by id, checks that it
//! is a `KIND_LEXICON` entry of one `FORMAT_TEXT` file, and reads that
//! file. The licence check and the size, SHA-256 and SHA-512 checks are the
//! zoo's, so a lexicon handed out here holds the bytes the index records.
//! [`EN_UNIGRAMS`] and [`EN_BIGRAMS`] are the English lists the built-in
//! index pins (docs/licences.md, "Frequency lexicons").

use std::fmt;
use std::path::PathBuf;

use turbo_ngram::NGramLanguageModel;
use turbo_zoo::{Acceptances, ArtifactId, Format, Kind, UserId, Zoo, ZooError};

use super::config::SymSpellConfig;
use super::dictionary::{FrequencyDictionary, MalformedLine};
use super::symspell::SymSpell;

/// SymSpell's English unigram list: 82,834 words load from it (the 82,765 in
/// the id is upstream's file name).
pub const EN_UNIGRAMS: &str = "symspell-frequency-dictionary-en-82-765";

/// SymSpell's English bigram list: 242,342 pairs (the 243,342 in the id
/// is upstream's file name).
pub const EN_BIGRAMS: &str = "symspell-frequency-bigramdictionary-en-243-342";

/// Why a lexicon could not be fetched, read or loaded.
#[derive(Debug)]
pub enum LexiconError {
    /// The zoo refused the id or could not hand out its files.
    Zoo(ZooError),
    /// The artifact is not one text file of kind `KIND_LEXICON`.
    NotALexicon { id: String, kind: Kind, format: Format, files: usize },
    /// The verified file could not be read.
    Read { path: PathBuf, error: std::io::Error },
    /// The file is not UTF-8.
    NotUtf8 { path: PathBuf },
    /// A line of the list does not have the shape its loader expects.
    Malformed { id: String, line: MalformedLine },
    /// The language model refused an entry.
    LanguageModel(turbo_ngram::Error),
}

impl fmt::Display for LexiconError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LexiconError::Zoo(e) => write!(f, "{e}"),
            LexiconError::NotALexicon { id, kind, format, files } => {
                write!(f, "{id} is {kind} {format} with {files} files, not one KIND_LEXICON FORMAT_TEXT file")
            }
            LexiconError::Read { path, error } => write!(f, "reading {}: {error}", path.display()),
            LexiconError::NotUtf8 { path } => write!(f, "{} is not UTF-8", path.display()),
            LexiconError::Malformed { id, line } => write!(f, "{id}: {line}"),
            LexiconError::LanguageModel(e) => write!(f, "language model: {e}"),
        }
    }
}

impl std::error::Error for LexiconError {}

impl From<ZooError> for LexiconError {
    fn from(e: ZooError) -> LexiconError {
        LexiconError::Zoo(e)
    }
}

/// A frequency list's text, as the zoo verified it, under its zoo id.
#[derive(Clone, Debug)]
pub struct Lexicon {
    id: ArtifactId,
    text: String,
}

impl Lexicon {
    /// The lexicon `id` from `zoo`, for `user`, reading the acceptances
    /// through `load` only when the entry needs some ([`Zoo::fetch_with`]).
    pub fn fetch(
        zoo: &Zoo<'_>,
        id: &str,
        user: &UserId,
        load: impl FnOnce() -> Result<Acceptances, ZooError>,
    ) -> Result<Lexicon, LexiconError> {
        let a = zoo.index().get(id)?;
        if a.kind != Kind::Lexicon || a.format != Format::Text || a.files.len() != 1 {
            return Err(LexiconError::NotALexicon {
                id: id.to_owned(),
                kind: a.kind,
                format: a.format,
                files: a.files.len(),
            });
        }
        let fetched = zoo.fetch_with(id, user, load)?;
        let path = fetched.single().expect("one file, checked above").to_path_buf();
        let bytes = std::fs::read(&path).map_err(|error| LexiconError::Read { path: path.clone(), error })?;
        let text = String::from_utf8(bytes).map_err(|_| LexiconError::NotUtf8 { path })?;
        Ok(Lexicon { id: fetched.artifact.id.clone(), text })
    }

    pub fn id(&self) -> &str {
        self.id.as_str()
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    fn malformed(&self, line: MalformedLine) -> LexiconError {
        LexiconError::Malformed { id: self.id.to_string(), line }
    }
}

impl SymSpell {
    /// An engine with `config` indexing the `word count` lines of
    /// `unigrams`, and the `w1 w2 count` lines of `bigrams` for
    /// [`SymSpell::lookup_compound`].
    pub fn from_lexicons(
        config: SymSpellConfig,
        unigrams: &Lexicon,
        bigrams: Option<&Lexicon>,
    ) -> Result<SymSpell, LexiconError> {
        let mut engine = SymSpell::new(config);
        FrequencyDictionary::load_unigrams(&mut engine, unigrams.text()).map_err(|l| unigrams.malformed(l))?;
        if let Some(b) = bigrams {
            FrequencyDictionary::load_bigrams(&mut engine, b.text()).map_err(|l| b.malformed(l))?;
        }
        Ok(engine)
    }
}

/// A bigram language model for [`super::ContextSpellCorrector`] from a
/// unigram and a bigram list.
///
/// The lists count words and pairs on one scale, as SymSpell's own
/// compound lookup reads them (both against one corpus size). The model
/// keeps counts as `i32` and its corpus size as their `i32` sum, while the
/// SymSpell lists count in the tens of billions, so every count is divided
/// by one divisor, the smallest that keeps the sum in range, rounded to
/// nearest and kept at 1 or more. The decoder scores a pair by its count
/// over its first word's and a word by its count over the sum, so one
/// divisor leaves both ratios as the lists give them, up to that rounding.
/// A word repeated in a list has its counts summed.
pub fn language_model(unigrams: &Lexicon, bigrams: &Lexicon) -> Result<NGramLanguageModel, LexiconError> {
    let mut total: u128 = 0;
    let mut entries: u128 = 0;
    FrequencyDictionary::read_unigrams(unigrams.text(), |_, c| {
        total += u128::from(c);
        entries += 1;
    })
    .map_err(|l| unigrams.malformed(l))?;
    FrequencyDictionary::read_bigrams(bigrams.text(), |_, _, c| {
        total += u128::from(c);
        entries += 1;
    })
    .map_err(|l| bigrams.malformed(l))?;
    // Rounding adds at most one to each entry, so the room left for the
    // divided counts is the range less one per entry.
    let room = (i32::MAX as u128).saturating_sub(entries).max(1);
    let divisor = total.div_ceil(room).max(1);
    let scale = |c: u64| -> i32 {
        let scaled = (u128::from(c) + divisor / 2) / divisor;
        scaled.clamp(1, i32::MAX as u128) as i32
    };
    let mut lm = NGramLanguageModel::new(2).map_err(LexiconError::LanguageModel)?;
    let mut failed = None;
    let mut put = |lm: &mut NGramLanguageModel, ngram: &[&str], c: u64| {
        if failed.is_some() {
            return;
        }
        let held = lm.get_count(ngram);
        // The model's own `add`, not the language model's, which would also
        // count every shorter gram inside.
        let step = if held == 0 { lm.model_mut().add(ngram) } else { Ok(()) };
        if let Err(e) = step.and_then(|()| lm.set_count(ngram, held.saturating_add(scale(c)))) {
            failed = Some(e);
        }
    };
    FrequencyDictionary::read_unigrams(unigrams.text(), |w, c| put(&mut lm, &[w], c)).expect("read above");
    FrequencyDictionary::read_bigrams(bigrams.text(), |w1, w2, c| put(&mut lm, &[w1, w2], c)).expect("read above");
    match failed {
        Some(e) => Err(LexiconError::LanguageModel(e)),
        None => Ok(lm),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lexicon(id: &str, text: &str) -> Lexicon {
        Lexicon { id: id.parse().unwrap(), text: text.to_owned() }
    }

    #[test]
    fn counts_past_i32_are_divided_by_one_divisor() {
        let uni = lexicon("u", "the 23135851162\nof 13151942776\nrare 12714\nrarer 12\n");
        let bi = lexicon("b", "of the 177045273024\nthe of 3\n");
        let lm = language_model(&uni, &bi).unwrap();
        assert!(lm.number_of_grams() > 0, "the sum stays in range");
        let (the, of) = (lm.get_count(&["the"]), lm.get_count(&["of"]));
        let ratio = f64::from(the) / f64::from(of);
        assert!((ratio - 23135851162.0 / 13151942776.0).abs() < 1e-6, "{ratio}");
        // The sum is about 2.1e11, so the divisor is about 100.
        assert_eq!(lm.get_count(&["rare"]), 127);
        assert_eq!(lm.get_count(&["rarer"]), 1, "a small count stays at one");
        assert_eq!(lm.get_count(&["the", "of"]), 1);
        assert_eq!(lm.order(), 2);
    }

    #[test]
    fn small_counts_are_kept_as_they_are_and_repeats_summed() {
        let uni = lexicon("u", "cat 5\ncat 2\ndog 3\n");
        let bi = lexicon("b", "the cat 4\n");
        let lm = language_model(&uni, &bi).unwrap();
        assert_eq!((lm.get_count(&["cat"]), lm.get_count(&["dog"])), (7, 3));
        assert_eq!(lm.get_count(&["the", "cat"]), 4);
        assert_eq!(lm.number_of_grams(), 14);
    }

    #[test]
    fn a_malformed_line_names_its_lexicon() {
        let uni = lexicon("u", "cat 5\n");
        let bi = lexicon("bigrams-x", "the 4\n");
        let e = language_model(&uni, &bi).unwrap_err();
        assert!(e.to_string().starts_with("bigrams-x: malformed dictionary line 1"), "{e}");
        let e = SymSpell::from_lexicons(SymSpellConfig::default(), &uni, Some(&bi)).unwrap_err();
        assert!(matches!(e, LexiconError::Malformed { ref id, .. } if id == "bigrams-x"), "{e}");
    }
}
