//! `MorfologikDictionaryReader` and `PoliMorfDictionaryReader`, ported from
//! `opennlp.tools.formats`.

use std::collections::HashMap;
use std::path::Path;

use turbo_analysis::{Annotator, Error, ModelRef, Refs, View};
use turbo_analysis_layers::{Lemmas, PosTags, Strs, Tokens};
use turbo_unicode::java::code_point::is_whitespace;
use turbo_unicode::java::to_lowercase;

use crate::charset::DictionaryCharset;
use crate::encoding::{BaseFormEncoding, decode_base_form};
use crate::error::MorfologikError;
use crate::fsa::FsaReader;
use crate::metadata::{MorfologikInputConversion, MorfologikMetadata};

/// Sentinel returned by [`ExpandedDictionaryLemmatizer::lemmatize`] when a `(word, tag)` pair
/// is not present in the dictionary.
pub const NOT_FOUND: &str = "O";

/// In-memory dictionary lemmatizer produced by [`MorfologikDictionaryReader`] and
/// [`PoliMorfDictionaryReader`], matching OpenNLP's `DictionaryLemmatizer` and
/// `MorfologikInputConversion.ConvertedLemmatizer`.
#[derive(Debug, Clone)]
pub struct ExpandedDictionaryLemmatizer {
    entries: HashMap<String, HashMap<String, Vec<String>>>,
    conversion: MorfologikInputConversion,
    model_ref: ModelRef,
}

impl ExpandedDictionaryLemmatizer {
    pub(crate) fn new(
        entries: HashMap<String, HashMap<String, Vec<String>>>,
        conversion: MorfologikInputConversion,
        parts_for_digest: &[&[u8]],
        id_prefix: &str,
    ) -> Self {
        let model_ref = ModelRef {
            id: id_prefix.to_owned(),
            revision: String::new(),
            sha256: crate::identity::digest(parts_for_digest),
        };
        Self { entries, conversion, model_ref }
    }

    /// Number of `(lowercased_form, tag)` keys in the dictionary map (`getDictMap().size()`).
    pub fn len(&self) -> usize {
        self.entries.values().map(HashMap::len).sum()
    }

    /// Whether the dictionary map is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Inserts or replaces the lemma list for `(lowercased_form, tag)` in the dictionary map
    /// (`getDictMap().put(List.of(form, tag), lemmas)`).
    /// The provenance digest includes the ordered edits after the original input.
    pub fn put(&mut self, lowercased_form: impl Into<String>, tag: impl Into<String>, lemmas: Vec<String>) {
        let word = lowercased_form.into();
        let tag = tag.into();
        let mut parts = vec![self.model_ref.sha256.as_bytes(), word.as_bytes(), tag.as_bytes()];
        parts.extend(lemmas.iter().map(String::as_bytes));
        self.model_ref.sha256 = crate::identity::digest(&parts);
        self.entries.entry(word).or_default().insert(tag, lemmas);
    }

    /// Looks up the first lemma for `(word, tag)`, returning [`NOT_FOUND`] (`"O"`) on a miss.
    pub fn lemma<'a>(&'a self, word: &str, tag: &str) -> &'a str {
        self.lemmas_for(word, tag).and_then(|v| v.first()).map_or(NOT_FOUND, String::as_str)
    }

    /// Looks up all lemmas for `(word, tag)` in first-seen order, or `None` on a miss.
    pub fn lemmas_for(&self, word: &str, tag: &str) -> Option<&[String]> {
        let key_word = self.conversion.convert_cow(word);
        self.entries.get(key_word.as_ref())?.get(tag).map(Vec::as_slice)
    }

    /// `DictionaryLemmatizer.lemmatize(String[] tokens, String[] tags)`: returns the first
    /// lemma for each token and tag, or `"O"` when absent.
    pub fn lemmatize<S1: AsRef<str>, S2: AsRef<str>>(&self, tokens: &[S1], tags: &[S2]) -> Vec<String> {
        assert_eq!(tokens.len(), tags.len(), "tokens and tags must have the same length");
        tokens.iter().zip(tags.iter()).map(|(tok, tag)| self.lemma(tok.as_ref(), tag.as_ref()).to_owned()).collect()
    }

    /// `DictionaryLemmatizer.lemmatize(List<String> tokens, List<String> tags)`: returns
    /// the list of all lemmas for each token and tag, or `vec!["O"]` when absent.
    pub fn lemmatize_all<S1: AsRef<str>, S2: AsRef<str>>(&self, tokens: &[S1], tags: &[S2]) -> Vec<Vec<String>> {
        assert_eq!(tokens.len(), tags.len(), "tokens and tags must have the same length");
        tokens
            .iter()
            .zip(tags.iter())
            .map(|(tok, tag)| match self.lemmas_for(tok.as_ref(), tag.as_ref()) {
                Some(list) if !list.is_empty() => list.to_vec(),
                _ => vec![NOT_FOUND.to_owned()],
            })
            .collect()
    }

    /// Provenance reference for this dictionary.
    pub fn model_ref(&self) -> &ModelRef {
        &self.model_ref
    }
}

impl Annotator for ExpandedDictionaryLemmatizer {
    const ID: &'static str = "turbo-morfologik:dictionary-lemmatizer";
    const VERSION: &'static str = "2";
    type Reads = (View, Tokens, PosTags);
    type Writes = (Lemmas,);

    fn annotate(&self, (view, tokens, tags): Refs<'_, Self>) -> Result<(Lemmas,), Error> {
        if tokens.spans.view() != view.id() {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!(
                    "the tokens are in view {}, not in view {} that is lemmatized",
                    tokens.spans.view().0,
                    view.id().0
                ),
            });
        }
        if tokens.spans.len() != tags.tags.len() {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!("{} tokens but {} POS tags", tokens.spans.len(), tags.tags.len()),
            });
        }
        let text = view.text();
        let mut out = Strs::default();
        for ((_, span), &tag) in tokens.spans.iter().zip(&tags.tags) {
            let word = text.get(span.start as usize..span.end as usize).ok_or_else(|| Error::Annotator {
                annotator: Self::ID,
                detail: format!("token span {}..{} out of bounds", span.start, span.end),
            })?;
            if usize::from(tag.0) >= tags.tagset.len() {
                return Err(Error::Annotator {
                    annotator: Self::ID,
                    detail: format!("POS tag {} outside tag set", tag.0),
                });
            }
            out.push(self.lemma(word, tags.tagset.label(tag)));
        }
        Ok((Lemmas { lemma: out },))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.model_ref.clone())
    }
}

/// Collects distinct lemmas in first-seen order per `(lowercased_form, tag)`
/// (`opennlp.tools.formats.LemmatizerEntries`).
#[derive(Default)]
struct LemmatizerEntries {
    entries: HashMap<String, HashMap<String, Vec<String>>>,
}

impl LemmatizerEntries {
    fn add(&mut self, form: &str, tag: String, lemma: String) {
        let mut lower = String::with_capacity(form.len());
        to_lowercase(form, &mut lower);
        let list = self.entries.entry(lower).or_default().entry(tag).or_default();
        if !list.contains(&lemma) {
            list.push(lemma);
        }
    }
}

/// Builds an [`ExpandedDictionaryLemmatizer`] from a Morfologik FSA5 or CFSA2 dictionary
/// (`opennlp.tools.formats.MorfologikDictionaryReader`).
pub struct MorfologikDictionaryReader;

impl MorfologikDictionaryReader {
    /// Reads a Morfologik dictionary from automaton bytes using an explicit separator,
    /// base-form encoder, and character encoding.
    pub fn read_with_options(
        dictionary: &[u8],
        separator: u8,
        encoding: BaseFormEncoding,
        charset: DictionaryCharset,
    ) -> Result<ExpandedDictionaryLemmatizer, MorfologikError> {
        let entries = Self::read_entries(dictionary, separator, encoding, charset)?;
        Ok(ExpandedDictionaryLemmatizer::new(
            entries.entries,
            MorfologikInputConversion::default(),
            &[dictionary, &[separator], encoding.as_str().as_bytes(), charset.canonical_name().as_bytes()],
            "opennlp:MorfologikDictionaryReader",
        ))
    }

    /// Reads a Morfologik dictionary from automaton bytes and `.info` metadata bytes.
    /// Metadata is validated before reading the automaton.
    pub fn read(dictionary: &[u8], info: &[u8]) -> Result<ExpandedDictionaryLemmatizer, MorfologikError> {
        let meta = MorfologikMetadata::parse(info)?;
        let entries = Self::read_entries(dictionary, meta.separator, meta.encoder, meta.charset)?;
        Ok(ExpandedDictionaryLemmatizer::new(
            entries.entries,
            meta.input_conversion,
            &[dictionary, info],
            "opennlp:MorfologikDictionaryReader",
        ))
    }

    /// Reads a Morfologik dictionary from a `.dict` file path and its sibling `.info` file.
    pub fn read_path(dict_path: impl AsRef<Path>) -> Result<ExpandedDictionaryLemmatizer, MorfologikError> {
        let dict_path = dict_path.as_ref();
        let info_path = MorfologikMetadata::expected_metadata_path(dict_path);
        let info_bytes = std::fs::read(&info_path)
            .map_err(|e| MorfologikError::format(format!("failed to read {}: {e}", info_path.display())))?;
        let meta = MorfologikMetadata::parse(&info_bytes)?;
        let dict_bytes = std::fs::read(dict_path)
            .map_err(|e| MorfologikError::format(format!("failed to read {}: {e}", dict_path.display())))?;
        let entries = Self::read_entries(&dict_bytes, meta.separator, meta.encoder, meta.charset)?;
        Ok(ExpandedDictionaryLemmatizer::new(
            entries.entries,
            meta.input_conversion,
            &[&dict_bytes, &info_bytes],
            "opennlp:MorfologikDictionaryReader",
        ))
    }

    fn read_entries(
        dictionary: &[u8],
        separator: u8,
        encoding: BaseFormEncoding,
        charset: DictionaryCharset,
    ) -> Result<LemmatizerEntries, MorfologikError> {
        let automaton = FsaReader::from_bytes(dictionary)?;
        let mut entries = LemmatizerEntries::default();
        automaton
            .try_for_each_sequence(|sequence| Self::add_entry(sequence, separator, encoding, charset, &mut entries))?;
        Ok(entries)
    }

    fn add_entry(
        sequence: &[u8],
        separator: u8,
        encoding: BaseFormEncoding,
        charset: DictionaryCharset,
        entries: &mut LemmatizerEntries,
    ) -> Result<(), MorfologikError> {
        let Some(first_sep) = index_of(sequence, separator, 0) else {
            return Err(MorfologikError::format("morfologik entry has no separator"));
        };
        let second_sep = index_of(sequence, separator, first_sep + 1 + encoding.control_bytes());
        let base_end = second_sep.unwrap_or(sequence.len());

        let form_bytes = &sequence[..first_sep];
        let encoded_base = &sequence[first_sep + 1..base_end];
        let base_bytes = decode_base_form(form_bytes, encoded_base, encoding)
            .map_err(|e| MorfologikError::format(format!("malformed morfologik entry: {e}")))?;

        let surface = charset.decode(form_bytes)?;
        let lemma = charset.decode(&base_bytes)?;
        let tag = match second_sep {
            None => String::new(),
            Some(sec) => charset.decode(&sequence[sec + 1..])?,
        };
        entries.add(&surface, tag, lemma);
        Ok(())
    }
}

pub(crate) fn index_of(slice: &[u8], value: u8, from: usize) -> Option<usize> {
    if from >= slice.len() {
        return None;
    }
    slice[from..].iter().position(|&b| b == value).map(|p| from + p)
}

/// Builds an [`ExpandedDictionaryLemmatizer`] from a tab-separated `surfaceForm\tlemma\ttag`
/// table (`opennlp.tools.formats.PoliMorfDictionaryReader`).
pub struct PoliMorfDictionaryReader;

impl PoliMorfDictionaryReader {
    /// Reads a UTF-8 `surfaceForm\tlemma\ttag` dictionary into an [`ExpandedDictionaryLemmatizer`].
    pub fn read(dictionary: &[u8]) -> Result<ExpandedDictionaryLemmatizer, MorfologikError> {
        Self::read_with_charset(dictionary, DictionaryCharset::Utf8)
    }

    /// Reads a `surfaceForm\tlemma\ttag` dictionary in `charset` into an
    /// [`ExpandedDictionaryLemmatizer`].
    pub fn read_with_charset(
        dictionary: &[u8],
        charset: DictionaryCharset,
    ) -> Result<ExpandedDictionaryLemmatizer, MorfologikError> {
        let text = charset.decode(dictionary)?;
        let mut entries = LemmatizerEntries::default();
        for (idx, line) in java_lines(&text).enumerate() {
            let line_number = idx + 1;
            if is_blank_opennlp(line) {
                continue;
            }
            let Some(first_tab) = line.find('\t') else {
                return Err(MorfologikError::format(format!(
                    "PoliMorf line {line_number} has fewer than 3 tab-separated fields: {line}"
                )));
            };
            let rest = &line[first_tab + 1..];
            let Some(second_rel) = rest.find('\t') else {
                return Err(MorfologikError::format(format!(
                    "PoliMorf line {line_number} has fewer than 3 tab-separated fields: {line}"
                )));
            };
            let second_tab = first_tab + 1 + second_rel;
            let after_second = &line[second_tab + 1..];
            let tag_end = after_second.find('\t').map_or(line.len(), |third_rel| second_tab + 1 + third_rel);

            let form = &line[..first_tab];
            let lemma = &line[first_tab + 1..second_tab];
            let tag = &line[second_tab + 1..tag_end];
            entries.add(form, tag.to_owned(), lemma.to_owned());
        }
        Ok(ExpandedDictionaryLemmatizer::new(
            entries.entries,
            MorfologikInputConversion::default(),
            &[dictionary, charset.canonical_name().as_bytes()],
            "opennlp:PoliMorfDictionaryReader",
        ))
    }
}

/// Checks whether `line` is empty or consists entirely of code points for which
/// `StringUtil.isWhitespace(int)` (`Character.isWhitespace` or `Character.isSpaceChar`,
/// including `U+00A0` and `U+202F`) is true.
fn is_blank_opennlp(line: &str) -> bool {
    use turbo_unicode::java::code_point::is_space_char;
    line.chars().all(|c| {
        let cp = c as u32;
        is_whitespace(cp) || is_space_char(cp)
    })
}

fn java_lines(text: &str) -> impl Iterator<Item = &str> {
    let mut rest = text;
    std::iter::from_fn(move || {
        if rest.is_empty() {
            return None;
        }
        let end = rest.find(['\n', '\r']).unwrap_or(rest.len());
        let line = &rest[..end];
        rest = &rest[end..];
        if rest.starts_with("\r\n") {
            rest = &rest[2..];
        } else if !rest.is_empty() {
            rest = &rest[1..];
        }
        Some(line)
    })
}
