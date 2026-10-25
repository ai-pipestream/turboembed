//! WordNet Morphy morphological analyzer (`MorphyExceptions` and `MorphyLemmatizer`).

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use crate::error::WordNetError;
use crate::folding::{fold, split_on_spaces};
use crate::lexicon::LexicalKnowledgeBase;
use crate::pos::WordNetPos;

/// Output emitted for a token whose lemma is unknown (`"O"`), matching OpenNLP's `MorphyLemmatizer.UNKNOWN_LEMMA`.
pub const UNKNOWN_LEMMA: &str = "O";

/// Irregular inflection tables (`noun.exc`, `verb.exc`, `adj.exc`, `adv.exc`).
#[derive(Debug, Clone, Default)]
pub struct MorphyExceptions {
    by_pos: HashMap<WordNetPos, HashMap<String, Vec<String>>>,
}

impl MorphyExceptions {
    /// Visits normalized forms in POS/key order, retaining each lemma list's
    /// preference order for configuration identity.
    pub(crate) fn visit_entries(&self, mut visit: impl FnMut(WordNetPos, &str, &[String])) {
        for pos in WordNetPos::ALL {
            if let Some(table) = self.by_pos.get(&pos) {
                let mut entries: Vec<_> = table.iter().collect();
                entries.sort_unstable_by_key(|(form, _)| *form);
                for (form, bases) in entries {
                    visit(pos, form, bases);
                }
            }
        }
    }

    /// Creates an empty exception table.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Loads all four `.exc` files (`noun.exc`, `verb.exc`, `adj.exc`, `adv.exc`) from `directory`.
    ///
    /// Returns an error if `directory` is not an existing directory, if any of the four
    /// exception files is missing, or if any line has fewer than 2 whitespace-separated tokens.
    pub fn load(directory: impl AsRef<Path>) -> Result<Self, WordNetError> {
        let dir = directory.as_ref();
        if !dir.is_dir() {
            return Err(WordNetError::invalid(format!(
                "directory does not exist or is not a directory: {}",
                dir.display()
            )));
        }
        let mut by_pos = HashMap::with_capacity(4);
        for pos in WordNetPos::ALL {
            let file_name = format!("{}.exc", pos.file_suffix());
            let file_path = dir.join(&file_name);
            if !file_path.is_file() {
                return Err(WordNetError::format(format!("Missing exception list file: {}", file_path.display())));
            }
            let file = File::open(&file_path)?;
            by_pos.insert(pos, Self::parse_named_reader(BufReader::new(file), &file_name)?);
        }
        Ok(Self { by_pos })
    }

    /// Creates a [`MorphyExceptions`] table from a per-POS map of inflected forms to base lemmas.
    pub fn from_map(map: HashMap<WordNetPos, HashMap<String, Vec<String>>>) -> Self {
        let mut by_pos = HashMap::with_capacity(map.len());
        for (pos, table) in map {
            let mut copy = HashMap::with_capacity(table.len());
            for (k, v) in table {
                copy.insert(fold(&k), v);
            }
            by_pos.insert(pos, copy);
        }
        Self { by_pos }
    }

    /// Parses a single `.exc` stream into a map from folded inflected form to base forms.
    pub fn parse_reader<R: Read>(reader: R) -> Result<HashMap<String, Vec<String>>, WordNetError> {
        Self::parse_named_reader(reader, "inline.exc")
    }

    fn parse_named_reader<R: Read>(reader: R, file_name: &str) -> Result<HashMap<String, Vec<String>>, WordNetError> {
        let mut out: HashMap<String, Vec<String>> = HashMap::new();
        let buf = BufReader::new(reader);
        for (idx, line_res) in buf.lines().enumerate() {
            let line = line_res?;
            if line.is_empty() {
                continue;
            }
            let tokens = split_on_spaces(&line);
            if tokens.len() < 2 {
                return Err(WordNetError::format(format!(
                    "Malformed exception list {file_name} at line {}: expected an inflected form and at least one base form, got: {line}",
                    idx + 1
                )));
            }
            let inflected = fold(tokens[0]);
            let mut bases = Vec::with_capacity(tokens.len() - 1);
            for &tok in &tokens[1..] {
                bases.push(fold(tok));
            }
            out.entry(inflected).or_insert(bases);
        }
        Ok(out)
    }

    /// Looks up irregular base forms for `inflected_form` and `pos`.
    pub fn lookup(&self, inflected_form: &str, pos: WordNetPos) -> &[String] {
        let Some(table) = self.by_pos.get(&pos) else {
            return &[];
        };
        let key = fold(inflected_form);
        table.get(&key).map_or(&[], Vec::as_slice)
    }
}

const NOUN_RULES: &[(&str, &str)] = &[
    ("s", ""),
    ("ses", "s"),
    ("xes", "x"),
    ("zes", "z"),
    ("ches", "ch"),
    ("shes", "sh"),
    ("men", "man"),
    ("ies", "y"),
];

const VERB_RULES: &[(&str, &str)] =
    &[("s", ""), ("ies", "y"), ("es", "e"), ("es", ""), ("ed", "e"), ("ed", ""), ("ing", "e"), ("ing", "")];

const ADJECTIVE_RULES: &[(&str, &str)] = &[("er", ""), ("est", ""), ("er", "e"), ("est", "e")];

/// WordNet's *Morphy* morphological analyzer (`opennlp.wordnet.MorphyLemmatizer`).
///
/// Combines irregular exception lists ([`MorphyExceptions`]) with POS-specific suffix
/// detachment rules validated against a [`LexicalKnowledgeBase`].
#[derive(Clone, Debug)]
pub struct MorphyLemmatizer<L: LexicalKnowledgeBase> {
    lexicon: L,
    exceptions: MorphyExceptions,
}

impl<L: LexicalKnowledgeBase> MorphyLemmatizer<L> {
    /// Creates a new [`MorphyLemmatizer`] backed by `lexicon` and `exceptions`.
    pub fn new(lexicon: L, exceptions: MorphyExceptions) -> Self {
        Self { lexicon, exceptions }
    }

    /// Returns a reference to the underlying [`LexicalKnowledgeBase`].
    pub fn lexicon(&self) -> &L {
        &self.lexicon
    }

    /// Returns a reference to the [`MorphyExceptions`] table.
    pub fn exceptions(&self) -> &MorphyExceptions {
        &self.exceptions
    }

    /// Maps a POS tag (Penn Treebank `NN`, `VBZ`, `JJ`, `RB`, UD `NOUN`, `VERB`, `ADJ`, `ADV`,
    /// or single-letter WordNet code `n`, `v`, `a`, `s`, `r`) onto [`WordNetPos`].
    pub fn pos_from_tag(tag: &str) -> Option<WordNetPos> {
        let first = *tag.as_bytes().first()?;
        if !first.is_ascii_alphabetic() {
            return None;
        }
        let upper = tag.to_ascii_uppercase();
        if upper.starts_with("ADJ") {
            return Some(WordNetPos::Adjective);
        }
        if upper.starts_with("ADV") {
            return Some(WordNetPos::Adverb);
        }
        match upper.as_bytes()[0] {
            b'N' => Some(WordNetPos::Noun),
            b'V' => Some(WordNetPos::Verb),
            b'J' => Some(WordNetPos::Adjective),
            b'A' | b'S' => (tag.len() == 1).then_some(WordNetPos::Adjective),
            b'R' => Some(WordNetPos::Adverb),
            _ => None,
        }
    }

    /// Lemmatizes a single `(token, pos_tag)` pair, returning the first candidate or `"O"`.
    pub fn lemmatize_one(&self, token: &str, tag: &str) -> String {
        self.lemmas_of(token, tag).into_iter().next().unwrap_or_else(|| UNKNOWN_LEMMA.to_string())
    }

    /// Lemmatizes parallel slices of `tokens` and `tags`, returning the primary lemma (or `"O"`)
    /// at each position.
    ///
    /// # Errors
    ///
    /// Returns [`WordNetError::InvalidInput`] if `tokens.len() != tags.len()`.
    pub fn lemmatize(&self, tokens: &[impl AsRef<str>], tags: &[impl AsRef<str>]) -> Result<Vec<String>, WordNetError> {
        if tokens.len() != tags.len() {
            return Err(WordNetError::invalid(format!(
                "toks and tags must have the same length, got {} and {}",
                tokens.len(),
                tags.len()
            )));
        }
        let mut out = Vec::with_capacity(tokens.len());
        for (tok, tag) in tokens.iter().zip(tags.iter()) {
            out.push(self.lemmatize_one(tok.as_ref(), tag.as_ref()));
        }
        Ok(out)
    }

    /// Lemmatizes parallel slices of `tokens` and `tags`, returning all candidate lemmas
    /// (or `vec!["O"]` when unknown) at each position.
    ///
    /// # Errors
    ///
    /// Returns [`WordNetError::InvalidInput`] if `tokens.len() != tags.len()`.
    pub fn lemmatize_candidates(
        &self,
        tokens: &[impl AsRef<str>],
        tags: &[impl AsRef<str>],
    ) -> Result<Vec<Vec<String>>, WordNetError> {
        if tokens.len() != tags.len() {
            return Err(WordNetError::invalid(format!(
                "toks and tags must have the same size, got {} and {}",
                tokens.len(),
                tags.len()
            )));
        }
        let mut out = Vec::with_capacity(tokens.len());
        for (tok, tag) in tokens.iter().zip(tags.iter()) {
            let cands = self.lemmas_of(tok.as_ref(), tag.as_ref());
            if cands.is_empty() {
                out.push(vec![UNKNOWN_LEMMA.to_string()]);
            } else {
                out.push(cands);
            }
        }
        Ok(out)
    }

    fn lemmas_of(&self, token: &str, tag: &str) -> Vec<String> {
        let Some(pos) = Self::pos_from_tag(tag) else {
            return Vec::new();
        };
        let folded = fold(token);
        let irregular = self.exceptions.lookup(&folded, pos);
        if !irregular.is_empty() {
            return irregular.to_vec();
        }
        let mut candidates = Vec::with_capacity(2);
        if self.lexicon.contains(&folded, pos) {
            candidates.push(folded.clone());
        }
        let rules = match pos {
            WordNetPos::Noun => NOUN_RULES,
            WordNetPos::Verb => VERB_RULES,
            WordNetPos::Adjective => ADJECTIVE_RULES,
            WordNetPos::Adverb => &[],
        };
        for &(suffix, replacement) in rules {
            if folded.len() > suffix.len() && folded.ends_with(suffix) {
                let stem_len = folded.len() - suffix.len();
                let mut candidate = String::with_capacity(stem_len + replacement.len());
                candidate.push_str(&folded[..stem_len]);
                candidate.push_str(replacement);
                if !candidates.contains(&candidate) && self.lexicon.contains(&candidate, pos) {
                    candidates.push(candidate);
                }
            }
        }
        candidates
    }
}
