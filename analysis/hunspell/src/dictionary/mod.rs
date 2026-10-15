//! A loaded dictionary: OpenNLP's `HunspellDictionary`.

mod build;
pub(crate) mod case;
mod index;
pub(crate) mod rules;

use std::path::Path;

use crate::LoadMode;
use crate::affix::Fields;
use crate::compound_rule::CompoundRule;
use crate::conversion::Conversion;
use crate::error::{LoadError, Stream};
use crate::flag::FlagSet;
use crate::hash::FoldMap;
use crate::read::{self, bytes::AFFIX_SOURCE, bytes::UnsupportedDirective};
use crate::settings::{CompoundPattern, Settings, WordBreak};
use crate::text::remove_ignored;

use case::CaseMapping;
pub(crate) use index::AffixIndex;

/// An entry of the word list.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct EntryId(u32);

/// A distinct flag set; entries with equal flags share one.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct FlagSetId(u32);

/// One line of the word list.
#[derive(Clone, Debug)]
pub(crate) struct Entry {
    pub(crate) flags: FlagSetId,
    pub(crate) morphology: Option<Fields>,
}

/// The entries of a word: the listed ones, then the hidden capitalized
/// ones when all-uppercase input may match them. OpenNLP's `lookup`
/// result, without building a list when both kinds exist.
#[derive(Copy, Clone, Debug, Default)]
pub(crate) struct Homonyms<'d> {
    listed: &'d [EntryId],
    hidden: &'d [EntryId],
}

impl<'d> Homonyms<'d> {
    pub(crate) fn is_empty(&self) -> bool {
        self.listed.is_empty() && self.hidden.is_empty()
    }

    pub(crate) fn first(&self) -> Option<EntryId> {
        self.listed.first().or(self.hidden.first()).copied()
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = EntryId> + 'd {
        self.listed.iter().chain(self.hidden).copied()
    }
}

/// A Hunspell dictionary loaded from its affix file (`.aff`) and word list
/// (`.dic`). Immutable once loaded and safe to share between threads.
///
/// [`Dictionary::from_bytes`] and [`Dictionary::open`] read both files with
/// OpenNLP's reader, so a file OpenNLP refuses is refused here with the
/// same message ([`LoadError`]).
pub struct Dictionary {
    pub(crate) listed: FoldMap<Box<str>, Box<[EntryId]>>,
    /// The capitalized forms of mixed-case and flagged all-uppercase words,
    /// which all-uppercase input outside compounds also matches.
    pub(crate) hidden: FoldMap<Box<str>, Box<[EntryId]>>,
    pub(crate) entries: Box<[Entry]>,
    pub(crate) flag_sets: Box<[FlagSet]>,
    pub(crate) suffixes: AffixIndex,
    pub(crate) prefixes: AffixIndex,
    pub(crate) settings: Settings,
    pub(crate) case: CaseMapping,
    /// `LANG` names Hungarian, which turns on its compound rules.
    pub(crate) hungarian: bool,
    pub(crate) ignored: Box<str>,
    pub(crate) input_conversion: Conversion,
    pub(crate) output_conversion: Conversion,
    /// The `REP` table and the `ph:` fields, which `CHECKCOMPOUNDREP`
    /// checks compounds against.
    pub(crate) replacements: Conversion,
    pub(crate) compound_rules: Box<[CompoundRule]>,
    pub(crate) compound_patterns: Box<[CompoundPattern]>,
    pub(crate) word_breaks: Box<[WordBreak]>,
    unsupported: Box<[UnsupportedDirective]>,
    /// The longest UTF-16 length a word with a space can have a reading at,
    /// `None` when none can.
    longest_spaced_form: Option<usize>,
    /// The longest UTF-16 length any word can have a direct or affixed
    /// reading at.
    longest_reading: usize,
    /// A hash of both files and the load mode.
    fingerprint: u64,
}

impl std::fmt::Debug for Dictionary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dictionary")
            .field("words", &self.listed.len())
            .field("prefixes", &self.prefixes.len())
            .field("suffixes", &self.suffixes.len())
            .field("unsupported", &self.unsupported)
            .finish_non_exhaustive()
    }
}

impl Dictionary {
    /// Loads a dictionary from the contents of its affix file and word
    /// list. A diagnostic names the affix file `affix stream`.
    pub fn from_bytes(affix: &[u8], words: &[u8], mode: LoadMode) -> Result<Dictionary, LoadError> {
        read::load(affix.to_vec(), || Ok(words.to_vec()), mode, AFFIX_SOURCE)
    }

    /// Loads a dictionary from its two files. A diagnostic names the affix
    /// file by its path.
    pub fn open(affix: impl AsRef<Path>, words: impl AsRef<Path>, mode: LoadMode) -> Result<Dictionary, LoadError> {
        let affix = affix.as_ref();
        let affix_file = std::fs::File::open(affix)?;
        let words_file = std::fs::File::open(words.as_ref())?;
        let bytes = read::read_bounded(affix_file, Stream::Affix)?;
        read::load(bytes, || read::read_bounded(words_file, Stream::Dictionary), mode, &affix.display().to_string())
    }

    /// The directives a [`LoadMode::AllowPartial`] load skipped, each name
    /// once at its first line; empty after a strict load.
    pub fn unsupported_directives(&self) -> &[UnsupportedDirective] {
        &self.unsupported
    }

    /// A hash of both files' bytes and the load mode, which tells
    /// dictionaries apart in an annotator's configuration hash.
    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    /// The number of distinct words of the word list.
    pub fn word_count(&self) -> usize {
        self.listed.len()
    }

    /// The listed entries of `word`.
    pub(crate) fn lookup(&self, word: &str) -> Homonyms<'_> {
        Homonyms { listed: self.listed.get(word).map_or(&[], |e| e), hidden: &[] }
    }

    /// The listed entries of `word`, and its hidden capitalized ones when
    /// `include_hidden`.
    pub(crate) fn lookup_with_hidden(&self, word: &str, include_hidden: bool) -> Homonyms<'_> {
        let mut h = self.lookup(word);
        if include_hidden {
            h.hidden = self.hidden.get(word).map_or(&[], |e| e);
        }
        h
    }

    /// The entries of the word a reading selected: the listed ones, or the
    /// hidden ones when it has none.
    pub(crate) fn homonyms(&self, root: &str) -> &[EntryId] {
        match self.listed.get(root) {
            Some(e) => e,
            None => self.hidden.get(root).map_or(&[], |e| e),
        }
    }

    pub(crate) fn entry(&self, id: EntryId) -> &Entry {
        &self.entries[id.0 as usize]
    }

    /// The flags of an entry.
    #[inline]
    pub(crate) fn flags(&self, id: EntryId) -> &FlagSet {
        &self.flag_sets[self.entries[id.0 as usize].flags.0 as usize]
    }

    /// Whether two entries have equal flags.
    #[inline]
    pub(crate) fn same_flags(&self, a: EntryId, b: EntryId) -> bool {
        self.entries[a.0 as usize].flags == self.entries[b.0 as usize].flags
    }

    /// The lookup form of a word: the input conversion applied, then the
    /// ignored characters removed.
    pub(crate) fn input_form<'a>(&self, word: &'a str) -> std::borrow::Cow<'a, str> {
        match self.input_conversion.apply(word) {
            std::borrow::Cow::Borrowed(w) => remove_ignored(w, &self.ignored),
            std::borrow::Cow::Owned(w) => std::borrow::Cow::Owned(remove_ignored(&w, &self.ignored).into_owned()),
        }
    }

    /// A stem with the output conversion applied.
    pub(crate) fn output_form<'a>(&self, stem: &'a str) -> std::borrow::Cow<'a, str> {
        self.output_conversion.apply(stem)
    }

    /// Whether a word holding a space and `utf16_len` units long can have
    /// a reading.
    pub(crate) fn may_read_spaced_form(&self, utf16_len: usize) -> bool {
        self.longest_spaced_form.is_some_and(|max| utf16_len <= max)
    }

    /// Whether a word `utf16_len` units long can have a direct or affixed
    /// reading. OpenNLP has no such bound; it reads such a word and finds
    /// nothing.
    pub(crate) fn may_read(&self, utf16_len: usize) -> bool {
        utf16_len <= self.longest_reading
    }

    /// Whether `text` is too long for [`Dictionary::may_read`], judged by
    /// its length in bytes alone: three bytes of UTF-8 are at least one
    /// UTF-16 unit, so this answers false for some texts it allows.
    pub(crate) fn too_long_to_read(&self, text: &str) -> bool {
        text.len() / 3 > self.longest_reading
    }
}
