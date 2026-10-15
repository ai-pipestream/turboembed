//! The lattice tokenizer for Japanese and Korean, a port of OpenNLP's
//! `opennlp.tools.tokenize.lattice.LatticeTokenizer` and the MeCab
//! dictionary reader `MecabDictionary`.
//!
//! The text is split at whitespace, and each whitespace-free stretch is
//! cut into the morphemes of the path through the word lattice with the
//! lowest sum of word costs and connection costs (a Viterbi search). The
//! dictionary is a MeCab source directory: the lexicon files `*.csv`,
//! `matrix.def`, `char.def` and `unk.def`. Text the lexicon does not cover
//! gets unknown word candidates from the character categories of
//! `char.def` and the templates of `unk.def`. The feature columns are
//! kept as the dictionary has them. IPADIC (in a UTF-8 copy) and
//! mecab-ko-dic are the published dictionaries checked against the JVM.
//!
//! [`LatticeTokenizer`] is the annotator: it writes `Tokens` and, from
//! each morpheme's first feature column, `PosTags`. The segmentation is
//! the Java's, checked against files the JVM wrote (docs/segmentation.md).
//! [`LatticeAnalyzer`] is the same tokenizer with the dictionary's format
//! ([`DictionaryFormat`]), and also writes `Readings`: the reading,
//! pronunciation and lemma columns of the format, which OpenNLP leaves in
//! the feature string.

mod chardef;
mod dictionary;
mod entries;
mod error;
mod format;
mod limits;
mod matrix;
mod number;
mod tokenizer;
mod trie;

pub use dictionary::{DictionaryFiles, EntryId, MecabDictionary};
pub use entries::Columns;
pub use error::{CharDefProblem, DictionaryError, DictionaryFile, EntryProblem, NoCandidate};
pub use format::{DictionaryFormat, FeatureMap, FormatError, IpadicColumn, KoDicColumn, UnidicColumn, UnidicLayout};
pub use limits::DictionaryLimits;
pub use tokenizer::{LatticeTokenizer, Morpheme, Scratch};

use turbo_analysis::{Annotator, Error, ModelRef, Refs, View};
use turbo_analysis_layers::{PosTags, Sentences, Strs, Tokens};
use turbo_tokenize_layers::Readings;
use turbo_unicode::WhitespaceMode;

use crate::fnv1a;
use crate::unit::tokenize_units_with;

/// The OpenNLP class the dictionary reader ports, the id of the model.
pub const DICTIONARY_NAME: &str = "MecabDictionary";

impl LatticeTokenizer {
    /// The tokens of `view` (each sentence on its own, when there are
    /// sentences), handing each token's entry to `entry` in token order.
    fn segment(
        &self,
        annotator: &'static str,
        view: &View,
        sentences: Option<&Sentences>,
        mut entry: impl FnMut(EntryId),
    ) -> Result<Tokens, Error> {
        let mut scratch = Scratch::new();
        let mut morphemes = Vec::new();
        let mut failed = None;
        let tokens = tokenize_units_with(
            annotator,
            view,
            sentences,
            |text, base, out: &mut Vec<(turbo_analysis::Span, EntryId)>| {
                if failed.is_some() {
                    return;
                }
                morphemes.clear();
                if let Err(e) = self.analyze_with(text, &mut scratch, &mut morphemes) {
                    failed = Some(NoCandidate { offset: e.offset + base as usize });
                    return;
                }
                out.extend(morphemes.iter().map(|m| (m.span, m.entry)));
            },
            &mut entry,
        )?;
        if let Some(e) = failed {
            return Err(Error::Annotator { annotator, detail: e.to_string() });
        }
        Ok(tokens)
    }

    fn pos_tags(&self, tags: Vec<turbo_analysis_layers::Tag>) -> PosTags {
        PosTags { tagset: self.dictionary.pos_tags().clone(), tags, prob: None }
    }

    fn model_ref(&self) -> ModelRef {
        ModelRef {
            id: format!("opennlp:{DICTIONARY_NAME}"),
            revision: String::new(),
            sha256: self.dictionary.sha256().to_hex(),
        }
    }

    fn whitespace_hash(&self) -> [u8; 2] {
        let mode = |m: WhitespaceMode| match m {
            WhitespaceMode::Legacy => 0,
            WhitespaceMode::Unicode => 1,
        };
        [mode(self.whitespace), mode(self.dictionary.whitespace())]
    }
}

impl Annotator for LatticeTokenizer {
    const ID: &'static str = "turbo-tokenize:lattice";
    type Reads = (View, Option<Sentences>);
    type Writes = (Tokens, PosTags);

    fn annotate(&self, (view, sentences): Refs<'_, Self>) -> Result<(Tokens, PosTags), Error> {
        let mut tags = Vec::new();
        let tokens = self.segment(Self::ID, view, sentences, |entry| tags.push(self.dictionary.tag(entry)))?;
        Ok((tokens, self.pos_tags(tags)))
    }

    /// The dictionary: `opennlp:MecabDictionary` as the id, no revision (a
    /// dictionary directory carries none), and the SHA-256 of the manifest
    /// of its files ([`MecabDictionary::sha256`]).
    fn model(&self) -> Option<ModelRef> {
        Some(self.model_ref())
    }

    /// The whitespace mode of the text and the one the dictionary's lines
    /// were split in.
    fn config_hash(&self) -> u64 {
        fnv1a(&self.whitespace_hash())
    }
}

/// The lattice tokenizer with its dictionary's format: it writes the
/// tokens and part of speech tags [`LatticeTokenizer`] writes, and the
/// [`Readings`] of each token from the format's columns.
#[derive(Debug, Clone)]
pub struct LatticeAnalyzer {
    tokenizer: LatticeTokenizer,
    format: DictionaryFormat,
}

impl LatticeAnalyzer {
    /// The analyzer over `tokenizer`, whose dictionary's lexicon entries
    /// must have the columns of `format` ([`DictionaryFormat::check`]).
    pub fn new(tokenizer: LatticeTokenizer, format: DictionaryFormat) -> Result<LatticeAnalyzer, FormatError> {
        format.check(tokenizer.dictionary())?;
        Ok(LatticeAnalyzer { tokenizer, format })
    }

    pub fn tokenizer(&self) -> &LatticeTokenizer {
        &self.tokenizer
    }

    pub fn format(&self) -> DictionaryFormat {
        self.format
    }
}

impl Annotator for LatticeAnalyzer {
    const ID: &'static str = "turbo-tokenize:lattice-readings";
    type Reads = (View, Option<Sentences>);
    type Writes = (Tokens, PosTags, Readings);

    fn annotate(&self, (view, sentences): Refs<'_, Self>) -> Result<(Tokens, PosTags, Readings), Error> {
        let t = &self.tokenizer;
        let d = &t.dictionary;
        let map = self.format.map();
        let column = |c: Option<usize>| c.map(|_| Strs::default());
        let (mut reading, mut pronunciation, mut lemma) =
            (column(map.reading), column(map.pronunciation), column(map.lemma));
        let mut tags = Vec::new();
        let tokens = t.segment(Self::ID, view, sentences, |entry| {
            tags.push(d.tag(entry));
            let push = |out: &mut Option<Strs>, c: Option<usize>| {
                if let Some(out) = out {
                    out.push(format::value(d, entry, c).unwrap_or(""));
                }
            };
            push(&mut reading, map.reading);
            push(&mut pronunciation, map.pronunciation);
            push(&mut lemma, map.lemma);
        })?;
        Ok((tokens, t.pos_tags(tags), Readings { reading, pronunciation, lemma }))
    }

    /// The dictionary, as [`LatticeTokenizer`] names it.
    fn model(&self) -> Option<ModelRef> {
        Some(self.tokenizer.model_ref())
    }

    /// The whitespace modes, as [`LatticeTokenizer`] hashes them, and the
    /// format.
    fn config_hash(&self) -> u64 {
        let [a, b] = self.tokenizer.whitespace_hash();
        let format = match self.format {
            DictionaryFormat::Ipadic => 0,
            DictionaryFormat::MecabKoDic => 1,
            DictionaryFormat::Unidic(UnidicLayout::Source17) => 2,
            DictionaryFormat::Unidic(UnidicLayout::Full26) => 3,
            DictionaryFormat::Unidic(UnidicLayout::Full29) => 4,
        };
        fnv1a(&[a, b, format])
    }
}
