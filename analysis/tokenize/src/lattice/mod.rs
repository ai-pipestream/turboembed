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

mod chardef;
mod dictionary;
mod entries;
mod error;
mod limits;
mod matrix;
mod number;
mod tokenizer;
mod trie;

pub use dictionary::{DictionaryFiles, EntryId, MecabDictionary};
pub use entries::Columns;
pub use error::{CharDefProblem, DictionaryError, DictionaryFile, EntryProblem, NoCandidate};
pub use limits::DictionaryLimits;
pub use tokenizer::{LatticeTokenizer, Morpheme, Scratch};

use turbo_analysis::{Annotator, Error, ModelRef, Refs, View};
use turbo_analysis_layers::{PosTags, Sentences, Tokens};
use turbo_unicode::WhitespaceMode;

use crate::fnv1a;
use crate::unit::tokenize_units_with;

/// The OpenNLP class the dictionary reader ports, the id of the model.
pub const DICTIONARY_NAME: &str = "MecabDictionary";

impl Annotator for LatticeTokenizer {
    const ID: &'static str = "turbo-tokenize:lattice";
    type Reads = (View, Option<Sentences>);
    type Writes = (Tokens, PosTags);

    fn annotate(&self, (view, sentences): Refs<'_, Self>) -> Result<(Tokens, PosTags), Error> {
        let mut scratch = Scratch::new();
        let mut morphemes = Vec::new();
        let mut failed = None;
        let mut tags = Vec::new();
        let tokens = tokenize_units_with(
            Self::ID,
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
            |entry| tags.push(self.dictionary.tag(entry)),
        )?;
        if let Some(e) = failed {
            return Err(Error::Annotator { annotator: Self::ID, detail: e.to_string() });
        }
        let pos = PosTags { tagset: self.dictionary.pos_tags().clone(), tags, prob: None };
        Ok((tokens, pos))
    }

    /// The dictionary: `opennlp:MecabDictionary` as the id, no revision (a
    /// dictionary directory carries none), and the SHA-256 of the manifest
    /// of its files ([`MecabDictionary::sha256`]).
    fn model(&self) -> Option<ModelRef> {
        Some(ModelRef {
            id: format!("opennlp:{DICTIONARY_NAME}"),
            revision: String::new(),
            sha256: self.dictionary.sha256().to_hex(),
        })
    }

    /// The whitespace mode of the text and the one the dictionary's lines
    /// were split in.
    fn config_hash(&self) -> u64 {
        let mode = |m: WhitespaceMode| match m {
            WhitespaceMode::Legacy => 0,
            WhitespaceMode::Unicode => 1,
        };
        fnv1a(&[mode(self.whitespace), mode(self.dictionary.whitespace())])
    }
}
