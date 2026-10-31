//! The dependency annotator over a bundle's biaffine parser head
//! (`HEAD_BIAFFINE_PARSER`): each sentence's words are pooled on the
//! device as for the taggers, then scored, decoded to a tree and labelled
//! by the core's `BiaffineParser` on the host.

use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};

use turbo::bundle::Bundle;
use turbo::head::BiaffineParser;
use turbo::manifest::HeadKind;
use turbo::status::Error as CoreError;
use turbo_analysis::{Annotator, Error, Execution, Fnv1a, Ix, Layer, ModelRef, Refs, TagSet, View};
use turbo_analysis_embed::EncoderOptions;
use turbo_analysis_layers::{Dependencies, Sentences, Subwords, Tag, Tokens};

use crate::annotate::{HeadRunner, find_head, model_ref, refuse};
use crate::words::WordEncoder;

pub use turbo::head::run::Parse;

/// Writes the dependency layer from a biaffine parser head. A word under
/// the root has no head; each sentence has one such word when the head
/// decodes by MST or Eisner.
pub struct DependencyParser {
    words: Mutex<WordEncoder>,
    parser: BiaffineParser,
    relations: Arc<TagSet>,
    model: ModelRef,
    execution: Execution,
    width: usize,
}

impl DependencyParser {
    /// The head named `name` in the bundle at `dir`, or with no name the
    /// one head whose `layer` is the dependency layer's id.
    pub fn open(dir: &Path, name: Option<&str>, o: &EncoderOptions) -> Result<DependencyParser, CoreError> {
        let bundle = Bundle::open(dir)?;
        let spec = find_head(&bundle, name, Dependencies::<Tokens>::ID, HeadKind::BiaffineParser)?;
        let parser = BiaffineParser::load(&bundle, &spec.name)?;
        let words = WordEncoder::open(dir, o)?;
        if words.hidden() != parser.hidden() {
            return Err(refuse(format!(
                "the head reads {} values a word, the encoder gives {}",
                parser.hidden(),
                words.hidden()
            )));
        }
        let labels: Vec<&str> = parser.labels().iter().map(String::as_str).collect();
        Ok(DependencyParser {
            relations: TagSet::new(&spec.name, &labels),
            model: model_ref(&bundle),
            execution: words.execution().clone(),
            width: words.width(),
            words: Mutex::new(words),
            parser,
        })
    }

    pub fn parser(&self) -> &BiaffineParser {
        &self.parser
    }

    pub fn relations(&self) -> &Arc<TagSet> {
        &self.relations
    }

    /// Each sequence of words parsed.
    pub fn parse(&self, sequences: &[Vec<&str>]) -> Result<Vec<Parse>, CoreError> {
        let pooling = self.parser.spec().word_pooling;
        let features = self.words.lock().unwrap_or_else(PoisonError::into_inner).features(sequences, pooling)?;
        parse_with(&self.parser, &features)
    }

    /// The dependency layer over every token of the view, the pieces
    /// from `subwords` when the encoder reads them (`WordEncoder::reads`).
    pub fn dependencies(
        &self,
        view: &View,
        tokens: &Tokens,
        sentences: Option<&Sentences>,
        subwords: Option<&Subwords>,
    ) -> Result<Dependencies<Tokens>, String> {
        let seqs = HeadRunner::sequences(view, tokens, sentences)?;
        let pooling = self.parser.spec().word_pooling;
        let mut words = self.words.lock().unwrap_or_else(PoisonError::into_inner);
        let features = words.token_features(view, tokens, &seqs, subwords, pooling)?;
        drop(words);
        let parses = parse_with(&self.parser, &features).map_err(|e| e.to_string())?;
        dep_layer(tokens, &seqs, &parses, &self.relations)
    }
}

impl Annotator for DependencyParser {
    const ID: &'static str = "turbo-tagger-dependencies";
    type Reads = (View, Tokens, Option<Subwords>, Option<Sentences>);
    type Writes = (Dependencies<Tokens>,);

    fn annotate(&self, (view, tokens, subwords, sentences): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        self.dependencies(view, tokens, sentences, subwords)
            .map(|d| (d,))
            .map_err(|detail| Error::Annotator { annotator: Self::ID, detail })
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.model.clone())
    }

    fn execution(&self) -> Option<Execution> {
        Some(self.execution.clone())
    }

    fn config_hash(&self) -> u64 {
        Fnv1a::new().str(&self.parser.spec().name).u64(self.width as u64).finish()
    }
}

pub(crate) use turbo::head::run::parse_sequences as parse_with;

/// The dependency layer from each sequence's parse.
pub(crate) fn dep_layer(
    tokens: &Tokens,
    seqs: &[Vec<usize>],
    parses: &[Parse],
    relations: &Arc<TagSet>,
) -> Result<Dependencies<Tokens>, String> {
    let n = tokens.spans.len();
    let (mut head, mut relation) = (vec![None; n], vec![Tag(0); n]);
    let mut seen = vec![false; n];
    for (s, p) in seqs.iter().zip(parses) {
        for (k, &i) in s.iter().enumerate() {
            let h = p.heads[k] as usize;
            head[i] = (h > 0).then(|| Ix::new(s[h - 1] as u32));
            relation[i] = Tag(p.relations[k] as u16);
            seen[i] = true;
        }
    }
    if seen.iter().any(|s| !s) {
        return Err("a token lies outside every sentence".into());
    }
    Ok(Dependencies { head, relation, relations: relations.clone() })
}
