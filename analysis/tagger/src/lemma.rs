//! The lemma annotator over a bundle's edit-tree lemmatizer head
//! (`HEAD_EDIT_TREE_LEMMATIZER`): each sentence's words are pooled on the
//! device as for the taggers; with a `pos_input`, the named tagger head
//! tags them first on the host; the core's `EditTreeLemmatizer` then picks
//! each word's tree and applies it.
//!
//! A [`LemmaDictionary`] may stand over the head: where it lists a lemma
//! for a word and its tag, that lemma is written and the head's is not. A
//! dictionary is any `Fn(form, tag) -> Option<String>`, so a Morfologik or
//! WordNet Morphy lookup plugs in as a closure; a lookup that answers a
//! miss with a placeholder (Morphy's `"O"`) maps it to None.

use std::path::Path;
use std::sync::{Mutex, PoisonError};

use turbo::bundle::Bundle;
use turbo::head::{EditTreeLemmatizer, TokenTagger, lemmatizer, tagger};
use turbo::manifest::HeadKind;
use turbo::status::Error as CoreError;
use turbo_analysis::{Annotator, Error, Execution, Fnv1a, Ix, Layer, ModelRef, Refs, View};
use turbo_analysis_embed::EncoderOptions;
use turbo_analysis_layers::{Lemmas, Sentences, Strs, Subwords, Tokens};

use crate::annotate::{HeadRunner, find_head, model_ref, refuse};
use crate::words::WordEncoder;

/// A lemma lookup that overrides the head where it has an answer.
pub trait LemmaDictionary: Send + Sync {
    /// The lemma listed for `form` under `tag` (the tag head's label, None
    /// when the lemmatizer reads no tags), or None to keep the head's.
    fn lemma(&self, form: &str, tag: Option<&str>) -> Option<String>;
}

impl<F: Fn(&str, Option<&str>) -> Option<String> + Send + Sync> LemmaDictionary for F {
    fn lemma(&self, form: &str, tag: Option<&str>) -> Option<String> {
        self(form, tag)
    }
}

/// A dictionary and the name that keys it in the config hash.
struct Dictionary {
    id: String,
    lookup: Box<dyn LemmaDictionary>,
}

/// Writes the lemma layer from an edit-tree lemmatizer head.
pub struct Lemmatizer {
    words: Mutex<WordEncoder>,
    head: EditTreeLemmatizer,
    /// The tagger the head's `pos_input` names.
    pos: Option<TokenTagger>,
    dictionary: Option<Dictionary>,
    model: ModelRef,
    execution: Execution,
    width: usize,
}

impl Lemmatizer {
    /// The head named `name` in the bundle at `dir`, or with no name the
    /// one head whose `layer` is the lemma layer's id, and the tagger
    /// head its `pos_input` names.
    pub fn open(dir: &Path, name: Option<&str>, o: &EncoderOptions) -> Result<Lemmatizer, CoreError> {
        let bundle = Bundle::open(dir)?;
        let spec = find_head(&bundle, name, Lemmas::ID, HeadKind::EditTreeLemmatizer)?;
        let head = EditTreeLemmatizer::load(&bundle, &spec.name)?;
        let pos = match &spec.pos_input {
            Some(p) => Some(TokenTagger::load(&bundle, &p.head)?),
            None => None,
        };
        let words = WordEncoder::open(dir, o)?;
        for hidden in std::iter::once(head.hidden()).chain(pos.as_ref().map(TokenTagger::hidden)) {
            if hidden != words.hidden() {
                return Err(refuse(format!(
                    "the head reads {hidden} values a word, the encoder gives {}",
                    words.hidden()
                )));
            }
        }
        Ok(Lemmatizer {
            model: model_ref(&bundle),
            execution: words.execution().clone(),
            width: words.width(),
            words: Mutex::new(words),
            head,
            pos,
            dictionary: None,
        })
    }

    /// This lemmatizer with `lookup` over the head, `id` naming it in the
    /// config hash (a dictionary file's digest, say).
    pub fn with_dictionary(mut self, id: &str, lookup: impl LemmaDictionary + 'static) -> Lemmatizer {
        self.dictionary = Some(Dictionary { id: id.to_owned(), lookup: Box::new(lookup) });
        self
    }

    pub fn head(&self) -> &EditTreeLemmatizer {
        &self.head
    }

    /// The tagger the head reads tags from, if it reads any.
    pub fn pos(&self) -> Option<&TokenTagger> {
        self.pos.as_ref()
    }

    /// Each sequence of words lemmatized.
    pub fn lemmatize(&self, sequences: &[Vec<&str>]) -> Result<Vec<Vec<String>>, CoreError> {
        let mut words = self.words.lock().unwrap_or_else(PoisonError::into_inner);
        let mut read = |pooling| words.features(sequences, pooling);
        let (head, pos) = self.features(&mut read)?;
        drop(words);
        self.lemmas_of(&head, pos.as_deref(), sequences)
    }

    /// The word features the head reads and, when the tagger pools words
    /// another way, the tagger's.
    #[allow(clippy::type_complexity)]
    fn features<E>(
        &self,
        read: &mut dyn FnMut(turbo::manifest::WordPooling) -> Result<Vec<Vec<f32>>, E>,
    ) -> Result<(Vec<Vec<f32>>, Option<Vec<Vec<f32>>>), E> {
        let pooling = self.head.spec().word_pooling;
        let head = read(pooling)?;
        let pos = match &self.pos {
            Some(t) if t.spec().word_pooling != pooling => Some(read(t.spec().word_pooling)?),
            _ => None,
        };
        Ok((head, pos))
    }

    /// Each sequence's lemmas from its features, `forms` its words' text.
    fn lemmas_of(
        &self,
        head: &[Vec<f32>],
        pos: Option<&[Vec<f32>]>,
        forms: &[Vec<&str>],
    ) -> Result<Vec<Vec<String>>, CoreError> {
        let hidden = self.head.hidden();
        let (mut scores, mut tags, mut trees) = (Vec::new(), Vec::new(), Vec::new());
        let (mut tw, mut lw) = (tagger::Work::default(), lemmatizer::Work::default());
        let mut out = Vec::with_capacity(head.len());
        for (s, f) in head.iter().enumerate() {
            let n = f.len() / hidden;
            let ids: Vec<u32> = (0..n as u32).collect();
            let words = &forms[s];
            if words.len() != n {
                return Err(refuse(format!("{} forms for {n} words", words.len())));
            }
            let tagged = match &self.pos {
                Some(t) => {
                    let pf = pos.map_or(f, |p| &p[s]);
                    t.scores(pf, &ids, &mut scores, &mut tw)?;
                    t.decode(&scores, n, &mut tags)?;
                    Some(&tags[..])
                }
                None => None,
            };
            self.head.scores(f, &ids, tagged, &mut scores, &mut lw)?;
            self.head.decode(&scores, words, &mut trees)?;
            let mut lemmas = self.head.lemmas(&trees, words)?;
            if let Some(d) = &self.dictionary {
                let labels = self.pos.as_ref().map(TokenTagger::labels);
                for (k, lemma) in lemmas.iter_mut().enumerate() {
                    let tag = labels.zip(tagged).map(|(l, t)| l[t[k] as usize].as_str());
                    if let Some(listed) = d.lookup.lemma(words[k], tag) {
                        *lemma = listed;
                    }
                }
            }
            out.push(lemmas);
        }
        Ok(out)
    }

    /// The lemma layer over every token of the view, the pieces from
    /// `subwords` when the encoder reads them (`WordEncoder::reads`).
    pub fn lemmas(
        &self,
        view: &View,
        tokens: &Tokens,
        sentences: Option<&Sentences>,
        subwords: Option<&Subwords>,
    ) -> Result<Lemmas, String> {
        let seqs = HeadRunner::sequences(view, tokens, sentences)?;
        let mut words = self.words.lock().unwrap_or_else(PoisonError::into_inner);
        let mut read = |pooling| words.token_features(view, tokens, &seqs, subwords, pooling);
        let (head, pos) = self.features(&mut read)?;
        drop(words);
        let text = view.text();
        let forms: Vec<Vec<&str>> = seqs
            .iter()
            .map(|s| s.iter().map(|&i| &text[tokens.spans.get(Ix::new(i as u32)).range()]).collect())
            .collect();
        let lemmas = self.lemmas_of(&head, pos.as_deref(), &forms).map_err(|e| e.to_string())?;
        let mut of: Vec<Option<String>> = vec![None; tokens.spans.len()];
        for (s, l) in seqs.iter().zip(lemmas) {
            for (&i, lemma) in s.iter().zip(l) {
                of[i] = Some(lemma);
            }
        }
        let mut lemma = Strs::default();
        for l in of {
            lemma.push(&l.ok_or("a token lies outside every sentence")?);
        }
        Ok(Lemmas { lemma })
    }
}

impl Annotator for Lemmatizer {
    const ID: &'static str = "turbo-tagger-lemmas";
    type Reads = (View, Tokens, Option<Subwords>, Option<Sentences>);
    type Writes = (Lemmas,);

    fn annotate(&self, (view, tokens, subwords, sentences): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        self.lemmas(view, tokens, sentences, subwords)
            .map(|l| (l,))
            .map_err(|detail| Error::Annotator { annotator: Self::ID, detail })
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.model.clone())
    }

    fn execution(&self) -> Option<Execution> {
        Some(self.execution.clone())
    }

    fn config_hash(&self) -> u64 {
        let dictionary = self.dictionary.as_ref().map_or("", |d| d.id.as_str());
        Fnv1a::new()
            .str(&self.head.spec().name)
            .u64(self.width as u64)
            .u64(self.dictionary.is_some() as u64)
            .str(dictionary)
            .finish()
    }
}
