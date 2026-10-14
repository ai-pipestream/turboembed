//! `LemmatizerME.lemmatize`: the best lemma class sequence of each
//! sentence by beam search, decoded into lemmas, and the annotator that
//! writes them as `Lemmas` with a `Confidence` per token.

use std::sync::Arc;

use turbo_analysis::{Annotator, Error, ModelRef, Refs, View};
use turbo_analysis_layers::{Confidence, Lemmas, PosTags, Strs, Tokens};
use turbo_sequence::{AcceptAll, BeamSearch};

use crate::context::{LemmaContext, Tagged};
use crate::edit_script::decode_lemma;
use crate::model::LemmatizerModel;

/// Lemmatizes the tokens of each sentence with a maxent model, given
/// their part-of-speech tags: a lemma and its probability per token.
///
/// The search is `BeamSearch.bestSequence` over
/// `DefaultLemmatizerContextGenerator` with a validator that accepts every
/// class, as `LemmatizerME.lemmatize(String[], String[])` runs it; the
/// classes found are applied to the lowercased tokens
/// (`LemmatizerME.decodeLemmas`), so a lemma is lowercase and `_` where
/// the class leaves nothing. The probabilities (`probs()`) are the JVM's
/// doubles bit for bit. A sentence is lemmatized on its own.
#[derive(Debug)]
pub struct MaxentLemmatizer {
    model: Arc<LemmatizerModel>,
}

impl MaxentLemmatizer {
    pub fn new(model: impl Into<Arc<LemmatizerModel>>) -> MaxentLemmatizer {
        MaxentLemmatizer { model: model.into() }
    }

    /// The model it runs.
    pub fn lemmatizer_model(&self) -> &LemmatizerModel {
        &self.model
    }

    /// `predictSES` for one sentence: appends the best lemma class of
    /// each tagged word and its probability, or appends nothing and
    /// returns false when no sequence scored above the search's minimum
    /// (where `lemmatize` returns an array of nulls).
    pub fn classes<'m>(&'m self, words: &[Tagged<'_>], classes: &mut Vec<&'m str>, probs: &mut Vec<f64>) -> bool {
        let search = BeamSearch::new(self.model.beam_size(), self.model.maxent());
        let Some(best) = search.best_sequence(words, &mut LemmaContext, &AcceptAll) else {
            return false;
        };
        classes.extend_from_slice(best.outcomes());
        probs.extend_from_slice(best.probs());
        true
    }

    /// `lemmatize` for one sentence: appends a lemma and its probability
    /// per tagged word, or appends nothing and returns `Ok(false)` when no
    /// class sequence was found. A class the decoder cannot read is an
    /// error, which no class of a trained model is.
    pub fn lemmatize(&self, words: &[Tagged<'_>], lemmas: &mut Strs, probs: &mut Vec<f64>) -> Result<bool, Error> {
        let mut classes = Vec::with_capacity(words.len());
        if !self.classes(words, &mut classes, probs) {
            return Ok(false);
        }
        for (&(word, _), class) in words.iter().zip(&classes) {
            let lemma = decode_lemma(word, class).map_err(|e| Error::Annotator {
                annotator: Self::ID,
                detail: format!("{word:?} with the class {class:?}: {e}"),
            })?;
            lemmas.push(&lemma);
        }
        Ok(true)
    }
}

/// The `(from, to)` token ranges of each sentence of `tokens`, or one
/// range over all of them when they were not made per sentence.
pub(crate) fn sentence_ranges(tokens: &Tokens) -> Vec<(usize, usize)> {
    let n = tokens.spans.columns().0.len();
    let mut ranges = Vec::new();
    let mut from = 0;
    while from < n {
        let to = match &tokens.sentence {
            Some(col) => from + col[from..].partition_point(|&s| s == col[from]),
            None => n,
        };
        ranges.push((from, to));
        from = to;
    }
    ranges
}

/// Checks that the tokens are spans of the view and the tags are one per
/// token, and gives the word of each token.
pub(crate) fn words<'d>(
    annotator: &'static str,
    view: &'d View,
    tokens: &'d Tokens,
    tags: &PosTags,
) -> Result<Vec<&'d str>, Error> {
    if tokens.spans.view() != view.id() {
        return Err(Error::Annotator {
            annotator,
            detail: format!(
                "the tokens are in view {}, not in view {} that is lemmatized",
                tokens.spans.view().0,
                view.id().0
            ),
        });
    }
    let text = view.text();
    let (starts, ends) = tokens.spans.columns();
    if tags.tags.len() != starts.len() {
        return Err(Error::Annotator {
            annotator,
            detail: format!("{} tags for {} tokens", tags.tags.len(), starts.len()),
        });
    }
    Ok(starts.iter().zip(ends).map(|(&s, &e)| &text[s as usize..e as usize]).collect())
}

impl Annotator for MaxentLemmatizer {
    const ID: &'static str = "turbo-lemmatizer:maxent";
    type Reads = (View, Tokens, PosTags);
    type Writes = (Lemmas, Confidence<Lemmas>);

    /// Lemmatizes each sentence of the tokens (each run of tokens with the
    /// same sentence, or all the tokens when they were not made per
    /// sentence) with the tags' labels. The tokens must be spans of the
    /// view read.
    fn annotate(&self, (view, tokens, tags): Refs<'_, Self>) -> Result<(Lemmas, Confidence<Lemmas>), Error> {
        let words = words(Self::ID, view, tokens, tags)?;
        let mut lemmas = Strs::default();
        let mut probs = Vec::with_capacity(words.len());
        let mut tagged: Vec<Tagged<'_>> = Vec::new();
        for (from, to) in sentence_ranges(tokens) {
            tagged.clear();
            tagged.extend((from..to).map(|i| (words[i], tags.tagset.label(tags.tags[i]))));
            if !self.lemmatize(&tagged, &mut lemmas, &mut probs)? {
                return Err(Error::Annotator {
                    annotator: Self::ID,
                    detail: format!("no lemma class sequence of tokens {from}..{to} scored above the minimum"),
                });
            }
        }
        Ok((Lemmas { lemma: lemmas }, Confidence::new(probs)))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.model.model_ref().clone())
    }

    /// FNV-1a over the beam size, the one setting a `LemmatizerME` is
    /// built with.
    fn config_hash(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in (self.model.beam_size() as u64).to_le_bytes() {
            h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
        h
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use turbo_analysis::{Analyzer, Document, Ix, Key, Provenance, Span, Spans};
    use turbo_analysis_layers::{Sentences, Tag, TagSet};

    use super::*;

    fn english() -> MaxentLemmatizer {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/models/opennlp-en-ud-ewt-lemmas-1.3-2.5.4.bin");
        MaxentLemmatizer::new(LemmatizerModel::read(path).unwrap())
    }

    fn lemmas(l: &MaxentLemmatizer, words: &[(&str, &str)]) -> Vec<String> {
        let (mut out, mut probs) = (Strs::default(), Vec::new());
        assert!(l.lemmatize(words, &mut out, &mut probs).unwrap());
        assert_eq!(out.len(), probs.len());
        assert!(probs.iter().all(|p| (0.0..=1.0).contains(p)));
        out.iter().map(str::to_string).collect()
    }

    #[test]
    fn plain_english_lemmas() {
        let l = english();
        let got = lemmas(&l, &[("The", "DET"), ("dogs", "NOUN"), ("were", "AUX"), ("barking", "VERB"), (".", "PUNCT")]);
        assert_eq!(got, ["the", "dog", "be", "bark", "."]);
        assert_eq!(lemmas(&l, &[]), Vec::<String>::new());
    }

    #[test]
    fn classes_are_the_models_outcomes() {
        let l = english();
        let (mut classes, mut probs) = (Vec::new(), Vec::new());
        assert!(l.classes(&[("dogs", "NOUN")], &mut classes, &mut probs));
        assert_eq!(classes, ["D0s"]);
        assert_eq!(probs.len(), 1);
    }

    #[test]
    fn the_annotator_runs_per_sentence_on_given_tokens_and_tags() {
        let text = "Dogs bark. Cats slept.";
        let mut doc = Document::new(text);
        let view = doc.get::<View>().unwrap().id();
        let mut sentences = Spans::new(view);
        sentences.push(Span::new(0, 10));
        sentences.push(Span::new(11, 22));
        let mut spans = Spans::new(view);
        for (a, b) in [(0, 4), (5, 9), (9, 10), (11, 15), (16, 21), (21, 22)] {
            spans.push(Span::new(a, b));
        }
        let tokens = Tokens { spans, sentence: Some([0, 0, 0, 1, 1, 1].map(Ix::new).to_vec()) };
        let tagset = TagSet::new("ud", &["NOUN", "VERB", "PUNCT"]);
        let tags = PosTags { tagset, tags: [0, 1, 2, 0, 1, 2].map(Tag).to_vec(), prob: None };
        let by = || Provenance::by("test");
        doc.add(&Key::<Sentences>::DEFAULT, Sentences { spans: sentences }, by()).unwrap();
        doc.add(&Key::<Tokens>::DEFAULT, tokens, by()).unwrap();
        doc.add(&Key::<PosTags>::DEFAULT, tags, by()).unwrap();
        let analyzer = Analyzer::builder()
            .given(Key::<Tokens>::DEFAULT)
            .given(Key::<PosTags>::DEFAULT)
            .add(english())
            .build()
            .unwrap();
        assert_eq!(analyzer.run(std::slice::from_mut(&mut doc)), [Ok(())]);
        let got = doc.get::<Lemmas>().unwrap();
        assert_eq!(got.lemma.iter().collect::<Vec<_>>(), ["dog", "bark", ".", "cat", "sleep", "."]);
        let confidence = doc.get::<Confidence<Lemmas>>().unwrap();
        assert_eq!(confidence.value.len(), 6);
        let p = doc.provenance(Key::<Lemmas>::DEFAULT.id()).unwrap();
        assert_eq!(p.annotator, MaxentLemmatizer::ID);
        assert_eq!(p.model.as_ref().map(|m| m.id.as_str()), Some("opennlp:StatisticalLemmatizer:en"));
        assert_eq!(p.reads[1..], [Key::<Tokens>::DEFAULT.id(), Key::<PosTags>::DEFAULT.id()]);
    }
}
