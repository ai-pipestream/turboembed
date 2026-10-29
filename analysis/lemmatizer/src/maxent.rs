//! `LemmatizerME.lemmatize`: the best lemma class sequence of each
//! sentence by beam search, decoded into lemmas, and the annotator that
//! writes them as `Lemmas` with a `Confidence` per token.

use std::sync::Arc;

use turbo_analysis::{Annotator, Error, ModelRef, Refs, View};
use turbo_analysis_layers::{Confidence, Lemmas, PosTags, Strs, Tokens};
use turbo_maxent::JavaString;
use turbo_sequence::{AcceptAll, BeamSearch, TokenTag};

use crate::LemmaEncoding;
use crate::context::LemmaContext;
use crate::edit_script::DecodeScratch;
use crate::model::LemmatizerModel;

/// Lemmatizes the tokens of each sentence with a maxent model, given
/// their part-of-speech tags: a lemma and its probability per token.
///
/// The search is `BeamSearch.bestSequence` over
/// `DefaultLemmatizerContextGenerator` with a validator that accepts every
/// class, as `LemmatizerME.lemmatize(String[], String[])` runs it; the
/// classes use the model's decoder. OpenNLP models lowercase tokens;
/// native models preserve case. An empty decoded lemma is `_`. The probabilities (`probs()`) are the JVM's
/// doubles bit for bit. A sentence is lemmatized on its own.
///
/// The classes are the model's outcomes as [`JavaString`]s: a class edits
/// the UTF-16 units of a word, so a model trained on a word or lemma with
/// a character outside the BMP can hold a class with half of a surrogate
/// pair, which is searched for, written into the next token's features
/// and decoded as the JVM does it. A lemma that is left holding half a
/// pair has it replaced by U+FFFD, since the layer holds text.
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
    pub fn classes<'m>(
        &'m self,
        words: &[TokenTag<'_>],
        classes: &mut Vec<&'m JavaString>,
        probs: &mut Vec<f64>,
    ) -> bool {
        Self::classes_with(&self.search(), words, classes, probs)
    }

    /// The search over the model, built once per call of `classes` and
    /// once per document of `annotate`.
    fn search(&self) -> BeamSearch<'_, JavaString> {
        let Ok(search) = BeamSearch::<JavaString>::new(self.model.beam_size(), self.model.maxent());
        search
    }

    /// One sentence through `search`: what `classes` appends.
    fn classes_with<'m>(
        search: &BeamSearch<'m, JavaString>,
        words: &[TokenTag<'_>],
        classes: &mut Vec<&'m JavaString>,
        probs: &mut Vec<f64>,
    ) -> bool {
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
    pub fn lemmatize(&self, words: &[TokenTag<'_>], lemmas: &mut Strs, probs: &mut Vec<f64>) -> Result<bool, Error> {
        let mut classes = Vec::with_capacity(words.len());
        let mut next_probs = Vec::with_capacity(words.len());
        if !self.classes(words, &mut classes, &mut next_probs) {
            return Ok(false);
        }
        decode_classes_into(words, &classes, &next_probs, lemmas, probs, self.model.encoding())?;
        Ok(true)
    }
}

fn decode_classes_into(
    words: &[TokenTag<'_>],
    classes: &[&JavaString],
    next_probs: &[f64],
    lemmas: &mut Strs,
    probs: &mut Vec<f64>,
    encoding: LemmaEncoding,
) -> Result<(), Error> {
    let mut scratch = DecodeScratch::default();
    let mut decoded = Vec::with_capacity(words.len());
    for (item, class) in words.iter().zip(classes) {
        let word = item.token;
        let lemma = scratch.decode_with_case(word, class, encoding == LemmaEncoding::CaseSensitiveV1).map_err(|e| {
            Error::Annotator {
                annotator: MaxentLemmatizer::ID,
                detail: format!("{word:?} with the class {class:?}: {e}"),
            }
        })?;
        decoded.push(lemma.to_owned());
    }
    for lemma in &decoded {
        lemmas.push(lemma);
    }
    probs.extend_from_slice(next_probs);
    Ok(())
}

/// The `(from, to)` token ranges of each sentence of `tokens`, or one
/// range over all of them when they were not made per sentence.
pub(crate) fn sentence_ranges(tokens: &Tokens) -> Vec<(usize, usize)> {
    tokens.sentence_ranges().map(|(_, r)| (r.start.get(), r.end.get())).collect()
}

/// Checks that the tokens are valid UTF-8 spans of the view and the tags
/// are one in-bounds tag per token, and gives the word of each token.
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
    let mut out = Vec::with_capacity(starts.len());
    for (i, ((&s, &e), &tag)) in starts.iter().zip(ends).zip(&tags.tags).enumerate() {
        let word = (s <= e).then(|| text.get(s as usize..e as usize)).flatten().ok_or_else(|| Error::Annotator {
            annotator,
            detail: format!("token {i} span {s}..{e} is not a valid slice of a {}-byte view", text.len()),
        })?;
        if usize::from(tag.0) >= tags.tagset.len() {
            return Err(Error::Annotator {
                annotator,
                detail: format!("token {i} tag {} is outside tagset of {}", tag.0, tags.tagset.len()),
            });
        }
        out.push(word);
    }
    Ok(out)
}

impl Annotator for MaxentLemmatizer {
    const ID: &'static str = "turbo-lemmatizer:maxent";
    const VERSION: &'static str = "2";
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
        let mut tagged: Vec<TokenTag<'_>> = Vec::new();
        let mut scratch = DecodeScratch::default();
        let search = self.search();
        let mut classes: Vec<&JavaString> = Vec::new();
        for (from, to) in sentence_ranges(tokens) {
            tagged.clear();
            tagged.extend((from..to).map(|i| TokenTag { token: words[i], tag: tags.tagset.label(tags.tags[i]) }));
            classes.clear();
            if !Self::classes_with(&search, &tagged, &mut classes, &mut probs) {
                return Err(Error::Annotator {
                    annotator: Self::ID,
                    detail: format!("no lemma class sequence of tokens {from}..{to} scored above the minimum"),
                });
            }
            for (item, class) in tagged.iter().zip(&classes) {
                let word = item.token;
                let lemma = scratch
                    .decode_with_case(word, class, self.model.encoding() == LemmaEncoding::CaseSensitiveV1)
                    .map_err(|e| Error::Annotator {
                        annotator: Self::ID,
                        detail: format!("{word:?} with the class {class:?}: {e}"),
                    })?;
                lemmas.push(lemma);
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
        turbo_analysis::Fnv1a::new().u64(self.model.beam_size() as u64).finish()
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn annotator_version_tracks_validation_changes() {
        assert_eq!(MaxentLemmatizer::VERSION, "2");
    }

    #[test]
    fn decode_classes_into_leaves_outputs_untouched_on_decode_error() {
        let mut lemmas = Strs::default();
        lemmas.push("prior");
        let mut probs = vec![0.9];
        let words = [TokenTag { token: "dogs", tag: "NOUN" }, TokenTag { token: "x", tag: "NOUN" }];
        let c0 = JavaString::new("D0s");
        let c1 = JavaString::new("D0");
        let classes = [&c0, &c1];
        let err = decode_classes_into(&words, &classes, &[0.8, 0.7], &mut lemmas, &mut probs, LemmaEncoding::OpenNlp)
            .unwrap_err();
        assert!(matches!(err, Error::Annotator { annotator: MaxentLemmatizer::ID, .. }));
        assert_eq!(lemmas.iter().collect::<Vec<_>>(), ["prior"]);
        assert_eq!(probs, [0.9]);
    }
}
