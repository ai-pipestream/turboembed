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
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in (self.model.beam_size() as u64).to_le_bytes() {
            h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
        h
    }
}

#[cfg(test)]
mod tests {
    use turbo_analysis::{Analyzer, Document, Ix, Key, Provenance, Span, Spans};
    use turbo_analysis_layers::{Sentences, Tag, TagSet};

    use super::*;

    /// The lemmatizer over the English model, or `None` when it is not
    /// downloaded.
    fn english() -> Option<MaxentLemmatizer> {
        let path = turbo_goldens::published_model("opennlp-en-ud-ewt-lemmas-1.3-2.5.4.bin")?;
        Some(MaxentLemmatizer::new(LemmatizerModel::read(path).unwrap()))
    }

    fn lemmas(l: &MaxentLemmatizer, words: &[(&str, &str)]) -> Vec<String> {
        let words: Vec<TokenTag<'_>> = words.iter().map(|&(token, tag)| TokenTag { token, tag }).collect();
        let (mut out, mut probs) = (Strs::default(), Vec::new());
        assert!(l.lemmatize(&words, &mut out, &mut probs).unwrap());
        assert_eq!(out.len(), probs.len());
        assert!(probs.iter().all(|p| (0.0..=1.0).contains(p)));
        out.iter().map(str::to_string).collect()
    }

    #[test]
    fn annotator_version_tracks_validation_changes() {
        assert_eq!(MaxentLemmatizer::VERSION, "2");
    }

    #[test]
    fn plain_english_lemmas() {
        let Some(l) = english() else { return };
        let got = lemmas(&l, &[("The", "DET"), ("dogs", "NOUN"), ("were", "AUX"), ("barking", "VERB"), (".", "PUNCT")]);
        assert_eq!(got, ["the", "dog", "be", "bark", "."]);
        assert_eq!(lemmas(&l, &[]), Vec::<String>::new());
    }

    #[test]
    fn classes_are_the_models_outcomes() {
        let Some(l) = english() else { return };
        let (mut classes, mut probs) = (Vec::new(), Vec::new());
        assert!(l.classes(&[TokenTag { token: "dogs", tag: "NOUN" }], &mut classes, &mut probs));
        assert_eq!(classes, [&JavaString::new("D0s")]);
        assert_eq!(probs.len(), 1);
    }

    #[test]
    fn the_annotator_runs_per_sentence_on_given_tokens_and_tags() {
        let Some(lemmatizer) = english() else { return };
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
            .add(lemmatizer)
            .build()
            .unwrap();
        assert_eq!(analyzer.run(std::slice::from_mut(&mut doc)), [Ok(())]);
        let got = doc.get::<Lemmas>().unwrap();
        assert_eq!(got.lemma.iter().collect::<Vec<_>>(), ["dog", "bark", ".", "cat", "sleep", "."]);
        let confidence = doc.get::<Confidence<Lemmas>>().unwrap();
        assert_eq!(confidence.value.len(), 6);
        let p = doc.provenance(Key::<Lemmas>::DEFAULT.id()).unwrap();
        assert_eq!(p.annotator, MaxentLemmatizer::ID);
        assert_eq!(p.version, "2");
        assert_eq!(p.model.as_ref().map(|m| m.id.as_str()), Some("opennlp:StatisticalLemmatizer:en"));
        assert_eq!(p.reads[1..], [Key::<Tokens>::DEFAULT.id(), Key::<PosTags>::DEFAULT.id()]);
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

    #[test]
    fn maxent_annotator_refuses_invalid_token_spans_without_panicking() {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../goldens/data/fixtures/lemma-fixture-rust.bin");
        let lemmatizer = MaxentLemmatizer::new(LemmatizerModel::read(path).unwrap());
        let doc = Document::new("café");
        let view = doc.get::<View>().unwrap();
        let tagset = TagSet::new("ud", &["NOUN"]);
        let tags = PosTags { tagset, tags: vec![Tag(0)], prob: None };

        // Out-of-bounds token span (end > text.len()).
        let mut oob_spans = Spans::new(view.id());
        oob_spans.push(Span::new(0, 99));
        let oob_tokens = Tokens { spans: oob_spans, sentence: None };
        let err = lemmatizer.annotate((view, &oob_tokens, &tags)).err().unwrap();
        assert!(err.to_string().contains("not a valid slice"), "unexpected error: {err}");

        // Mid-codepoint UTF-8 boundary ('é' is bytes 3..5 in "café").
        let mut mid_spans = Spans::new(view.id());
        mid_spans.push(Span::new(0, 4));
        let mid_tokens = Tokens { spans: mid_spans, sentence: None };
        let err = lemmatizer.annotate((view, &mid_tokens, &tags)).err().unwrap();
        assert!(err.to_string().contains("not a valid slice"), "unexpected error: {err}");
    }
}
