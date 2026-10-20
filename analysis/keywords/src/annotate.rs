//! The analysis steps: [`YakeAnnotator`] and [`TextRankAnnotator`] write
//! the `turbo:keywords` layer from the document's sentences, tokens,
//! terms and (for TextRank) POS tags. Both read the text view their
//! spans must live in, and refuse spans that do not slice it.

use turbo_analysis::{Annotator, Error, Refs, Span, View};
use turbo_analysis_layers::{PosTags, Sentences, Terms, Tokens};

use crate::layer::{Keywords, TEXTRANK, YAKE, algorithm_tagset};
use crate::ranked::RankedKeyword;
use crate::{Stopwords, TextRankExtractor, YakeExtractor};

/// Runs one extractor and writes its survivors, one layer item per
/// occurrence span, the items in document order.
fn emit(view: &View, ranked: Vec<RankedKeyword>, algorithm: &str) -> Keywords {
    let mut kws = Keywords::new(view.id());
    let tag = algorithm_tagset().tag(algorithm).expect("the shared algorithm tag set has both labels");
    let mut items: Vec<(Span, &str, f64, u16)> = Vec::with_capacity(ranked.iter().map(|k| k.occurrences.len()).sum());
    for kw in &ranked {
        for &span in &kw.occurrences {
            items.push((span, kw.phrase.as_str(), kw.salience, kw.terms));
        }
    }
    items.sort_by_key(|(span, ..)| (span.start, span.end));
    for (span, phrase, score, terms) in items {
        kws.push(span, phrase, score, tag, terms);
    }
    kws
}

/// Every span of `tokens` must be a UTF-8 slice of `view`'s text; a
/// foreign view or a broken span fails the document rather than scoring
/// garbage.
fn check_spans(view: &View, tokens: &Tokens, who: &'static str) -> Result<(), Error> {
    if tokens.spans.view() != view.id() {
        return Err(Error::Annotator { annotator: who, detail: "the tokens are in another view".into() });
    }
    let text = view.text();
    for (i, span) in tokens.spans.iter() {
        let (s, e) = (span.start as usize, span.end as usize);
        if s > e || e > text.len() || !text.is_char_boundary(s) || !text.is_char_boundary(e) {
            return Err(Error::Annotator {
                annotator: who,
                detail: format!(
                    "token {} span {s}..{e} is not a valid UTF-8 slice of view text of length {}",
                    i.get(),
                    text.len()
                ),
            });
        }
    }
    Ok(())
}

/// YAKE over the document's sentences, tokens and terms, writing
/// `turbo:keywords` tagged `yake`. The stopword filter is not a layer:
/// it is the extractor's configuration, one of the bundled lists or
/// the caller's own words.
#[derive(Clone, Debug)]
pub struct YakeAnnotator {
    extractor: YakeExtractor,
}

impl YakeAnnotator {
    pub fn new(stopwords: Stopwords) -> YakeAnnotator {
        YakeAnnotator { extractor: YakeExtractor::new(stopwords) }
    }

    pub fn extractor(&self) -> &YakeExtractor {
        &self.extractor
    }

    pub fn with_window_size(mut self, window_size: usize) -> YakeAnnotator {
        self.extractor = self.extractor.with_window_size(window_size);
        self
    }

    pub fn with_ngrams(mut self, min_ngram: usize, max_ngram: usize) -> YakeAnnotator {
        self.extractor = self.extractor.with_ngrams(min_ngram, max_ngram);
        self
    }

    pub fn with_top_n(mut self, top_n: usize) -> YakeAnnotator {
        self.extractor = self.extractor.with_top_n(top_n);
        self
    }

    pub fn with_threshold(mut self, threshold: f64) -> YakeAnnotator {
        self.extractor = self.extractor.with_threshold(threshold);
        self
    }

    pub fn with_dedup(mut self, dedup: crate::Dedup) -> YakeAnnotator {
        self.extractor = self.extractor.with_dedup(dedup);
        self
    }
}

impl Annotator for YakeAnnotator {
    const ID: &'static str = "turbo-keywords:yake";
    type Reads = (View, Sentences, Tokens, Terms);
    type Writes = (Keywords,);

    fn annotate(&self, (view, sentences, tokens, terms): Refs<'_, Self>) -> Result<(Keywords,), Error> {
        check_spans(view, tokens, Self::ID)?;
        if sentences.spans.view() != view.id() {
            return Err(Error::Annotator { annotator: Self::ID, detail: "the sentences are in another view".into() });
        }
        let ranked = self.extractor.extract(sentences, tokens, terms)?;
        Ok((emit(view, ranked, YAKE),))
    }

    fn config_hash(&self) -> u64 {
        self.extractor.config_hash()
    }
}

/// TextRank over the document's sentences, tokens, POS tags and terms,
/// writing `turbo:keywords` tagged `textrank`.
#[derive(Clone, Debug, Default)]
pub struct TextRankAnnotator {
    extractor: TextRankExtractor,
}

impl TextRankAnnotator {
    pub fn new() -> TextRankAnnotator {
        TextRankAnnotator::default()
    }

    pub fn extractor(&self) -> &TextRankExtractor {
        &self.extractor
    }

    pub fn with_window(mut self, window: usize) -> TextRankAnnotator {
        self.extractor = self.extractor.with_window(window);
        self
    }

    pub fn with_top_n(mut self, top_n: usize) -> TextRankAnnotator {
        self.extractor = self.extractor.with_top_n(top_n);
        self
    }

    pub fn with_damping(mut self, damping: f64) -> TextRankAnnotator {
        self.extractor = self.extractor.with_damping(damping);
        self
    }

    pub fn with_epsilon(mut self, epsilon: f64) -> TextRankAnnotator {
        self.extractor = self.extractor.with_epsilon(epsilon);
        self
    }
}

impl Annotator for TextRankAnnotator {
    const ID: &'static str = "turbo-keywords:textrank";
    type Reads = (View, Sentences, Tokens, PosTags, Terms);
    type Writes = (Keywords,);

    fn annotate(&self, (view, sentences, tokens, pos_tags, terms): Refs<'_, Self>) -> Result<(Keywords,), Error> {
        check_spans(view, tokens, Self::ID)?;
        if sentences.spans.view() != view.id() {
            return Err(Error::Annotator { annotator: Self::ID, detail: "the sentences are in another view".into() });
        }
        let ranked = self.extractor.extract(sentences, tokens, pos_tags, terms)?;
        Ok((emit(view, ranked, TEXTRANK),))
    }

    fn config_hash(&self) -> u64 {
        self.extractor.config_hash()
    }
}
