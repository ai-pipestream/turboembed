//! Document-level ([`PolarityAnnotator`]) and sentence-level
//! ([`SentencePolarityAnnotator`]) polarity annotators combining VADER
//! (`LexiconSentiment`) and learned [`PolarityVectorScorer`] heads.

use std::hash::{Hash, Hasher};

use turbo_analysis::{Annotator, Error, Ix, IxRange, Layer, ModelRef, Refs, Tag, View};
use turbo_analysis_layers::{Categories, Dependencies, Opinions, Polarity, PolarityScores, Sentences, Tokens, Vectors};
use turbo_normalize_layers::Emoji;

use crate::calibrated::CalibratableCategorizer;
use crate::scorers::{PolarityStage, PolarityVectorScorer};
use crate::sentiment::lexicon::{COMPOUND_THRESHOLD, VaderScore};
use crate::sentiment::validation;

use super::head::{POLARITY_NAMES, polarity_index, validate_polarity_scores};
use super::{PolarityAnnotator, SentencePolarityAnnotator};

/// Smoothing floor applied when converting VADER's compound decision into a
/// calibrated three-class probability distribution in `(0, 1)`.
const VADER_PROB_SMOOTHING: f64 = 0.03;

/// Converts a [`VaderScore`] into normalized three-class [`PolarityScores`]
/// in `(0, 1)` summing to `1.0` whose argmax (`scores.label()`) always matches
/// VADER's compound-threshold decision (`score.polarity()`).
pub fn vader_to_polarity_scores(score: VaderScore) -> (Polarity, PolarityScores) {
    let polarity = score.polarity();
    let c = if score.compound.is_finite() { score.compound.clamp(-1.0, 1.0) } else { 0.0 };
    let pos_raw = if score.pos.is_finite() && score.pos > 0.0 { score.pos } else { 0.0 };
    let neg_raw = if score.neg.is_finite() && score.neg > 0.0 { score.neg } else { 0.0 };
    let neu_raw = if score.neu.is_finite() && score.neu > 0.0 { score.neu } else { 0.0 };

    let (p_pos, p_neg, p_neu) = match polarity {
        Polarity::Positive => {
            let t = ((c - COMPOUND_THRESHOLD) / (1.0 - COMPOUND_THRESHOLD)).clamp(0.0, 1.0);
            let p_pos = 0.88 + 0.06 * t;
            let excess = 1.0 - p_pos - 2.0 * VADER_PROB_SMOOTHING;
            let other_sum = neg_raw + neu_raw;
            let neg_share = if other_sum > 0.0 { (neg_raw / other_sum).clamp(0.0, 1.0) } else { 0.5 };
            let p_neg = VADER_PROB_SMOOTHING + excess * neg_share;
            let p_neu = VADER_PROB_SMOOTHING + excess * (1.0 - neg_share);
            (p_pos, p_neg, p_neu)
        }
        Polarity::Negative => {
            let t = ((-c - COMPOUND_THRESHOLD) / (1.0 - COMPOUND_THRESHOLD)).clamp(0.0, 1.0);
            let p_neg = 0.88 + 0.06 * t;
            let excess = 1.0 - p_neg - 2.0 * VADER_PROB_SMOOTHING;
            let other_sum = pos_raw + neu_raw;
            let pos_share = if other_sum > 0.0 { (pos_raw / other_sum).clamp(0.0, 1.0) } else { 0.5 };
            let p_pos = VADER_PROB_SMOOTHING + excess * pos_share;
            let p_neu = VADER_PROB_SMOOTHING + excess * (1.0 - pos_share);
            (p_pos, p_neg, p_neu)
        }
        Polarity::Neutral => {
            let u = (c / COMPOUND_THRESHOLD).clamp(-1.0, 1.0);
            let p_neu = 0.94 - 0.06 * u.abs();
            let excess = 1.0 - p_neu - 2.0 * VADER_PROB_SMOOTHING;
            let pos_share = 0.5 * (1.0 + u);
            let p_pos = VADER_PROB_SMOOTHING + excess * pos_share;
            let p_neg = VADER_PROB_SMOOTHING + excess * (1.0 - pos_share);
            (p_pos, p_neg, p_neu)
        }
    };
    let scores = PolarityScores::new(p_pos, p_neg, p_neu);
    (polarity, scores)
}

/// Blends lexicon and learned [`PolarityScores`] with `head_weight` in `0.0..=1.0`.
pub fn blend_polarity_scores(
    lexicon: PolarityScores,
    learned: PolarityScores,
    head_weight: f64,
) -> Result<(Polarity, PolarityScores), String> {
    if !head_weight.is_finite() || !(0.0..=1.0).contains(&head_weight) {
        return Err(format!("polarity head_weight must be in 0.0..=1.0, got {head_weight}"));
    }
    let lw = 1.0 - head_weight;
    let blended = PolarityScores::new(
        lw * lexicon.positive + head_weight * learned.positive,
        lw * lexicon.negative + head_weight * learned.negative,
        lw * lexicon.neutral + head_weight * learned.neutral,
    );
    let normalized = validate_polarity_scores(blended)?;
    Ok((normalized.label(), normalized))
}

impl PolarityAnnotator {
    /// Scores a document directly and returns `(Polarity, PolarityScores)`.
    pub fn score_document(
        &self,
        view: &View,
        sentences: Option<&Sentences>,
        tokens: Option<&Tokens>,
        deps: Option<&Dependencies>,
        emoji: Option<&Emoji>,
        vectors: Option<&Vectors<Sentences>>,
    ) -> Result<(Polarity, PolarityScores), Error> {
        validate_head_weight(Self::ID, self.head_weight)?;

        let lex_result = if matches!(self.stage, PolarityStage::Lexicon | PolarityStage::Hybrid) {
            Some(self.score_document_lexicon(view, sentences, tokens, deps, emoji)?)
        } else {
            None
        };

        let head_result = if matches!(self.stage, PolarityStage::Learned | PolarityStage::Hybrid) {
            let scorer = self.vector_scorer.as_deref().ok_or_else(|| Error::Annotator {
                annotator: Self::ID,
                detail: "learned or hybrid polarity stage requires a PolarityVectorScorer".into(),
            })?;
            let vecs = vectors.ok_or_else(|| Error::Annotator {
                annotator: Self::ID,
                detail: "learned or hybrid polarity stage requires Vectors<Sentences>".into(),
            })?;
            validation::checked(Self::ID, vecs)?;
            if let Some(s) = sentences
                && vecs.len() != s.len()
            {
                return Err(Error::Annotator {
                    annotator: Self::ID,
                    detail: format!(
                        "Vectors<Sentences> length {} does not match sentences length {}",
                        vecs.len(),
                        s.len()
                    ),
                });
            }
            if Layer::len(vecs) == 0 {
                return Err(Error::Annotator {
                    annotator: Self::ID,
                    detail: "Vectors<Sentences> layer is empty".into(),
                });
            }
            check_encoder_model(Self::ID, scorer, &vecs.model)?;
            let mut pos_sum = 0.0f64;
            let mut neg_sum = 0.0f64;
            let mut neu_sum = 0.0f64;
            let mut non_empty = 0usize;
            for i in 0..Layer::len(vecs) {
                let ix = Ix::<Sentences>::new(i as u32);
                if vecs.is_empty(ix) {
                    continue;
                }
                let raw = scorer
                    .score_polarity(vecs.row(ix))
                    .map_err(|detail| Error::Annotator { annotator: Self::ID, detail })?;
                let validated =
                    validate_polarity_scores(raw).map_err(|detail| Error::Annotator { annotator: Self::ID, detail })?;
                pos_sum += validated.positive;
                neg_sum += validated.negative;
                neu_sum += validated.neutral;
                non_empty += 1;
            }
            if non_empty == 0 {
                return Err(Error::Annotator {
                    annotator: Self::ID,
                    detail: "Vectors<Sentences> has no non-empty sentence rows".into(),
                });
            }
            let n = non_empty as f64;
            let pooled = validate_polarity_scores(PolarityScores::new(pos_sum / n, neg_sum / n, neu_sum / n))
                .map_err(|detail| Error::Annotator { annotator: Self::ID, detail })?;
            Some((pooled.label(), pooled))
        } else {
            None
        };

        match (self.stage, lex_result, head_result) {
            (PolarityStage::Lexicon, Some(res), _) => Ok(res),
            (PolarityStage::Learned, _, Some(res)) => Ok(res),
            (PolarityStage::Hybrid, Some((_, lex_scores)), Some((_, head_scores))) => {
                blend_polarity_scores(lex_scores, head_scores, self.head_weight)
                    .map_err(|detail| Error::Annotator { annotator: Self::ID, detail })
            }
            _ => Err(Error::Annotator { annotator: Self::ID, detail: "unreachable polarity stage state".into() }),
        }
    }

    fn score_document_lexicon(
        &self,
        view: &View,
        sentences: Option<&Sentences>,
        tokens: Option<&Tokens>,
        deps: Option<&Dependencies>,
        emoji: Option<&Emoji>,
    ) -> Result<(Polarity, PolarityScores), Error> {
        let text = view.text();
        validate_lexicon_inputs(Self::ID, view, sentences, tokens, deps, emoji)?;

        if let (Some(sents), Some(toks)) = (sentences, tokens)
            && !sents.is_empty()
        {
            let tok_count = toks.len();
            let mut tok_cursor = 0usize;
            let mut pos_sum = 0.0f64;
            let mut neg_sum = 0.0f64;
            let mut neu_sum = 0.0f64;
            let mut compound_sum = 0.0f64;
            let n = sents.len() as f64;

            for (_, sent_span) in sents.spans.iter() {
                while tok_cursor < tok_count && toks.spans.get(Ix::new(tok_cursor as u32)).end <= sent_span.start {
                    tok_cursor += 1;
                }
                let sent_tok_start = tok_cursor;
                while tok_cursor < tok_count && toks.spans.get(Ix::new(tok_cursor as u32)).start < sent_span.end {
                    tok_cursor += 1;
                }
                let sent_tok_end = tok_cursor;
                let range = IxRange::new(sent_tok_start as u32, sent_tok_end as u32);
                let res = self.lexicon.score_sentence(text, sent_span, toks, range, deps, emoji);
                pos_sum += res.score.pos;
                neg_sum += res.score.neg;
                neu_sum += res.score.neu;
                compound_sum += res.score.compound;
            }
            let doc_vader =
                VaderScore { pos: pos_sum / n, neg: neg_sum / n, neu: neu_sum / n, compound: compound_sum / n };
            return Ok(vader_to_polarity_scores(doc_vader));
        }

        if let Some(toks) = tokens
            && !toks.is_empty()
        {
            let doc_span = turbo_analysis::Span::new(0, text.len() as u32);
            let range = IxRange::new(0, toks.len() as u32);
            let res = self.lexicon.score_sentence(text, doc_span, toks, range, deps, emoji);
            return Ok(vader_to_polarity_scores(res.score));
        }

        Ok(vader_to_polarity_scores(self.lexicon.score_text(text)))
    }
}

impl Annotator for PolarityAnnotator {
    const ID: &'static str = "turbo-doccat:polarity";
    const VERSION: &'static str = "1";
    type Reads =
        (View, Option<Sentences>, Option<Tokens>, Option<Dependencies>, Option<Emoji>, Option<Vectors<Sentences>>);
    type Writes = (Categories,);

    fn annotate(
        &self,
        (view, sentences, tokens, deps, emoji, vectors): Refs<'_, Self>,
    ) -> Result<(Categories,), Error> {
        let (polarity, scores) = self.score_document(view, sentences, tokens, deps, emoji, vectors)?;
        let mut categories = Categories::with_capacity(self.categories.clone(), 1);
        let best_tag = Tag(polarity_index(polarity) as u16);
        categories.push(best_tag, &[scores.positive, scores.negative, scores.neutral]);
        Ok((categories,))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.effective_model_ref())
    }

    fn config_hash(&self) -> u64 {
        let mut h = crate::hash::FoldHasher::default();
        self.stage.hash(&mut h);
        self.head_weight.to_bits().hash(&mut h);
        Annotator::config_hash(&self.lexicon).hash(&mut h);
        if let Some(scorer) = &self.vector_scorer {
            scorer.config_hash().hash(&mut h);
        }
        h.finish()
    }
}

impl CalibratableCategorizer for PolarityAnnotator {
    fn calibration_labels(&self) -> Vec<String> {
        POLARITY_NAMES.iter().map(|&s| s.to_owned()).collect()
    }
}

impl Annotator for SentencePolarityAnnotator {
    const ID: &'static str = "turbo-doccat:sentence-polarity";
    const VERSION: &'static str = "1";
    type Reads = (View, Sentences, Option<Tokens>, Option<Dependencies>, Option<Emoji>, Option<Vectors<Sentences>>);
    type Writes = (Opinions, Categories<Sentences>);

    fn annotate(
        &self,
        (view, sentences, tokens, deps, emoji, vectors): Refs<'_, Self>,
    ) -> Result<(Opinions, Categories<Sentences>), Error> {
        validate_head_weight(Self::ID, self.head_weight)?;
        let text = view.text();
        let need_lexicon = matches!(self.stage, PolarityStage::Lexicon | PolarityStage::Hybrid);
        let need_head = matches!(self.stage, PolarityStage::Learned | PolarityStage::Hybrid);
        validate_lexicon_inputs(Self::ID, view, Some(sentences), tokens, deps, emoji)?;
        if tokens.is_none() && need_lexicon {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: "lexicon or hybrid sentence polarity requires Tokens".into(),
            });
        }

        let scorer_and_vecs = if need_head {
            let scorer = self.vector_scorer.as_deref().ok_or_else(|| Error::Annotator {
                annotator: Self::ID,
                detail: "learned or hybrid sentence polarity requires a PolarityVectorScorer".into(),
            })?;
            let vecs = vectors.ok_or_else(|| Error::Annotator {
                annotator: Self::ID,
                detail: "learned or hybrid sentence polarity requires Vectors<Sentences>".into(),
            })?;
            validation::checked(Self::ID, vecs)?;
            if vecs.len() != sentences.len() {
                return Err(Error::Annotator {
                    annotator: Self::ID,
                    detail: format!(
                        "Vectors<Sentences> length {} does not match sentences length {}",
                        vecs.len(),
                        sentences.len()
                    ),
                });
            }
            check_encoder_model(Self::ID, scorer, &vecs.model)?;
            Some((scorer, vecs))
        } else {
            None
        };

        let mut opinions = Opinions::new();
        let mut categories = Categories::<Sentences>::with_capacity(self.categories.clone(), sentences.len());
        let tok_count = tokens.map_or(0, Tokens::len);
        let mut tok_cursor = 0usize;

        for (sent_ix, sent_span) in sentences.spans.iter() {
            let lex_pair = if let Some(toks) = tokens {
                while tok_cursor < tok_count && toks.spans.get(Ix::new(tok_cursor as u32)).end <= sent_span.start {
                    tok_cursor += 1;
                }
                let sent_tok_start = tok_cursor;
                while tok_cursor < tok_count && toks.spans.get(Ix::new(tok_cursor as u32)).start < sent_span.end {
                    tok_cursor += 1;
                }
                let sent_tok_end = tok_cursor;
                let range = IxRange::new(sent_tok_start as u32, sent_tok_end as u32);
                let result = self.lexicon.score_sentence(text, sent_span, toks, range, deps, emoji);
                for op in result.opinions {
                    opinions.push(op);
                }
                Some(vader_to_polarity_scores(result.score))
            } else {
                None
            };

            let head_pair = if let Some((scorer, vecs)) = scorer_and_vecs {
                if vecs.is_empty(sent_ix) {
                    Some((Polarity::Neutral, PolarityScores::new(0.0, 0.0, 1.0)))
                } else {
                    let raw = scorer
                        .score_polarity(vecs.row(sent_ix))
                        .map_err(|detail| Error::Annotator { annotator: Self::ID, detail })?;
                    let validated = validate_polarity_scores(raw)
                        .map_err(|detail| Error::Annotator { annotator: Self::ID, detail })?;
                    Some((validated.label(), validated))
                }
            } else {
                None
            };

            let (polarity, scores) = match (self.stage, lex_pair, head_pair) {
                (PolarityStage::Lexicon, Some(pair), _) => pair,
                (PolarityStage::Learned, _, Some(pair)) => pair,
                (PolarityStage::Hybrid, Some((_, lex_scores)), Some((_, head_scores))) => {
                    blend_polarity_scores(lex_scores, head_scores, self.head_weight)
                        .map_err(|detail| Error::Annotator { annotator: Self::ID, detail })?
                }
                _ => {
                    return Err(Error::Annotator {
                        annotator: Self::ID,
                        detail: "missing required inputs for sentence polarity stage".into(),
                    });
                }
            };

            let best_tag = Tag(polarity_index(polarity) as u16);
            categories.push(best_tag, &[scores.positive, scores.negative, scores.neutral]);
        }

        Ok((opinions, categories))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.effective_model_ref())
    }

    fn config_hash(&self) -> u64 {
        let mut h = crate::hash::FoldHasher::default();
        self.stage.hash(&mut h);
        self.head_weight.to_bits().hash(&mut h);
        Annotator::config_hash(&self.lexicon).hash(&mut h);
        if let Some(scorer) = &self.vector_scorer {
            scorer.config_hash().hash(&mut h);
        }
        h.finish()
    }
}

fn validate_head_weight(annotator: &'static str, weight: f64) -> Result<(), Error> {
    if !weight.is_finite() || !(0.0..=1.0).contains(&weight) {
        return Err(Error::Annotator {
            annotator,
            detail: format!("polarity head_weight must be in 0.0..=1.0, got {weight}"),
        });
    }
    Ok(())
}

fn check_encoder_model(
    annotator: &'static str,
    scorer: &dyn PolarityVectorScorer,
    actual: &ModelRef,
) -> Result<(), Error> {
    if let Some(expected) = scorer.model()
        && &expected != actual
    {
        return Err(Error::Annotator {
            annotator,
            detail: format!("polarity head expects encoder {expected:?}, got {actual:?}"),
        });
    }
    Ok(())
}

fn validate_lexicon_inputs(
    annotator: &'static str,
    view: &View,
    sentences: Option<&Sentences>,
    tokens: Option<&Tokens>,
    deps: Option<&Dependencies>,
    emoji: Option<&Emoji>,
) -> Result<(), Error> {
    let text = view.text();
    if let Some(s) = sentences {
        if s.spans.view() != view.id() {
            return Err(Error::Annotator { annotator, detail: "sentences are not in the given view".into() });
        }
        validation::checked(annotator, s)?;
        validation::contiguous(annotator, &s.spans, text)?;
    }
    if let Some(t) = tokens {
        if t.spans.view() != view.id() {
            return Err(Error::Annotator { annotator, detail: "tokens are not in the given view".into() });
        }
        validation::tokens(annotator, t, sentences, text)?;
    }
    if let Some(e) = emoji {
        if e.spans.view() != view.id() {
            return Err(Error::Annotator { annotator, detail: "emoji layer is not in the given view".into() });
        }
        validation::checked(annotator, e)?;
        validation::contiguous(annotator, &e.spans, text)?;
    }
    if let Some(d) = deps {
        validation::checked(annotator, d)?;
        let Some(t) = tokens else {
            return Err(Error::Annotator { annotator, detail: "dependencies require tokens".into() });
        };
        if d.len() != t.len() {
            return Err(Error::Annotator { annotator, detail: "dependencies must align with tokens".into() });
        }
    }
    Ok(())
}
