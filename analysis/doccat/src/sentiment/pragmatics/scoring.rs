use super::super::validation;
use super::*;

impl PragmaticsAnnotator {
    /// Scores every sentence in `sentences` and builds the [`Pragmatics`]
    /// layer.
    #[allow(clippy::too_many_arguments)]
    pub fn score_document(
        &self,
        view: &View,
        sentences: &Sentences,
        tokens: &Tokens,
        opinions: Option<&Opinions>,
        categories: Option<&Categories<Sentences>>,
        emoji: Option<&Emoji>,
        vectors: Option<&Vectors<Sentences>>,
    ) -> Result<Pragmatics, Error> {
        if !self.positive_threshold.is_finite()
            || !self.min_positive_polarity.is_finite()
            || !self.min_negative_polarity.is_finite()
        {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: "pragmatics thresholds must be finite".into(),
            });
        }
        if sentences.spans.view() != view.id() {
            return Err(Error::Annotator { annotator: Self::ID, detail: "sentences are not in the given view".into() });
        }
        if tokens.spans.view() != view.id() {
            return Err(Error::Annotator { annotator: Self::ID, detail: "tokens are not in the given view".into() });
        }
        if let Some(e) = emoji
            && e.spans.view() != view.id()
        {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: "emoji layer is not in the given view".into(),
            });
        }
        if let Some(cats) = categories
            && cats.len() != sentences.len()
        {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!(
                    "Categories<Sentences> length ({}) does not match Sentences length ({})",
                    cats.len(),
                    sentences.len()
                ),
            });
        }
        if let Some(vecs) = vectors
            && vecs.len() != sentences.len()
        {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!(
                    "Vectors<Sentences> length ({}) does not match Sentences length ({})",
                    vecs.len(),
                    sentences.len()
                ),
            });
        }

        let text = view.text();
        validation::tokens(Self::ID, tokens, Some(sentences), text)?;

        if let Some(emoji) = emoji {
            validation::checked(Self::ID, emoji)?;
            validation::contiguous(Self::ID, &emoji.spans, text)?;
        }
        if let Some(cats) = categories {
            validation::checked(Self::ID, cats)?;
        }
        if let Some(vectors) = vectors {
            validation::vectors(Self::ID, vectors, sentences.len())?;
        }
        if let Some(ops) = opinions {
            validation::checked(Self::ID, ops)?;
            if ops.tokens.iter().any(|range| range.end.get() > tokens.len()) {
                return Err(Error::Annotator {
                    annotator: Self::ID,
                    detail: "opinion token range out of bounds".into(),
                });
            }
        }
        let lower_tokens: Vec<String> = (0..tokens.len())
            .map(|i| {
                let sp = tokens.spans.get(Ix::new(i as u32));
                clean_token(&text[sp.range()])
            })
            .collect();

        let mut out = Pragmatics::new();
        for s_idx in 0..sentences.len() {
            let sent_ix = Ix::<Sentences>::new(s_idx as u32);
            let sent_span = sentences.spans.get(sent_ix);
            let tok_range = sentence_token_range(s_idx, sent_span, tokens);

            let head_score = self.score_head_for_sentence(s_idx, categories, vectors)?;

            if tok_range.is_empty() {
                let score = head_score.unwrap_or(0.0).clamp(0.0, 1.0);
                out.try_push(sentences, tokens, PragmaticsItem { score, evidence: None })
                    .map_err(|detail| Error::Annotator { annotator: Self::ID, detail })?;
                continue;
            }

            let emoji_spans = collect_sentence_emoji(sent_span, tok_range, text, tokens, emoji);
            let riloff = self.detect_riloff_incongruity(tok_range, &lower_tokens, opinions, &emoji_spans);
            let emoji_mismatch =
                self.detect_emoji_mismatch(s_idx, tok_range, &lower_tokens, opinions, categories, &emoji_spans);

            let explicit = match (riloff, emoji_mismatch) {
                (Some((s1, inc1)), Some((s2, inc2))) => {
                    let combined = 1.0 - (1.0 - s1) * (1.0 - s2 * 0.5);
                    if s2 > s1 {
                        Some((combined.clamp(0.0, 1.0), inc2))
                    } else {
                        Some((combined.clamp(0.0, 1.0), inc1))
                    }
                }
                (Some(r), None) => Some(r),
                (None, Some(e)) => Some(e),
                (None, None) => None,
            };

            let item = match (explicit, head_score) {
                (Some((contrast_score, incongruity)), Some(h)) => {
                    let score = if h >= self.positive_threshold {
                        (1.0 - (1.0 - contrast_score) * (1.0 - h * 0.35)).clamp(0.0, 1.0)
                    } else {
                        contrast_score.clamp(0.0, 1.0)
                    };
                    PragmaticsItem { score, evidence: Some(incongruity) }
                }
                (Some((contrast_score, incongruity)), None) => {
                    PragmaticsItem { score: contrast_score.clamp(0.0, 1.0), evidence: Some(incongruity) }
                }
                (None, Some(h)) => PragmaticsItem { score: h.clamp(0.0, 1.0), evidence: None },
                (None, None) => PragmaticsItem { score: 0.0, evidence: None },
            };

            out.try_push(sentences, tokens, item).map_err(|detail| Error::Annotator { annotator: Self::ID, detail })?;
        }

        Ok(out)
    }

    fn score_head_for_sentence(
        &self,
        sentence_index: usize,
        categories: Option<&Categories<Sentences>>,
        vectors: Option<&Vectors<Sentences>>,
    ) -> Result<Option<f64>, Error> {
        let mut best: Option<f64> = None;

        let ix = Ix::<Sentences>::new(sentence_index as u32);
        if let (Some(scorer), Some(vecs)) = (&self.vector_scorer, vectors)
            && !vecs.is_empty(ix)
        {
            // A row the encoder marked empty is zeros, not evidence: the
            // scorer never sees it. Vectors from another encoder are a
            // different feature space, so the scorer's model, when it
            // names one, must be the vectors' model.
            if let Some(m) = scorer.model_ref()
                && m != vecs.model
            {
                return Err(Error::Annotator {
                    annotator: Self::ID,
                    detail: format!(
                        "sentence vectors are from model {} (revision {}), scorer is for {} (revision {})",
                        vecs.model.id, vecs.model.revision, m.id, m.revision
                    ),
                });
            }
            let s = scorer
                .score_sentence_vector(vecs.row(ix))
                .map_err(|detail| Error::Annotator { annotator: Self::ID, detail })?;
            if !s.is_finite() || !(0.0..=1.0).contains(&s) {
                return Err(Error::Annotator {
                    annotator: Self::ID,
                    detail: format!(
                        "PragmaticsVectorScorer returned score {s} outside 0..=1 on sentence {sentence_index}"
                    ),
                });
            }
            best = Some(s);
        }

        if let Some(cats) = categories {
            let mut cat_prag: Option<f64> = None;
            for label in ["sarcasm", "irony", "sarcastic", "ironic"] {
                if let Some(v) = cats.score_of(sentence_index, label)
                    && v.is_finite()
                    && (0.0..=1.0).contains(&v)
                {
                    cat_prag = Some(cat_prag.map_or(v, |prev| prev.max(v)));
                }
            }
            if let Some(v) = cat_prag {
                best = Some(best.map_or(v, |prev| prev.max(v)));
            }
        }

        Ok(best)
    }
}
