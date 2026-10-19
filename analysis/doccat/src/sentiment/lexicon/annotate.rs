use super::super::validation;
use super::*;

impl Annotator for LexiconSentiment {
    const ID: &'static str = "turbo-doccat:lexicon-sentiment";
    const VERSION: &'static str = "1";
    type Reads = (View, Sentences, Tokens, Option<Dependencies>, Option<Emoji>);
    type Writes = (Opinions, Categories<Sentences>);

    fn annotate(
        &self,
        (view, sentences, tokens, deps, emoji): Refs<'_, Self>,
    ) -> Result<(Opinions, Categories<Sentences>), Error> {
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

        let text = view.text();
        validation::tokens(Self::ID, tokens, Some(sentences), text)?;

        if let Some(emoji) = emoji {
            validation::checked(Self::ID, emoji)?;
            validation::contiguous(Self::ID, &emoji.spans, text)?;
        }
        if let Some(deps) = deps {
            validation::checked(Self::ID, deps)?;
            if deps.len() != tokens.len() {
                return Err(Error::Annotator {
                    annotator: Self::ID,
                    detail: "dependencies must align with tokens".into(),
                });
            }
        }
        let mut opinions = Opinions::new();
        let mut categories = Categories::<Sentences>::with_capacity(self.categories.clone(), sentences.len());

        let tok_count = tokens.len();
        let mut tok_cursor = 0usize;

        for (_, sent_span) in sentences.spans.iter() {
            while tok_cursor < tok_count && tokens.spans.get(Ix::new(tok_cursor as u32)).end <= sent_span.start {
                tok_cursor += 1;
            }
            let sent_tok_start = tok_cursor;
            while tok_cursor < tok_count && tokens.spans.get(Ix::new(tok_cursor as u32)).start < sent_span.end {
                tok_cursor += 1;
            }
            let sent_tok_end = tok_cursor;
            let range = IxRange::new(sent_tok_start as u32, sent_tok_end as u32);

            let result = self.score_sentence(text, sent_span, tokens, range, deps, emoji);
            for op in result.opinions {
                opinions.push(op);
            }

            let (pos, neg, mut neu) = (result.score.pos, result.score.neg, result.score.neu);
            if pos == 0.0 && neg == 0.0 && neu == 0.0 {
                neu = 1.0;
            }
            // The best category follows the compound threshold, not the largest
            // proportion: "not bad" is mostly neutral tokens but classifies positive.
            let best_tag = match result.score.polarity() {
                Polarity::Positive => Tag(0),
                Polarity::Negative => Tag(1),
                Polarity::Neutral => Tag(2),
            };
            categories.push(best_tag, &[pos, neg, neu]);
        }

        Ok((opinions, categories))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.model_ref.clone())
    }

    fn config_hash(&self) -> u64 {
        let mut h = crate::hash::FoldHasher::default();
        self.negation_mode.hash(&mut h);
        self.use_opennlp_emoji.hash(&mut h);
        self.alpha.to_bits().hash(&mut h);
        self.lexicon.sha256_hex().hash(&mut h);
        h.finish()
    }
}
