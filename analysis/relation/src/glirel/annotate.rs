use turbo_analysis::{Annotator, Edges, Error, Ix, ModelRef, Refs, View};
use turbo_analysis_layers::{Entities, Tokens};

use super::GlirelRelations;
use crate::Relations;

impl GlirelRelations {
    pub fn configure(&mut self, labels: Vec<String>, threshold: f32) -> Result<&mut Self, String> {
        self.config.set(labels, threshold)?;
        Ok(self)
    }
}

impl Annotator for GlirelRelations {
    const ID: &'static str = "turbo:glirel-relations";
    type Reads = (View, Tokens, Option<Entities>);
    type Writes = (Entities, Relations, Edges<Entities, Entities>);

    fn annotate(&self, (view, tokens, existing): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let fail = |detail: String| Error::Annotator { annotator: Self::ID, detail };
        if self.config.labels.is_empty() {
            return Err(fail("configure relation labels before annotation".into()));
        }
        let (entities, _) = crate::neural::entities(view, tokens, existing, &[]).map_err(fail)?;
        let mut found = Vec::new();
        if let Some(existing) = existing {
            let labels: Vec<&str> = self.config.labels.iter().map(String::as_str).collect();
            let text = view.text();
            let mut words = Vec::new();
            if tokens.spans.has_discontinuous() {
                words = super::words::split(text);
            } else {
                for (_, span) in tokens.spans.iter() {
                    let raw = text
                        .get(span.start as usize..span.end as usize)
                        .ok_or_else(|| fail("a token outside the view".into()))?;
                    words.push(super::words::Word {
                        text: raw,
                        start: span.start as usize,
                        end: span.end as usize,
                        subword_len: 0,
                    });
                }
            }
            let pass = self.pass_words(words, &labels).map_err(|e| fail(e.to_string()))?;
            let mut original = Vec::new();
            let mut ranges = Vec::new();
            for i in 0..existing.spans.len() {
                // The model accepts one contiguous word range per entity. Retain
                // fragmented entities in the output, but never score their envelope.
                if existing.spans.is_discontinuous(Ix::new(i as u32)) {
                    continue;
                }
                let span = existing.spans.get(Ix::new(i as u32));
                if let Some(range) = super::words_of(&pass.prompt().words, span) {
                    original.push(i as u32);
                    ranges.push(range);
                }
            }
            let pairs: Vec<(usize, usize)> = (0..ranges.len())
                .flat_map(|a| (0..ranges.len()).filter(move |&b| a != b).map(move |b| (a, b)))
                .collect();
            let logits = self.logits(&pass, &ranges, &pairs).map_err(|e| fail(e.to_string()))?;
            for (p, &(head, tail)) in pairs.iter().enumerate() {
                for (label, &logit) in logits[p * labels.len()..(p + 1) * labels.len()].iter().enumerate() {
                    let score = turbo::head::sigmoid(logit);
                    if score > self.config.threshold {
                        found.push((Ix::new(original[head]), Ix::new(original[tail]), label, score));
                    }
                }
            }
        }
        let (relations, edges) = crate::neural::layers(view, &entities, &self.config.labels, &found).map_err(fail)?;
        Ok((entities, relations, edges))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.model.clone())
    }

    fn config_hash(&self) -> u64 {
        self.config.hash()
    }
}
