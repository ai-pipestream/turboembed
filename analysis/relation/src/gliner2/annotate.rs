use turbo_analysis::{Annotator, Edges, Error, ModelRef, Refs, Span, View};
use turbo_analysis_layers::{Entities, Tokens};

use super::Gliner2Relations;
use crate::Relations;

impl Gliner2Relations {
    pub fn configure(&mut self, labels: Vec<String>, threshold: f32) -> Result<&mut Self, String> {
        self.config.set(labels, threshold)?;
        Ok(self)
    }
}

impl Annotator for Gliner2Relations {
    const ID: &'static str = "turbo:gliner2-relations";
    type Reads = (View, Tokens, Option<Entities>);
    type Writes = (Entities, Relations, Edges<Entities, Entities>);

    fn annotate(&self, (view, tokens, existing): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let fail = |detail: String| Error::Annotator { annotator: Self::ID, detail };
        if self.config.labels.is_empty() {
            return Err(fail("configure relation labels before annotation".into()));
        }
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
                let folded = super::words::lowercase(raw);
                words.push(super::words::Word {
                    text: folded,
                    start: span.start as usize,
                    end: span.end as usize,
                    subword_len: 0,
                });
            }
        }
        let prompt = super::prompt::build_words(&self.tokenizer, &self.specials, text, &labels, words);
        let pass = self.pass_prompt(prompt).map_err(|e| fail(e.to_string()))?;
        let extracted = super::decode(&pass, self.config.threshold).map_err(|e| fail(e.to_string()))?;
        let len = view.text().len();
        let clip = |found: super::Found| {
            let end = found.end.min(len);
            (found.start < end).then_some(Span::new(found.start as u32, end as u32))
        };
        let spans: Vec<Option<Span>> = extracted.iter().flat_map(|r| [clip(r.head), clip(r.tail)]).collect();
        let needed: Vec<Span> = spans.iter().flatten().copied().collect();
        let (entities, mapping) = crate::neural::entities(view, tokens, existing, &needed).map_err(fail)?;
        let mut at = 0;
        let mut next = || {
            let i = mapping[at];
            at += 1;
            i
        };
        let mut found = Vec::new();
        for (r, ends) in extracted.iter().zip(spans.as_chunks::<2>().0) {
            let head = ends[0].map(|_| next());
            let tail = ends[1].map(|_| next());
            if let (Some(Some(head)), Some(Some(tail))) = (head, tail) {
                found.push((head, tail, r.label, r.head.score.min(r.tail.score)));
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
