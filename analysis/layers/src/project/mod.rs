//! [`ProjectableLayer`] implementations for the standard layers in
//! `turbo-analysis-layers`.

mod discourse;
mod verify;

use std::sync::Arc;

use turbo_analysis::{Document, Error, ProjectableLayer, Span, ViewId};

use self::verify::{
    check_invariants, verify_levels, verify_sentence_ixs, verify_sentence_ranges, verify_token_ixs, verify_token_ranges,
};
use crate::{
    Chunks, Entities, Levels, Money, Numbers, Ordinals, PhraseChunks, Phrases, Pii, Quantities, Quotations, Sentences,
    Structure, Subwords, Temporals, Times, Tokens,
};

fn map_opt_spans(doc: &Document, from: ViewId, to: ViewId, spans: &[Option<Span>]) -> Result<Vec<Option<Span>>, Error> {
    spans.iter().map(|s| s.map(|span| doc.map_span(from, to, span)).transpose()).collect()
}

impl ProjectableLayer for Structure {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        let out = Structure {
            spans: doc.project_spans_monotonic(&self.spans, to)?,
            kind: self.kind.clone(),
            parent: self.parent.clone(),
            depth: self.depth.clone(),
        };
        check_invariants(&out)?;
        Ok(out)
    }
}

impl ProjectableLayer for Sentences {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        let out = Sentences { spans: doc.project_spans_monotonic(&self.spans, to)? };
        check_invariants(&out)?;
        Ok(out)
    }
}

impl ProjectableLayer for Tokens {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        let from = self.spans.view();
        let out = Tokens { spans: doc.project_spans_monotonic(&self.spans, to)?, sentence: self.sentence.clone() };
        check_invariants(&out)?;
        if let Some(sents) = &out.sentence {
            verify_sentence_ixs(doc, from, to, &out.spans, sents)?;
        }
        Ok(out)
    }
}

impl ProjectableLayer for Subwords {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        let from = self.spans.view();
        let out = Subwords {
            spans: doc.project_spans_monotonic(&self.spans, to)?,
            ids: self.ids.clone(),
            model: self.model.clone(),
            token: self.token.clone(),
        };
        check_invariants(&out)?;
        if let Some(toks) = &out.token {
            verify_token_ixs(doc, from, to, &out.spans, toks, false)?;
        }
        Ok(out)
    }
}

impl ProjectableLayer for Chunks {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        let from = self.spans.view();
        let out = Chunks {
            spans: doc.project_spans_monotonic(&self.spans, to)?,
            level: self.level.clone(),
            parent: self.parent.clone(),
            sentences: self.sentences.clone(),
            cut: self.cut.clone(),
            cost: self.cost.clone(),
            model_tokens: self.model_tokens.clone(),
            over_budget: self.over_budget.clone(),
            prompt_overhead: self.prompt_overhead,
        };
        check_invariants(&out)?;
        verify_sentence_ranges(doc, from, to, &out.spans, &out.sentences)?;
        Ok(out)
    }
}

impl ProjectableLayer for Levels {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        let from = self.spans.view();
        let out = Levels {
            spans: doc.project_spans_monotonic(&self.spans, to)?,
            unit: self.unit.clone(),
            parent: self.parent.clone(),
        };
        check_invariants(&out)?;
        verify_levels(doc, from, to, &out)?;
        Ok(out)
    }
}

impl ProjectableLayer for Entities {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        let from = self.spans.view();
        let out = Entities {
            spans: doc.project_spans_monotonic(&self.spans, to)?,
            tokens: self.tokens.clone(),
            kind: self.kind.clone(),
            kinds: Arc::clone(&self.kinds),
            score: self.score.clone(),
        };
        check_invariants(&out)?;
        verify_token_ranges(doc, from, to, &out.spans, &out.tokens)?;
        Ok(out)
    }
}

impl ProjectableLayer for PhraseChunks {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        let from = self.spans.view();
        let out = PhraseChunks {
            spans: doc.project_spans_monotonic(&self.spans, to)?,
            tokens: self.tokens.clone(),
            kind: self.kind.clone(),
            kinds: Arc::clone(&self.kinds),
        };
        check_invariants(&out)?;
        verify_token_ranges(doc, from, to, &out.spans, &out.tokens)?;
        Ok(out)
    }
}

impl ProjectableLayer for Phrases {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        let from = self.spans.view();
        let out = Phrases {
            spans: doc.project_spans_monotonic(&self.spans, to)?,
            tokens: self.tokens.clone(),
            kind: self.kind.clone(),
            kinds: Arc::clone(&self.kinds),
            head: self.head.clone(),
            parent: self.parent.clone(),
        };
        check_invariants(&out)?;
        verify_token_ranges(doc, from, to, &out.spans, &out.tokens)?;
        Ok(out)
    }
}

impl ProjectableLayer for Numbers {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        let from = self.spans.view();
        let spans = doc.project_spans_monotonic(&self.spans, to)?;
        let mut numeral_start = Vec::with_capacity(self.numeral_start.len());
        let mut numeral_end = Vec::with_capacity(self.numeral_end.len());
        for i in 0..self.numeral_start.len() {
            let num = doc.map_span(from, to, Span::new(self.numeral_start[i], self.numeral_end[i]))?;
            numeral_start.push(num.start);
            numeral_end.push(num.end);
        }
        let out = Numbers {
            spans,
            numeral_start,
            numeral_end,
            negative: self.negative.clone(),
            coefficient: self.coefficient.clone(),
            scale: self.scale.clone(),
            multiplier: self.multiplier.clone(),
            marker: self.marker.clone(),
            notation: self.notation.clone(),
        };
        check_invariants(&out)?;
        Ok(out)
    }
}

impl ProjectableLayer for Money {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        let out = Money {
            spans: doc.project_spans_monotonic(&self.spans, to)?,
            negative: self.negative.clone(),
            coefficient: self.coefficient.clone(),
            scale: self.scale.clone(),
            currency: self.currency.clone(),
        };
        check_invariants(&out)?;
        Ok(out)
    }
}

impl ProjectableLayer for Quantities {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        let out = Quantities {
            spans: doc.project_spans_monotonic(&self.spans, to)?,
            negative: self.negative.clone(),
            coefficient: self.coefficient.clone(),
            scale: self.scale.clone(),
            unit: self.unit.clone(),
            normalized_negative: self.normalized_negative.clone(),
            normalized_coefficient: self.normalized_coefficient.clone(),
            normalized_scale: self.normalized_scale.clone(),
            base_unit: self.base_unit.clone(),
        };
        check_invariants(&out)?;
        Ok(out)
    }
}

impl ProjectableLayer for Times {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        let out = Times {
            spans: doc.project_spans_monotonic(&self.spans, to)?,
            kind: self.kind.clone(),
            grain: self.grain.clone(),
            modifier: self.modifier.clone(),
            start: self.start.clone(),
            end: self.end.clone(),
            duration_ms: self.duration_ms.clone(),
            ambiguous: self.ambiguous.clone(),
            timex: self.timex.clone(),
        };
        check_invariants(&out)?;
        Ok(out)
    }
}

impl ProjectableLayer for Temporals {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        let out = Temporals {
            spans: doc.project_spans_monotonic(&self.spans, to)?,
            granularity: self.granularity.clone(),
            origin: self.origin.clone(),
            year: self.year.clone(),
            period: self.period.clone(),
            day: self.day.clone(),
        };
        check_invariants(&out)?;
        Ok(out)
    }
}

impl ProjectableLayer for Ordinals {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        let out = Ordinals {
            spans: doc.project_spans_monotonic(&self.spans, to)?,
            base: self.base.clone(),
            offset: self.offset.clone(),
        };
        check_invariants(&out)?;
        Ok(out)
    }
}

impl ProjectableLayer for Pii {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        let out = Pii {
            spans: doc.project_spans_monotonic(&self.spans, to)?,
            kind: self.kind.clone(),
            confidence: self.confidence.clone(),
            normalized: self.normalized.clone(),
        };
        check_invariants(&out)?;
        Ok(out)
    }
}

impl ProjectableLayer for Quotations {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        let from = self.spans.view();
        let out = Quotations {
            spans: doc.project_spans_monotonic(&self.spans, to)?,
            speaker: map_opt_spans(doc, from, to, &self.speaker)?,
            cue_verb: map_opt_spans(doc, from, to, &self.cue_verb)?,
            speaker_chain: self.speaker_chain.clone(),
            kind: self.kind.clone(),
            confidence: self.confidence.clone(),
        };
        check_invariants(&out)?;
        Ok(out)
    }
}
