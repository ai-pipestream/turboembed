//! [`ProjectableLayer`] implementations for the standard layers in
//! `turbo-analysis-layers`.

use std::sync::Arc;

use turbo_analysis::{Document, Error, Key, ProjectableLayer, Span, Spans, ViewId};

use crate::{
    Abbreviations, Aspects, ChunkTableContext, ChunkTableHeaders, Chunks, Containment, Edus, Entities, EventArgument,
    Events, KeyValuePair, KeyValuePairs, LanguageSpans, Levels, LexicalExpansions, Matches, Modality, ModalityTag,
    Money, Numbers, Opinions, Ordinals, PhraseChunks, Phrases, Pii, Quantities, Quotations, Sentences, SpanQuality,
    Structure, Subwords, Tables, Temporals, TextQuality, Times, TokenVectorWindows, Tokens, WordNetSenses, Words,
};

fn map_opt_spans(doc: &Document, from: ViewId, to: ViewId, spans: &[Option<Span>]) -> Result<Vec<Option<Span>>, Error> {
    spans.iter().map(|s| s.map(|span| doc.map_span(from, to, span)).transpose()).collect()
}

impl ProjectableLayer for Structure {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        Ok(Structure {
            spans: doc.project_spans_monotonic(&self.spans, to)?,
            kind: self.kind.clone(),
            parent: self.parent.clone(),
            depth: self.depth.clone(),
        })
    }
}

impl ProjectableLayer for Sentences {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        Ok(Sentences { spans: doc.project_spans_monotonic(&self.spans, to)? })
    }
}

impl ProjectableLayer for Tokens {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        Ok(Tokens { spans: doc.project_spans_monotonic(&self.spans, to)?, sentence: self.sentence.clone() })
    }
}

impl ProjectableLayer for Subwords {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        Ok(Subwords {
            spans: doc.project_spans_monotonic(&self.spans, to)?,
            ids: self.ids.clone(),
            model: self.model.clone(),
            token: self.token.clone(),
        })
    }
}

impl ProjectableLayer for Chunks {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        Ok(Chunks {
            spans: doc.project_spans_monotonic(&self.spans, to)?,
            level: self.level.clone(),
            parent: self.parent.clone(),
            sentences: self.sentences.clone(),
            cut: self.cut.clone(),
            cost: self.cost.clone(),
            model_tokens: self.model_tokens.clone(),
            over_budget: self.over_budget.clone(),
            prompt_overhead: self.prompt_overhead,
        })
    }
}

impl ProjectableLayer for Levels {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        Ok(Levels {
            spans: doc.project_spans_monotonic(&self.spans, to)?,
            unit: self.unit.clone(),
            parent: self.parent.clone(),
        })
    }
}

impl ProjectableLayer for Entities {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        Ok(Entities {
            spans: doc.project_spans_monotonic(&self.spans, to)?,
            tokens: self.tokens.clone(),
            kind: self.kind.clone(),
            kinds: Arc::clone(&self.kinds),
            score: self.score.clone(),
        })
    }
}

impl ProjectableLayer for PhraseChunks {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        Ok(PhraseChunks {
            spans: doc.project_spans_monotonic(&self.spans, to)?,
            tokens: self.tokens.clone(),
            kind: self.kind.clone(),
            kinds: Arc::clone(&self.kinds),
        })
    }
}

impl ProjectableLayer for Phrases {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        Ok(Phrases {
            spans: doc.project_spans_monotonic(&self.spans, to)?,
            tokens: self.tokens.clone(),
            kind: self.kind.clone(),
            kinds: Arc::clone(&self.kinds),
            head: self.head.clone(),
            parent: self.parent.clone(),
        })
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
        Ok(Numbers {
            spans,
            numeral_start,
            numeral_end,
            negative: self.negative.clone(),
            coefficient: self.coefficient.clone(),
            scale: self.scale.clone(),
            multiplier: self.multiplier.clone(),
            marker: self.marker.clone(),
            notation: self.notation.clone(),
        })
    }
}

impl ProjectableLayer for Money {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        Ok(Money {
            spans: doc.project_spans_monotonic(&self.spans, to)?,
            negative: self.negative.clone(),
            coefficient: self.coefficient.clone(),
            scale: self.scale.clone(),
            currency: self.currency.clone(),
        })
    }
}

impl ProjectableLayer for Quantities {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        Ok(Quantities {
            spans: doc.project_spans_monotonic(&self.spans, to)?,
            negative: self.negative.clone(),
            coefficient: self.coefficient.clone(),
            scale: self.scale.clone(),
            unit: self.unit.clone(),
            normalized_negative: self.normalized_negative.clone(),
            normalized_coefficient: self.normalized_coefficient.clone(),
            normalized_scale: self.normalized_scale.clone(),
            base_unit: self.base_unit.clone(),
        })
    }
}

impl ProjectableLayer for Times {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        Ok(Times {
            spans: doc.project_spans_monotonic(&self.spans, to)?,
            kind: self.kind.clone(),
            grain: self.grain.clone(),
            modifier: self.modifier.clone(),
            start: self.start.clone(),
            end: self.end.clone(),
            duration_ms: self.duration_ms.clone(),
            ambiguous: self.ambiguous.clone(),
            timex: self.timex.clone(),
        })
    }
}

impl ProjectableLayer for Temporals {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        Ok(Temporals {
            spans: doc.project_spans_monotonic(&self.spans, to)?,
            granularity: self.granularity.clone(),
            origin: self.origin.clone(),
            year: self.year.clone(),
            period: self.period.clone(),
            day: self.day.clone(),
        })
    }
}

impl ProjectableLayer for Ordinals {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        Ok(Ordinals {
            spans: doc.project_spans_monotonic(&self.spans, to)?,
            base: self.base.clone(),
            offset: self.offset.clone(),
        })
    }
}

impl ProjectableLayer for Pii {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        Ok(Pii {
            spans: doc.project_spans_monotonic(&self.spans, to)?,
            kind: self.kind.clone(),
            confidence: self.confidence.clone(),
            normalized: self.normalized.clone(),
        })
    }
}

impl ProjectableLayer for Quotations {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        let from = self.spans.view();
        Ok(Quotations {
            spans: doc.project_spans_monotonic(&self.spans, to)?,
            speaker: map_opt_spans(doc, from, to, &self.speaker)?,
            cue_verb: map_opt_spans(doc, from, to, &self.cue_verb)?,
            speaker_chain: self.speaker_chain.clone(),
            kind: self.kind.clone(),
            confidence: self.confidence.clone(),
        })
    }
}

impl ProjectableLayer for Matches {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        Ok(Matches {
            spans: doc.project_spans_monotonic(&self.spans, to)?,
            tokens: self.tokens.clone(),
            label: self.label.clone(),
            labels: Arc::clone(&self.labels),
            rule: self.rule.clone(),
        })
    }
}

impl ProjectableLayer for LanguageSpans {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        Ok(LanguageSpans {
            spans: doc.project_spans_monotonic(&self.spans, to)?,
            language: self.language.clone(),
            languages: Arc::clone(&self.languages),
            script: self.script.clone(),
            confidence: self.confidence.clone(),
        })
    }
}

impl ProjectableLayer for Abbreviations {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        Ok(Abbreviations { spans: doc.project_spans_monotonic(&self.spans, to)?, form: self.form.clone() })
    }
}

impl ProjectableLayer for WordNetSenses {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        Ok(WordNetSenses {
            spans: doc.project_spans_monotonic(&self.spans, to)?,
            token: self.token.clone(),
            synset_id: self.synset_id.clone(),
            sense_id: self.sense_id.clone(),
            pos: self.pos.clone(),
            lemma: self.lemma.clone(),
            synset_lemmas: self.synset_lemmas.clone(),
            synset_lemma_row: self.synset_lemma_row.clone(),
            gloss: self.gloss.clone(),
            sense_rank: self.sense_rank.clone(),
            lexicon_id: Arc::clone(&self.lexicon_id),
        })
    }
}

impl ProjectableLayer for LexicalExpansions {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        Ok(LexicalExpansions {
            spans: doc.project_spans_monotonic(&self.spans, to)?,
            token: self.token.clone(),
            term: self.term.clone(),
            relation: self.relation.clone(),
            depth: self.depth.clone(),
            sense_rank: self.sense_rank.clone(),
            weight: self.weight.clone(),
            lexicon_id: Arc::clone(&self.lexicon_id),
            synset_id: self.synset_id.clone(),
        })
    }
}

impl ProjectableLayer for Tables {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        Ok(Tables {
            spans: doc.project_spans_monotonic(&self.spans, to)?,
            cells: self.cells.clone(),
            table_bounds: doc.project_spans_monotonic(&self.table_bounds, to)?,
            rows: doc.project_spans_monotonic(&self.rows, to)?,
            row_table: self.row_table.clone(),
            row_ix: self.row_ix.clone(),
        })
    }
}

impl ProjectableLayer for KeyValuePairs {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        let from = self.spans.view();
        let spans = doc.project_spans_monotonic(&self.spans, to)?;
        let pairs = self
            .pairs
            .iter()
            .map(|p| {
                Ok(KeyValuePair {
                    key: doc.map_span(from, to, p.key)?,
                    value: doc.map_span(from, to, p.value)?,
                    separator: doc.map_span(from, to, p.separator)?,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(KeyValuePairs { spans, pairs })
    }
}

impl ProjectableLayer for ChunkTableHeaders {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        let from = self.spans.view();
        let spans = doc.project_spans_monotonic(&self.spans, to)?;
        let contexts = self
            .contexts
            .iter()
            .map(|c| {
                let cells =
                    c.header_cells.iter().map(|&s| doc.map_span(from, to, s)).collect::<Result<Vec<_>, Error>>()?;
                Ok(ChunkTableContext {
                    chunk_ix: c.chunk_ix,
                    table_id: c.table_id,
                    is_continuation: c.is_continuation,
                    header_cells: cells.into_boxed_slice(),
                    header_prefix: c.header_prefix.clone(),
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(ChunkTableHeaders { spans, contexts })
    }
}

impl ProjectableLayer for Edus {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        Ok(Edus { spans: doc.project_spans_monotonic(&self.spans, to)?, sentence: self.sentence.clone() })
    }
}

impl ProjectableLayer for TextQuality {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        let _ = doc.project_spans_monotonic(&self.spans, to)?;
        let mut spans = Spans::with_capacity(to, 1);
        spans.push(Span::new(0, doc.text(to).len() as u32));
        Ok(TextQuality { spans, metrics: self.metrics })
    }
}

impl ProjectableLayer for SpanQuality {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        Ok(SpanQuality { spans: doc.project_spans_monotonic(&self.spans, to)?, metrics: self.metrics.clone() })
    }
}

impl ProjectableLayer for Events {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        let from = self.spans.view();
        let spans = doc.project_spans_monotonic(&self.spans, to)?;
        let arguments = self
            .arguments
            .iter()
            .map(|a| Ok(EventArgument { span: doc.map_span(from, to, a.span)?, ..*a }))
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(Events {
            spans,
            lemma: self.lemma.clone(),
            sentence: self.sentence.clone(),
            confidence: self.confidence.clone(),
            arg_end: self.arg_end.clone(),
            arguments,
        })
    }
}

impl ProjectableLayer for Modality {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        let from = self.spans.view();
        let spans = doc.project_spans_monotonic(&self.spans, to)?;
        let cues = map_opt_spans(doc, from, to, &self.cue)?;
        let mut out = Modality::new(to);
        for (i, scope) in spans.iter() {
            let k = i.get();
            out.push(
                scope,
                ModalityTag {
                    polarity: self.polarity[k],
                    epistemic: self.epistemic[k],
                    deontic: self.deontic[k],
                    cue: cues[k],
                    confidence: self.confidence[k],
                },
            );
        }
        out.spans = spans;
        Ok(out)
    }
}

impl ProjectableLayer for Containment {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        Ok(Containment {
            spans: doc.project_spans_monotonic(&self.spans, to)?,
            ancestor_id: self.ancestor_id.clone(),
            ancestor_name: self.ancestor_name.clone(),
            ancestor_kind: self.ancestor_kind.clone(),
            chain_end: self.chain_end.clone(),
        })
    }
}

impl ProjectableLayer for TokenVectorWindows {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        Ok(TokenVectorWindows { spans: doc.project_spans_monotonic(&self.spans, to)? })
    }
}

impl ProjectableLayer for Opinions {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        let from = doc.get_key(&self.tokens_key).map_or(ViewId::SOURCE, |t| t.spans.view());
        let _ = doc.lca_view(from, to)?;
        let mut out = self.clone();
        if !self.tokens_key.name.is_empty()
            && doc.get_key(&self.tokens_key).is_some_and(|t| t.spans.view() != to)
            && doc.get::<Tokens>().is_some_and(|t| t.spans.view() == to)
        {
            out.tokens_key = Key::DEFAULT;
        }
        Ok(out)
    }
}

impl ProjectableLayer for Aspects {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        let from = doc
            .get_key(&self.tokens_key)
            .map(|t| t.spans.view())
            .or_else(|| doc.get_key(&self.sentences_key).map(|s| s.spans.view()))
            .unwrap_or(ViewId::SOURCE);
        let _ = doc.lca_view(from, to)?;
        let mut out = self.clone();
        if !self.tokens_key.name.is_empty()
            && doc.get_key(&self.tokens_key).is_some_and(|t| t.spans.view() != to)
            && doc.get::<Tokens>().is_some_and(|t| t.spans.view() == to)
        {
            out.tokens_key = Key::DEFAULT;
        }
        if !self.sentences_key.name.is_empty()
            && doc.get_key(&self.sentences_key).is_some_and(|s| s.spans.view() != to)
            && doc.get::<Sentences>().is_some_and(|s| s.spans.view() == to)
        {
            out.sentences_key = Key::DEFAULT;
        }
        Ok(out)
    }
}

impl ProjectableLayer for Words {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        let from = doc.get_key(&self.tokens_key).map_or(ViewId::SOURCE, |t| t.spans.view());
        let _ = doc.lca_view(from, to)?;
        let subspan = self.subspan.as_ref().map(|subs| map_opt_spans(doc, from, to, subs)).transpose()?;
        let mut tokens_key = self.tokens_key;
        if !tokens_key.name.is_empty()
            && doc.get_key(&tokens_key).is_some_and(|t| t.spans.view() != to)
            && doc.get::<Tokens>().is_some_and(|t| t.spans.view() == to)
        {
            tokens_key = Key::DEFAULT;
        }
        Ok(Words { tokens: self.tokens.clone(), form: self.form.clone(), subspan, tokens_key })
    }
}
