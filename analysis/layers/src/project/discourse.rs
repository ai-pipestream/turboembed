//! [`ProjectableLayer`] implementations for lexical, table, discourse, event,
//! quality, and token-indexed layers in `turbo-analysis-layers`.

use std::sync::Arc;

use turbo_analysis::{Document, Error, ProjectableLayer, Span, Spans, ViewId};

use super::map_opt_spans;
use super::verify::{
    check_invariants, retarget_span_key, verify_chunk_table_headers, verify_events, verify_sentence_ixs,
    verify_token_ixs, verify_token_ranges,
};
use crate::{
    Abbreviations, Aspects, ChunkTableContext, ChunkTableHeaders, Containment, Edus, EventArgument, Events,
    KeyValuePair, KeyValuePairs, LanguageSpans, LexicalExpansions, Matches, Modality, ModalityTag, Opinions,
    SpanQuality, Tables, TextQuality, TokenVectorWindows, WordNetSenses, Words,
};

impl ProjectableLayer for Matches {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        let from = self.spans.view();
        let out = Matches {
            spans: doc.project_spans_monotonic(&self.spans, to)?,
            tokens: self.tokens.clone(),
            label: self.label.clone(),
            labels: Arc::clone(&self.labels),
            rule: self.rule.clone(),
        };
        check_invariants(&out)?;
        verify_token_ranges(doc, from, to, &out.spans, &out.tokens)?;
        Ok(out)
    }
}

impl ProjectableLayer for LanguageSpans {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        let out = LanguageSpans {
            spans: doc.project_spans_monotonic(&self.spans, to)?,
            language: self.language.clone(),
            languages: Arc::clone(&self.languages),
            script: self.script.clone(),
            confidence: self.confidence.clone(),
        };
        check_invariants(&out)?;
        Ok(out)
    }
}

impl ProjectableLayer for Abbreviations {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        let out = Abbreviations { spans: doc.project_spans_monotonic(&self.spans, to)?, form: self.form.clone() };
        check_invariants(&out)?;
        Ok(out)
    }
}

impl ProjectableLayer for WordNetSenses {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        let from = self.spans.view();
        let out = WordNetSenses {
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
        };
        check_invariants(&out)?;
        verify_token_ixs(doc, from, to, &out.spans, &out.token, true)?;
        Ok(out)
    }
}

impl ProjectableLayer for LexicalExpansions {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        let from = self.spans.view();
        let out = LexicalExpansions {
            spans: doc.project_spans_monotonic(&self.spans, to)?,
            token: self.token.clone(),
            term: self.term.clone(),
            relation: self.relation.clone(),
            depth: self.depth.clone(),
            sense_rank: self.sense_rank.clone(),
            weight: self.weight.clone(),
            lexicon_id: Arc::clone(&self.lexicon_id),
            synset_id: self.synset_id.clone(),
        };
        check_invariants(&out)?;
        verify_token_ixs(doc, from, to, &out.spans, &out.token, true)?;
        Ok(out)
    }
}

impl ProjectableLayer for Tables {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        let out = Tables {
            spans: doc.project_spans_monotonic(&self.spans, to)?,
            cells: self.cells.clone(),
            table_bounds: doc.project_spans_monotonic(&self.table_bounds, to)?,
            rows: doc.project_spans_monotonic(&self.rows, to)?,
            row_table: self.row_table.clone(),
            row_ix: self.row_ix.clone(),
        };
        check_invariants(&out)?;
        Ok(out)
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
        let out = KeyValuePairs { spans, pairs };
        check_invariants(&out)?;
        Ok(out)
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
        let out = ChunkTableHeaders { spans, contexts };
        check_invariants(&out)?;
        verify_chunk_table_headers(doc, from, to, &out)?;
        Ok(out)
    }
}

impl ProjectableLayer for Edus {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        let from = self.spans.view();
        let out = Edus { spans: doc.project_spans_monotonic(&self.spans, to)?, sentence: self.sentence.clone() };
        check_invariants(&out)?;
        verify_sentence_ixs(doc, from, to, &out.spans, &out.sentence)?;
        Ok(out)
    }
}

impl ProjectableLayer for TextQuality {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        let _ = doc.project_spans_monotonic(&self.spans, to)?;
        let mut spans = Spans::with_capacity(to, 1);
        spans.push(Span::new(0, doc.text(to).len() as u32));
        let out = TextQuality { spans, metrics: self.metrics };
        check_invariants(&out)?;
        Ok(out)
    }
}

impl ProjectableLayer for SpanQuality {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        let out = SpanQuality { spans: doc.project_spans_monotonic(&self.spans, to)?, metrics: self.metrics.clone() };
        check_invariants(&out)?;
        Ok(out)
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
        let out = Events {
            spans,
            lemma: self.lemma.clone(),
            sentence: self.sentence.clone(),
            confidence: self.confidence.clone(),
            arg_end: self.arg_end.clone(),
            arguments,
        };
        check_invariants(&out)?;
        verify_events(doc, from, to, &out)?;
        Ok(out)
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
        check_invariants(&out)?;
        Ok(out)
    }
}

impl ProjectableLayer for Containment {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        let out = Containment {
            spans: doc.project_spans_monotonic(&self.spans, to)?,
            ancestor_id: self.ancestor_id.clone(),
            ancestor_name: self.ancestor_name.clone(),
            ancestor_kind: self.ancestor_kind.clone(),
            chain_end: self.chain_end.clone(),
        };
        check_invariants(&out)?;
        Ok(out)
    }
}

impl ProjectableLayer for TokenVectorWindows {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        let out = TokenVectorWindows { spans: doc.project_spans_monotonic(&self.spans, to)? };
        check_invariants(&out)?;
        Ok(out)
    }
}

impl ProjectableLayer for Opinions {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        let (_, tokens_key) = retarget_span_key(doc, self.tokens_key, to, |t| &t.spans)?;
        let out = Opinions { tokens_key, ..self.clone() };
        check_invariants(&out)?;
        Ok(out)
    }
}

impl ProjectableLayer for Aspects {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        let mut out = self.clone();
        if self.tokens.iter().any(Option::is_some) || !self.tokens_key.name.is_empty() {
            out.tokens_key = retarget_span_key(doc, self.tokens_key, to, |t| &t.spans)?.1;
        }
        if self.sentence.iter().any(Option::is_some) || !self.sentences_key.name.is_empty() {
            out.sentences_key = retarget_span_key(doc, self.sentences_key, to, |s| &s.spans)?.1;
        }
        check_invariants(&out)?;
        Ok(out)
    }
}

impl ProjectableLayer for Words {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        let (from, tokens_key) = retarget_span_key(doc, self.tokens_key, to, |t| &t.spans)?;
        let subspan = self.subspan.as_ref().map(|subs| map_opt_spans(doc, from, to, subs)).transpose()?;
        let out = Words { tokens: self.tokens.clone(), form: self.form.clone(), subspan, tokens_key };
        check_invariants(&out)?;
        Ok(out)
    }
}
