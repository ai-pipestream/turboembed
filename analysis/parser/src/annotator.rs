//! Pipeline annotator and command-line helper for constituency parsing.
//!
//! [`ParserAnnotator`] implements [`turbo_analysis::Annotator`], reading
//! [`View`], [`Sentences`], and [`Tokens`] and writing [`Phrases`]
//! (`opennlp:phrases`) with phrase categories, token ranges, head token indices,
//! and parent phrase links.
//!
//! [`ParserTool`] mirrors `opennlp.tools.cmdline.parser.ParserTool`, spacing
//! untokenized parentheses, splitting on whitespace, running bottom-up parsing,
//! and mapping spans back to the input line's character/byte coordinates.

use std::sync::Arc;

use turbo_analysis::{Annotator, Error, Ix, IxRange, ModelRef, Refs, Span, Spans, View};
use turbo_analysis_layers::{Phrases, Sentences, Tag, TagSet, Tokens};

use crate::bottom_up::{Parser, ParserEngine, ParserFactory};
use crate::model::ParserModel;
use crate::parse::{INC_NODE, Parse, TOK_NODE, TOP_NODE, space_untokenized_parens};

/// Constituency parser annotator (`turbo-parser:constituency`).
///
/// For each sentence in [`Sentences`], collects its covered [`Tokens`], builds
/// a flat [`Parse::from_tokens`] tree, runs [`ParserEngine::parse`], and records
/// every non-POS constituent node (excluding synthetic `TOP`, `INC`, and `TK`
/// nodes) in pre-order into [`Phrases`].
#[derive(Debug)]
pub struct ParserAnnotator {
    model: Arc<ParserModel>,
    engine: ParserEngine,
}

impl ParserAnnotator {
    /// Creates a `ParserAnnotator` with OpenNLP's default beam search parameters
    /// (`beamSize = 20`, `advancePercentage = 0.95`).
    pub fn new(model: impl Into<Arc<ParserModel>>) -> Self {
        let model = model.into();
        let engine = ParserFactory::create(Arc::clone(&model));
        Self { model, engine }
    }

    /// Creates a `ParserAnnotator` with explicit `beam_size` and `advance_percentage`.
    pub fn with_params(model: impl Into<Arc<ParserModel>>, beam_size: usize, advance_percentage: f64) -> Self {
        let model = model.into();
        let engine = ParserFactory::create_with_params(Arc::clone(&model), beam_size, advance_percentage);
        Self { model, engine }
    }

    /// Borrows the underlying [`ParserEngine`].
    pub fn engine(&self) -> &ParserEngine {
        &self.engine
    }

    /// Borrows the underlying [`ParserModel`].
    pub fn parser_model(&self) -> &ParserModel {
        &self.model
    }
}

impl Annotator for ParserAnnotator {
    const ID: &'static str = "turbo-parser:constituency";
    type Reads = (View, Sentences, Tokens);
    type Writes = (Phrases,);

    fn annotate(&self, (view, sentences, tokens): Refs<'_, Self>) -> Result<(Phrases,), Error> {
        if tokens.spans.view() != view.id() {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!("the tokens are in view {}, not in view {}", tokens.spans.view().0, view.id().0),
            });
        }
        let text = view.text();
        let mut spans = Spans::<Phrases>::new(view.id());
        let mut token_ranges = Vec::new();
        let mut raw_kinds: Vec<String> = Vec::new();
        let mut unique_kinds: Vec<String> = Vec::new();
        let mut heads = Vec::new();
        let mut parents = Vec::new();

        let (sent_starts, sent_ends) = sentences.spans.columns();
        let (tok_starts, tok_ends) = tokens.spans.columns();
        let num_toks = tok_starts.len();
        let mut tok_cursor = 0usize;

        for si in 0..sent_starts.len() {
            let s_start = sent_starts[si];
            let s_end = sent_ends[si];
            while tok_cursor < num_toks && tok_ends[tok_cursor] <= s_start {
                tok_cursor += 1;
            }
            let sent_tok_start = tok_cursor;
            while tok_cursor < num_toks && tok_starts[tok_cursor] < s_end {
                tok_cursor += 1;
            }
            let sent_tok_end = tok_cursor;
            if sent_tok_start >= sent_tok_end {
                continue;
            }

            let sent_tok_spans: Vec<Span> =
                (sent_tok_start..sent_tok_end).map(|ti| Span::new(tok_starts[ti], tok_ends[ti])).collect();
            let words: Vec<&str> = sent_tok_spans.iter().map(|sp| &text[sp.range()]).collect();

            let Ok(flat) = Parse::from_tokens(&words) else {
                continue;
            };
            let Some(parsed) = self.engine.parse(&flat) else {
                continue;
            };

            let word_byte_starts: Vec<u32> = flat.children().iter().map(|ch| ch.span().start).collect();

            collect_phrases(
                &parsed,
                None,
                sent_tok_start,
                &sent_tok_spans,
                &word_byte_starts,
                &mut spans,
                &mut token_ranges,
                &mut raw_kinds,
                &mut unique_kinds,
                &mut heads,
                &mut parents,
            );
        }

        let kind_refs: Vec<&str> = unique_kinds.iter().map(String::as_str).collect();
        let kinds = TagSet::new("penn-phrase", &kind_refs);
        let kind: Vec<Tag> = raw_kinds.iter().map(|k| kinds.tag(k).expect("interned phrase kind")).collect();

        Ok((Phrases { spans, tokens: token_ranges, kind, kinds, head: heads, parent: parents },))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.model.model_ref().clone())
    }
}

#[allow(clippy::too_many_arguments)]
fn collect_phrases(
    node: &Parse,
    parent_phrase: Option<Ix<Phrases>>,
    sent_tok_start: usize,
    sent_tok_spans: &[Span],
    word_byte_starts: &[u32],
    spans: &mut Spans<Phrases>,
    token_ranges: &mut Vec<IxRange<Tokens>>,
    raw_kinds: &mut Vec<String>,
    unique_kinds: &mut Vec<String>,
    heads: &mut Vec<Ix<Tokens>>,
    parents: &mut Vec<Option<Ix<Phrases>>>,
) {
    if node.is_pos_tag() || node.children().is_empty() {
        return;
    }

    let node_type = node.kind();
    let is_synthetic = node_type == TOP_NODE || node_type == INC_NODE || node_type == TOK_NODE;

    let current_phrase = if !is_synthetic {
        if let Some((first_tok, end_tok)) = covered_token_range(node.span(), word_byte_starts, sent_tok_spans.len()) {
            let doc_start = sent_tok_spans[first_tok].start;
            let doc_end = sent_tok_spans[end_tok - 1].end;
            let head_local =
                resolve_head_token_index(node, word_byte_starts).unwrap_or(first_tok).clamp(first_tok, end_tok - 1);

            let phrase_ix = spans.push(Span::new(doc_start, doc_end));
            token_ranges.push(IxRange::new((sent_tok_start + first_tok) as u32, (sent_tok_start + end_tok) as u32));
            if !unique_kinds.iter().any(|k| k == node_type) {
                unique_kinds.push(node_type.to_owned());
            }
            raw_kinds.push(node_type.to_owned());
            heads.push(Ix::new((sent_tok_start + head_local) as u32));
            parents.push(parent_phrase);
            Some(phrase_ix)
        } else {
            parent_phrase
        }
    } else {
        parent_phrase
    };

    for child in node.children() {
        collect_phrases(
            child,
            current_phrase,
            sent_tok_start,
            sent_tok_spans,
            word_byte_starts,
            spans,
            token_ranges,
            raw_kinds,
            unique_kinds,
            heads,
            parents,
        );
    }
}

fn covered_token_range(span: Span, word_byte_starts: &[u32], num_tokens: usize) -> Option<(usize, usize)> {
    if num_tokens == 0 || word_byte_starts.is_empty() {
        return None;
    }
    let first = word_byte_starts.partition_point(|&start| start < span.start);
    let end = word_byte_starts.partition_point(|&start| start < span.end);
    if first < end && end <= num_tokens { Some((first, end)) } else { None }
}

fn resolve_head_token_index(node: &Parse, word_byte_starts: &[u32]) -> Option<usize> {
    let head = node.head();
    let idx = head.head_index;
    if idx >= 0 && (idx as usize) < word_byte_starts.len() {
        return Some(idx as usize);
    }
    word_byte_starts.binary_search(&head.span.start).ok()
}

/// Command-line helper mirroring `opennlp.tools.cmdline.parser.ParserTool`.
pub struct ParserTool;

impl ParserTool {
    /// Parses a whitespace-tokenized line of text into up to `num_parses`
    /// constituent trees whose spans refer to byte offsets in `line`.
    ///
    /// Mirrors `ParserTool.parseLine(String line, Parser parser, int numParses)`:
    /// 1. Spaces untokenized parentheses via [`space_untokenized_parens`].
    /// 2. Splits into tokens using Java `StringTokenizer` default delimiters
    ///    (`" \t\n\r\f"`).
    /// 3. Builds a flat `INC` tree with [`TOK_NODE`] children, runs
    ///    `parser.parse_n(&flat, num_parses)`, and adjusts every node's span
    ///    and the shared backing text back to `line`.
    pub fn parse_line(line: &str, parser: &dyn Parser, num_parses: usize) -> Vec<Parse> {
        Self::parse_line_with_mode(line, parser, turbo_unicode::WhitespaceMode::Unicode, num_parses)
    }

    /// Like [`Self::parse_line`], with an explicit [`turbo_unicode::WhitespaceMode`].
    pub fn parse_line_with_mode(
        line: &str,
        parser: &dyn Parser,
        mode: turbo_unicode::WhitespaceMode,
        num_parses: usize,
    ) -> Vec<Parse> {
        if line.trim().is_empty() || num_parses == 0 {
            return Vec::new();
        }
        let spaced = space_untokenized_parens(line, mode);
        let raw_tokens: Vec<&str> = spaced.split([' ', '\t', '\n', '\r', '\x0C']).filter(|s| !s.is_empty()).collect();
        if raw_tokens.is_empty() {
            return Vec::new();
        }

        // Spacing only inserts spaces, so every token is a substring of `line`, in order.
        let mut line_spans = Vec::with_capacity(raw_tokens.len());
        let mut search_from = 0usize;
        for &tok in &raw_tokens {
            let start = search_from + line[search_from..].find(tok).expect("each token is a substring of line");
            let end = start + tok.len();
            line_spans.push(Span::new(start as u32, end as u32));
            search_from = end;
        }

        let Ok(flat) = Parse::from_tokens(&raw_tokens) else {
            return Vec::new();
        };
        let synthetic_starts: Vec<u32> = flat.children().iter().map(|ch| ch.span().start).collect();
        let synthetic_ends: Vec<u32> = flat.children().iter().map(|ch| ch.span().end).collect();

        let shared_line: Arc<str> = Arc::from(line);
        let mut parses = parser.parse_n(&flat, num_parses);
        for p in &mut parses {
            remap_parse_to_line(p, &shared_line, &synthetic_starts, &synthetic_ends, &line_spans);
        }
        parses
    }
}

fn remap_parse_to_line(
    node: &mut Parse,
    shared_line: &Arc<str>,
    synthetic_starts: &[u32],
    synthetic_ends: &[u32],
    line_spans: &[Span],
) {
    let old_span = node.span();
    let new_start = synthetic_starts
        .binary_search(&old_span.start)
        .ok()
        .and_then(|i| line_spans.get(i).map(|s| s.start))
        .unwrap_or_else(|| {
            let idx = synthetic_starts.partition_point(|&s| s < old_span.start);
            line_spans.get(idx).map(|s| s.start).unwrap_or(0)
        });
    let new_end =
        synthetic_ends.binary_search(&old_span.end).ok().and_then(|i| line_spans.get(i).map(|s| s.end)).unwrap_or_else(
            || {
                let idx = synthetic_ends.partition_point(|&e| e < old_span.end);
                line_spans.get(idx).or_else(|| line_spans.last()).map(|s| s.end).unwrap_or(shared_line.len() as u32)
            },
        );

    node.replace_text_and_span(Arc::clone(shared_line), Span::new(new_start, new_end));
    for child in node.children_mut() {
        remap_parse_to_line(child, shared_line, synthetic_starts, synthetic_ends, line_spans);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Returns the flat token tree unchanged, so the test sees only `parse_line`'s remapping.
    struct Echo;

    impl Parser for Echo {
        fn parse_n(&self, tokens: &Parse, _num_parses: usize) -> Vec<Parse> {
            vec![tokens.clone()]
        }
    }

    fn token_spans(line: &str) -> Vec<(u32, u32)> {
        let parses = ParserTool::parse_line(line, &Echo, 1);
        parses[0].children().iter().map(|c| (c.span().start, c.span().end)).collect()
    }

    #[test]
    fn parse_line_maps_each_token_to_its_own_bytes_in_the_line() {
        // A literal `-LRB-` token is its own five bytes, not the next `(` in the line.
        assert_eq!(token_spans("-LRB- x ("), [(0, 5), (6, 7), (8, 9)]);
        assert_eq!(token_spans("a(b) é"), [(0, 1), (1, 2), (2, 3), (3, 4), (5, 7)]);
    }
}
