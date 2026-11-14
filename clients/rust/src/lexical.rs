//! In-process zero-RPC lexical analyzer with composed `View` alignments,
//! position-length token lattices (`#2340`), and bidirectional UTF-8, UTF-16,
//! and Unicode code-point span projection (`#2338`).

use std::sync::Arc;

use turbo_analysis::{
    Alignment, Annotator, CodePointOffset, Ix, Layer, OffsetTable, Span, Spans, Utf16Offset, View, ViewId,
};
use turbo_analysis_layers::{Sentences, Stems, Strs, TokenArcKind, TokenGraph, TokenGraphBuilder, Tokens};
use turbo_matcher::{Input as MatcherInput, PhraseSynonymExpander};
use turbo_normalize::{
    AccentFold, CaseFold, CaseFoldMode, Chain, Dehyphenate, HtmlEntityDecodeNormalizer, HtmlTagStripNormalizer,
    Invisible, Mojibake, UnicodeForm, Whitespace, WhitespaceMode, emit_normalize_lattice, normalize,
};
use turbo_sentdetect::RuleSentenceDetector;
use turbo_stem::{PorterStemmer, SnowballStemmer, Stemmer};
use turbo_tokenize::{SimpleTokenizer, TokenizerOptions, WhitespaceTokenizer, WordTokenizer};
use turbo_unicode::category::{GeneralCategory, general_category};

use crate::error::{ClientError, Result};
use crate::fingerprint::{
    AnalysisFingerprint, FieldAnalysisConfig, NormalizerKind, StemmerKind, TermSourceKind, TokenizerKind,
};

/// Half-open span projected simultaneously onto UTF-8 byte offsets, UTF-16 code
/// units, and Unicode scalar values of `ViewId::SOURCE`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ProjectedSpanSet {
    /// `(start, end)` UTF-8 byte offsets in the original source text.
    pub utf8: (u32, u32),
    /// `(start, end)` UTF-16 code unit offsets in the original source text.
    pub utf16: (u32, u32),
    /// `(start, end)` Unicode code-point offsets in the original source text.
    pub code_point: (u32, u32),
}

impl ProjectedSpanSet {
    pub(crate) fn from_source_span(source: &str, table: &OffsetTable, span: Span) -> Self {
        let s = span.start;
        let e = span.end;
        Self {
            utf8: (s, e),
            utf16: (table.to_utf16(source, s).0, table.to_utf16(source, e).0),
            code_point: (table.to_code_point(source, s).0, table.to_code_point(source, e).0),
        }
    }
}

/// Sentence span projected back onto `ViewId::SOURCE`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct AnalyzedSentence {
    /// Zero-based sentence index within the field.
    pub index: u32,
    /// Original-text UTF-8, UTF-16, and code-point offsets.
    pub spans: ProjectedSpanSet,
}

/// One arc in the analyzed field's position-length lexical lattice (`#2340`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AnalyzedTokenArc {
    /// Normalized/stemmed term text ready for posting insertion or query lookup.
    pub term: String,
    /// Zero-based start position in the token lattice.
    pub pos: u32,
    /// Number of positions spanned (`>= 1`; `> 1` for multi-token compounds/synonyms).
    pub pos_len: u32,
    /// Original-text UTF-8, UTF-16, and code-point offsets in `ViewId::SOURCE`.
    pub spans: ProjectedSpanSet,
    /// Sentence ordinal enclosing this occurrence (`0..sentences.len()`).
    pub sentence_index: u32,
    /// Origin classification of this lattice arc.
    pub kind: TokenArcKind,
}

/// Complete in-process lexical analysis output for one field value.
#[derive(Clone, Debug)]
pub struct AnalyzedField {
    /// UTF-8 byte length of the original input text.
    pub source_utf8_len: u32,
    /// Normalized child view text.
    pub normalized_text: String,
    /// Composed alignment mapping normalized child bytes <-> `ViewId::SOURCE` bytes.
    pub alignment: Alignment,
    /// Total 1-best token count prior to empty-term filtering (preserves exact BM25 ordinals).
    pub token_stream_len: u32,
    /// Lattice position horizon (`TokenGraph::num_positions`).
    pub num_positions: u32,
    /// Monotonic original-text sentence spans covering every emitted token arc.
    pub sentences: Vec<AnalyzedSentence>,
    /// Position-length lexical lattice arcs projected onto original source coordinates.
    pub arcs: Vec<AnalyzedTokenArc>,
    /// Deterministic cryptographic fingerprint of the configuration that produced this field.
    pub fingerprint: AnalysisFingerprint,
}

impl AnalyzedField {
    /// Converts a UTF-16 span in `ViewId::SOURCE` back to exact UTF-8 byte offsets.
    pub fn utf16_to_source_utf8(&self, source: &str, utf16_start: u32, utf16_end: u32) -> Result<(u32, u32)> {
        let table = OffsetTable::build(source);
        let s = table.from_utf16(source, Utf16Offset(utf16_start))?;
        let e = table.from_utf16(source, Utf16Offset(utf16_end))?;
        if s > e {
            return Err(ClientError::InvalidArgument(format!("inverted UTF-16 span {utf16_start}..{utf16_end}")));
        }
        Ok((s, e))
    }

    /// Converts a code-point span in `ViewId::SOURCE` back to exact UTF-8 byte offsets.
    pub fn code_point_to_source_utf8(&self, source: &str, cp_start: u32, cp_end: u32) -> Result<(u32, u32)> {
        let table = OffsetTable::build(source);
        let s = table.from_code_point(source, CodePointOffset(cp_start))?;
        let e = table.from_code_point(source, CodePointOffset(cp_end))?;
        if s > e {
            return Err(ClientError::InvalidArgument(format!("inverted code-point span {cp_start}..{cp_end}")));
        }
        Ok((s, e))
    }

    /// Projects a UTF-8 byte span from `ViewId::SOURCE` forward into the normalized view.
    pub fn source_to_normalized_span(&self, source_span: Span) -> Span {
        self.alignment.span_to_child(source_span)
    }

    /// Projects a UTF-8 byte span from the normalized view back into `ViewId::SOURCE`.
    pub fn normalized_to_source_span(&self, norm_span: Span) -> Span {
        self.alignment.span_to_parent(norm_span)
    }
}

/// Folded primary field paired with its case-preserving unstemmed twin.
#[derive(Clone, Debug)]
pub struct DualAnalyzedField {
    /// Folded/stemmed primary analysis.
    pub primary: AnalyzedField,
    /// Case-preserving unstemmed twin over the same original source text.
    pub cased: AnalyzedField,
}

fn build_chain(steps: &[NormalizerKind]) -> Chain {
    let mut chain = Chain::new();
    for step in steps {
        chain = match step {
            NormalizerKind::StripInvisible => chain.then(Invisible),
            NormalizerKind::Whitespace => chain.then(Whitespace::new(WhitespaceMode::Collapse)),
            NormalizerKind::FullCaseFold => chain.then(CaseFold::new(CaseFoldMode::Full)),
            NormalizerKind::SimpleCaseFold => chain.then(CaseFold::new(CaseFoldMode::Simple)),
            NormalizerKind::Nfkc => chain.then(UnicodeForm::nfkc()),
            NormalizerKind::Nfc => chain.then(UnicodeForm::nfc()),
            NormalizerKind::AccentFold => chain.then(AccentFold::new()),
            NormalizerKind::Dehyphenate => chain.then(Dehyphenate::new()),
            NormalizerKind::HtmlEntityDecode => chain.then(HtmlEntityDecodeNormalizer::new()),
            NormalizerKind::HtmlTagStrip => chain.then(HtmlTagStripNormalizer::new()),
            NormalizerKind::Mojibake => chain.then(Mojibake),
        };
    }
    chain
}

fn stem_word(stemmer: StemmerKind, word: &str) -> String {
    match stemmer {
        StemmerKind::None => word.to_owned(),
        StemmerKind::Porter => PorterStemmer.stem(word),
        StemmerKind::Snowball(alg) => SnowballStemmer::new(alg).stem(word),
    }
}

/// Thread-safe, zero-RPC lexical analyzer compiled from a [`FieldAnalysisConfig`].
pub struct LexicalAnalyzer {
    config: FieldAnalysisConfig,
    fingerprint: AnalysisFingerprint,
    chain: Chain,
    sent_chain: Chain,
    sent: RuleSentenceDetector,
    synonyms: Option<Arc<PhraseSynonymExpander>>,
}

impl LexicalAnalyzer {
    /// Compiles an in-process [`LexicalAnalyzer`] from `config`.
    pub fn new(config: FieldAnalysisConfig) -> Self {
        let fingerprint = config.fingerprint();
        let chain = build_chain(&config.normalizers);
        let sent_norms: Vec<NormalizerKind> =
            config.normalizers.iter().copied().filter(|&n| n != NormalizerKind::FullCaseFold).collect();
        let sent_chain = build_chain(&sent_norms);
        let sent = RuleSentenceDetector::new(&config.sentence_language);
        Self { config, fingerprint, chain, sent_chain, sent, synonyms: None }
    }

    /// Attaches a multi-token [`PhraseSynonymExpander`] (`turbo-matcher`) to
    /// inject `TokenArcKind::Synonym` lattice arcs during lexical analysis.
    pub fn with_synonyms(mut self, expander: Arc<PhraseSynonymExpander>) -> Self {
        self.synonyms = Some(expander);
        self
    }

    /// Active field configuration.
    pub fn config(&self) -> &FieldAnalysisConfig {
        &self.config
    }

    /// Deterministic cryptographic fingerprint of this analyzer.
    pub fn fingerprint(&self) -> AnalysisFingerprint {
        self.fingerprint
    }

    /// Normalizes a prefix query token through the configured normalizer chain
    /// without applying word stemming.
    pub fn normalize_prefix(&self, prefix: &str) -> String {
        if self.chain.is_empty() {
            return prefix.to_owned();
        }
        let (norm, _) = normalize(&self.chain, prefix);
        norm
    }

    /// Analyzes `source` in-process and projects all sentence and lexical
    /// lattice spans back onto `ViewId::SOURCE`.
    pub fn analyze(&self, source: &str) -> Result<AnalyzedField> {
        let src_len = u32::try_from(source.len())
            .map_err(|_| ClientError::InvalidArgument("source text exceeds u32 byte offset limit".into()))?;
        let (norm_text, alignment) = if self.chain.is_empty() {
            (source.to_owned(), Alignment::identity())
        } else {
            normalize(&self.chain, source)
        };
        let (sent_text, sent_alignment) = if self.sent_chain.is_empty() {
            (source.to_owned(), Alignment::identity())
        } else {
            normalize(&self.sent_chain, source)
        };
        let sent_view = View::derived_root_for_local(&sent_text);
        let (sent_layer,) = self.sent.annotate((&sent_view,))?;
        let offset_table = OffsetTable::build(source);
        let sentences = project_sentences(source, &offset_table, &sent_alignment, &sent_layer, src_len);

        let work_view = View::derived_root_for_local(&norm_text);
        let tokens = self.tokenize_view(&work_view)?;
        let token_stream_len = tokens.len() as u32;
        let stems = self.compute_stems(work_view.text(), &tokens);

        let graph = self.build_lattice(work_view.text(), &alignment, &tokens, stems.as_ref())?;
        let arcs = project_arcs(
            source,
            &offset_table,
            &alignment,
            &graph,
            &sentences,
            self.config.stemmer,
            self.config.term_source,
        );

        Ok(AnalyzedField {
            source_utf8_len: src_len,
            normalized_text: norm_text,
            alignment,
            token_stream_len,
            num_positions: graph.num_positions,
            sentences,
            arcs,
            fingerprint: self.fingerprint,
        })
    }

    /// Analyzes both the configured primary field and its `cased_twin()` in-process.
    pub fn analyze_dual(&self, source: &str) -> Result<DualAnalyzedField> {
        let primary = self.analyze(source)?;
        let cased_analyzer = LexicalAnalyzer::new(self.config.cased_twin());
        let cased = cased_analyzer.analyze(source)?;
        Ok(DualAnalyzedField { primary, cased })
    }

    /// Analyzes a batch of documents in parallel across at most `max_workers`
    /// scoped worker threads, preserving input order deterministically.
    pub fn analyze_batch(&self, texts: &[&str], max_workers: usize) -> Vec<Result<AnalyzedField>> {
        let workers = max_workers.clamp(1, 8).min(texts.len().max(1));
        if workers <= 1 || texts.len() <= 1 {
            return texts.iter().map(|t| self.analyze(t)).collect();
        }
        let chunk_size = texts.len().div_ceil(workers);
        std::thread::scope(|scope| {
            let handles: Vec<_> = texts
                .chunks(chunk_size)
                .map(|slice| scope.spawn(move || slice.iter().map(|t| self.analyze(t)).collect::<Vec<_>>()))
                .collect();
            let mut out = Vec::with_capacity(texts.len());
            for h in handles {
                match h.join() {
                    Ok(part) => out.extend(part),
                    Err(_) => out.push(Err(ClientError::Internal("lexical worker thread panicked".into()))),
                }
            }
            out
        })
    }

    fn tokenize_view(&self, view: &View) -> Result<Tokens> {
        let opts = TokenizerOptions::default();
        match self.config.tokenizer {
            TokenizerKind::Whitespace => Ok(WhitespaceTokenizer::new(opts).annotate((view, None))?.0),
            TokenizerKind::Uax29 => Ok(WordTokenizer::new().annotate((view, None))?.0),
            TokenizerKind::Simple => Ok(SimpleTokenizer::new(opts).annotate((view, None))?.0),
        }
    }

    fn compute_stems(&self, view_text: &str, tokens: &Tokens) -> Option<Stems> {
        if self.config.term_source == TermSourceKind::Tokens {
            return None;
        }
        let mut col = Strs::default();
        let (starts, ends) = tokens.spans.columns();
        for (&s, &e) in starts.iter().zip(ends) {
            let slice = view_text.get(s as usize..e as usize).unwrap_or_default();
            let base = if self.config.term_source == TermSourceKind::NormalizedStems {
                slice.to_lowercase()
            } else {
                slice.to_owned()
            };
            col.push(&stem_word(self.config.stemmer, &base));
        }
        Some(Stems { stem: col })
    }

    fn build_lattice(
        &self,
        view_text: &str,
        alignment: &Alignment,
        tokens: &Tokens,
        stems: Option<&Stems>,
    ) -> Result<TokenGraph> {
        let mut builder = TokenGraphBuilder::from_tokens(view_text, tokens, None).map_err(ClientError::Internal)?;
        if self.config.emit_lattice {
            emit_normalize_lattice(view_text, alignment, tokens, &mut builder);
        }
        if let Some(exp) = &self.synonyms {
            let mut input = MatcherInput::new(view_text, tokens)
                .map_err(|e| ClientError::Internal(format!("synonym input: {e}")))?;
            if let Some(s) = stems {
                input = input.with_stems(&s.stem).map_err(|e| ClientError::Internal(format!("synonym stems: {e}")))?;
            }
            exp.emit(&input, tokens, &mut builder)
                .map_err(|e| ClientError::Internal(format!("synonym lattice: {e}")))?;
        }
        builder.finish().map_err(ClientError::Internal)
    }
}

trait LocalRootView {
    fn derived_root_for_local(text: &str) -> View;
}

impl LocalRootView for View {
    fn derived_root_for_local(text: &str) -> View {
        let doc = turbo_analysis::Document::new(text);
        doc.get::<View>().expect("Document::new always holds ViewId::SOURCE").clone()
    }
}

fn project_sentences(
    source: &str,
    table: &OffsetTable,
    alignment: &Alignment,
    sentences: &Sentences,
    src_len: u32,
) -> Vec<AnalyzedSentence> {
    if src_len == 0 {
        return Vec::new();
    }
    let mut rows = Vec::with_capacity(sentences.len().max(1));
    let mut prev_end = 0u32;
    for i in 0..sentences.len() {
        let mapped =
            extend_stripped_marks(source, alignment, alignment.span_to_parent(sentences.spans.get(Ix::new(i as u32))));
        let start = mapped.start.max(prev_end).min(mapped.end);
        let end = mapped.end.max(start);
        if start < end {
            rows.push(AnalyzedSentence {
                index: rows.len() as u32,
                spans: ProjectedSpanSet::from_source_span(source, table, Span::new(start, end)),
            });
            prev_end = end;
        }
    }
    if rows.is_empty() {
        let _ = Spans::<Sentences>::new(ViewId::SOURCE);
        rows.push(AnalyzedSentence {
            index: 0,
            spans: ProjectedSpanSet::from_source_span(source, table, Span::new(0, src_len)),
        });
    }
    rows
}

fn project_arcs(
    source: &str,
    table: &OffsetTable,
    alignment: &Alignment,
    graph: &TokenGraph,
    sentences: &[AnalyzedSentence],
    stemmer: StemmerKind,
    term_source: TermSourceKind,
) -> Vec<AnalyzedTokenArc> {
    let mut out = Vec::with_capacity(graph.len());
    for (_, arc) in graph.iter() {
        let term = if term_source == TermSourceKind::Tokens {
            arc.term.to_owned()
        } else if term_source == TermSourceKind::NormalizedStems {
            stem_word(stemmer, &arc.term.to_lowercase())
        } else {
            stem_word(stemmer, arc.term)
        };
        if term.is_empty() {
            continue;
        }
        let mapped = extend_stripped_marks(source, alignment, alignment.span_to_parent(arc.span));
        let sent_idx = find_enclosing_sentence(sentences, mapped.start);
        out.push(AnalyzedTokenArc {
            term,
            pos: arc.position,
            pos_len: arc.position_length,
            spans: ProjectedSpanSet::from_source_span(source, table, mapped),
            sentence_index: sent_idx,
            kind: arc.kind,
        });
    }
    out
}

fn extend_stripped_marks(source: &str, alignment: &Alignment, span: Span) -> Span {
    if span.is_empty() {
        return span;
    }
    let mut end = span.end;
    if let Some(tail) = source.get(end as usize..) {
        for ch in tail.chars() {
            let is_mark =
                matches!(general_category(ch), GeneralCategory::Mn | GeneralCategory::Mc | GeneralCategory::Me);
            let next_end = end + ch.len_utf8() as u32;
            if is_mark && alignment.span_to_child(Span::new(end, next_end)).is_empty() {
                end = next_end;
            } else {
                break;
            }
        }
    }
    Span::new(span.start, end)
}

fn find_enclosing_sentence(sentences: &[AnalyzedSentence], byte_start: u32) -> u32 {
    if sentences.is_empty() {
        return 0;
    }
    let idx = sentences.partition_point(|s| s.spans.utf8.1 <= byte_start);
    sentences.get(idx).map_or_else(|| sentences.last().map_or(0, |s| s.index), |s| s.index)
}
