//! Multi-label incivility and toxicity annotator (`#323`).
//!
//! Scores sentences, retrieval chunks, or structure blocks across the six
//! multi-label dimensions of [`Incivility`] (`turbo:incivility` from `#319`),
//! following the Oasis (ICDMW 2019) and HKBU (ICA 2026) label set:
//! - [`IncivilityLabel::Insult`] (`"insult"`)
//! - [`IncivilityLabel::Profanity`] (`"profanity"`)
//! - [`IncivilityLabel::Threat`] (`"threat"`)
//! - [`IncivilityLabel::IdentityAttack`] (`"identity_attack"`)
//! - [`IncivilityLabel::Intolerance`] (`"intolerance"`)
//! - [`IncivilityLabel::SevereToxicity`] (`"severe_toxicity"`)
//!
//! [`IncivilityAnnotator`] (`ID = "turbo-doccat:incivility"`) combines:
//! 1. **Lexicon and phrase matching (`IncivilityLexicon`)**: deterministic,
//!    zero-regex token and multi-token phrase matching over `opennlp:tokens`
//!    that assigns per-label weights and records the exact [`IxRange<Tokens>`]
//!    evidence span for every matched expression.
//! 2. **Multi-label embedding head (`IncivilityVectorScorer`)**: scores
//!    `Vectors<Sentences>` or `Vectors<Chunks>` with a multi-label sigmoid
//!    classifier head or prototype centroid scorer from `#321`. When the
//!    embedding head predicts a positive label (`score >= threshold`) that had
//!    no direct lexicon match in that unit, the annotator attributes evidence
//!    to the unit's highest-salience content token span so every positive
//!    label carries at least one non-empty [`IxRange<Tokens>`] evidence span.
//! 3. **Optional reply-context conflict scoring (`with_reply_context`)**:
//!    following Opinion Conflicts (Zhang et al., 2018, arXiv:1809.00289),
//!    when `use_reply_context` is enabled and `turbo:structure` ([`Structure`])
//!    is present with a parent block or a preceding [`Block::Quote`] block,
//!    scores antagonistic second-person reply patterns against the quoted or
//!    parent block. [`ModelRef::id`] ends in `:reply-context` or
//!    `:no-reply-context` after the setting; it does not say whether a
//!    [`Structure`] layer was present on the document.

use std::collections::BTreeMap;
use std::sync::Arc;

use turbo_analysis::{Annotator, Error, Ix, IxRange, Layer, ModelRef, Refs, View};
use turbo_analysis_layers::{
    Block, Chunks, Incivility, IncivilityItem, IncivilityLabel, IncivilityScores, IncivilityTarget, Sentences,
    Structure, Tokens, Vectors,
};
use turbo_maxent::sha256;

/// Default score threshold (`0.5`) at or above which an [`IncivilityLabel`] is
/// considered positive and must carry at least one [`IxRange<Tokens>`]
/// evidence span.
pub const DEFAULT_INCIVILITY_THRESHOLD: f64 = 0.5;

/// One `(label, token_range)` evidence span attributing an [`IncivilityLabel`]
/// to a contiguous token slice.
pub type IncivilityEvidenceSpan = (IncivilityLabel, IxRange<Tokens>);

/// A multi-label embedding head that scores a sentence or chunk vector
/// (`&[f32]`) into [`IncivilityScores`] (`0.0..=1.0` per label). Neither
/// `ClassifierHead` nor `LabelCentroids` in `turbo-analysis-embed` implements
/// it; a wrapper calls their `score_incivility_vector` (see the adapters in
/// `analysis/embed/tests/heads.rs`). When [`IncivilityVectorScorer::model`]
/// names an encoder, the annotator refuses vectors from any other encoder.
pub trait IncivilityVectorScorer: Send + Sync {
    /// Scores a single embedding vector `vec` (`&[f32]`) and returns its six
    /// [`IncivilityScores`] values in `0.0..=1.0`.
    fn score_incivility(&self, vec: &[f32]) -> Result<IncivilityScores, String>;

    /// Optional [`ModelRef`] describing the underlying embedding head.
    fn model(&self) -> Option<ModelRef> {
        None
    }

    /// Deterministic configuration hash for provenance tracking.
    fn config_hash(&self) -> u64 {
        0
    }
}

/// Which unit span layer [`IncivilityAnnotator`] scores.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum IncivilityUnit {
    /// Score each sentence in `opennlp:sentences` (`IncivilityTarget::Sentence`).
    #[default]
    Sentences = 0,
    /// Score each retrieval chunk in `turbo:chunks` (`IncivilityTarget::Chunk`).
    Chunks = 1,
    /// Score each structure block in `turbo:structure` (`IncivilityTarget::Block`).
    Blocks = 2,
}

/// Deterministic phrase lexicon for [`IncivilityAnnotator`], with a canonical
/// SHA-256 content digest for [`ModelRef::sha256`].
#[derive(Clone, Debug)]
pub struct IncivilityLexicon {
    /// Canonical map from `(tokens, label)` to weight in `0.0..=1.0`.
    entries: BTreeMap<(Vec<String>, IncivilityLabel), f64>,
    sha256: String,
}

impl Default for IncivilityLexicon {
    fn default() -> Self {
        Self::bundled()
    }
}

impl IncivilityLexicon {
    /// Builds an empty [`IncivilityLexicon`].
    pub fn empty() -> Self {
        let mut lex = IncivilityLexicon { entries: BTreeMap::new(), sha256: String::new() };
        lex.recompute_digest();
        lex
    }

    /// Loads the default English incivility phrase table covering the six
    /// Oasis / HKBU dimensions.
    pub fn bundled() -> Self {
        let mut entries = BTreeMap::new();
        for &(phrase, label, weight) in DEFAULT_INCIVILITY_PHRASES {
            let toks: Vec<String> = phrase.split_whitespace().map(|s| s.to_ascii_lowercase()).collect();
            if !toks.is_empty() {
                entries.insert((toks, label), weight.clamp(0.0, 1.0));
            }
        }
        let mut lex = IncivilityLexicon { entries, sha256: String::new() };
        lex.recompute_digest();
        lex
    }

    /// Inserts or updates a phrase rule (`phrase` split on ASCII whitespace and
    /// lowercased) for `label` with `weight` clamped to `0.0..=1.0`, and
    /// recomputes the canonical SHA-256 digest.
    pub fn insert(&mut self, phrase: &str, label: IncivilityLabel, weight: f64) {
        let toks: Vec<String> = phrase
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'').to_ascii_lowercase())
            .filter(|w| !w.is_empty())
            .collect();
        if !toks.is_empty() {
            self.entries.insert((toks, label), weight.clamp(0.0, 1.0));
            self.recompute_digest();
        }
    }

    /// Number of `(phrase, label)` entries in the lexicon.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the lexicon has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Canonical SHA-256 hex digest of the lexicon's entries.
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    fn recompute_digest(&mut self) {
        let mut canonical = String::new();
        for ((toks, label), &weight) in &self.entries {
            canonical.push_str(&toks.join(" "));
            canonical.push('\t');
            canonical.push_str(label.name());
            canonical.push('\t');
            canonical.push_str(&format!("{weight:.6}\n"));
        }
        self.sha256 = sha256(canonical.as_bytes()).to_hex();
    }
}

/// Multi-label incivility and toxicity annotator (`turbo-doccat:incivility`).
///
/// Reads `(View, Tokens, Option<Sentences>, Option<Chunks>, Option<Structure>, Option<Vectors<Sentences>>, Option<Vectors<Chunks>>)`
/// and writes [`Incivility`] (`turbo:incivility`).
#[derive(Clone)]
pub struct IncivilityAnnotator {
    /// Phrase lexicon providing per-label scores and exact token-range
    /// evidence spans.
    pub lexicon: Arc<IncivilityLexicon>,
    /// Which unit layer to score (`Sentences`, `Chunks`, or `Blocks`).
    pub unit: IncivilityUnit,
    /// Optional multi-label embedding head over `Vectors<Sentences>` or
    /// `Vectors<Chunks>`.
    pub vector_scorer: Option<Arc<dyn IncivilityVectorScorer>>,
    /// Score threshold (`0.5` by default) at or above which a label is
    /// considered positive and guaranteed to carry at least one evidence span.
    pub threshold: f64,
    /// Whether to score reply-context conflicts against parent or preceding
    /// quote blocks when `turbo:structure` is present on the document.
    pub use_reply_context: bool,
}

impl Default for IncivilityAnnotator {
    fn default() -> Self {
        Self::new()
    }
}

impl IncivilityAnnotator {
    /// Creates an [`IncivilityAnnotator`] scoring `Sentences` with the bundled
    /// lexicon, `threshold = 0.5`, and `use_reply_context = false`.
    pub fn new() -> Self {
        IncivilityAnnotator {
            lexicon: Arc::new(IncivilityLexicon::bundled()),
            unit: IncivilityUnit::Sentences,
            vector_scorer: None,
            threshold: DEFAULT_INCIVILITY_THRESHOLD,
            use_reply_context: false,
        }
    }

    /// Sets the [`IncivilityLexicon`].
    pub fn with_lexicon(mut self, lexicon: Arc<IncivilityLexicon>) -> Self {
        self.lexicon = lexicon;
        self
    }

    /// Sets the target unit layer (`Sentences`, `Chunks`, or `Blocks`).
    pub fn with_unit(mut self, unit: IncivilityUnit) -> Self {
        self.unit = unit;
        self
    }

    /// Attaches a multi-label [`IncivilityVectorScorer`] (such as a
    /// `ClassifierHead` or `LabelCentroids` head from `turbo-analysis-embed`).
    pub fn with_vector_scorer(mut self, scorer: Arc<dyn IncivilityVectorScorer>) -> Self {
        self.vector_scorer = Some(scorer);
        self
    }

    /// Sets the positive label threshold (clamped to `0.05..=0.99`).
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold.clamp(0.05, 0.99);
        self
    }

    /// Enables or disables reply-context conflict scoring over `turbo:structure`
    /// (`use_reply_context = false` by default, recorded in provenance).
    pub fn with_reply_context(mut self, enabled: bool) -> Self {
        self.use_reply_context = enabled;
        self
    }

    /// Scores a single unit spanning `tok_start..tok_end` in `tokens`,
    /// combining lexicon matches, optional reply-context conflict against
    /// `parent_tok_range`, and optional embedding vector `unit_vec`.
    pub fn score_unit(
        &self,
        token_lower: &[String],
        tok_start: usize,
        tok_end: usize,
        parent_tok_range: Option<(usize, usize)>,
        unit_vec: Option<&[f32]>,
    ) -> Result<(IncivilityScores, Vec<IncivilityEvidenceSpan>), String> {
        let mut scores = IncivilityScores::default();
        let mut evidence: Vec<IncivilityEvidenceSpan> = Vec::new();

        if tok_start < tok_end {
            // 1. Scan lexicon phrases inside `tok_start..tok_end`.
            let (lex_scores, lex_ev) = scan_lexicon(&self.lexicon, token_lower, tok_start, tok_end);
            for (lbl, val) in lex_scores.iter() {
                if val > scores.get(lbl) {
                    scores.set(lbl, val);
                }
            }
            evidence.extend(lex_ev);

            // 2. Reply-context conflict scoring (Opinion Conflicts, Zhang et al. 2018)
            // when enabled and a parent/quote token range is present.
            if self.use_reply_context
                && let Some((p_start, p_end)) = parent_tok_range
                && p_start < p_end
            {
                let (ctx_scores, ctx_ev) = score_reply_conflict(token_lower, p_start, p_end, tok_start, tok_end);
                for (lbl, val) in ctx_scores.iter() {
                    if val > 0.0 {
                        let combined = 1.0 - (1.0 - scores.get(lbl)) * (1.0 - val);
                        scores.set(lbl, combined.clamp(0.0, 1.0));
                    }
                }
                evidence.extend(ctx_ev);
            }
        }

        // 3. Score embedding vector when present and `self.vector_scorer` is configured.
        if let (Some(scorer), Some(vec)) = (&self.vector_scorer, unit_vec) {
            let head_scores = scorer.score_incivility(vec)?;
            for (lbl, h_val) in head_scores.iter() {
                let l_val = scores.get(lbl);
                // If both lexicon and head fire, combine probabilistically; otherwise take the stronger signal.
                let merged = if l_val >= 0.25 && h_val >= 0.25 {
                    (1.0 - (1.0 - l_val) * (1.0 - h_val)).clamp(0.0, 1.0)
                } else {
                    l_val.max(h_val).clamp(0.0, 1.0)
                };
                scores.set(lbl, merged);
            }
        }

        // 4. Co-occurrence rule for SevereToxicity: when Threat co-occurs with
        // Insult, Profanity, IdentityAttack, or Intolerance at >= threshold,
        // elevate SevereToxicity and merge their evidence spans.
        let max_other = scores.insult.max(scores.profanity).max(scores.identity_attack).max(scores.intolerance);
        if scores.threat >= self.threshold && max_other >= self.threshold {
            let sev = (0.55 * scores.threat + 0.45 * max_other).clamp(self.threshold, 0.99);
            if sev > scores.severe_toxicity {
                scores.severe_toxicity = sev;
            }
            let supporting: Vec<IxRange<Tokens>> = evidence
                .iter()
                .filter(|(l, _)| {
                    matches!(
                        l,
                        IncivilityLabel::Threat
                            | IncivilityLabel::Insult
                            | IncivilityLabel::Profanity
                            | IncivilityLabel::IdentityAttack
                            | IncivilityLabel::Intolerance
                    )
                })
                .map(|(_, r)| *r)
                .collect();
            for r in supporting {
                evidence.push((IncivilityLabel::SevereToxicity, r));
            }
        }

        // 5. Guarantee that every positive label (`score >= self.threshold`) has at least
        // one non-empty `IxRange<Tokens>` evidence span inside `tok_start..tok_end`.
        if tok_start < tok_end {
            let fallback_span = select_salient_token_span(token_lower, tok_start, tok_end);
            for (lbl, val) in scores.iter() {
                if val >= self.threshold && !evidence.iter().any(|(l, _)| *l == lbl) {
                    evidence.push((lbl, fallback_span));
                }
            }
        }

        evidence.sort_unstable_by_key(|&(l, r)| (l, r.start, r.end));
        evidence.dedup();
        Ok((scores, evidence))
    }
}

impl Annotator for IncivilityAnnotator {
    const ID: &'static str = "turbo-doccat:incivility";
    type Reads = (
        View,
        Tokens,
        Option<Sentences>,
        Option<Chunks>,
        Option<Structure>,
        Option<Vectors<Sentences>>,
        Option<Vectors<Chunks>>,
    );
    type Writes = (Incivility,);

    fn annotate(
        &self,
        (view, tokens, sentences, chunks, structure, sent_vecs, chunk_vecs): Refs<'_, Self>,
    ) -> Result<Self::Writes, Error> {
        let fail = |detail: String| Error::Annotator { annotator: Self::ID, detail };
        if tokens.spans.view() != view.id() {
            return Err(fail("tokens are in another view".into()));
        }
        if sentences.is_some_and(|s| s.spans.view() != view.id()) {
            return Err(fail("sentences are in another view".into()));
        }
        if chunks.is_some_and(|c| c.spans.view() != view.id()) {
            return Err(fail("chunks are in another view".into()));
        }
        if structure.is_some_and(|st| st.spans.view() != view.id()) {
            return Err(fail("structure is in another view".into()));
        }
        let text = view.text();
        let n_tokens = tokens.len();
        let (tok_starts, tok_ends) = tokens.spans.columns();

        let mut token_lower: Vec<String> = Vec::with_capacity(n_tokens);
        for t in 0..n_tokens {
            let slice = text.get(tok_starts[t] as usize..tok_ends[t] as usize).unwrap_or("");
            token_lower.push(slice.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'').to_ascii_lowercase());
        }

        let scorer_model = self.vector_scorer.as_ref().and_then(|s| s.model());
        let check_encoder = |vectors_model: &ModelRef| -> Result<(), Error> {
            if let Some(m) = &scorer_model
                && *vectors_model != *m
            {
                return Err(fail(format!(
                    "vectors are from model {} (revision {}), the scorer is for {} (revision {})",
                    vectors_model.id, vectors_model.revision, m.id, m.revision
                )));
            }
            Ok(())
        };
        if let Some(v) = sent_vecs
            && self.unit == IncivilityUnit::Sentences
        {
            check_encoder(&v.model)?;
        }
        if let Some(v) = chunk_vecs
            && self.unit == IncivilityUnit::Chunks
        {
            check_encoder(&v.model)?;
        }

        let mut out = Incivility::new();

        match self.unit {
            IncivilityUnit::Sentences => {
                let sents = sentences.ok_or_else(|| {
                    fail("IncivilityUnit::Sentences requires a Sentences layer on the document".into())
                })?;
                let (s_starts, s_ends) = sents.spans.columns();
                let mut tok_cursor = 0usize;
                for s_idx in 0..sents.len() {
                    let s_start_byte = s_starts[s_idx];
                    let s_end_byte = s_ends[s_idx];
                    while tok_cursor < n_tokens && tok_ends[tok_cursor] <= s_start_byte {
                        tok_cursor += 1;
                    }
                    let tok_start = tok_cursor;
                    while tok_cursor < n_tokens && tok_starts[tok_cursor] < s_end_byte {
                        tok_cursor += 1;
                    }
                    let tok_end = tok_cursor;

                    let parent_range = if self.use_reply_context {
                        structure.and_then(|st| {
                            find_parent_or_quote_token_range(st, s_start_byte, s_end_byte, tok_starts, tok_ends)
                        })
                    } else {
                        None
                    };

                    let sent_ix = Ix::<Sentences>::new(s_idx as u32);
                    let u_vec = sent_vecs.and_then(|v| {
                        if s_idx < v.len() && !v.is_empty(sent_ix) { Some(v.row(sent_ix)) } else { None }
                    });

                    let (scores, evidence) =
                        self.score_unit(&token_lower, tok_start, tok_end, parent_range, u_vec).map_err(fail)?;
                    out.try_push(
                        tokens,
                        Some(sents),
                        chunks,
                        structure,
                        IncivilityItem { target: IncivilityTarget::Sentence(sent_ix), scores, evidence: &evidence },
                    )
                    .map_err(fail)?;
                }
            }
            IncivilityUnit::Chunks => {
                let ch = chunks
                    .ok_or_else(|| fail("IncivilityUnit::Chunks requires a Chunks layer on the document".into()))?;
                let (c_starts, c_ends) = ch.spans.columns();
                let mut tok_cursor = 0usize;
                for c_idx in 0..ch.len() {
                    let c_start_byte = c_starts[c_idx];
                    let c_end_byte = c_ends[c_idx];
                    while tok_cursor < n_tokens && tok_ends[tok_cursor] <= c_start_byte {
                        tok_cursor += 1;
                    }
                    let tok_start = tok_cursor;
                    while tok_cursor < n_tokens && tok_starts[tok_cursor] < c_end_byte {
                        tok_cursor += 1;
                    }
                    let tok_end = tok_cursor;

                    let parent_range = if self.use_reply_context {
                        structure.and_then(|st| {
                            find_parent_or_quote_token_range(st, c_start_byte, c_end_byte, tok_starts, tok_ends)
                        })
                    } else {
                        None
                    };

                    let chunk_ix = Ix::<Chunks>::new(c_idx as u32);
                    let u_vec = chunk_vecs.and_then(|v| {
                        if c_idx < v.len() && !v.is_empty(chunk_ix) { Some(v.row(chunk_ix)) } else { None }
                    });

                    let (scores, evidence) =
                        self.score_unit(&token_lower, tok_start, tok_end, parent_range, u_vec).map_err(fail)?;
                    out.try_push(
                        tokens,
                        sentences,
                        Some(ch),
                        structure,
                        IncivilityItem { target: IncivilityTarget::Chunk(chunk_ix), scores, evidence: &evidence },
                    )
                    .map_err(fail)?;
                }
            }
            IncivilityUnit::Blocks => {
                let st = structure
                    .ok_or_else(|| fail("IncivilityUnit::Blocks requires a Structure layer on the document".into()))?;
                let (b_starts, b_ends) = st.spans.columns();
                for b_idx in 0..st.len() {
                    let b_start_byte = b_starts[b_idx];
                    let b_end_byte = b_ends[b_idx];
                    let tok_start = tok_ends.partition_point(|&e| e <= b_start_byte);
                    let tok_end = tok_starts.partition_point(|&s| s < b_end_byte);

                    let parent_range = if self.use_reply_context {
                        find_block_parent_or_quote_token_range(st, b_idx, tok_starts, tok_ends)
                    } else {
                        None
                    };

                    let block_ix = Ix::<Structure>::new(b_idx as u32);
                    let (scores, evidence) =
                        self.score_unit(&token_lower, tok_start, tok_end, parent_range, None).map_err(fail)?;
                    out.try_push(
                        tokens,
                        sentences,
                        chunks,
                        Some(st),
                        IncivilityItem { target: IncivilityTarget::Block(block_ix), scores, evidence: &evidence },
                    )
                    .map_err(fail)?;
                }
            }
        }

        Ok((out,))
    }

    fn model(&self) -> Option<ModelRef> {
        let ctx_suffix = if self.use_reply_context { "reply-context" } else { "no-reply-context" };
        if let Some(scorer) = &self.vector_scorer
            && let Some(m) = scorer.model()
        {
            return Some(ModelRef {
                id: format!("{}+lexicon:{ctx_suffix}", m.id),
                revision: m.revision,
                sha256: m.sha256,
            });
        }
        Some(ModelRef {
            id: format!("oasis-incivility:lexicon:{ctx_suffix}"),
            revision: "1".into(),
            sha256: self.lexicon.sha256().to_owned(),
        })
    }

    fn config_hash(&self) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for &b in self.lexicon.sha256().as_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h ^= (self.unit as u64).wrapping_mul(0x9e3779b97f4a7c15);
        h ^= self.threshold.to_bits();
        h ^= u64::from(self.use_reply_context);
        if let Some(s) = &self.vector_scorer {
            h ^= s.config_hash();
        }
        h
    }
}

/// Default English phrase rules covering the six Oasis / HKBU incivility
/// dimensions (`insult`, `profanity`, `threat`, `identity_attack`,
/// `intolerance`, `severe_toxicity`).
const DEFAULT_INCIVILITY_PHRASES: &[(&str, IncivilityLabel, f64)] = &[
    // Insult
    ("idiot", IncivilityLabel::Insult, 0.85),
    ("idiots", IncivilityLabel::Insult, 0.85),
    ("moron", IncivilityLabel::Insult, 0.88),
    ("morons", IncivilityLabel::Insult, 0.88),
    ("imbecile", IncivilityLabel::Insult, 0.88),
    ("stupid", IncivilityLabel::Insult, 0.78),
    ("dumb", IncivilityLabel::Insult, 0.75),
    ("pathetic", IncivilityLabel::Insult, 0.76),
    ("loser", IncivilityLabel::Insult, 0.80),
    ("losers", IncivilityLabel::Insult, 0.80),
    ("clown", IncivilityLabel::Insult, 0.75),
    ("clowns", IncivilityLabel::Insult, 0.75),
    ("scumbag", IncivilityLabel::Insult, 0.90),
    ("scumbags", IncivilityLabel::Insult, 0.90),
    ("worthless", IncivilityLabel::Insult, 0.82),
    ("incompetent", IncivilityLabel::Insult, 0.74),
    ("brainless", IncivilityLabel::Insult, 0.84),
    ("ignorant fool", IncivilityLabel::Insult, 0.86),
    ("utterly clueless", IncivilityLabel::Insult, 0.78),
    ("complete fraud", IncivilityLabel::Insult, 0.82),
    ("total fraud", IncivilityLabel::Insult, 0.82),
    ("you are a joke", IncivilityLabel::Insult, 0.80),
    ("shut up", IncivilityLabel::Insult, 0.72),
    ("jerk", IncivilityLabel::Insult, 0.76),
    ("coward", IncivilityLabel::Insult, 0.76),
    ("parasite", IncivilityLabel::Insult, 0.86),
    // Profanity
    ("damn", IncivilityLabel::Profanity, 0.68),
    ("damned", IncivilityLabel::Profanity, 0.68),
    ("hell", IncivilityLabel::Profanity, 0.62),
    ("crap", IncivilityLabel::Profanity, 0.66),
    ("crappy", IncivilityLabel::Profanity, 0.65),
    ("shit", IncivilityLabel::Profanity, 0.90),
    ("shitty", IncivilityLabel::Profanity, 0.88),
    ("bullshit", IncivilityLabel::Profanity, 0.92),
    ("fuck", IncivilityLabel::Profanity, 0.95),
    ("fucking", IncivilityLabel::Profanity, 0.95),
    ("fucked", IncivilityLabel::Profanity, 0.94),
    ("asshole", IncivilityLabel::Profanity, 0.92),
    ("assholes", IncivilityLabel::Profanity, 0.92),
    ("bastard", IncivilityLabel::Profanity, 0.86),
    ("bastards", IncivilityLabel::Profanity, 0.86),
    ("bitch", IncivilityLabel::Profanity, 0.90),
    ("wtf", IncivilityLabel::Profanity, 0.78),
    // Threat
    ("kill you", IncivilityLabel::Threat, 0.96),
    ("hurt you", IncivilityLabel::Threat, 0.90),
    ("destroy you", IncivilityLabel::Threat, 0.88),
    ("beat you up", IncivilityLabel::Threat, 0.92),
    ("beat you", IncivilityLabel::Threat, 0.88),
    ("hunt you down", IncivilityLabel::Threat, 0.95),
    ("make you pay", IncivilityLabel::Threat, 0.82),
    ("watch your back", IncivilityLabel::Threat, 0.86),
    ("you will regret this", IncivilityLabel::Threat, 0.80),
    ("burn it down", IncivilityLabel::Threat, 0.88),
    ("burn your house", IncivilityLabel::Threat, 0.96),
    ("punch you", IncivilityLabel::Threat, 0.90),
    ("smash your face", IncivilityLabel::Threat, 0.95),
    ("shoot you", IncivilityLabel::Threat, 0.96),
    ("we will attack", IncivilityLabel::Threat, 0.88),
    // IdentityAttack
    ("subhuman", IncivilityLabel::IdentityAttack, 0.95),
    ("vermin", IncivilityLabel::IdentityAttack, 0.90),
    ("those people are parasites", IncivilityLabel::IdentityAttack, 0.94),
    ("go back to your country", IncivilityLabel::IdentityAttack, 0.92),
    ("inferior race", IncivilityLabel::IdentityAttack, 0.96),
    ("filthy immigrants", IncivilityLabel::IdentityAttack, 0.95),
    ("second class citizens", IncivilityLabel::IdentityAttack, 0.86),
    ("evil religion", IncivilityLabel::IdentityAttack, 0.88),
    ("deport them all", IncivilityLabel::IdentityAttack, 0.86),
    ("these foreigners are criminals", IncivilityLabel::IdentityAttack, 0.92),
    // Intolerance
    ("not welcome here", IncivilityLabel::Intolerance, 0.82),
    ("ban their kind", IncivilityLabel::Intolerance, 0.90),
    ("no place for them", IncivilityLabel::Intolerance, 0.84),
    ("silence all dissenters", IncivilityLabel::Intolerance, 0.82),
    ("zero tolerance for their beliefs", IncivilityLabel::Intolerance, 0.85),
    ("strip their rights", IncivilityLabel::Intolerance, 0.90),
    ("ban them from speaking", IncivilityLabel::Intolerance, 0.84),
    ("purge them from society", IncivilityLabel::Intolerance, 0.92),
    ("lock them all up", IncivilityLabel::Intolerance, 0.84),
    ("exclude their group", IncivilityLabel::Intolerance, 0.82),
    // SevereToxicity
    ("die in a fire", IncivilityLabel::SevereToxicity, 0.96),
    ("rot in hell", IncivilityLabel::SevereToxicity, 0.92),
    ("drop dead", IncivilityLabel::SevereToxicity, 0.90),
    ("you deserve to die", IncivilityLabel::SevereToxicity, 0.97),
    ("wipe them out", IncivilityLabel::SevereToxicity, 0.95),
    ("exterminate them", IncivilityLabel::SevereToxicity, 0.98),
    ("kill yourself", IncivilityLabel::SevereToxicity, 0.98),
];

fn scan_lexicon(
    lexicon: &IncivilityLexicon,
    token_lower: &[String],
    tok_start: usize,
    tok_end: usize,
) -> (IncivilityScores, Vec<(IncivilityLabel, IxRange<Tokens>)>) {
    let mut scores = IncivilityScores::default();
    let mut evidence: Vec<(IncivilityLabel, IxRange<Tokens>)> = Vec::new();

    for ((pat, label), &weight) in &lexicon.entries {
        let pat_len = pat.len();
        if pat_len == 0 || tok_start + pat_len > tok_end {
            continue;
        }
        let mut i = tok_start;
        while i + pat_len <= tok_end {
            let mut matched = true;
            for k in 0..pat_len {
                if token_lower[i + k] != pat[k] {
                    matched = false;
                    break;
                }
            }
            if matched {
                // Guard against negated benign adjectives like `"not stupid"` or `"not dumb"`.
                let negated_mild = pat_len == 1
                    && matches!(pat[0].as_str(), "stupid" | "dumb" | "hell" | "damn")
                    && i > tok_start
                    && matches!(token_lower[i - 1].as_str(), "not" | "never" | "no");
                if !negated_mild {
                    let prev = scores.get(*label);
                    let combined =
                        if prev > 0.0 { (1.0 - (1.0 - prev) * (1.0 - weight)).clamp(0.0, 1.0) } else { weight };
                    scores.set(*label, combined);
                    evidence.push((*label, IxRange::new(i as u32, (i + pat_len) as u32)));
                }
                i += pat_len;
            } else {
                i += 1;
            }
        }
    }

    (scores, evidence)
}

/// Opinion-conflict patterns (Zhang et al., 2018, arXiv:1809.00289) that become
/// uncivil when directed at a parent/quoted post in a reply context.
const REPLY_CONFLICT_PATTERNS: &[(&[&str], IncivilityLabel, f64)] = &[
    (&["you", "have", "no", "clue"], IncivilityLabel::Insult, 0.78),
    (&["you", "know", "nothing"], IncivilityLabel::Insult, 0.76),
    (&["you", "are", "delusional"], IncivilityLabel::Insult, 0.82),
    (&["you", "are", "lying"], IncivilityLabel::Insult, 0.76),
    (&["you", "always", "lie"], IncivilityLabel::Insult, 0.78),
    (&["your", "argument", "is", "garbage"], IncivilityLabel::Insult, 0.80),
    (&["stop", "spreading", "lies"], IncivilityLabel::Insult, 0.75),
    (&["hypocrite"], IncivilityLabel::Insult, 0.76),
    (&["hypocrites"], IncivilityLabel::Insult, 0.76),
    (&["people", "like", "you", "do", "not", "belong"], IncivilityLabel::Intolerance, 0.84),
    (&["your", "kind", "should", "be", "silenced"], IncivilityLabel::Intolerance, 0.86),
];

fn score_reply_conflict(
    token_lower: &[String],
    parent_start: usize,
    parent_end: usize,
    reply_start: usize,
    reply_end: usize,
) -> (IncivilityScores, Vec<(IncivilityLabel, IxRange<Tokens>)>) {
    let mut scores = IncivilityScores::default();
    let mut evidence = Vec::new();
    let parent_has_content = (parent_start..parent_end).any(|t| !token_lower[t].is_empty());
    if !parent_has_content {
        return (scores, evidence);
    }

    for &(pat, label, weight) in REPLY_CONFLICT_PATTERNS {
        let m = pat.len();
        if m == 0 || reply_start + m > reply_end {
            continue;
        }
        for i in reply_start..=(reply_end - m) {
            if (0..m).all(|k| token_lower[i + k] == pat[k]) {
                if weight > scores.get(label) {
                    scores.set(label, weight);
                }
                evidence.push((label, IxRange::new(i as u32, (i + m) as u32)));
            }
        }
    }

    (scores, evidence)
}

fn find_parent_or_quote_token_range(
    structure: &Structure,
    unit_start_byte: u32,
    unit_end_byte: u32,
    tok_starts: &[u32],
    tok_ends: &[u32],
) -> Option<(usize, usize)> {
    let (b_starts, b_ends) = structure.spans.columns();
    // Find the innermost non-Quote block containing `unit_start_byte..unit_end_byte`.
    let mut enclosing_block: Option<usize> = None;
    for b_idx in 0..structure.len() {
        if b_starts[b_idx] <= unit_start_byte && unit_end_byte <= b_ends[b_idx] {
            if structure.kind[b_idx] == Block::Quote {
                // Inside the quote itself, not the reply to the quote.
                return None;
            }
            enclosing_block = Some(b_idx);
        }
    }
    let b_idx = enclosing_block?;
    find_block_parent_or_quote_token_range(structure, b_idx, tok_starts, tok_ends)
}

fn find_block_parent_or_quote_token_range(
    structure: &Structure,
    b_idx: usize,
    tok_starts: &[u32],
    tok_ends: &[u32],
) -> Option<(usize, usize)> {
    if structure.kind[b_idx] == Block::Quote {
        return None;
    }
    let (b_starts, b_ends) = structure.spans.columns();
    // 1. Check if the immediately preceding block is a `Block::Quote`.
    if b_idx > 0 && structure.kind[b_idx - 1] == Block::Quote {
        let q_start = b_starts[b_idx - 1];
        let q_end = b_ends[b_idx - 1];
        let ts = tok_ends.partition_point(|&e| e <= q_start);
        let te = tok_starts.partition_point(|&s| s < q_end);
        if ts < te {
            return Some((ts, te));
        }
    }
    // 2. Check if the enclosing parent block has an earlier child `Block::Quote` or earlier sibling paragraph.
    if let Some(parent_ix) = structure.parent[b_idx] {
        let p_idx = parent_ix.get();
        for prev in (p_idx + 1)..b_idx {
            if structure.parent[prev] == Some(parent_ix)
                && matches!(structure.kind[prev], Block::Quote | Block::Paragraph)
            {
                let q_start = b_starts[prev];
                let q_end = b_ends[prev];
                let ts = tok_ends.partition_point(|&e| e <= q_start);
                let te = tok_starts.partition_point(|&s| s < q_end);
                if ts < te {
                    return Some((ts, te));
                }
            }
        }
    }
    None
}

fn select_salient_token_span(token_lower: &[String], tok_start: usize, tok_end: usize) -> IxRange<Tokens> {
    let mut best_idx = tok_start;
    let mut best_len = 0usize;
    for (idx, w) in token_lower[tok_start..tok_end].iter().enumerate() {
        let tok_idx = tok_start + idx;
        if w.is_empty() || is_stopword(w) {
            continue;
        }
        if w.len() > best_len {
            best_len = w.len();
            best_idx = tok_idx;
        }
    }
    IxRange::new(best_idx as u32, (best_idx + 1) as u32)
}

fn is_stopword(w: &str) -> bool {
    matches!(
        w,
        "the"
            | "a"
            | "an"
            | "and"
            | "or"
            | "but"
            | "is"
            | "are"
            | "was"
            | "were"
            | "be"
            | "been"
            | "to"
            | "of"
            | "in"
            | "on"
            | "for"
            | "with"
            | "at"
            | "by"
            | "from"
            | "this"
            | "that"
            | "these"
            | "those"
            | "it"
            | "its"
            | "i"
            | "we"
            | "you"
            | "they"
            | "he"
            | "she"
    )
}

/// Per-label evaluation metrics (`precision`, `recall`, `f1`) on an
/// [`Incivility`] evaluation set.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct LabelMetrics {
    pub tp: usize,
    pub fp: usize,
    pub fn_count: usize,
    pub tn: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

/// Multi-label evaluation summary across all six [`IncivilityLabel`]
/// dimensions.
#[derive(Clone, Debug, PartialEq)]
pub struct IncivilityMetrics {
    pub per_label: [(IncivilityLabel, LabelMetrics); 6],
    pub macro_precision: f64,
    pub macro_recall: f64,
    pub macro_f1: f64,
    pub micro_precision: f64,
    pub micro_recall: f64,
    pub micro_f1: f64,
    /// Number of positive predictions (`score >= threshold`) that carried at
    /// least one non-empty [`IxRange<Tokens>`] evidence span.
    pub positive_with_evidence: usize,
    /// Total number of positive predictions (`score >= threshold`).
    pub total_positive_predictions: usize,
}

/// Gold multi-label annotation for one unit in [`evaluate_incivility_corpus`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GoldIncivilityUnit {
    pub positive_labels: Vec<IncivilityLabel>,
}

/// Evaluates predicted [`Incivility`] items against `gold` at `threshold`,
/// computing per-label, macro, and micro precision/recall/F1 and checking
/// evidence span coverage on all positive predictions.
pub fn evaluate_incivility_corpus(
    gold: &[GoldIncivilityUnit],
    predicted: &Incivility,
    threshold: f64,
) -> IncivilityMetrics {
    assert_eq!(gold.len(), predicted.len(), "gold and predicted unit counts must match");
    let mut per_label = IncivilityLabel::ALL.map(|l| (l, LabelMetrics::default()));
    let mut positive_with_evidence = 0usize;
    let mut total_positive_predictions = 0usize;

    for (u_idx, g) in gold.iter().enumerate() {
        let ix = Ix::<Incivility>::new(u_idx as u32);
        let scores = predicted.scores(ix);
        let ev: Vec<(IncivilityLabel, IxRange<Tokens>)> = predicted.evidence_of(ix).collect();

        for (l_idx, &label) in IncivilityLabel::ALL.iter().enumerate() {
            let want = g.positive_labels.contains(&label);
            let got = scores.get(label) >= threshold;
            if got {
                total_positive_predictions += 1;
                if ev.iter().any(|&(el, r)| el == label && !r.is_empty()) {
                    positive_with_evidence += 1;
                }
            }
            let m = &mut per_label[l_idx].1;
            match (want, got) {
                (true, true) => m.tp += 1,
                (false, true) => m.fp += 1,
                (true, false) => m.fn_count += 1,
                (false, false) => m.tn += 1,
            }
        }
    }

    let mut sum_p = 0.0;
    let mut sum_r = 0.0;
    let mut sum_f1 = 0.0;
    let mut tot_tp = 0usize;
    let mut tot_fp = 0usize;
    let mut tot_fn = 0usize;

    for (_, m) in &mut per_label {
        m.precision = if m.tp + m.fp == 0 { 1.0 } else { (m.tp as f64) / ((m.tp + m.fp) as f64) };
        m.recall = if m.tp + m.fn_count == 0 { 1.0 } else { (m.tp as f64) / ((m.tp + m.fn_count) as f64) };
        m.f1 =
            if m.precision + m.recall == 0.0 { 0.0 } else { 2.0 * m.precision * m.recall / (m.precision + m.recall) };
        sum_p += m.precision;
        sum_r += m.recall;
        sum_f1 += m.f1;
        tot_tp += m.tp;
        tot_fp += m.fp;
        tot_fn += m.fn_count;
    }

    let micro_precision = if tot_tp + tot_fp == 0 { 1.0 } else { (tot_tp as f64) / ((tot_tp + tot_fp) as f64) };
    let micro_recall = if tot_tp + tot_fn == 0 { 1.0 } else { (tot_tp as f64) / ((tot_tp + tot_fn) as f64) };
    let micro_f1 = if micro_precision + micro_recall == 0.0 {
        0.0
    } else {
        2.0 * micro_precision * micro_recall / (micro_precision + micro_recall)
    };

    IncivilityMetrics {
        per_label,
        macro_precision: sum_p / 6.0,
        macro_recall: sum_r / 6.0,
        macro_f1: sum_f1 / 6.0,
        micro_precision,
        micro_recall,
        micro_f1,
        positive_with_evidence,
        total_positive_predictions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_analysis::{Analyzer, Document, Key, Provenance, Span, Spans, ViewId};
    use turbo_analysis_layers::Cut;

    fn spans_from_pairs<L: Layer>(pairs: impl IntoIterator<Item = (u32, u32)>) -> Spans<L> {
        let mut spans = Spans::new(ViewId::SOURCE);
        for (s, e) in pairs {
            spans.push(Span::new(s, e));
        }
        spans
    }

    fn make_doc_with_sentences(text: &str, sent_ranges: &[(u32, u32)]) -> Document {
        let mut doc = Document::new(text);
        let spans = spans_from_pairs::<Sentences>(sent_ranges.iter().copied());
        let mut t_spans = Spans::new(ViewId::SOURCE);
        let mut t_sent = Vec::new();
        for (s_idx, &(s, e)) in sent_ranges.iter().enumerate() {
            let slice = &text[s as usize..e as usize];
            let mut idx = 0usize;
            while idx < slice.len() {
                let ch = slice[idx..].chars().next().unwrap();
                let ch_len = ch.len_utf8();
                if ch.is_whitespace() {
                    idx += ch_len;
                    continue;
                }
                if ch.is_ascii_punctuation() && ch != '\'' {
                    t_spans.push(Span::new(s + idx as u32, s + (idx + ch_len) as u32));
                    t_sent.push(Ix::new(s_idx as u32));
                    idx += ch_len;
                    continue;
                }
                let tok_start = idx;
                idx += ch_len;
                while idx < slice.len() {
                    let c2 = slice[idx..].chars().next().unwrap();
                    if c2.is_whitespace() || (c2.is_ascii_punctuation() && c2 != '\'') {
                        break;
                    }
                    idx += c2.len_utf8();
                }
                t_spans.push(Span::new(s + tok_start as u32, s + idx as u32));
                t_sent.push(Ix::new(s_idx as u32));
            }
        }
        doc.add(&Key::<Sentences>::DEFAULT, Sentences { spans }, Provenance::by("test")).unwrap();
        doc.add(&Key::<Tokens>::DEFAULT, Tokens { spans: t_spans, sentence: Some(t_sent) }, Provenance::by("test"))
            .unwrap();
        doc
    }

    #[test]
    fn lexicon_digest_updates_on_insert_and_is_canonical() {
        let mut lex1 = IncivilityLexicon::bundled();
        let d1 = lex1.sha256().to_owned();
        assert_eq!(d1.len(), 64);

        lex1.insert("  Absolute   Clown! ", IncivilityLabel::Insult, 0.85);
        let d2 = lex1.sha256().to_owned();
        assert_ne!(d1, d2);

        let mut lex2 = IncivilityLexicon::bundled();
        lex2.insert("absolute clown", IncivilityLabel::Insult, 0.85);
        assert_eq!(d2, lex2.sha256());
    }

    #[test]
    fn multi_label_co_occurrence_and_evidence_spans_per_label() {
        // Single sentence triggering Profanity ("fucking"), Insult ("idiot"),
        // Threat ("hunt you down"), and co-occurrence SevereToxicity.
        let text = "You fucking idiot, we will hunt you down!";
        let mut doc = make_doc_with_sentences(text, &[(0, text.len() as u32)]);
        let pipe = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .given(Key::<Tokens>::DEFAULT)
            .add(IncivilityAnnotator::new())
            .build()
            .unwrap();
        for res in pipe.run(std::slice::from_mut(&mut doc)) {
            res.unwrap();
        }

        let prov = doc.provenance(Key::<Incivility>::DEFAULT.id()).unwrap();
        assert_eq!(prov.annotator, "turbo-doccat:incivility");
        assert_eq!(prov.model.as_ref().unwrap().id, "oasis-incivility:lexicon:no-reply-context");

        let inciv = doc.get::<Incivility>().unwrap();
        assert_eq!(inciv.len(), 1);
        let s0 = Ix::new(0);
        let scores = inciv.scores(s0);
        assert!(scores.profanity >= 0.90, "profanity={}", scores.profanity);
        assert!(scores.insult >= 0.80, "insult={}", scores.insult);
        assert!(scores.threat >= 0.90, "threat={}", scores.threat);
        assert!(scores.severe_toxicity >= 0.85, "severe_toxicity={}", scores.severe_toxicity);

        let ev: Vec<(IncivilityLabel, IxRange<Tokens>)> = inciv.evidence_of(s0).collect();
        // Tokens: 0="You", 1="fucking", 2="idiot", 3=",", 4="we", 5="will", 6="hunt", 7="you", 8="down", 9="!"
        assert!(ev.contains(&(IncivilityLabel::Profanity, IxRange::new(1, 2))));
        assert!(ev.contains(&(IncivilityLabel::Insult, IxRange::new(2, 3))));
        assert!(ev.contains(&(IncivilityLabel::Threat, IxRange::new(6, 9))));
        assert!(ev.iter().any(|&(l, _)| l == IncivilityLabel::SevereToxicity));
    }

    #[test]
    fn vector_scorer_and_chunk_unit_guarantee_evidence_spans() {
        struct FixedScorer;
        impl IncivilityVectorScorer for FixedScorer {
            fn score_incivility(&self, vec: &[f32]) -> Result<IncivilityScores, String> {
                let mut s = IncivilityScores::default();
                if vec.first().copied().unwrap_or(0.0) > 0.5 {
                    s.intolerance = 0.88;
                }
                Ok(s)
            }

            fn model(&self) -> Option<ModelRef> {
                Some(ModelRef { id: "fixed-head".into(), revision: "1".into(), sha256: "00".into() })
            }
        }

        // Sentence with no lexicon hits, scored positive on `Intolerance` by the vector head.
        let text = "Exclude every outsider from civic life.";
        let mut doc = make_doc_with_sentences(text, &[(0, text.len() as u32)]);
        let chunks = Chunks {
            spans: spans_from_pairs::<Chunks>([(0, text.len() as u32)]),
            level: vec![0],
            parent: vec![None],
            sentences: vec![IxRange::new(0, 1)],
            cut: vec![Cut::Sentence],
            cost: vec![1000],
            model_tokens: vec![6],
            over_budget: vec![false],
        };
        doc.add(&Key::<Chunks>::DEFAULT, chunks, Provenance::by("test")).unwrap();

        let vecs = Vectors::<Chunks>::new(
            2,
            vec![0.9, 0.1],
            vec![],
            ModelRef { id: "fixed-head".into(), revision: "1".into(), sha256: "00".into() },
        );
        doc.add(&Key::<Vectors<Chunks>>::DEFAULT, vecs, Provenance::by("test")).unwrap();

        let ann =
            IncivilityAnnotator::new().with_unit(IncivilityUnit::Chunks).with_vector_scorer(Arc::new(FixedScorer));
        let pipe = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .given(Key::<Tokens>::DEFAULT)
            .given(Key::<Chunks>::DEFAULT)
            .given(Key::<Vectors<Chunks>>::DEFAULT)
            .add(ann)
            .build()
            .unwrap();
        for res in pipe.run(std::slice::from_mut(&mut doc)) {
            res.unwrap();
        }

        let inciv = doc.get::<Incivility>().unwrap();
        assert_eq!(inciv.len(), 1);
        let c0 = Ix::new(0);
        assert_eq!(inciv.target[0], IncivilityTarget::Chunk(Ix::new(0)));
        assert!(inciv.scores(c0).intolerance >= 0.85);
        // Even though no lexicon phrase matched, the positive head label still receives a
        // non-empty salient content token range inside the chunk.
        let ev: Vec<(IncivilityLabel, IxRange<Tokens>)> = inciv.evidence_of(c0).collect();
        assert!(!ev.is_empty());
        assert!(ev.iter().any(|&(l, r)| l == IncivilityLabel::Intolerance && !r.is_empty()));
    }

    #[test]
    fn sentence_vectors_from_another_encoder_are_refused() {
        struct NamedScorer;
        impl IncivilityVectorScorer for NamedScorer {
            fn score_incivility(&self, _vec: &[f32]) -> Result<IncivilityScores, String> {
                Ok(IncivilityScores::default())
            }

            fn model(&self) -> Option<ModelRef> {
                Some(ModelRef { id: "head-encoder".into(), revision: "1".into(), sha256: "00".into() })
            }
        }

        let text = "The patch fixes the parser.";
        let mut doc = make_doc_with_sentences(text, &[(0, text.len() as u32)]);
        let vecs = Vectors::<Sentences>::new(
            2,
            vec![0.5, 0.5],
            vec![],
            ModelRef { id: "other-encoder".into(), revision: "1".into(), sha256: "11".into() },
        );
        doc.add(&Key::<Vectors<Sentences>>::DEFAULT, vecs, Provenance::by("test")).unwrap();
        let pipe = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .given(Key::<Tokens>::DEFAULT)
            .given(Key::<Vectors<Sentences>>::DEFAULT)
            .add(IncivilityAnnotator::new().with_vector_scorer(Arc::new(NamedScorer)))
            .build()
            .unwrap();
        let err = pipe.run(std::slice::from_mut(&mut doc)).remove(0).unwrap_err();
        assert!(err.to_string().contains("other-encoder"), "{err}");
        assert!(err.to_string().contains("head-encoder"), "{err}");
    }

    #[test]
    fn reply_context_scores_conflict_against_quote_block_when_enabled() {
        let quote = "The new transit schedule adds two evening buses.";
        let reply = "You have no clue what you are talking about.";
        let text = format!("{quote} {reply}");
        let q_end = quote.len() as u32;
        let r_start = q_end + 1;
        let r_end = text.len() as u32;

        // 1. Default (`use_reply_context = false`): provenance records `:no-reply-context`,
        // and the reply is not scored against the quote.
        let mut doc_no_ctx = make_doc_with_sentences(&text, &[(0, q_end), (r_start, r_end)]);
        let structure1 = Structure {
            spans: spans_from_pairs::<Structure>([(0, q_end), (r_start, r_end)]),
            kind: vec![Block::Quote, Block::Paragraph],
            parent: vec![None, None],
            depth: vec![1, 0],
        };
        doc_no_ctx.add(&Key::<Structure>::DEFAULT, structure1, Provenance::by("test")).unwrap();

        let pipe_no_ctx = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .given(Key::<Tokens>::DEFAULT)
            .given(Key::<Structure>::DEFAULT)
            .add(IncivilityAnnotator::new())
            .build()
            .unwrap();
        for res in pipe_no_ctx.run(std::slice::from_mut(&mut doc_no_ctx)) {
            res.unwrap();
        }
        let prov_no = doc_no_ctx.provenance(Key::<Incivility>::DEFAULT.id()).unwrap();
        assert_eq!(prov_no.model.as_ref().unwrap().id, "oasis-incivility:lexicon:no-reply-context");
        let inciv_no = doc_no_ctx.get::<Incivility>().unwrap();
        assert!(inciv_no.scores(Ix::new(1)).insult < 0.5);

        // 2. With `with_reply_context(true)`: provenance records `:reply-context`,
        // and the reply sentence triggers `Insult` on `"You have no clue"`.
        let mut doc_ctx = make_doc_with_sentences(&text, &[(0, q_end), (r_start, r_end)]);
        let structure2 = Structure {
            spans: spans_from_pairs::<Structure>([(0, q_end), (r_start, r_end)]),
            kind: vec![Block::Quote, Block::Paragraph],
            parent: vec![None, None],
            depth: vec![1, 0],
        };
        doc_ctx.add(&Key::<Structure>::DEFAULT, structure2, Provenance::by("test")).unwrap();

        let pipe_ctx = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .given(Key::<Tokens>::DEFAULT)
            .given(Key::<Structure>::DEFAULT)
            .add(IncivilityAnnotator::new().with_reply_context(true))
            .build()
            .unwrap();
        for res in pipe_ctx.run(std::slice::from_mut(&mut doc_ctx)) {
            res.unwrap();
        }
        let prov_ctx = doc_ctx.provenance(Key::<Incivility>::DEFAULT.id()).unwrap();
        assert_eq!(prov_ctx.model.as_ref().unwrap().id, "oasis-incivility:lexicon:reply-context");
        let inciv_ctx = doc_ctx.get::<Incivility>().unwrap();
        assert!(inciv_ctx.scores(Ix::new(0)).insult < 0.5);
        assert!(inciv_ctx.scores(Ix::new(1)).insult >= 0.75);
        let ev: Vec<(IncivilityLabel, IxRange<Tokens>)> = inciv_ctx.evidence_of(Ix::new(1)).collect();
        assert!(ev.iter().any(|&(l, r)| l == IncivilityLabel::Insult && !r.is_empty()));
    }

    #[test]
    fn regression_fixture_per_label_precision_recall_and_evidence_invariants() {
        let rows: &[(&str, &[IncivilityLabel])] = &[
            ("You are an incompetent moron and a clown.", &[IncivilityLabel::Insult]),
            ("This whole deployment is absolute bullshit.", &[IncivilityLabel::Profanity]),
            ("Watch your back, we will hunt you down.", &[IncivilityLabel::Threat]),
            ("Those people are parasites and subhuman vermin.", &[IncivilityLabel::IdentityAttack]),
            ("Ban their kind, there is no place for them here.", &[IncivilityLabel::Intolerance]),
            (
                "You fucking scumbag, we will kill you!",
                &[
                    IncivilityLabel::Insult,
                    IncivilityLabel::Profanity,
                    IncivilityLabel::Threat,
                    IncivilityLabel::SevereToxicity,
                ],
            ),
            ("You deserve to die, rot in hell.", &[IncivilityLabel::Profanity, IncivilityLabel::SevereToxicity]),
            ("The patch fixes the buffer alignment issue in the parser.", &[]),
            ("That problem is not stupid at all, thank you for checking.", &[]),
            ("We respectfully disagree with the proposed schedule change.", &[]),
        ];

        let mut full_text = String::new();
        let mut sent_spans = Vec::new();
        let mut gold = Vec::new();
        for &(s, labels) in rows {
            if !full_text.is_empty() {
                full_text.push(' ');
            }
            let start = full_text.len() as u32;
            full_text.push_str(s);
            let end = full_text.len() as u32;
            sent_spans.push((start, end));
            gold.push(GoldIncivilityUnit { positive_labels: labels.to_vec() });
        }

        let mut doc = make_doc_with_sentences(&full_text, &sent_spans);
        let pipe = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .given(Key::<Tokens>::DEFAULT)
            .add(IncivilityAnnotator::new())
            .build()
            .unwrap();
        for res in pipe.run(std::slice::from_mut(&mut doc)) {
            res.unwrap();
        }

        let inciv = doc.get::<Incivility>().unwrap();
        let metrics = evaluate_incivility_corpus(&gold, inciv, DEFAULT_INCIVILITY_THRESHOLD);
        assert_eq!(metrics.positive_with_evidence, metrics.total_positive_predictions);
        assert!(metrics.total_positive_predictions >= 10);
        for (lbl, m) in &metrics.per_label {
            assert_eq!(m.fp, 0, "{lbl}: unexpected false positive");
            assert_eq!(m.fn_count, 0, "{lbl}: unexpected false negative");
            assert!((m.f1 - 1.0).abs() < 1e-12, "{lbl}: expected F1 1.0, got {}", m.f1);
        }
    }

    #[test]
    fn foreign_view_layers_are_rejected() {
        let doc = Document::new("Hello world.");
        let view = doc.get::<View>().unwrap();
        let (clean_text, align) = turbo_analysis::ViewBuilder::new(view.text()).finish();
        let child_view = View::derived(view, clean_text, align);
        let foreign = child_view.id();

        let ok_tokens = Tokens { spans: spans_from_pairs([(0, 5), (6, 11), (11, 12)]), sentence: None };
        let bad_tokens = Tokens {
            spans: {
                let mut s = Spans::new(foreign);
                s.push(Span::new(0, 5));
                s
            },
            sentence: None,
        };
        let bad_sentences = Sentences {
            spans: {
                let mut s = Spans::new(foreign);
                s.push(Span::new(0, 12));
                s
            },
        };
        let bad_chunks = Chunks {
            spans: {
                let mut s = Spans::new(foreign);
                s.push(Span::new(0, 12));
                s
            },
            level: vec![0],
            parent: vec![None],
            sentences: vec![IxRange::new(0, 1)],
            cut: vec![Cut::DocumentStart],
            cost: vec![0],
            model_tokens: vec![3],
            over_budget: vec![false],
        };
        let bad_structure = Structure {
            spans: {
                let mut s = Spans::new(foreign);
                s.push(Span::new(0, 12));
                s
            },
            kind: vec![Block::Paragraph],
            parent: vec![None],
            depth: vec![0],
        };

        let annotator = IncivilityAnnotator::new();
        let err = annotator.annotate((view, &bad_tokens, None, None, None, None, None)).unwrap_err();
        assert!(err.to_string().contains("tokens are in another view"), "{err}");

        let err = annotator.annotate((view, &ok_tokens, Some(&bad_sentences), None, None, None, None)).unwrap_err();
        assert!(err.to_string().contains("sentences are in another view"), "{err}");

        let err = annotator.annotate((view, &ok_tokens, None, Some(&bad_chunks), None, None, None)).unwrap_err();
        assert!(err.to_string().contains("chunks are in another view"), "{err}");

        let err = annotator.annotate((view, &ok_tokens, None, None, Some(&bad_structure), None, None)).unwrap_err();
        assert!(err.to_string().contains("structure is in another view"), "{err}");
    }
}
