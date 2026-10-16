//! Aspect and opinion extraction for aspect-based sentiment analysis
//! (`turbo:aspects` and `turbo:opinions`, #322).
//!
//! Extracts explicit aspect terms (`AspectTarget::Term`), implicit sentence
//! aspects (`AspectTarget::Implicit`), and opinion expressions (`Opinions`),
//! links each aspect to its backing opinion expressions (`Ix<Opinions>`),
//! assigns an optional aspect category from a [`TagSet`], and scores three-way
//! [`PolarityScores`] (`positive`, `negative`, `neutral`) per aspect.
//!
//! This module implements the rule-based path (`RuleAspectSentiment`):
//!
//! - **Double propagation over dependency relations** following Qiu, Liu, Bu,
//!   and Chen (2011, *Opinion Word Expansion and Target Extraction through
//!   Double Propagation*, Computational Linguistics 37(1)):
//!   - **R1 (`O -> T`)**: Known opinion words extract modified aspect targets
//!     via direct syntactic relations (`amod`, `nsubj`, `obj`, `dobj`, `obl`,
//!     `nmod`, `acl`, `xcomp`, `acomp`) or shared heads (`O <- H -> T` in
//!     copular and verb-complement constructions).
//!   - **R2 (`T -> O`)**: Extracted aspect targets extract predicate and
//!     modifier opinion words attached to them.
//!   - **R3 (`T -> T'`)**: Extracted aspect targets propagate to conjoined
//!     noun targets (`conj`, `appos`) within the same polarity clause.
//!   - **R4 (`O -> O'`)**: Opinion words propagate to conjoined predicates and
//!     adjectives (`conj`), inverting polarity across adversative conjunctions.
//! - **Clause-aware noun-phrase and lexical fallback**: When `opennlp:dependencies`
//!   is absent (or alongside sparse trees), `RuleAspectSentiment` uses
//!   `opennlp:chunks` (`PhraseChunks` `NP` spans), `opennlp:pos` (`PosTags`),
//!   and compound aspect phrase tables, segmented at clause boundaries (`but`,
//!   `however`, `yet`, `although`, `though`, `whereas`, `while`, `;`, `,`) so
//!   contrastive sentences such as *"battery life is great but the screen
//!   scratches"* yield two aspects with opposite polarity.
//! - **Implicit aspects**: When a sentence contains opinion expressions that do
//!   not attach to any explicit aspect term in that sentence (for example,
//!   *"Way too expensive and completely overpriced."*), an implicit aspect
//!   (`AspectTarget::Implicit { sentence }`) is emitted, anchored to the
//!   sentence, linked to those opinions, and categorized into the aspect
//!   `TagSet`.

mod link;
mod propagation;
mod tables;

use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use turbo_analysis::{Annotator, Error, Ix, Layer, ModelRef, Refs, Tag, TagSet, View};
use turbo_analysis_layers::{
    Aspect, AspectTarget, Aspects, Dependencies, Opinion, Opinions, PhraseChunks, Polarity, PolarityScores, PosTags,
    Sentences, Tokens,
};

use super::lexicon::VaderLexicon;

pub use tables::DEFAULT_ASPECT_CATEGORIES;

use link::{
    compute_aspect_polarity, dedup_aspects, dedup_opinions, link_opinions_to_aspect, ranges_overlap, span_text,
};
use propagation::{ExtractedOpinion, assign_clauses};
use tables::{
    IMPLICIT_OPINION_CATEGORY_CUES, clean_word_lower, domain_opinion_valence, is_pure_aspect_noun,
    single_word_aspect_category,
};

/// Optional scorer that assigns an aspect category [`Tag`] from a [`TagSet`]
/// to an aspect (explicit or implicit) given its surface text and enclosing
/// sentence text.
///
/// Implemented in `turbo-analysis-embed` by [`LabelCentroids`]-backed category
/// scorers as well as keyword/centroid hybrid scorers.
pub trait AspectCategoryScorer: Send + Sync + fmt::Debug {
    /// The aspect category [`TagSet`] produced by this scorer.
    fn categories(&self) -> &Arc<TagSet>;

    /// Scores the aspect described by `aspect_text` (empty for implicit
    /// aspects), `opinion_text`, and `sentence_text`, returning the best
    /// category [`Tag`] if confidence exceeds the scorer's threshold.
    fn score_category(&self, aspect_text: &str, opinion_text: &str, sentence_text: &str) -> Option<Tag>;

    /// Deterministic configuration digest for provenance.
    fn config_hash(&self) -> u64;
}

/// Rule-based aspect and opinion extractor (`turbo-doccat:aspect-sentiment-rule`,
/// `#322`).
///
/// Runs Qiu et al. (2011) double propagation over `opennlp:dependencies` when
/// present, combined with clause-aware noun-phrase (`opennlp:chunks`), POS
/// (`opennlp:pos`), and compound aspect lexicon extraction, and writes
/// `(Opinions, Aspects)`.
#[derive(Clone)]
pub struct RuleAspectSentiment {
    lexicon: Arc<VaderLexicon>,
    categories: Arc<TagSet>,
    category_scorer: Option<Arc<dyn AspectCategoryScorer>>,
    emit_implicit_aspects: bool,
}

impl fmt::Debug for RuleAspectSentiment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RuleAspectSentiment")
            .field("categories", &self.categories.name)
            .field("has_category_scorer", &self.category_scorer.is_some())
            .field("emit_implicit_aspects", &self.emit_implicit_aspects)
            .finish()
    }
}

impl Default for RuleAspectSentiment {
    fn default() -> Self {
        Self::new()
    }
}

impl RuleAspectSentiment {
    /// Creates a new `RuleAspectSentiment` with the bundled VADER lexicon and
    /// [`DEFAULT_ASPECT_CATEGORIES`].
    pub fn new() -> Self {
        Self {
            lexicon: VaderLexicon::bundled(),
            categories: TagSet::new("aspect-categories", DEFAULT_ASPECT_CATEGORIES),
            category_scorer: None,
            emit_implicit_aspects: true,
        }
    }

    /// Uses a custom aspect category [`TagSet`].
    pub fn with_categories(mut self, categories: Arc<TagSet>) -> Self {
        self.categories = categories;
        self
    }

    /// Attaches a custom [`AspectCategoryScorer`] (such as a centroid-based
    /// scorer from `turbo-analysis-embed`).
    pub fn with_category_scorer(mut self, scorer: Arc<dyn AspectCategoryScorer>) -> Self {
        self.categories = Arc::clone(scorer.categories());
        self.category_scorer = Some(scorer);
        self
    }

    /// Configures whether sentences with opinion expressions but no explicit
    /// aspect term emit an [`AspectTarget::Implicit`] entry.
    pub fn with_implicit_aspects(mut self, enabled: bool) -> Self {
        self.emit_implicit_aspects = enabled;
        self
    }

    /// Returns the aspect category [`TagSet`].
    pub fn categories(&self) -> &Arc<TagSet> {
        &self.categories
    }

    /// Looks up the prior sentiment valence of a lowercased token in the
    /// domain opinion lexicon first, then the VADER lexicon.
    pub fn token_opinion_valence(&self, lower: &str) -> Option<f64> {
        // Do not treat aspect head nouns with neutral/mild VADER scores (like "value" or "support")
        // as opinion expressions when they appear as aspect nouns, unless they are strong opinions.
        if let Some(score) = domain_opinion_valence(lower) {
            return Some(score);
        }
        if let Some(v) = self.lexicon.get(lower)
            && v.abs() >= 0.8
            && !is_pure_aspect_noun(lower)
        {
            return Some(v);
        }
        None
    }
}

impl Annotator for RuleAspectSentiment {
    const ID: &'static str = "turbo-doccat:aspect-sentiment-rule";
    type Reads = (View, Sentences, Tokens, Option<Dependencies>, Option<PhraseChunks>, Option<PosTags>);
    type Writes = (Opinions, Aspects);

    fn annotate(
        &self,
        (view, sentences, tokens, deps, phrase_chunks, pos_tags): Refs<'_, Self>,
    ) -> Result<Self::Writes, Error> {
        let fail = |detail: String| Error::Annotator { annotator: Self::ID, detail };
        let text = view.text();
        let n_tokens = tokens.len();
        let (tok_starts, tok_ends) = tokens.spans.columns();
        let (sent_starts, sent_ends) = sentences.spans.columns();

        let mut token_lower: Vec<String> = Vec::with_capacity(n_tokens);
        for t in 0..n_tokens {
            let s = tok_starts[t] as usize;
            let e = tok_ends[t] as usize;
            let slice = text.get(s..e).unwrap_or("");
            token_lower.push(clean_word_lower(slice));
        }

        struct PendingAspect {
            target: AspectTarget,
            category: Option<Tag>,
            polarity: PolarityScores,
            opinions: Vec<Ix<Opinions>>,
        }

        let mut all_opinions: Vec<Opinion> = Vec::new();
        let mut pending_aspects: Vec<PendingAspect> = Vec::new();

        let mut tok_cursor = 0usize;
        for s_idx in 0..sentences.len() {
            let s_start_byte = sent_starts[s_idx];
            let s_end_byte = sent_ends[s_idx];

            while tok_cursor < n_tokens && tok_ends[tok_cursor] <= s_start_byte {
                tok_cursor += 1;
            }
            let sent_tok_start = tok_cursor;
            while tok_cursor < n_tokens && tok_starts[tok_cursor] < s_end_byte {
                tok_cursor += 1;
            }
            let sent_tok_end = tok_cursor;
            if sent_tok_start >= sent_tok_end {
                continue;
            }

            let sent_ix = Ix::<Sentences>::new(s_idx as u32);
            let clause_ids = assign_clauses(&token_lower, text, tok_starts, tok_ends, sent_tok_start, sent_tok_end);

            // Step 1: Seed opinions and explicit aspect candidates in the sentence.
            let (mut sent_opinions, mut sent_aspects) = self.extract_sentence_candidates(
                text,
                tok_starts,
                tok_ends,
                &token_lower,
                sent_tok_start,
                sent_tok_end,
                &clause_ids,
                deps,
                phrase_chunks,
                pos_tags,
            );

            // Step 2: If Dependencies are present, run Double Propagation (Qiu et al. 2011)
            // to expand targets from opinions (R1), opinions from targets (R2),
            // conjoined targets (R3), and conjoined opinions (R4).
            if let Some(d) = deps {
                self.run_double_propagation(
                    d,
                    pos_tags,
                    &token_lower,
                    sent_tok_start,
                    sent_tok_end,
                    &clause_ids,
                    &mut sent_opinions,
                    &mut sent_aspects,
                );
            }

            // Sort and deduplicate opinions by token range in ascending order.
            sent_opinions.sort_by_key(|o| (o.tokens.start.get(), o.tokens.end.get()));
            dedup_opinions(&mut sent_opinions);

            // Sort and deduplicate explicit aspects by token range in ascending order,
            // removing any aspect span that overlaps an opinion span.
            sent_aspects.retain(|a| !sent_opinions.iter().any(|o| ranges_overlap(a.tokens, o.tokens)));
            sent_aspects.sort_by_key(|a| (a.tokens.start.get(), a.tokens.end.get()));
            dedup_aspects(&mut sent_aspects);

            // Record the sentence's opinions in `all_opinions` and remember their global `Ix<Opinions>`.
            let base_op_idx = all_opinions.len();
            for op in &sent_opinions {
                all_opinions.push(Opinion {
                    tokens: op.tokens,
                    polarity: op.polarity,
                    intensity: op.intensity,
                    negated: op.negated,
                });
            }

            let sent_text = text.get(s_start_byte as usize..s_end_byte as usize).unwrap_or("");

            if !sent_aspects.is_empty() {
                // Step 3: Link each explicit aspect to its governing/modifying opinions
                // (via dependency paths first when `deps` is present, then same-clause
                // proximity, then sentence fallback).
                for (a_pos, asp) in sent_aspects.iter().enumerate() {
                    let linked_local = link_opinions_to_aspect(
                        asp,
                        a_pos,
                        &sent_aspects,
                        &sent_opinions,
                        deps,
                        &token_lower,
                        sent_tok_start,
                        sent_tok_end,
                    );
                    let linked_ixs: Vec<Ix<Opinions>> =
                        linked_local.iter().map(|&k| Ix::new((base_op_idx + k) as u32)).collect();
                    let polarity = compute_aspect_polarity(&linked_local, &sent_opinions);
                    let asp_text = span_text(text, tok_starts, tok_ends, asp.tokens);
                    let op_text = linked_local
                        .first()
                        .map(|&k| span_text(text, tok_starts, tok_ends, sent_opinions[k].tokens))
                        .unwrap_or("");
                    let category = self.resolve_category(
                        asp.category_hint,
                        asp_text,
                        op_text,
                        sent_text,
                        &token_lower[asp.tokens.start.get()..asp.tokens.end.get()],
                    );
                    pending_aspects.push(PendingAspect {
                        target: AspectTarget::Term { tokens: asp.tokens, sentence: Some(sent_ix) },
                        category,
                        polarity,
                        opinions: linked_ixs,
                    });
                }
            } else if self.emit_implicit_aspects && !sent_opinions.is_empty() {
                // Step 4: Implicit aspect anchored to the sentence when opinion expressions
                // exist without any explicit aspect term in the sentence.
                let all_local: Vec<usize> = (0..sent_opinions.len()).collect();
                let linked_ixs: Vec<Ix<Opinions>> =
                    all_local.iter().map(|&k| Ix::new((base_op_idx + k) as u32)).collect();
                let polarity = compute_aspect_polarity(&all_local, &sent_opinions);
                let op_text = span_text(text, tok_starts, tok_ends, sent_opinions[0].tokens);
                let category = self.resolve_implicit_category(&sent_opinions, &token_lower, op_text, sent_text);
                pending_aspects.push(PendingAspect {
                    target: AspectTarget::Implicit { sentence: sent_ix },
                    category,
                    polarity,
                    opinions: linked_ixs,
                });
            }
        }

        let opinions_layer = Opinions::from_items(tokens, all_opinions).map_err(fail)?;
        let mut aspects_layer = Aspects::new(Arc::clone(&self.categories));
        for p in &pending_aspects {
            aspects_layer
                .try_push(
                    tokens,
                    Some(sentences),
                    Some(&opinions_layer),
                    Aspect { target: p.target, category: p.category, polarity: p.polarity, opinions: &p.opinions },
                )
                .map_err(fail)?;
        }

        Ok((opinions_layer, aspects_layer))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(ModelRef {
            id: "double-propagation:vader".to_string(),
            revision: "1.0.0".to_string(),
            sha256: format!("{:016x}", self.config_hash()),
        })
    }

    fn config_hash(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        Self::ID.hash(&mut h);
        self.categories.name.hash(&mut h);
        for i in 0..self.categories.len() {
            self.categories.label(Tag(i as u16)).hash(&mut h);
        }
        self.emit_implicit_aspects.hash(&mut h);
        if let Some(scorer) = &self.category_scorer {
            scorer.config_hash().hash(&mut h);
        }
        h.finish()
    }
}

impl RuleAspectSentiment {
    fn resolve_category(
        &self,
        hint: Option<&'static str>,
        aspect_text: &str,
        opinion_text: &str,
        sentence_text: &str,
        aspect_tokens_lower: &[String],
    ) -> Option<Tag> {
        if let Some(scorer) = &self.category_scorer
            && let Some(tag) = scorer.score_category(aspect_text, opinion_text, sentence_text)
        {
            return Some(tag);
        }
        if let Some(h) = hint
            && let Some(tag) = self.categories.tag(h)
        {
            return Some(tag);
        }
        // Check if any token in the aspect directly matches a label in `self.categories`.
        for w in aspect_tokens_lower {
            if let Some(tag) = self.categories.tag(w.as_str()) {
                return Some(tag);
            }
            if let Some(cat) = single_word_aspect_category(w)
                && let Some(tag) = self.categories.tag(cat)
            {
                return Some(tag);
            }
        }
        self.categories.tag("general")
    }

    fn resolve_implicit_category(
        &self,
        opinions: &[ExtractedOpinion],
        token_lower: &[String],
        opinion_text: &str,
        sentence_text: &str,
    ) -> Option<Tag> {
        if let Some(scorer) = &self.category_scorer
            && let Some(tag) = scorer.score_category("", opinion_text, sentence_text)
        {
            return Some(tag);
        }
        for op in opinions {
            let w = token_lower[op.head_token].as_str();
            if let Some(&(_, cat)) = IMPLICIT_OPINION_CATEGORY_CUES.iter().find(|&&(cue, _)| cue == w)
                && let Some(tag) = self.categories.tag(cat)
            {
                return Some(tag);
            }
        }
        self.categories.tag("general")
    }
}

/// Precision, recall, and F1 metrics for aspect term extraction and
/// aspect-opinion-polarity triplet extraction (`#322`).
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct AbsaMetrics {
    pub aspect_precision: f64,
    pub aspect_recall: f64,
    pub aspect_f1: f64,
    pub opinion_precision: f64,
    pub opinion_recall: f64,
    pub opinion_f1: f64,
    pub triplet_precision: f64,
    pub triplet_recall: f64,
    pub triplet_f1: f64,
}

/// Gold or predicted aspect-opinion-polarity triplet in a document:
/// `(aspect_tokens_or_none_for_implicit, sentence_index, opinion_tokens, polarity)`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct AbsaTriplet {
    pub aspect_tokens: Option<(u32, u32)>,
    pub sentence: u32,
    pub opinion_tokens: (u32, u32),
    pub polarity: Polarity,
}

/// Extracts all [`AbsaTriplet`]s from a document's `(Opinions, Aspects)` layers.
pub fn extract_triplets(opinions: &Opinions, aspects: &Aspects) -> Vec<AbsaTriplet> {
    let mut out = Vec::new();
    for i in 0..aspects.len() {
        let ix = Ix::<Aspects>::new(i as u32);
        let asp = aspects.get(ix);
        let (aspect_tokens, sentence) = match asp.target {
            AspectTarget::Term { tokens, sentence } => (
                Some((tokens.start.get() as u32, tokens.end.get() as u32)),
                sentence.map(|s| s.get() as u32).unwrap_or(0),
            ),
            AspectTarget::Implicit { sentence } => (None, sentence.get() as u32),
        };
        let pol = asp.polarity.label();
        for &op_ix in asp.opinions {
            let op = opinions.get(op_ix);
            out.push(AbsaTriplet {
                aspect_tokens,
                sentence,
                opinion_tokens: (op.tokens.start.get() as u32, op.tokens.end.get() as u32),
                polarity: pol,
            });
        }
    }
    out
}

/// Computes micro-averaged precision, recall, and F1 for explicit aspect spans,
/// opinion spans, and aspect-opinion-polarity triplets across a corpus of
/// documents.
pub fn evaluate_absa_corpus(gold: &[Vec<AbsaTriplet>], predicted: &[Vec<AbsaTriplet>]) -> AbsaMetrics {
    assert_eq!(gold.len(), predicted.len(), "gold and predicted document counts must match");
    let mut asp_tp = 0usize;
    let mut asp_fp = 0usize;
    let mut asp_fn = 0usize;

    let mut op_tp = 0usize;
    let mut op_fp = 0usize;
    let mut op_fn = 0usize;

    let mut tri_tp = 0usize;
    let mut tri_fp = 0usize;
    let mut tri_fn = 0usize;

    for (g_doc, p_doc) in gold.iter().zip(predicted) {
        let mut g_aspects: Vec<(Option<(u32, u32)>, u32)> =
            g_doc.iter().map(|t| (t.aspect_tokens, t.sentence)).collect();
        g_aspects.sort_unstable();
        g_aspects.dedup();

        let mut p_aspects: Vec<(Option<(u32, u32)>, u32)> =
            p_doc.iter().map(|t| (t.aspect_tokens, t.sentence)).collect();
        p_aspects.sort_unstable();
        p_aspects.dedup();

        for a in &p_aspects {
            if g_aspects.contains(a) {
                asp_tp += 1;
            } else {
                asp_fp += 1;
            }
        }
        for a in &g_aspects {
            if !p_aspects.contains(a) {
                asp_fn += 1;
            }
        }

        let mut g_ops: Vec<(u32, u32)> = g_doc.iter().map(|t| t.opinion_tokens).collect();
        g_ops.sort_unstable();
        g_ops.dedup();

        let mut p_ops: Vec<(u32, u32)> = p_doc.iter().map(|t| t.opinion_tokens).collect();
        p_ops.sort_unstable();
        p_ops.dedup();

        for o in &p_ops {
            if g_ops.contains(o) {
                op_tp += 1;
            } else {
                op_fp += 1;
            }
        }
        for o in &g_ops {
            if !p_ops.contains(o) {
                op_fn += 1;
            }
        }

        for t in p_doc {
            if g_doc.contains(t) {
                tri_tp += 1;
            } else {
                tri_fp += 1;
            }
        }
        for t in g_doc {
            if !p_doc.contains(t) {
                tri_fn += 1;
            }
        }
    }

    let (aspect_precision, aspect_recall, aspect_f1) = prf1(asp_tp, asp_fp, asp_fn);
    let (opinion_precision, opinion_recall, opinion_f1) = prf1(op_tp, op_fp, op_fn);
    let (triplet_precision, triplet_recall, triplet_f1) = prf1(tri_tp, tri_fp, tri_fn);

    AbsaMetrics {
        aspect_precision,
        aspect_recall,
        aspect_f1,
        opinion_precision,
        opinion_recall,
        opinion_f1,
        triplet_precision,
        triplet_recall,
        triplet_f1,
    }
}

fn prf1(tp: usize, fp: usize, fneg: usize) -> (f64, f64, f64) {
    let p = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
    let r = if tp + fneg > 0 { tp as f64 / (tp + fneg) as f64 } else { 0.0 };
    let f1 = if p + r > 0.0 { 2.0 * p * r / (p + r) } else { 0.0 };
    (p, r, f1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_analysis::{Analyzer, Document, IxRange, Key, Provenance, Span, Spans, ViewId};

    fn tokenize_simple(text: &str) -> Tokens {
        let mut spans = Spans::<Tokens>::new(ViewId(0));
        let mut sentence = Vec::new();
        let bytes = text.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            if i >= bytes.len() {
                break;
            }
            let start = i;
            if bytes[i].is_ascii_alphanumeric() {
                while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'\'') {
                    i += 1;
                }
            } else {
                i += 1;
            }
            spans.push(Span::new(start as u32, i as u32));
            sentence.push(Ix::new(0));
        }
        Tokens { spans, sentence: Some(sentence) }
    }

    fn run_rule_absa(text: &str) -> Document {
        let mut doc = Document::new(text);
        let mut sents = Spans::<Sentences>::new(ViewId(0));
        sents.push(Span::new(0, text.len() as u32));
        doc.add(&Key::DEFAULT, Sentences { spans: sents }, Provenance::by("test")).unwrap();
        doc.add(&Key::DEFAULT, tokenize_simple(text), Provenance::by("test")).unwrap();

        let pipe = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .given(Key::<Tokens>::DEFAULT)
            .add(RuleAspectSentiment::new())
            .build()
            .unwrap();
        let mut docs = [doc];
        pipe.run(&mut docs).remove(0).unwrap();
        docs.into_iter().next().unwrap()
    }

    #[test]
    fn battery_life_is_great_but_screen_scratches_gives_opposite_polarity_aspects() {
        // Exact example from #322 summary:
        // "battery life is great but the screen scratches"
        let doc = run_rule_absa("battery life is great but the screen scratches");
        let tokens = doc.get::<Tokens>().unwrap();
        let opinions = doc.get::<Opinions>().unwrap();
        let aspects = doc.get::<Aspects>().unwrap();

        assert_eq!(aspects.len(), 2);
        assert_eq!(opinions.len(), 2);

        let a0 = aspects.get(Ix::new(0));
        let a1 = aspects.get(Ix::new(1));
        assert_eq!(a0.target, AspectTarget::Term { tokens: IxRange::new(0, 2), sentence: Some(Ix::new(0)) });
        assert_eq!(aspects.category_label(Ix::new(0)), Some("battery"));
        assert_eq!(a0.polarity.label(), Polarity::Positive);
        assert_eq!(a0.opinions, &[Ix::<Opinions>::new(0)]);
        assert_eq!(opinions.get(Ix::new(0)).tokens, IxRange::new(3, 4)); // "great"

        assert_eq!(a1.target, AspectTarget::Term { tokens: IxRange::new(6, 7), sentence: Some(Ix::new(0)) });
        assert_eq!(aspects.category_label(Ix::new(1)), Some("display"));
        assert_eq!(a1.polarity.label(), Polarity::Negative);
        assert_eq!(a1.opinions, &[Ix::<Opinions>::new(1)]);
        assert_eq!(opinions.get(Ix::new(1)).tokens, IxRange::new(7, 8)); // "scratches"
        let _ = tokens;
    }

    #[test]
    fn negation_with_a_curly_apostrophe_still_negates() {
        // A tokenizer that keeps contractions whole hands the extractor
        // tokens like "isn’t" (U+2019): the negation cues, written with
        // ASCII apostrophes, must still fire.
        let text = "The screen isn\u{2019}t great.";
        let mut doc = Document::new(text);
        let mut sents = Spans::<Sentences>::new(ViewId(0));
        sents.push(Span::new(0, text.len() as u32));
        doc.add(&Key::DEFAULT, Sentences { spans: sents }, Provenance::by("test")).unwrap();
        let mut spans = Spans::<Tokens>::new(ViewId(0));
        for (a, b) in [(0, 3), (4, 10), (11, 18), (19, 24), (24, 25)] {
            spans.push(Span::new(a, b));
        }
        let tokens = Tokens { spans, sentence: Some(vec![Ix::new(0); 5]) };
        doc.add(&Key::DEFAULT, tokens, Provenance::by("test")).unwrap();

        let pipe = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .given(Key::<Tokens>::DEFAULT)
            .add(RuleAspectSentiment::new())
            .build()
            .unwrap();
        let mut docs = [doc];
        pipe.run(&mut docs).remove(0).unwrap();
        let doc = &docs[0];
        let opinions = doc.get::<Opinions>().unwrap();
        let aspects = doc.get::<Aspects>().unwrap();
        assert_eq!(opinions.len(), 1);
        assert!(opinions.get(Ix::new(0)).negated, "the curly-apostrophe negation must reach the opinion");
        assert_eq!(aspects.len(), 1);
        assert_eq!(aspects.get(Ix::new(0)).polarity.label(), Polarity::Negative);
    }

    #[test]
    fn double_propagation_over_dependencies_extracts_new_targets_and_conjoined_aspects() {
        // "The sushi and tiramisu were delicious."
        // Notice "tiramisu" is not in SINGLE_WORD_ASPECT_TERMS, so R3 (`conj` from "sushi" or R12 from "delicious")
        // extracts it via Dependencies!
        let text = "The sushi and tiramisu were delicious.";
        let mut doc = Document::new(text);
        let mut sents = Spans::<Sentences>::new(ViewId(0));
        sents.push(Span::new(0, text.len() as u32));
        doc.add(&Key::DEFAULT, Sentences { spans: sents }, Provenance::by("test")).unwrap();
        doc.add(&Key::DEFAULT, tokenize_simple(text), Provenance::by("test")).unwrap();

        // Tokens: 0:"The", 1:"sushi", 2:"and", 3:"tiramisu", 4:"were", 5:"delicious", 6:"."
        let rels = TagSet::new("ud", &["root", "det", "nsubj", "cc", "conj", "cop", "punct"]);
        let deps = Dependencies::<Tokens> {
            head: vec![
                Some(Ix::new(1)), // The -> sushi
                Some(Ix::new(5)), // sushi -> delicious (nsubj)
                Some(Ix::new(3)), // and -> tiramisu
                Some(Ix::new(1)), // tiramisu -> sushi (conj)
                Some(Ix::new(5)), // were -> delicious (cop)
                None,             // delicious (root)
                Some(Ix::new(5)), // . -> delicious
            ],
            relation: vec![
                rels.tag("det").unwrap(),
                rels.tag("nsubj").unwrap(),
                rels.tag("cc").unwrap(),
                rels.tag("conj").unwrap(),
                rels.tag("cop").unwrap(),
                rels.tag("root").unwrap(),
                rels.tag("punct").unwrap(),
            ],
            relations: rels,
        };
        doc.add(&Key::DEFAULT, deps, Provenance::by("test")).unwrap();

        let absa_pipe = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .given(Key::<Tokens>::DEFAULT)
            .given(Key::<Dependencies>::DEFAULT)
            .add(RuleAspectSentiment::new())
            .build()
            .unwrap();
        let mut docs = [doc];
        absa_pipe.run(&mut docs).remove(0).unwrap();
        let doc = &docs[0];

        let aspects = doc.get::<Aspects>().unwrap();
        assert_eq!(aspects.len(), 2);
        assert_eq!(aspects.get(Ix::new(0)).target.tokens(), Some(IxRange::new(1, 2))); // "sushi"
        assert_eq!(aspects.get(Ix::new(1)).target.tokens(), Some(IxRange::new(3, 4))); // "tiramisu" extracted by R3/R1!
        assert_eq!(aspects.get(Ix::new(0)).polarity.label(), Polarity::Positive);
        assert_eq!(aspects.get(Ix::new(1)).polarity.label(), Polarity::Positive);
    }

    #[test]
    fn aspect_and_opinion_under_the_same_governor_link_without_panicking() {
        // "The battery works well": `nsubj(works, battery)` and `advmod(works, well)` share
        // the governor "works". Linking used to index an empty slice on this shape.
        let text = "The battery works well";
        let mut doc = Document::new(text);
        let mut sents = Spans::<Sentences>::new(ViewId(0));
        sents.push(Span::new(0, text.len() as u32));
        doc.add(&Key::DEFAULT, Sentences { spans: sents }, Provenance::by("test")).unwrap();
        doc.add(&Key::DEFAULT, tokenize_simple(text), Provenance::by("test")).unwrap();
        let rels = TagSet::new("ud", &["root", "det", "nsubj", "advmod"]);
        let deps = Dependencies::<Tokens> {
            head: vec![Some(Ix::new(1)), Some(Ix::new(2)), None, Some(Ix::new(2))],
            relation: vec![
                rels.tag("det").unwrap(),
                rels.tag("nsubj").unwrap(),
                rels.tag("root").unwrap(),
                rels.tag("advmod").unwrap(),
            ],
            relations: rels,
        };
        doc.add(&Key::DEFAULT, deps, Provenance::by("test")).unwrap();
        let pipe = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .given(Key::<Tokens>::DEFAULT)
            .given(Key::<Dependencies>::DEFAULT)
            .add(RuleAspectSentiment::new())
            .build()
            .unwrap();
        let mut docs = [doc];
        pipe.run(&mut docs).remove(0).unwrap();
        let aspects = docs[0].get::<Aspects>().unwrap();
        let opinions = docs[0].get::<Opinions>().unwrap();
        assert_eq!(opinions.get(Ix::new(0)).tokens, IxRange::new(3, 4)); // "well"
        let battery = (0..aspects.len())
            .map(|i| Ix::<Aspects>::new(i as u32))
            .find(|&ix| aspects.get(ix).target.tokens() == Some(IxRange::new(1, 2)))
            .expect("battery is an aspect");
        assert_eq!(aspects.get(battery).polarity.label(), Polarity::Positive);
        assert_eq!(aspects.get(battery).opinions, &[Ix::<Opinions>::new(0)]);
    }

    #[test]
    fn implicit_aspect_anchors_to_sentence_and_scores_category() {
        let doc = run_rule_absa("Way too expensive and completely overpriced.");
        let aspects = doc.get::<Aspects>().unwrap();
        let opinions = doc.get::<Opinions>().unwrap();
        assert_eq!(aspects.len(), 1);
        assert_eq!(opinions.len(), 2);
        let a0 = aspects.get(Ix::new(0));
        assert_eq!(a0.target, AspectTarget::Implicit { sentence: Ix::new(0) });
        assert_eq!(aspects.category_label(Ix::new(0)), Some("price"));
        assert_eq!(a0.polarity.label(), Polarity::Negative);
        assert_eq!(a0.opinions.len(), 2);
    }

    #[test]
    fn absa_benchmark_f1_for_aspect_terms_and_triplets() {
        // Ten project-written sentences covering laptop/hardware, restaurant/hospitality,
        // multi-aspect contrast, negation, and implicit aspects. The rules were written
        // beside these sentences, so this is a regression fixture, not a benchmark.
        let benchmark: &[(&str, &[AbsaTriplet])] = &[
            (
                "battery life is great but the screen scratches",
                &[
                    AbsaTriplet {
                        aspect_tokens: Some((0, 2)),
                        sentence: 0,
                        opinion_tokens: (3, 4),
                        polarity: Polarity::Positive,
                    },
                    AbsaTriplet {
                        aspect_tokens: Some((6, 7)),
                        sentence: 0,
                        opinion_tokens: (7, 8),
                        polarity: Polarity::Negative,
                    },
                ],
            ),
            (
                "The pizza was delicious and the staff was friendly.",
                &[
                    AbsaTriplet {
                        aspect_tokens: Some((1, 2)),
                        sentence: 0,
                        opinion_tokens: (3, 4),
                        polarity: Polarity::Positive,
                    },
                    AbsaTriplet {
                        aspect_tokens: Some((6, 7)),
                        sentence: 0,
                        opinion_tokens: (8, 9),
                        polarity: Polarity::Positive,
                    },
                ],
            ),
            (
                "The keyboard is responsive, however the fan noise is loud.",
                &[
                    AbsaTriplet {
                        aspect_tokens: Some((1, 2)),
                        sentence: 0,
                        opinion_tokens: (3, 4),
                        polarity: Polarity::Positive,
                    },
                    AbsaTriplet {
                        aspect_tokens: Some((7, 9)),
                        sentence: 0,
                        opinion_tokens: (10, 11),
                        polarity: Polarity::Negative,
                    },
                ],
            ),
            (
                "The customer service was not helpful.",
                &[AbsaTriplet {
                    aspect_tokens: Some((1, 3)),
                    sentence: 0,
                    opinion_tokens: (5, 6),
                    polarity: Polarity::Negative,
                }],
            ),
            (
                "The build quality is sturdy and the trackpad is smooth.",
                &[
                    AbsaTriplet {
                        aspect_tokens: Some((1, 3)),
                        sentence: 0,
                        opinion_tokens: (4, 5),
                        polarity: Polarity::Positive,
                    },
                    AbsaTriplet {
                        aspect_tokens: Some((7, 8)),
                        sentence: 0,
                        opinion_tokens: (9, 10),
                        polarity: Polarity::Positive,
                    },
                ],
            ),
            (
                "The pasta was soggy and the waiter was rude.",
                &[
                    AbsaTriplet {
                        aspect_tokens: Some((1, 2)),
                        sentence: 0,
                        opinion_tokens: (3, 4),
                        polarity: Polarity::Negative,
                    },
                    AbsaTriplet {
                        aspect_tokens: Some((6, 7)),
                        sentence: 0,
                        opinion_tokens: (8, 9),
                        polarity: Polarity::Negative,
                    },
                ],
            ),
            (
                "The camera takes crisp photos, but the speakers are tinny.",
                &[
                    AbsaTriplet {
                        aspect_tokens: Some((1, 2)),
                        sentence: 0,
                        opinion_tokens: (3, 4),
                        polarity: Polarity::Positive,
                    },
                    AbsaTriplet {
                        aspect_tokens: Some((4, 5)),
                        sentence: 0,
                        opinion_tokens: (3, 4),
                        polarity: Polarity::Positive,
                    },
                    AbsaTriplet {
                        aspect_tokens: Some((8, 9)),
                        sentence: 0,
                        opinion_tokens: (10, 11),
                        polarity: Polarity::Negative,
                    },
                ],
            ),
            (
                "The ambience was cozy and the wine list was generous.",
                &[
                    AbsaTriplet {
                        aspect_tokens: Some((1, 2)),
                        sentence: 0,
                        opinion_tokens: (3, 4),
                        polarity: Polarity::Positive,
                    },
                    AbsaTriplet {
                        aspect_tokens: Some((6, 8)),
                        sentence: 0,
                        opinion_tokens: (9, 10),
                        polarity: Polarity::Positive,
                    },
                ],
            ),
            (
                "The software is buggy and the interface is clunky.",
                &[
                    AbsaTriplet {
                        aspect_tokens: Some((1, 2)),
                        sentence: 0,
                        opinion_tokens: (3, 4),
                        polarity: Polarity::Negative,
                    },
                    AbsaTriplet {
                        aspect_tokens: Some((6, 7)),
                        sentence: 0,
                        opinion_tokens: (8, 9),
                        polarity: Polarity::Negative,
                    },
                ],
            ),
            (
                "Way too expensive and completely overpriced.",
                &[
                    AbsaTriplet {
                        aspect_tokens: None,
                        sentence: 0,
                        opinion_tokens: (2, 3),
                        polarity: Polarity::Negative,
                    },
                    AbsaTriplet {
                        aspect_tokens: None,
                        sentence: 0,
                        opinion_tokens: (5, 6),
                        polarity: Polarity::Negative,
                    },
                ],
            ),
        ];

        let mut gold_corpus = Vec::new();
        let mut pred_corpus = Vec::new();
        for &(text, gold_triplets) in benchmark {
            let doc = run_rule_absa(text);
            let ops = doc.get::<Opinions>().unwrap();
            let asps = doc.get::<Aspects>().unwrap();
            gold_corpus.push(gold_triplets.to_vec());
            pred_corpus.push(extract_triplets(ops, asps));
        }

        let metrics = evaluate_absa_corpus(&gold_corpus, &pred_corpus);
        assert!(
            metrics.aspect_f1 >= 0.95,
            "expected aspect F1 >= 0.95, got {:.3} (P={:.3}, R={:.3})",
            metrics.aspect_f1,
            metrics.aspect_precision,
            metrics.aspect_recall
        );
        assert!(
            metrics.opinion_f1 >= 0.95,
            "expected opinion F1 >= 0.95, got {:.3} (P={:.3}, R={:.3})",
            metrics.opinion_f1,
            metrics.opinion_precision,
            metrics.opinion_recall
        );
        assert!(
            metrics.triplet_f1 >= 0.95,
            "expected triplet F1 >= 0.95, got {:.3} (P={:.3}, R={:.3})",
            metrics.triplet_f1,
            metrics.triplet_precision,
            metrics.triplet_recall
        );
    }
}
