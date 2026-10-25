//! Context-aware spelling correction: Viterbi decoding over `SymSpell` and
//! confusable candidates scored with an `NGramLanguageModel`, while preserving
//! protected spans and recording exact `View` alignments.

use std::sync::Arc;

use turbo_analysis::{Annotator, Error, Overlap, Refs, Span, View, ViewBuilder};
use turbo_analysis_layers::{Abbreviations, Entities, Money, Pii, Quantities, Sentences, Temporals};
use turbo_ngram::NGramLanguageModel;
use turbo_normalize_layers::{Corrections, Placeholders, ProtectedSpans};
use turbo_unicode::java;

use super::casing;
use super::confusables::{ConfusableSet, ProtectedInputs, collect_protected_spans, to_lowercase};
use super::distance::EditDistance;
use super::guards::{is_number_like, is_url_like};
use super::suggestion::Verbosity;
use super::symspell::{LookupError, SymSpell};
use super::trellis::{Candidate, TokenSlot, collect_tokens, group_into_sentences, lm_fingerprint, viterbi_decode};
use crate::{Normalizer, hash_config};

/// Default minimum token length (in code points) for [`ContextSpellCorrector`],
/// allowing 2- and 3-character confusables (`"to"`/`"two"`, `"teh"`/`"the"`)
/// to be disambiguated by the language model while leaving 1-character tokens
/// (`"I"`, `"a"`) untouched.
pub const DEFAULT_CONTEXT_MIN_TOKEN_LENGTH: usize = 2;

/// Default weight $\lambda$ multiplying edit distance in the Viterbi objective
/// $\lambda \cdot d_{\text{edit}}(w_i, c_i) - \log P_{\text{LM}}(c_i \mid c_{i-n+1..i-1})$.
pub const DEFAULT_EDIT_WEIGHT: f64 = 1.5;

/// Default maximum number of candidates evaluated per token in the trellis.
pub const DEFAULT_MAX_CANDIDATES: usize = 8;

/// Default maximum Viterbi beam width retained per trellis step.
pub const DEFAULT_MAX_BEAM_WIDTH: usize = 32;

/// Context-aware spelling corrector and pipeline annotator.
///
/// Generates per-token candidate sets from [`SymSpell`] (within configurable
/// `max_edit_distance`) together with real-word confusable alternatives from
/// [`ConfusableSet`], builds a sentence trellis, and decodes the optimal
/// candidate sequence via Viterbi under the combined objective
/// $\lambda \cdot d_{\text{edit}}(w_i, c_i) - \log P_{\text{LM}}(c_i \mid c_{i-n+1..i-1})$.
///
/// When run as an [`Annotator`], any token overlapping [`ProtectedSpans`],
/// [`Placeholders`], [`Pii`], [`Temporals`], [`Money`], [`Quantities`],
/// [`Entities`], or [`Abbreviations`] is locked to its original text while
/// still contributing its surface form to surrounding language-model context.
#[derive(Clone, Debug)]
pub struct ContextSpellCorrector {
    engine: Arc<SymSpell>,
    lm: Arc<NGramLanguageModel>,
    lm_digest: u64,
    confusables: ConfusableSet,
    min_token_length: usize,
    max_edit_distance: usize,
    max_candidates: usize,
    max_beam_width: usize,
    edit_weight: f64,
    check_real_word_edits: bool,
    skip_numbers: bool,
    skip_urls: bool,
    keep_unknown_proper_nouns: bool,
    use_protected_spans: bool,
}

/// Alias for [`ContextSpellCorrector`] when referenced as a pipeline annotator.
pub type ContextSpellAnnotator = ContextSpellCorrector;

impl ContextSpellCorrector {
    /// Creates a context-aware spelling corrector backed by `engine` and `lm`,
    /// initialized with [`ConfusableSet::english_default`].
    pub fn new(engine: Arc<SymSpell>, lm: Arc<NGramLanguageModel>) -> ContextSpellCorrector {
        let max_edit_distance = engine.max_edit_distance();
        let lm_digest = lm_fingerprint(&lm);
        ContextSpellCorrector {
            engine,
            lm,
            lm_digest,
            confusables: ConfusableSet::english_default(),
            min_token_length: DEFAULT_CONTEXT_MIN_TOKEN_LENGTH,
            max_edit_distance,
            max_candidates: DEFAULT_MAX_CANDIDATES,
            max_beam_width: DEFAULT_MAX_BEAM_WIDTH,
            edit_weight: DEFAULT_EDIT_WEIGHT,
            check_real_word_edits: false,
            skip_numbers: true,
            skip_urls: true,
            keep_unknown_proper_nouns: false,
            use_protected_spans: true,
        }
    }

    pub fn engine(&self) -> &Arc<SymSpell> {
        &self.engine
    }

    pub fn language_model(&self) -> &Arc<NGramLanguageModel> {
        &self.lm
    }

    pub fn confusables(&self) -> &ConfusableSet {
        &self.confusables
    }

    /// Replaces the real-word confusable set.
    pub fn with_confusables(mut self, confusables: ConfusableSet) -> ContextSpellCorrector {
        self.confusables = confusables;
        self
    }

    /// Sets the minimum token length (in code points) eligible for correction.
    pub fn min_token_length(mut self, len: usize) -> ContextSpellCorrector {
        self.min_token_length = len.max(1);
        self
    }

    /// Sets the maximum edit distance queried against [`SymSpell`].
    pub fn max_edit_distance(mut self, distance: usize) -> Result<ContextSpellCorrector, LookupError> {
        self.engine.check(distance)?;
        self.max_edit_distance = distance;
        Ok(self)
    }

    /// Sets the maximum number of candidates retained per token in the trellis.
    pub fn max_candidates(mut self, count: usize) -> ContextSpellCorrector {
        self.max_candidates = count.max(1);
        self
    }

    /// Sets the maximum Viterbi beam width retained per trellis step.
    pub fn max_beam_width(mut self, width: usize) -> ContextSpellCorrector {
        self.max_beam_width = width.max(1);
        self
    }

    /// Alias for [`Self::max_beam_width`].
    pub fn with_max_beam_width(self, width: usize) -> ContextSpellCorrector {
        self.max_beam_width(width)
    }

    /// Sets the edit-distance weight $\lambda \ge 0$ in the Viterbi cost.
    pub fn edit_weight(mut self, weight: f64) -> ContextSpellCorrector {
        self.edit_weight = if weight.is_finite() && weight >= 0.0 { weight } else { DEFAULT_EDIT_WEIGHT };
        self
    }

    /// When enabled, also queries [`SymSpell`] for edit-distance neighbors of
    /// words that are already in the dictionary, in addition to [`ConfusableSet`].
    pub fn with_real_word_edits(mut self, enabled: bool) -> ContextSpellCorrector {
        self.check_real_word_edits = enabled;
        self
    }

    pub fn skip_numbers(mut self, skip: bool) -> ContextSpellCorrector {
        self.skip_numbers = skip;
        self
    }

    pub fn skip_urls(mut self, skip: bool) -> ContextSpellCorrector {
        self.skip_urls = skip;
        self
    }

    pub fn keep_unknown_proper_nouns(mut self, keep: bool) -> ContextSpellCorrector {
        self.keep_unknown_proper_nouns = keep;
        self
    }

    /// Controls whether the [`Annotator`] implementation automatically skips
    /// tokens overlapping protected span layers.
    pub fn use_protected_spans(mut self, enabled: bool) -> ContextSpellCorrector {
        self.use_protected_spans = enabled;
        self
    }

    fn options_hash(&self) -> u64 {
        hash_config(&(
            (
                self.min_token_length,
                self.max_edit_distance,
                self.max_candidates,
                self.max_beam_width,
                self.edit_weight.to_bits(),
                self.check_real_word_edits,
                self.skip_numbers,
                self.skip_urls,
            ),
            (
                self.keep_unknown_proper_nouns,
                self.use_protected_spans,
                self.engine.fingerprint(),
                self.confusables.fingerprint(),
                self.lm_digest,
            ),
        ))
    }

    fn run_sentences(
        &self,
        text: &str,
        out: &mut ViewBuilder<'_>,
        protected: Option<&[Span]>,
        sentence_spans: Option<&[Span]>,
        report: &mut dyn FnMut(Span, &str, usize),
    ) {
        let all_tokens = collect_tokens(text, protected);
        if all_tokens.is_empty() {
            return;
        }
        let sentences = group_into_sentences(text, &all_tokens, sentence_spans);
        let mut cased = String::new();
        for sent in sentences {
            let cand_sets: Vec<Vec<Candidate>> = sent.iter().map(|t| self.candidates_for(text, t)).collect();
            let decoded = viterbi_decode(&self.lm, self.edit_weight, self.max_beam_width, cand_sets);
            for (tok, cand) in sent.iter().zip(decoded.iter()) {
                if cand.is_original {
                    continue;
                }
                let original_core = &text[tok.core.clone()];
                cased.clear();
                casing::apply(original_core, &cand.word, &mut cased);
                if cased != original_core {
                    out.keep(tok.core.start);
                    out.replace(tok.core.end, &cased);
                    report(Span::new(tok.core.start as u32, tok.core.end as u32), &cased, cand.report_distance);
                }
            }
        }
    }

    fn candidates_for(&self, text: &str, tok: &TokenSlot) -> Vec<Candidate> {
        let core_str = &text[tok.core.clone()];
        let lower = to_lowercase(core_str);
        let original = Candidate { word: lower.clone(), trellis_edit_cost: 0.0, report_distance: 0, is_original: true };
        if tok.protected || !self.should_correct(core_str) {
            return vec![original];
        }

        let in_dict = self.engine.contains(&lower);
        let in_lm = self.lm.contains(&[lower.as_str()]);
        let known = in_dict || in_lm || !self.confusables.alternatives(&lower).is_empty();
        if self.keep_unknown_proper_nouns && !known && core_str.chars().next().is_some_and(casing::is_upper) {
            return vec![original];
        }

        let mut out: Vec<Candidate> = Vec::new();
        if known {
            out.push(original.clone());
        }
        if out.len() >= self.max_candidates {
            return out;
        }

        let metric = self.engine.config().metric();
        let mut confusable_cands: Vec<(usize, i32, u64, Candidate)> = Vec::new();
        for alt in self.confusables.alternatives(&lower) {
            if (self.engine.contains(alt) || self.lm.contains(&[alt.as_str()])) && !out.iter().any(|c| c.word == *alt) {
                let dist = metric.distance(&lower, alt, 255).unwrap_or(1).max(1);
                let lm_cnt = self.lm.get_count(&[alt.as_str()]);
                let dict_cnt = self.engine.count(alt).unwrap_or(0);
                confusable_cands.push((
                    dist,
                    lm_cnt,
                    dict_cnt,
                    Candidate { word: alt.clone(), trellis_edit_cost: 1.0, report_distance: dist, is_original: false },
                ));
            }
        }
        confusable_cands.sort_by(|a, b| {
            a.0.cmp(&b.0).then_with(|| b.1.cmp(&a.1)).then_with(|| b.2.cmp(&a.2)).then_with(|| a.3.word.cmp(&b.3.word))
        });
        for (_, _, _, cand) in confusable_cands {
            if out.len() >= self.max_candidates {
                break;
            }
            out.push(cand);
        }

        if (!in_dict || self.check_real_word_edits) && out.len() < self.max_candidates {
            let suggestions = self.engine.lookup_within(&lower, Verbosity::All, self.max_edit_distance);
            for s in suggestions {
                if out.len() >= self.max_candidates {
                    break;
                }
                if !out.iter().any(|c| c.word == s.term) {
                    let is_orig = s.term == lower;
                    out.push(Candidate {
                        word: s.term.to_owned(),
                        trellis_edit_cost: if is_orig { 0.0 } else { s.edit_distance as f64 },
                        report_distance: s.edit_distance,
                        is_original: is_orig,
                    });
                }
            }
        }

        if out.is_empty() {
            out.push(original);
        }
        out.truncate(self.max_candidates);
        out
    }

    fn should_correct(&self, core_str: &str) -> bool {
        if core_str.chars().count() < self.min_token_length {
            return false;
        }
        if self.skip_urls && is_url_like(core_str) {
            return false;
        }
        if self.skip_numbers && is_number_like(core_str) {
            return false;
        }
        core_str.chars().any(java::is_letter)
    }
}

impl Normalizer for ContextSpellCorrector {
    const ID: &'static str = "turbo:normalize:context-spell-correct";
    const VERSION: &'static str = "2";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        self.run_sentences(text, out, None, None, &mut |_, _, _| {});
    }

    fn config_hash(&self) -> u64 {
        self.options_hash()
    }
}

impl Annotator for ContextSpellCorrector {
    const ID: &'static str = <ContextSpellCorrector as Normalizer>::ID;
    const VERSION: &'static str = <ContextSpellCorrector as Normalizer>::VERSION;
    type Reads = (
        View,
        Option<ProtectedSpans>,
        Option<Placeholders>,
        Option<Pii>,
        Option<Temporals>,
        Option<Money>,
        Option<Quantities>,
        Option<Entities>,
        Option<Abbreviations>,
        Option<Sentences>,
    );
    type Writes = (View, Corrections);

    fn annotate(
        &self,
        (src, protected, placeholders, pii, temporals, money, quantities, entities, abbreviations, sentences): Refs<
            '_,
            Self,
        >,
    ) -> Result<(View, Corrections), Error> {
        let skip = if self.use_protected_spans {
            Some(collect_protected_spans(
                <Self as Annotator>::ID,
                src,
                &ProtectedInputs {
                    protected,
                    placeholders,
                    pii,
                    temporals,
                    money,
                    quantities,
                    entities,
                    abbreviations,
                },
            )?)
        } else {
            None
        };
        let sent_spans = sentences
            .map(|s| {
                let invalid = |detail| Error::Annotator { annotator: <Self as Annotator>::ID, detail };
                if s.spans.view() != src.id() {
                    return Err(invalid(format!(
                        "sentences are in view {}, text in view {}",
                        s.spans.view().0,
                        src.id().0
                    )));
                }
                s.spans.check(Overlap::None).map_err(invalid)?;
                let mut spans = Vec::with_capacity(s.spans.len());
                for (ix, span) in s.spans.iter() {
                    if s.spans.is_discontinuous(ix) || src.text().get(span.range()).is_none() {
                        return Err(invalid(format!(
                            "sentence {ix:?} must be a contiguous UTF-8 span within the view"
                        )));
                    }
                    spans.push(span);
                }
                Ok(spans)
            })
            .transpose()?;
        let text = src.text();
        let mut out = ViewBuilder::new(text);
        let mut corrections = Corrections::new(src.id());
        self.run_sentences(text, &mut out, skip.as_deref(), sent_spans.as_deref(), &mut |s, with, distance| {
            corrections.spans.push(s);
            corrections.suggestion.push(with);
            corrections.edit_distance.push(u8::try_from(distance).unwrap_or(u8::MAX));
        });
        let (normalized, alignment) = out.finish();
        Ok((View::derived(src, normalized, alignment), corrections))
    }

    fn config_hash(&self) -> u64 {
        self.options_hash()
    }
}
