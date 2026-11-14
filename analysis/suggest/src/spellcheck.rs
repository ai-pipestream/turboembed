//! Context-aware query spellchecker ("Did you mean?", issue #2341).
//!
//! Combines bounded Damerau-OSA candidate generation (`SymSpell` +
//! `ConfusableSet` + script/grapheme guards from `turbo-unicode`) with
//! `turbo-ngram` token-bigram/trigram Stupid Backoff Viterbi decoding.
//! Preserves quoted phrases (`"..."`), boolean operators (`AND`, `OR`, `NOT`,
//! `+`, `-`), fielded clauses (`field:value`), numbers, URLs, and caller
//! protected spans, while projecting exact UTF-8, UTF-16, and code-point
//! spans for every replacement token.

use std::ops::Range;
use std::sync::Arc;

use turbo_analysis::{CodePointSpan, OffsetTable, Span, Utf16Span};
use turbo_ngram::NGramLanguageModel;
use turbo_normalize::spellcheck::{ConfusableSet, SymSpell, SymSpellConfig, Verbosity};
use turbo_unicode::script::{Script, script};

use crate::fold::canonical_surface_lowercase;

const STUPID_BACKOFF_ALPHA: f64 = 0.4;
const UNSEEN_FLOOR: f64 = 1e-9;

/// Tuning parameters for [`QuerySpellchecker`].
#[derive(Clone, Debug)]
pub struct SpellcheckConfig {
    pub max_edit_distance: usize,
    pub max_candidates: usize,
    pub max_beam_width: usize,
    pub edit_weight: f64,
    pub real_word_penalty: f64,
    pub min_token_chars: usize,
    pub min_margin: f64,
}

impl Default for SpellcheckConfig {
    fn default() -> Self {
        SpellcheckConfig {
            max_edit_distance: 2,
            max_candidates: 10,
            max_beam_width: 32,
            edit_weight: 2.0,
            real_word_penalty: 0.75,
            min_token_chars: 3,
            min_margin: 0.25,
        }
    }
}

/// One suggested replacement token with exact source spans in UTF-8 bytes,
/// UTF-16 code units, and Unicode scalar values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpellEdit {
    pub utf8_span: Span,
    pub utf16_span: Utf16Span,
    pub code_point_span: CodePointSpan,
    pub original: String,
    pub replacement: String,
    pub edit_distance: u32,
}

/// Output of [`QuerySpellchecker::check_query`].
#[derive(Clone, Debug, PartialEq)]
pub struct QuerySpellResult {
    pub corrected_query: String,
    pub did_change: bool,
    pub confidence: f64,
    pub edits: Vec<SpellEdit>,
}

/// Multi-word context-aware search query spellchecker.
#[derive(Clone, Debug)]
pub struct QuerySpellchecker {
    symspell: Arc<SymSpell>,
    lm: Arc<NGramLanguageModel>,
    confusables: ConfusableSet,
    config: SpellcheckConfig,
}

impl QuerySpellchecker {
    /// Constructs a query spellchecker from prebuilt [`SymSpell`] and
    /// [`NGramLanguageModel`] tables.
    pub fn new(symspell: Arc<SymSpell>, lm: Arc<NGramLanguageModel>, config: SpellcheckConfig) -> Self {
        QuerySpellchecker { symspell, lm, confusables: ConfusableSet::english_default(), config }
    }

    /// Learns both [`SymSpell`] unigram frequencies and a token-bigram/trigram
    /// [`NGramLanguageModel`] directly from a slice of corpus documents.
    pub fn from_documents<S: AsRef<str>>(docs: &[S], config: SpellcheckConfig) -> Self {
        let sym_cfg = SymSpellConfig::builder()
            .max_dictionary_edit_distance(config.max_edit_distance.max(1))
            .prefix_length(7)
            .count_threshold(1)
            .build()
            .expect("valid symspell config");
        let mut sym = SymSpell::new(sym_cfg);
        let mut lm = NGramLanguageModel::new(3).expect("order 3");
        for doc in docs {
            let mut cur_sent: Vec<String> = Vec::new();
            for piece in doc.as_ref().split(['.', '!', '?', '\n', '\r', ';']) {
                cur_sent.clear();
                for raw in piece.split_whitespace() {
                    let trimmed = raw.trim_matches(|c: char| !c.is_alphabetic() && !c.is_ascii_digit() && c != '\'');
                    if trimmed.is_empty() {
                        continue;
                    }
                    let w = canonical_surface_lowercase(trimmed);
                    if w.chars().any(char::is_alphabetic) {
                        sym.add(&w, 1);
                        cur_sent.push(w);
                    }
                }
                if !cur_sent.is_empty() {
                    for pair in cur_sent.windows(2) {
                        sym.add_bigram(&pair[0], &pair[1], 1);
                    }
                    let _ = lm.add(&cur_sent);
                }
            }
        }
        Self::new(Arc::new(sym), Arc::new(lm), config)
    }

    /// Builds a [`QuerySpellchecker`] from `# turbo-vocabulary 1` TSV bytes
    /// produced by `TurboVocabularyService::LearnVocabulary`.
    pub fn from_vocabulary_tsv(bytes: &[u8], config: SpellcheckConfig) -> Result<Self, String> {
        let text = std::str::from_utf8(bytes).map_err(|e| format!("invalid UTF-8 vocabulary TSV: {e}"))?;
        let sym_cfg = SymSpellConfig::builder()
            .max_dictionary_edit_distance(config.max_edit_distance.max(1))
            .prefix_length(7)
            .count_threshold(1)
            .build()
            .map_err(|e| e.to_string())?;
        let mut sym = SymSpell::new(sym_cfg);
        let mut lm = NGramLanguageModel::new(3).map_err(|e| e.to_string())?;
        for line in text.lines() {
            if line.is_empty() || line.starts_with('#') || line.starts_with("term\tkind\t") {
                continue;
            }
            let cols: Vec<&str> = line.split('\t').collect();
            if cols.len() < 4 {
                continue;
            }
            let term = cols[0];
            let kind = cols[1];
            let tf: u64 = cols[2].parse().unwrap_or(0);
            if tf == 0 {
                continue;
            }
            let clamped = i32::try_from(tf.min(i32::MAX as u64)).unwrap_or(i32::MAX);
            match kind {
                "term" => {
                    sym.add(term, tf);
                    let single = [term];
                    let _ = lm.model_mut().add(&single);
                    let _ = lm.model_mut().set_count(&single, clamped);
                }
                "ngram" => {
                    let parts: Vec<&str> = term.split(' ').filter(|s| !s.is_empty()).collect();
                    if parts.len() == 2 {
                        sym.add_bigram(parts[0], parts[1], tf);
                    }
                    if !parts.is_empty() {
                        let _ = lm.model_mut().add(&parts);
                        let _ = lm.model_mut().set_count(&parts, clamped);
                    }
                }
                _ => {}
            }
        }
        Ok(Self::new(Arc::new(sym), Arc::new(lm), config))
    }

    /// Evaluates `query` and returns contextually coherent spellcheck edits,
    /// respecting query quotes, boolean operators, field prefixes, and
    /// `protected_spans`.
    pub fn check_query(&self, query: &str, protected_spans: &[Span]) -> QuerySpellResult {
        let slots = parse_query_slots(query, protected_spans);
        if slots.is_empty() {
            return QuerySpellResult {
                corrected_query: query.to_owned(),
                did_change: false,
                confidence: 1.0,
                edits: Vec::new(),
            };
        }
        let cands: Vec<Vec<Cand>> = slots.iter().map(|slot| self.slot_candidates(query, slot)).collect();
        let (chosen, best_cost, orig_cost) = self.decode_trellis(&cands);
        let gain = orig_cost - best_cost;
        if gain < self.config.min_margin || chosen.iter().all(|c| c.is_original) {
            return QuerySpellResult {
                corrected_query: query.to_owned(),
                did_change: false,
                confidence: 1.0,
                edits: Vec::new(),
            };
        }

        let offsets = OffsetTable::build(query);
        let mut corrected = String::with_capacity(query.len() + 8);
        let mut cursor = 0usize;
        let mut edits = Vec::new();
        for (slot, cand) in slots.iter().zip(chosen.iter()) {
            if cand.is_original {
                continue;
            }
            let orig_slice = &query[slot.core.clone()];
            let replacement = apply_casing(orig_slice, &cand.word);
            if replacement == orig_slice {
                continue;
            }
            corrected.push_str(&query[cursor..slot.core.start]);
            corrected.push_str(&replacement);
            cursor = slot.core.end;
            let u8_span = Span::new(slot.core.start as u32, slot.core.end as u32);
            let u16_span = offsets.span_to_utf16_typed(query, u8_span);
            let cp_span = offsets.span_to_code_point_typed(query, u8_span);
            edits.push(SpellEdit {
                utf8_span: u8_span,
                utf16_span: u16_span,
                code_point_span: cp_span,
                original: orig_slice.to_owned(),
                replacement,
                edit_distance: cand.distance as u32,
            });
        }
        if edits.is_empty() {
            return QuerySpellResult { corrected_query: query.to_owned(), did_change: false, confidence: 1.0, edits };
        }
        corrected.push_str(&query[cursor..]);
        let confidence = (1.0 / (1.0 + (-gain).exp())).clamp(0.0, 1.0);
        QuerySpellResult { corrected_query: corrected, did_change: true, confidence, edits }
    }

    fn slot_candidates(&self, query: &str, slot: &QuerySlot) -> Vec<Cand> {
        let orig = &query[slot.core.clone()];
        let lower = canonical_surface_lowercase(orig);
        let in_vocab = self.symspell.contains(&lower) || self.lm.get_count(&[lower.as_str()]) > 0;
        let orig_cand = Cand { word: lower.clone(), edit_cost: 0.0, distance: 0, is_original: true };
        if slot.protected || lower.chars().count() < self.config.min_token_chars {
            return vec![orig_cand];
        }
        let orig_script = dominant_script(&lower);
        let max_d = if in_vocab { 1.min(self.config.max_edit_distance) } else { self.config.max_edit_distance };

        let mut out = Vec::with_capacity(self.config.max_candidates + 1);
        out.push(orig_cand);
        if let Ok(suggs) = self.symspell.lookup(&lower, Verbosity::All, max_d) {
            for s in suggs {
                if s.term == lower || dominant_script(s.term) != orig_script {
                    continue;
                }
                let extra = if in_vocab { self.config.real_word_penalty } else { 0.0 };
                let cost = self.config.edit_weight * (s.edit_distance as f64) + extra;
                out.push(Cand {
                    word: s.term.to_owned(),
                    edit_cost: cost,
                    distance: s.edit_distance,
                    is_original: false,
                });
                if out.len() > self.config.max_candidates {
                    break;
                }
            }
        }
        for alt in self.confusables.alternatives(&lower) {
            if *alt != lower && !out.iter().any(|c| c.word == *alt) {
                out.push(Cand {
                    word: alt.clone(),
                    edit_cost: self.config.edit_weight * 0.8 + self.config.real_word_penalty,
                    distance: 1,
                    is_original: false,
                });
            }
        }
        out
    }

    fn token_neg_log_prob(&self, history: &[&str], word: &str) -> f64 {
        let order = self.lm.order().max(1);
        let start = history.len().saturating_sub(order - 1);
        let mut seq: Vec<&str> = Vec::with_capacity(order);
        seq.extend_from_slice(&history[start..]);
        seq.push(word);
        let p = self.stupid_backoff_prob(&seq).max(UNSEEN_FLOOR);
        -p.ln()
    }

    fn stupid_backoff_prob(&self, seq: &[&str]) -> f64 {
        let total = (self.lm.size() as f64).max(1.0);
        let count = self.lm.get_count(seq) as f64;
        if seq.len() > 1 {
            let prefix = &seq[..seq.len() - 1];
            let denom = self.lm.get_count(prefix) as f64;
            if count > 0.0 && denom > 0.0 {
                count / denom
            } else {
                STUPID_BACKOFF_ALPHA * self.stupid_backoff_prob(&seq[1..])
            }
        } else if count > 0.0 {
            count / total
        } else if let Some(sym_c) = self.symspell.count(seq[0]) {
            (sym_c as f64) / (self.symspell.corpus_word_count() as f64)
        } else {
            UNSEEN_FLOOR
        }
    }

    fn decode_trellis(&self, cands: &[Vec<Cand>]) -> (Vec<Cand>, f64, f64) {
        #[derive(Clone)]
        struct BeamPath {
            chosen: Vec<usize>,
            cost: f64,
        }
        let mut beam = vec![BeamPath { chosen: Vec::with_capacity(cands.len()), cost: 0.0 }];
        for (step, step_cands) in cands.iter().enumerate() {
            let mut next = Vec::with_capacity(beam.len() * step_cands.len());
            for path in &beam {
                let hist: Vec<&str> =
                    path.chosen.iter().enumerate().map(|(i, &ci)| cands[i][ci].word.as_str()).collect();
                for (ci, cand) in step_cands.iter().enumerate() {
                    let lm_cost = self.token_neg_log_prob(&hist, &cand.word);
                    let mut chosen = path.chosen.clone();
                    chosen.push(ci);
                    next.push(BeamPath { chosen, cost: path.cost + cand.edit_cost + lm_cost });
                }
            }
            next.sort_by(|a, b| a.cost.total_cmp(&b.cost));
            next.truncate(self.config.max_beam_width);
            beam = next;
            let _ = step;
        }
        let orig_words: Vec<&str> = cands.iter().map(|cs| cs[0].word.as_str()).collect();
        let mut orig_cost = 0.0;
        for i in 0..orig_words.len() {
            orig_cost += self.token_neg_log_prob(&orig_words[..i], orig_words[i]);
        }
        let best = &beam[0];
        let decoded: Vec<Cand> = best.chosen.iter().enumerate().map(|(i, &ci)| cands[i][ci].clone()).collect();
        (decoded, best.cost, orig_cost)
    }
}

struct QuerySlot {
    core: Range<usize>,
    protected: bool,
}

fn parse_query_slots(query: &str, protected_spans: &[Span]) -> Vec<QuerySlot> {
    let mut slots = Vec::new();
    let bytes = query.as_bytes();
    let mut i = 0usize;
    let mut in_quotes = false;
    while i < bytes.len() {
        let ch = query[i..].chars().next().unwrap();
        if ch == '"' {
            in_quotes = !in_quotes;
            i += 1;
            continue;
        }
        if ch.is_whitespace() {
            i += ch.len_utf8();
            continue;
        }
        let tok_start = i;
        while i < bytes.len() {
            let c = query[i..].chars().next().unwrap();
            if c.is_whitespace() || c == '"' {
                break;
            }
            i += c.len_utf8();
        }
        let raw = &query[tok_start..i];
        let is_operator = matches!(raw, "AND" | "OR" | "NOT");
        let is_fielded_or_url = raw.contains(':') || raw.contains('/') || raw.contains('@');
        let mut core_rel_start = 0usize;
        for c in raw.chars() {
            if matches!(c, '+' | '-' | '(' | '[') || (!c.is_alphabetic() && !c.is_ascii_digit()) {
                core_rel_start += c.len_utf8();
            } else {
                break;
            }
        }
        let mut core_rel_end = raw.len();
        for c in raw[core_rel_start..].chars().rev() {
            if !c.is_alphabetic() && !c.is_ascii_digit() {
                core_rel_end -= c.len_utf8();
            } else {
                break;
            }
        }
        if core_rel_start >= core_rel_end {
            continue;
        }
        let core = (tok_start + core_rel_start)..(tok_start + core_rel_end);
        let core_str = &query[core.clone()];
        let has_digit = core_str.chars().any(|c| c.is_numeric());
        let overlaps_protected = protected_spans
            .iter()
            .any(|p| !p.is_empty() && (p.start as usize) < core.end && (p.end as usize) > core.start);
        let protected = in_quotes || is_operator || is_fielded_or_url || has_digit || overlaps_protected;
        slots.push(QuerySlot { core, protected });
    }
    slots
}

#[derive(Clone)]
struct Cand {
    word: String,
    edit_cost: f64,
    distance: usize,
    is_original: bool,
}

fn dominant_script(word: &str) -> Script {
    for ch in word.chars() {
        let sc = script(ch);
        if sc != Script::Common && sc != Script::Inherited {
            return sc;
        }
    }
    Script::Common
}

fn apply_casing(orig: &str, lower_replacement: &str) -> String {
    let mut chars = orig.chars();
    let Some(first) = chars.next() else {
        return lower_replacement.to_owned();
    };
    if orig.chars().all(char::is_uppercase) && orig.chars().any(char::is_alphabetic) {
        return lower_replacement.to_uppercase();
    }
    if first.is_uppercase() && chars.all(|c| !c.is_uppercase()) {
        let mut r_chars = lower_replacement.chars();
        if let Some(rf) = r_chars.next() {
            let mut out = rf.to_uppercase().collect::<String>();
            out.push_str(r_chars.as_str());
            return out;
        }
    }
    lower_replacement.to_owned()
}
