//! BIO tag decoding, clause segmentation, negation detection, and opinion-to-aspect
//! linking helpers for [`super::LearnedAspectSentiment`].

use turbo_analysis::{Ix, IxRange, Layer, Span};
use turbo_analysis_layers::{Dependencies, PhraseChunks, Subwords, Tokens};

use turbo_doccat::sentiment::aspect::shared::{self, LinkPolicy, LinkedSpan};

use crate::window::pieces_of;

/// Clause-boundary adversatives for the learned decoder.
/// Deliberately narrower than the rule extractor's
/// `ASPECT_ADVERSATIVES` (no `even`/`despite`/`sadly` family),
/// matching what this decoder has always split on; `and`, `;`,
/// and `,` always split in the shared implementation.
const DECODE_ADVERSATIVES: &[&str] = &["but", "however", "yet", "although", "though", "whereas", "while", "except"];

pub(super) const TAG_O: u16 = 0;
pub(super) const TAG_B_ASP: u16 = 1;
pub(super) const TAG_I_ASP: u16 = 2;
pub(super) const TAG_B_OP: u16 = 3;
pub(super) const TAG_I_OP: u16 = 4;

#[derive(Clone, Debug)]
pub(super) struct DecodedOpinion {
    pub(super) tokens: IxRange<Tokens>,
    pub(super) head_token: usize,
    pub(super) polarity: f64,
    pub(super) intensity: f64,
    pub(super) negated: bool,
    pub(super) pos_prob: f64,
    pub(super) neg_prob: f64,
    pub(super) neu_prob: f64,
    pub(super) clause_id: usize,
}

#[derive(Clone, Debug)]
pub(super) struct DecodedAspect {
    pub(super) tokens: IxRange<Tokens>,
    pub(super) head_token: usize,
    pub(super) clause_id: usize,
}

pub(super) fn unit_l2(v: &[f32]) -> Vec<f32> {
    let norm_sq: f64 = v.iter().map(|&x| (x as f64) * (x as f64)).sum();
    if norm_sq <= 1e-24 {
        return v.to_vec();
    }
    let inv = (1.0 / norm_sq.sqrt()) as f32;
    v.iter().map(|&x| x * inv).collect()
}

/// Maps each token in `tokens` to its overlapping subword piece range
/// `start..end` in `subwords`.
pub(super) fn token_subword_ranges(tokens: &Tokens, subwords: &Subwords) -> Vec<std::ops::Range<u32>> {
    let n_tokens = tokens.len();
    if let Some(tok_col) = &subwords.token {
        let mut out = Vec::with_capacity(n_tokens);
        for t in 0..n_tokens {
            let target = Ix::<Tokens>::new(t as u32);
            let a = tok_col.partition_point(|&x| x < target);
            let b = a + tok_col[a..].partition_point(|&x| x == target);
            out.push(a as u32..b as u32);
        }
        return out;
    }
    let (sub_starts, sub_ends) = subwords.spans.columns();
    let (tok_starts, tok_ends) = tokens.spans.columns();
    let mut out = Vec::with_capacity(n_tokens);
    for t in 0..n_tokens {
        out.push(pieces_of(sub_starts, sub_ends, tok_starts[t], tok_ends[t]));
    }
    out
}

pub(super) fn project_subword_bio_to_tokens(
    tokens: &Tokens,
    subwords: &Subwords,
    piece_probs: &[[f64; 5]],
) -> Vec<u16> {
    let n_tokens = tokens.len();
    let tok_pieces = token_subword_ranges(tokens, subwords);
    let mut tok_tags = vec![TAG_O; n_tokens];
    for t in 0..n_tokens {
        let r = tok_pieces[t].clone();
        if r.is_empty() {
            tok_tags[t] = TAG_O;
            continue;
        }
        let first = r.start as usize;
        let mut avg = piece_probs[first];
        let count = (r.end - r.start) as f64;
        if count > 1.0 {
            for p in (r.start + 1)..r.end {
                let pp = &piece_probs[p as usize];
                for k in 0..5 {
                    avg[k] += pp[k];
                }
            }
            for val in &mut avg {
                *val /= count;
            }
        }
        let mut best_k = 0usize;
        for k in 1..5 {
            if avg[k] > avg[best_k] {
                best_k = k;
            }
        }
        tok_tags[t] = best_k as u16;
    }
    tok_tags
}

pub(super) fn token_range_span(range: IxRange<Tokens>, tokens: &Tokens) -> Span {
    let first = tokens.spans.get(range.start);
    let last = tokens.spans.get(Ix::new(range.end.get() as u32 - 1));
    Span::new(first.start, last.end)
}

pub(super) fn decode_bio_spans(
    tok_tags: &[u16],
    token_lower: &[String],
    clause_ids: &[usize],
    sent_start: usize,
    sent_end: usize,
    phrase_chunks: Option<&PhraseChunks>,
) -> (Vec<IxRange<Tokens>>, Vec<IxRange<Tokens>>) {
    let mut aspects: Vec<IxRange<Tokens>> = Vec::new();
    let mut opinions: Vec<IxRange<Tokens>> = Vec::new();

    let mut t = sent_start;
    while t < sent_end {
        let tag = tok_tags[t];
        if tag == TAG_B_ASP || tag == TAG_I_ASP {
            let start = t;
            let clause = clause_ids[t - sent_start];
            t += 1;
            while t < sent_end
                && (tok_tags[t] == TAG_I_ASP || tok_tags[t] == TAG_B_ASP)
                && clause_ids[t - sent_start] == clause
                && !token_lower[t].is_empty()
            {
                t += 1;
            }
            // Trim leading determiners or empty punctuation tokens if any were tagged.
            let mut s = start;
            while s < t && (token_lower[s].is_empty() || is_determiner(&token_lower[s])) {
                s += 1;
            }
            let mut e = t;
            while e > s && token_lower[e - 1].is_empty() {
                e -= 1;
            }
            if s < e {
                aspects.push(IxRange::new(s as u32, e as u32));
            }
        } else if tag == TAG_B_OP || tag == TAG_I_OP {
            let start = t;
            let clause = clause_ids[t - sent_start];
            t += 1;
            while t < sent_end
                && (tok_tags[t] == TAG_I_OP || tok_tags[t] == TAG_B_OP)
                && clause_ids[t - sent_start] == clause
                && !token_lower[t].is_empty()
            {
                t += 1;
            }
            let mut s = start;
            while s < t && token_lower[s].is_empty() {
                s += 1;
            }
            let mut e = t;
            while e > s && token_lower[e - 1].is_empty() {
                e -= 1;
            }
            if s < e {
                opinions.push(IxRange::new(s as u32, e as u32));
            }
        } else {
            t += 1;
        }
    }

    // If `PhraseChunks` (`opennlp:chunks`) is present, expand any single-token aspect inside an `NP`
    // chunk to include adjacent noun-compound tokens in that `NP` that are not tagged as opinions.
    if let Some(chunks) = phrase_chunks {
        let n_tokens = token_lower.len();
        for asp in &mut aspects {
            for (c_idx, crange) in chunks.tokens.iter().enumerate() {
                if !chunks.kinds.label(chunks.kind[c_idx]).eq_ignore_ascii_case("NP") {
                    continue;
                }
                if crange.end.get() > n_tokens {
                    continue;
                }
                if crange.start.get() <= asp.start.get() && asp.end.get() <= crange.end.get() {
                    let mut s = crange.start.get();
                    while s < asp.start.get()
                        && (token_lower[s].is_empty()
                            || is_determiner(&token_lower[s])
                            || opinions.iter().any(|o| o.start.get() <= s && s < o.end.get()))
                    {
                        s += 1;
                    }
                    if s < asp.start.get() {
                        asp.start = Ix::new(s as u32);
                    }
                    break;
                }
            }
        }
    }

    (aspects, opinions)
}

fn is_determiner(w: &str) -> bool {
    matches!(
        w,
        "the"
            | "a"
            | "an"
            | "this"
            | "that"
            | "these"
            | "those"
            | "my"
            | "our"
            | "your"
            | "his"
            | "her"
            | "its"
            | "their"
    )
}

pub(super) fn assign_clauses(
    token_lower: &[String],
    text: &str,
    tok_starts: &[u32],
    tok_ends: &[u32],
    sent_start: usize,
    sent_end: usize,
) -> Vec<usize> {
    shared::assign_clauses(token_lower, text, tok_starts, tok_ends, sent_start, sent_end, DECODE_ADVERSATIVES)
}

pub(super) fn is_token_negated(
    tok: usize,
    sent_start: usize,
    sent_end: usize,
    token_lower: &[String],
    clause_ids: &[usize],
    deps: Option<&Dependencies>,
) -> bool {
    shared::is_token_negated(tok, sent_start, sent_end, token_lower, clause_ids, deps, is_negation_word)
}

fn is_negation_word(w: &str) -> bool {
    matches!(
        w,
        "not"
            | "no"
            | "never"
            | "neither"
            | "nor"
            | "none"
            | "hardly"
            | "barely"
            | "scarcely"
            | "without"
            | "cannot"
            | "cant"
    ) || w.ends_with("n't")
}

impl LinkedSpan for DecodedAspect {
    fn tokens(&self) -> IxRange<Tokens> {
        self.tokens
    }
    fn head_token(&self) -> usize {
        self.head_token
    }
    fn clause_id(&self) -> usize {
        self.clause_id
    }
}

impl LinkedSpan for DecodedOpinion {
    fn tokens(&self) -> IxRange<Tokens> {
        self.tokens
    }
    fn head_token(&self) -> usize {
        self.head_token
    }
    fn clause_id(&self) -> usize {
        self.clause_id
    }
}

pub(super) fn link_decoded_opinions(
    asp: &DecodedAspect,
    asp_pos: usize,
    sent_aspects: &[DecodedAspect],
    sent_opinions: &[DecodedOpinion],
    deps: Option<&Dependencies>,
    sent_end: usize,
) -> Vec<usize> {
    // No adversative veto and no conjunction chains: the plain
    // shared-head dependency link this decoder has always used.
    // The empty token slice is never read under that policy.
    shared::link_opinions_to_aspect(
        asp,
        asp_pos,
        sent_aspects,
        sent_opinions,
        deps,
        &[],
        sent_end,
        LinkPolicy { adversatives: &[], conjunction_chains: false },
    )
}
