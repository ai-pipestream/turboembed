//! The attribution sieves of [`QuotationAnnotator`](super::QuotationAnnotator).

use turbo_analysis::{Groups, Ix, Layer, Span};
use turbo_analysis_layers::{Dependencies, Quotation, QuoteKind};

use super::QuotationConfig;
use crate::layer::Mentions;
use crate::quotes::{QuoteRegions, speech_verb};

pub(super) struct Input<'a> {
    pub spans: &'a [Span],
    pub forms: &'a [&'a str],
    pub deps: Option<&'a Dependencies>,
    pub mentions: Option<&'a Mentions>,
    pub chains: Option<&'a Groups<Mentions>>,
}

/// A found attribution: the verb token and the speaker's tokens.
#[derive(Clone, Copy)]
struct Cue {
    verb: usize,
    speaker: Option<(usize, usize)>,
}

impl Input<'_> {
    fn bytes(&self, first: usize, last: usize) -> Span {
        Span::new(self.spans[first].start, self.spans[last].end)
    }

    fn relation(&self, t: usize) -> &str {
        let d = self.deps.expect("dependencies");
        let label = d.relations.label(d.relation[t]);
        label.split(':').next().unwrap_or(label)
    }

    fn head(&self, t: usize) -> Option<usize> {
        self.deps.and_then(|d| d.head[t]).map(|h| h.get())
    }

    fn children(&self, t: usize) -> impl Iterator<Item = usize> + '_ {
        (0..self.spans.len()).filter(move |&c| self.head(c) == Some(t))
    }

    /// The first and last token of `t`'s subtree.
    fn subtree(&self, t: usize) -> (usize, usize) {
        let (mut lo, mut hi) = (t, t);
        for c in 0..self.spans.len() {
            let mut at = Some(c);
            for _ in 0..self.spans.len() {
                match at {
                    Some(a) if a == t => {
                        lo = lo.min(c);
                        hi = hi.max(c);
                        break;
                    }
                    Some(a) => at = self.head(a),
                    None => break,
                }
            }
        }
        (lo, hi)
    }

    fn word(&self, t: usize) -> bool {
        self.forms[t].chars().any(char::is_alphanumeric)
    }

    /// A speaker around token `t`: the mention covering it, else `t` with
    /// the capitalized words next to it.
    fn speaker_at(&self, t: usize) -> (usize, usize) {
        if let Some(m) = self.mention_at(t) {
            let tokens = self.mentions.expect("mentions").tokens[m];
            return (tokens.start.get(), tokens.end.get() - 1);
        }
        let cap = |i: usize| self.forms[i].chars().next().is_some_and(char::is_uppercase);
        let (mut lo, mut hi) = (t, t);
        if cap(t) {
            while lo > 0 && cap(lo - 1) {
                lo -= 1;
            }
            while hi + 1 < self.forms.len() && cap(hi + 1) {
                hi += 1;
            }
        }
        (lo, hi)
    }

    /// The shortest mention whose tokens cover `t`.
    fn mention_at(&self, t: usize) -> Option<usize> {
        let m = self.mentions?;
        (0..m.len())
            .filter(|&i| m.tokens[i].start.get() <= t && t < m.tokens[i].end.get())
            .min_by_key(|&i| (m.tokens[i].end.get() - m.tokens[i].start.get(), i))
    }

    /// The chain of the mention covering the speaker's tokens.
    fn chain_of(&self, (lo, hi): (usize, usize)) -> Option<u32> {
        let chains = self.chains?;
        let m = (lo..=hi).find_map(|t| self.mention_at(t))?;
        (0..chains.len()).find(|&g| chains.get(Ix::new(g as u32)).iter().any(|x| x.get() == m)).map(|g| g as u32)
    }

    fn subject_of(&self, verb: usize) -> Option<(usize, usize)> {
        let s = self.children(verb).find(|&c| self.relation(c) == "nsubj")?;
        Some(self.mention_at(s).map_or_else(|| self.subtree(s), |_| self.speaker_at(s)))
    }
}

pub(super) fn run(inp: &Input<'_>, config: QuotationConfig) -> Vec<Quotation> {
    let regions = QuoteRegions::scan(inp.forms);
    let mut out = Vec::new();
    for q in 0..regions.starts.len() {
        let (open, close) = (regions.starts[q], regions.ends[q]);
        let last = if inp.forms.get(close).is_some_and(|_| close > open && regions.quote_of[close] == 0) {
            close - 1
        } else {
            close
        };
        if last <= open {
            continue;
        }
        let (cue, confidence) = match by_dependency(inp, &regions, q, open + 1, last) {
            Some(c) => (Some(c), 0.9),
            None => match by_surface(inp, &regions, open, (last, close), config.max_cue_distance) {
                Some(c) => (Some(c), 0.75),
                None => (None, 0.3),
            },
        };
        let speaker = cue.and_then(|c| c.speaker);
        out.push(Quotation {
            content: inp.bytes(open + 1, last),
            speaker: speaker.map(|(a, b)| inp.bytes(a, b)),
            cue_verb: cue.map(|c| inp.bytes(c.verb, c.verb)),
            speaker_chain: speaker.and_then(|s| inp.chain_of(s)),
            kind: QuoteKind::Direct,
            confidence,
        });
    }
    if config.alternate_dialog {
        alternate(&mut out);
    }
    if config.indirect && inp.deps.is_some() {
        indirect(inp, &regions, &mut out);
    }
    out.sort_by_key(|q| (q.content.start, q.content.end));
    out
}

/// Sieve 1: a verb of speech outside quotes linked by an arc to the quote.
fn by_dependency(inp: &Input<'_>, regions: &QuoteRegions, q: usize, first: usize, last: usize) -> Option<Cue> {
    inp.deps?;
    let inside = |t: usize| regions.quote_of[t] == q + 1;
    let verb = (0..inp.spans.len())
        .filter(|&v| regions.quote_of[v] == 0 && speech_verb(inp.forms[v]))
        .filter(|&v| (first..=last).any(|c| inp.head(c) == Some(v)) || inp.head(v).is_some_and(inside))
        .min_by_key(|&v| v.abs_diff(first).min(v.abs_diff(last)))?;
    Some(Cue { verb, speaker: inp.subject_of(verb) })
}

/// Sieve 2: a verb of speech a few tokens after the closing mark or
/// before the opening one, its speaker next to it. A cue after a quote
/// that ends a sentence and leads into the next quote (`"Hi." Bob said,
/// "Hello."`) belongs to that next quote; a cue before a quote with a
/// sentence end between them belongs to an earlier one.
fn by_surface(
    inp: &Input<'_>,
    regions: &QuoteRegions,
    open: usize,
    (last, close): (usize, usize),
    max: usize,
) -> Option<Cue> {
    let n = inp.forms.len();
    let free = |t: usize| regions.quote_of[t] == 0 && !regions.is_mark(t);
    let stop = |t: usize| matches!(inp.forms[t], "." | "?" | "!");
    let after = (close + 1..n.min(close + 1 + max)).take_while(|&t| free(t)).find(|&t| speech_verb(inp.forms[t]));
    if let Some(v) = after {
        let leads_on = (v + 1..n.min(v + 2 + max))
            .find(|&t| !inp.word(t) && inp.forms[t] != ",")
            .is_some_and(|t| regions.starts.contains(&t));
        if !(stop(last) && leads_on) {
            let before = (close + 1..v).rev().find(|&t| inp.word(t));
            let next = (v + 1 < n && free(v + 1) && inp.word(v + 1)).then_some(v + 1);
            return Some(Cue { verb: v, speaker: before.or(next).map(|t| inp.speaker_at(t)) });
        }
    }
    let before = (open.saturating_sub(max)..open)
        .rev()
        .take_while(|&t| free(t) && !stop(t))
        .find(|&t| speech_verb(inp.forms[t]))?;
    let speaker =
        (0..before).rev().take_while(|&t| free(t) && !stop(t)).find(|&t| inp.word(t)).map(|t| inp.speaker_at(t));
    Some(Cue { verb: before, speaker })
}

/// Sieve 3, dialog: an unattributed quote after quotes of two different
/// chains takes the chain before last.
fn alternate(out: &mut [Quotation]) {
    for i in 2..out.len() {
        let (a, b) = (out[i - 2].speaker_chain, out[i - 1].speaker_chain);
        if out[i].cue_verb.is_none()
            && let (Some(a), Some(b)) = (a, b)
            && a != b
        {
            out[i].speaker_chain = Some(a);
            out[i].confidence = 0.5;
        }
    }
}

/// Indirect and mixed speech: the `ccomp` of a verb of speech outside
/// quotes, its leading `mark` left out; a complement wholly inside one
/// quote is that direct quote already.
fn indirect(inp: &Input<'_>, regions: &QuoteRegions, out: &mut Vec<Quotation>) {
    for v in 0..inp.spans.len() {
        if regions.quote_of[v] != 0 || !speech_verb(inp.forms[v]) {
            continue;
        }
        for c in inp.children(v).filter(|&c| inp.relation(c) == "ccomp") {
            let (mut lo, mut hi) = inp.subtree(c);
            while lo < hi && (inp.relation(lo) == "mark" || !inp.word(lo)) && !regions.is_mark(lo) {
                lo += 1;
            }
            while hi > lo && !inp.word(hi) && !regions.is_mark(hi) {
                hi -= 1;
            }
            let words: Vec<usize> = (lo..=hi).filter(|&t| inp.word(t)).collect();
            let quoted = words.iter().filter(|&&t| regions.quote_of[t] != 0).count();
            if words.is_empty() || quoted == words.len() {
                continue;
            }
            let speaker = inp.subject_of(v);
            let mixed = quoted > 0;
            out.push(Quotation {
                content: inp.bytes(lo, hi),
                speaker: speaker.map(|(a, b)| inp.bytes(a, b)),
                cue_verb: Some(inp.bytes(v, v)),
                speaker_chain: speaker.and_then(|s| inp.chain_of(s)),
                kind: if mixed { QuoteKind::Mixed } else { QuoteKind::Indirect },
                confidence: if mixed { 0.7 } else { 0.8 },
            });
            if mixed {
                let span = inp.bytes(lo, hi);
                out.retain(|q| q.kind != QuoteKind::Direct || q.content.start < span.start || q.content.end > span.end);
            }
        }
    }
}
