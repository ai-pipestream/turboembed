//! Token-stream flattening, form alignment, and sentence-list inputs
//! for the analysis benchmark scorer.

use std::collections::BTreeSet;

use super::{ConlluSentence, ConlluToken, Counts, Measure, MetricScore, SystemScores};
use crate::Result;

/// One flattened token: its row, whether it ends a sentence, and its
/// head as a flattened index (`Some(None)` for a root, `None` when the
/// head field is missing or points outside its sentence).
pub(super) struct FlatToken<'a> {
    pub(super) token: &'a ConlluToken,
    pub(super) sentence_end: bool,
    pub(super) head: Option<Option<usize>>,
}

/// One side's tokens flattened into one stream.
pub(super) struct Flattened<'a> {
    pub(super) tokens: Vec<FlatToken<'a>>,
    /// Document token ranges, split at `# newdoc`.
    pub(super) documents: Vec<(usize, usize)>,
    /// Sentence token ranges.
    pub(super) sentences: Vec<(usize, usize)>,
}

/// Flattens sentences into one token stream with its document and
/// sentence ranges.
pub(super) fn flatten(sentences: &[ConlluSentence]) -> Flattened<'_> {
    let mut flat = Vec::new();
    let mut docs = Vec::new();
    let mut sents = Vec::new();
    let mut doc_start = 0usize;
    for (si, sentence) in sentences.iter().enumerate() {
        if sentence.new_doc && si > 0 {
            docs.push((doc_start, flat.len()));
            doc_start = flat.len();
        }
        let start = flat.len();
        let len = sentence.tokens.len();
        for (i, token) in sentence.tokens.iter().enumerate() {
            let head = token.head.and_then(|h| {
                if h == 0 {
                    Some(None)
                } else if h >= 1 && (h as usize) <= len {
                    Some(Some(start + h as usize - 1))
                } else {
                    None
                }
            });
            flat.push(FlatToken { token, sentence_end: i + 1 == len, head });
        }
        sents.push((start, flat.len()));
    }
    docs.push((doc_start, flat.len()));
    Flattened { tokens: flat, documents: docs, sentences: sents }
}

/// How far the alignment looks ahead for a resync point after a form
/// mismatch, in tokens on either side.
const RESYNC_WINDOW: usize = 64;

/// Aligns two token streams by form: equal forms in order pair up;
/// after a mismatch (a split or merged token), the walk resyncs at the
/// nearest pair of equal forms whose following forms also agree, so a
/// frequent word cannot anchor a false sync. Tokens left unpaired are
/// the two tokenizations' real disagreements.
pub(super) fn align_forms(gold: &[FlatToken<'_>], pred: &[FlatToken<'_>]) -> Vec<(usize, usize)> {
    fn form<'a>(side: &[FlatToken<'a>], i: usize) -> Option<&'a str> {
        side.get(i).map(|t| t.token.form.as_str())
    }
    let mut pairs = Vec::new();
    let (mut g, mut p) = (0usize, 0usize);
    while g < gold.len() && p < pred.len() {
        if gold[g].token.form == pred[p].token.form {
            pairs.push((g, p));
            g += 1;
            p += 1;
            continue;
        }
        let mut resync = None;
        'search: for step in 1..=(2 * RESYNC_WINDOW) {
            for dg in 0..=step.min(RESYNC_WINDOW) {
                let dp = step - dg;
                if dp > RESYNC_WINDOW {
                    continue;
                }
                if form(gold, g + dg).is_some()
                    && form(gold, g + dg) == form(pred, p + dp)
                    && form(gold, g + dg + 1) == form(pred, p + dp + 1)
                {
                    resync = Some((dg, dp));
                    break 'search;
                }
            }
        }
        match resync {
            Some((dg, dp)) => {
                g += dg;
                p += dp;
            }
            None => {
                g += 1;
                p += 1;
            }
        }
    }
    pairs
}

/// Each gold document's `# text` lines joined with one space, with the
/// gold sentence-boundary ends in joined coordinates (the final
/// sentence's end excluded: the document ends there anyway). Documents
/// are kept apart so bracket and quote state cannot leak across them.
pub(super) fn joined_documents(gold: &[ConlluSentence]) -> Result<Vec<(String, BTreeSet<u32>)>> {
    let mut texts: Vec<Vec<&str>> = Vec::new();
    for (i, sentence) in gold.iter().enumerate() {
        let text = sentence.text.as_deref().ok_or_else(|| format!("gold sentence {i} has no # text line"))?;
        if sentence.new_doc || texts.is_empty() {
            texts.push(Vec::new());
        }
        texts.last_mut().expect("a document group").push(text);
    }
    let mut out = Vec::with_capacity(texts.len());
    for document in &texts {
        let mut gold_at = BTreeSet::new();
        let mut at = 0usize;
        for text in &document[..document.len().saturating_sub(1)] {
            at += text.len();
            gold_at.insert(at as u32);
            at += 1;
        }
        out.push((document.join(" "), gold_at));
    }
    Ok(out)
}

pub(super) fn add_boundary_counts(counts: &mut Counts, gold_at: &BTreeSet<u32>, pred_at: &BTreeSet<u32>) {
    counts.true_positives += gold_at.intersection(pred_at).count() as u64;
    counts.selected += pred_at.len() as u64;
    counts.target += gold_at.len() as u64;
}

/// Reads a sentence-list prediction: one sentence per line, documents
/// separated by blank lines, in gold document order.
pub fn parse_sentence_list(text: &str) -> Vec<Vec<String>> {
    let mut docs: Vec<Vec<String>> = Vec::new();
    let mut current: Vec<String> = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            if !current.is_empty() {
                docs.push(std::mem::take(&mut current));
            }
        } else {
            current.push(line.to_string());
        }
    }
    if !current.is_empty() {
        docs.push(current);
    }
    docs
}

/// Scores a sentence-only system (wtpsplit) whose output is sentences,
/// not tokens: the predicted sentences must tile each gold document's
/// joined text in order with only whitespace between them, and their
/// end positions are scored against the gold boundaries.
pub fn score_sentence_list(system: &str, gold: &[ConlluSentence], peer: &[Vec<String>]) -> Result<SystemScores> {
    let documents = joined_documents(gold)?;
    if peer.len() != documents.len() {
        return Err(format!("{system}: {} predicted documents, {} gold documents", peer.len(), documents.len()));
    }
    let mut counts = Counts::default();
    for (doc, ((joined, gold_at), sentences)) in documents.iter().zip(peer).enumerate() {
        let mut pred_at = BTreeSet::new();
        let mut cursor = 0usize;
        for (i, sentence) in sentences.iter().enumerate() {
            let sentence = sentence.trim();
            if sentence.is_empty() {
                return Err(format!("{system}: document {doc} sentence {i} is empty"));
            }
            let at = joined[cursor..].find(sentence).map(|p| cursor + p).ok_or_else(|| {
                format!("{system}: document {doc} sentence {i} not found in the document text in order")
            })?;
            if !joined[cursor..at].trim().is_empty() {
                return Err(format!("{system}: document {doc} sentence {i} skips text"));
            }
            cursor = at + sentence.len();
            if i + 1 < sentences.len() {
                pred_at.insert(cursor as u32);
            }
        }
        if !joined[cursor..].trim().is_empty() {
            return Err(format!("{system}: document {doc} leaves text unsplit"));
        }
        add_boundary_counts(&mut counts, gold_at, &pred_at);
    }
    Ok(SystemScores {
        system: system.to_string(),
        sentences: MetricScore::of(counts, Measure::FMeasure, None, 0),
        tokens: None,
        upos: None,
        xpos: None,
        feats: None,
        lemma: None,
        uas: None,
        las: None,
        entities: None,
        language: None,
        not_run: vec!["tokens/fields: a sentence-only system writes no token fields".into()],
        token_mismatch_sentences: 0,
        throughput_sentences_per_second: None,
    })
}
