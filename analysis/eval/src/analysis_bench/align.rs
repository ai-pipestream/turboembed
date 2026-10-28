//! Token-stream flattening and form alignment for the analysis
//! benchmark scorer.

use super::{ConlluSentence, ConlluToken};

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
