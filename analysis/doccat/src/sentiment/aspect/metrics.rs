use super::*;

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
