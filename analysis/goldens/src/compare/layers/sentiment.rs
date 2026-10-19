//! `GoldenCompare` and `From<&Layer> for GoldenLayer` for sentiment and discourse layers.

use turbo_analysis::{Ix, Layer};
use turbo_analysis_layers::{Aspects, Incivility, IncivilityLabel, Opinions, Pragmatics, Sentences, Strs};

use super::super::{Cx, GoldenCompare, Items, Labels, Mismatch, Names, TokenCols, Tolerance, compare_items};
use super::{anchor_spans, check_bool_u32, f64s, u32s};
use crate::format::ColumnType::{F64, Str, U32};
use crate::format::{Column, ColumnData, ColumnSpec, GoldenLayer};

/// Compares two layers that have multiple `f64` columns in row-major order
/// (`0..n`), checking tokens and labels first and then each `(name, rust,
/// golden)` float slice under `tol`.
pub(super) fn compare_multi_values<'a, 'b>(
    cx: &Cx<'_>,
    mut rust: Items<'a>,
    mut golden: Items<'b>,
    label_name: Option<&'static str>,
    value_cols: &[(&'static str, &'a [f64], &'b [f64])],
    tol: Tolerance,
) -> Result<(), Mismatch> {
    let n = rust.len.min(golden.len);
    let mut chosen = value_cols[0];
    'rows: for i in 0..n {
        if let (Some(r), Some(g)) = (rust.tokens, golden.tokens)
            && r.get(i) != g.get(i)
        {
            break 'rows;
        }
        if let (Some(r), Some(g)) = (rust.labels, golden.labels)
            && r.get(i) != g.get(i)
        {
            break 'rows;
        }
        for &(col_name, r, g) in value_cols {
            if !tol.accepts(r[i], g[i]) {
                chosen = (col_name, r, g);
                break 'rows;
            }
        }
    }
    rust.values = Some(chosen.1);
    golden.values = Some(chosen.2);
    compare_items(cx, &rust, &golden, Names { label: label_name, value: Some(chosen.0) }, tol)
}

impl GoldenCompare for Opinions {
    const COLUMNS: &'static [ColumnSpec] = &[
        ColumnSpec::new("token_start", U32),
        ColumnSpec::new("token_end", U32),
        ColumnSpec::new("polarity", F64),
        ColumnSpec::new("intensity", F64),
        ColumnSpec::new("negated", U32),
    ];

    fn compare(&self, cx: &Cx<'_>, golden: &GoldenLayer, tol: Tolerance) -> Result<(), Mismatch> {
        let negated = u32s(golden, "negated");
        check_bool_u32(cx, negated, "negated")?;
        let mut ours = Strs::default();
        for &neg in &self.negated {
            ours.push(if neg { "1" } else { "0" });
        }
        let mut theirs = Strs::default();
        for &neg in negated {
            theirs.push(if neg != 0 { "1" } else { "0" });
        }
        let rust = Items {
            len: self.len(),
            spans: None,
            context: None,
            tokens: Some(TokenCols::Ranges(&self.tokens)),
            units: None,
            labels: Some(Labels::Strs(&ours)),
            values: None,
        };
        let golden_items = Items {
            len: golden.len(),
            spans: None,
            context: None,
            tokens: Some(TokenCols::Columns { start: u32s(golden, "token_start"), end: u32s(golden, "token_end") }),
            units: None,
            labels: Some(Labels::Strs(&theirs)),
            values: None,
        };
        let cols = [
            ("polarity", self.polarity.as_slice(), f64s(golden, "polarity")),
            ("intensity", self.intensity.as_slice(), f64s(golden, "intensity")),
        ];
        compare_multi_values(cx, rust, golden_items, Some("negated"), &cols, tol)
    }
}

impl From<&Opinions> for GoldenLayer {
    fn from(layer: &Opinions) -> GoldenLayer {
        let token_start = layer.tokens.iter().map(|r| r.start.get() as u32).collect();
        let token_end = layer.tokens.iter().map(|r| r.end.get() as u32).collect();
        let negated = layer.negated.iter().map(|&b| u32::from(b)).collect();
        GoldenLayer {
            id: Opinions::ID.into(),
            anchor: Opinions::golden_anchor(),
            columns: vec![
                Column { name: "token_start".into(), data: ColumnData::U32(token_start) },
                Column { name: "token_end".into(), data: ColumnData::U32(token_end) },
                Column { name: "polarity".into(), data: ColumnData::F64(layer.polarity.clone()) },
                Column { name: "intensity".into(), data: ColumnData::F64(layer.intensity.clone()) },
                Column { name: "negated".into(), data: ColumnData::U32(negated) },
            ],
        }
    }
}

impl GoldenCompare for Aspects {
    const COLUMNS: &'static [ColumnSpec] = &[
        ColumnSpec::new("token_start", U32),
        ColumnSpec::new("token_end", U32),
        ColumnSpec::new("sentence", Str),
        ColumnSpec::new("category", Str),
        ColumnSpec::new("positive", F64),
        ColumnSpec::new("negative", F64),
        ColumnSpec::new("neutral", F64),
        ColumnSpec::new("opinions", Str),
    ];

    fn compare(&self, cx: &Cx<'_>, golden: &GoldenLayer, tol: Tolerance) -> Result<(), Mismatch> {
        let mut ours = Strs::default();
        for i in 0..self.len() {
            let ix = Ix::new(i as u32);
            let (ts, te) = self.tokens[i].map_or((0, 0), |r| (r.start.get() as u32, r.end.get() as u32));
            let sent = self.sentence[i].map_or_else(String::new, |s| s.get().to_string());
            let cat = self.category_label(ix).unwrap_or("");
            let ops = self.opinions_of(ix).iter().map(|o| o.get().to_string()).collect::<Vec<_>>().join(",");
            ours.push(&format!("tokens {ts}..{te} sentence {sent:?} category {cat:?} opinions [{ops}]"));
        }
        let (token_start, token_end) = (u32s(golden, "token_start"), u32s(golden, "token_end"));
        let sentence = golden.strs("sentence").expect("a str column");
        let category = golden.strs("category").expect("a str column");
        let opinions = golden.strs("opinions").expect("a str column");
        let mut theirs = Strs::default();
        for i in 0..golden.len() {
            theirs.push(&format!(
                "tokens {}..{} sentence {:?} category {:?} opinions [{}]",
                token_start[i],
                token_end[i],
                sentence.get(i),
                category.get(i),
                opinions.get(i)
            ));
        }
        let rust = Items {
            len: self.len(),
            spans: None,
            context: None,
            tokens: None,
            units: None,
            labels: Some(Labels::Strs(&ours)),
            values: None,
        };
        let golden_items = Items {
            len: golden.len(),
            spans: None,
            context: None,
            tokens: None,
            units: None,
            labels: Some(Labels::Strs(&theirs)),
            values: None,
        };
        let cols = [
            ("positive", self.positive.as_slice(), f64s(golden, "positive")),
            ("negative", self.negative.as_slice(), f64s(golden, "negative")),
            ("neutral", self.neutral.as_slice(), f64s(golden, "neutral")),
        ];
        compare_multi_values(cx, rust, golden_items, Some("aspect"), &cols, tol)
    }
}

impl From<&Aspects> for GoldenLayer {
    fn from(layer: &Aspects) -> GoldenLayer {
        let mut token_start = Vec::with_capacity(layer.len());
        let mut token_end = Vec::with_capacity(layer.len());
        let mut sentence = Strs::default();
        let mut category = Strs::default();
        let mut opinions = Strs::default();
        for i in 0..layer.len() {
            let ix = Ix::new(i as u32);
            let (ts, te) = layer.tokens[i].map_or((0, 0), |r| (r.start.get() as u32, r.end.get() as u32));
            token_start.push(ts);
            token_end.push(te);
            match layer.sentence[i] {
                Some(s) => sentence.push(&s.get().to_string()),
                None => sentence.push(""),
            }
            category.push(layer.category_label(ix).unwrap_or(""));
            let ops = layer.opinions_of(ix).iter().map(|o| o.get().to_string()).collect::<Vec<_>>().join(",");
            opinions.push(&ops);
        }
        GoldenLayer {
            id: Aspects::ID.into(),
            anchor: Aspects::golden_anchor(),
            columns: vec![
                Column { name: "token_start".into(), data: ColumnData::U32(token_start) },
                Column { name: "token_end".into(), data: ColumnData::U32(token_end) },
                Column { name: "sentence".into(), data: ColumnData::Str(sentence) },
                Column { name: "category".into(), data: ColumnData::Str(category) },
                Column { name: "positive".into(), data: ColumnData::F64(layer.positive.clone()) },
                Column { name: "negative".into(), data: ColumnData::F64(layer.negative.clone()) },
                Column { name: "neutral".into(), data: ColumnData::F64(layer.neutral.clone()) },
                Column { name: "opinions".into(), data: ColumnData::Str(opinions) },
            ],
        }
    }
}

impl GoldenCompare for Incivility {
    const COLUMNS: &'static [ColumnSpec] = &[
        ColumnSpec::new("target_kind", Str),
        ColumnSpec::new("target_index", U32),
        ColumnSpec::new("insult", F64),
        ColumnSpec::new("profanity", F64),
        ColumnSpec::new("threat", F64),
        ColumnSpec::new("identity_attack", F64),
        ColumnSpec::new("intolerance", F64),
        ColumnSpec::new("severe_toxicity", F64),
        ColumnSpec::new("evidence", Str),
    ];

    fn compare(&self, cx: &Cx<'_>, golden: &GoldenLayer, tol: Tolerance) -> Result<(), Mismatch> {
        let mut ours = Strs::default();
        for i in 0..self.len() {
            let ix = Ix::new(i as u32);
            let t = self.target[i];
            let ev = self
                .evidence_of(ix)
                .map(|(l, r)| format!("{}:{}..{}", l.name(), r.start.get(), r.end.get()))
                .collect::<Vec<_>>()
                .join(",");
            ours.push(&format!("{}:{} evidence [{ev}]", t.kind_name(), t.raw_index()));
        }
        let target_kind = golden.strs("target_kind").expect("a str column");
        let target_index = u32s(golden, "target_index");
        let evidence = golden.strs("evidence").expect("a str column");
        let mut theirs = Strs::default();
        for (i, &idx) in target_index.iter().enumerate().take(golden.len()) {
            theirs.push(&format!("{}:{idx} evidence [{}]", target_kind.get(i), evidence.get(i)));
        }
        let rust = Items {
            len: self.len(),
            spans: None,
            context: None,
            tokens: None,
            units: None,
            labels: Some(Labels::Strs(&ours)),
            values: None,
        };
        let golden_items = Items {
            len: golden.len(),
            spans: None,
            context: None,
            tokens: None,
            units: None,
            labels: Some(Labels::Strs(&theirs)),
            values: None,
        };
        let cols = [
            (IncivilityLabel::Insult.name(), self.insult.as_slice(), f64s(golden, "insult")),
            (IncivilityLabel::Profanity.name(), self.profanity.as_slice(), f64s(golden, "profanity")),
            (IncivilityLabel::Threat.name(), self.threat.as_slice(), f64s(golden, "threat")),
            (IncivilityLabel::IdentityAttack.name(), self.identity_attack.as_slice(), f64s(golden, "identity_attack")),
            (IncivilityLabel::Intolerance.name(), self.intolerance.as_slice(), f64s(golden, "intolerance")),
            (IncivilityLabel::SevereToxicity.name(), self.severe_toxicity.as_slice(), f64s(golden, "severe_toxicity")),
        ];
        compare_multi_values(cx, rust, golden_items, Some("target"), &cols, tol)
    }
}

impl From<&Incivility> for GoldenLayer {
    fn from(layer: &Incivility) -> GoldenLayer {
        let mut target_kind = Strs::default();
        let mut target_index = Vec::with_capacity(layer.len());
        let mut evidence = Strs::default();
        for i in 0..layer.len() {
            let ix = Ix::new(i as u32);
            let t = layer.target[i];
            target_kind.push(t.kind_name());
            target_index.push(t.raw_index());
            let ev = layer
                .evidence_of(ix)
                .map(|(l, r)| format!("{}:{}..{}", l.name(), r.start.get(), r.end.get()))
                .collect::<Vec<_>>()
                .join(",");
            evidence.push(&ev);
        }
        GoldenLayer {
            id: Incivility::ID.into(),
            anchor: Incivility::golden_anchor(),
            columns: vec![
                Column { name: "target_kind".into(), data: ColumnData::Str(target_kind) },
                Column { name: "target_index".into(), data: ColumnData::U32(target_index) },
                Column { name: "insult".into(), data: ColumnData::F64(layer.insult.clone()) },
                Column { name: "profanity".into(), data: ColumnData::F64(layer.profanity.clone()) },
                Column { name: "threat".into(), data: ColumnData::F64(layer.threat.clone()) },
                Column { name: "identity_attack".into(), data: ColumnData::F64(layer.identity_attack.clone()) },
                Column { name: "intolerance".into(), data: ColumnData::F64(layer.intolerance.clone()) },
                Column { name: "severe_toxicity".into(), data: ColumnData::F64(layer.severe_toxicity.clone()) },
                Column { name: "evidence".into(), data: ColumnData::Str(evidence) },
            ],
        }
    }
}

impl GoldenCompare for Pragmatics {
    const COLUMNS: &'static [ColumnSpec] = &[
        ColumnSpec::new("score", F64),
        ColumnSpec::new("pos_start", U32),
        ColumnSpec::new("pos_end", U32),
        ColumnSpec::new("neg_start", U32),
        ColumnSpec::new("neg_end", U32),
    ];

    fn compare(&self, cx: &Cx<'_>, golden: &GoldenLayer, tol: Tolerance) -> Result<(), Mismatch> {
        let mut ours = Strs::default();
        for ev in &self.evidence {
            let (ps, pe, ns, ne) = ev.map_or((0, 0, 0, 0), |inc| {
                (
                    inc.positive.start.get() as u32,
                    inc.positive.end.get() as u32,
                    inc.negative.start.get() as u32,
                    inc.negative.end.get() as u32,
                )
            });
            ours.push(&format!("pos {ps}..{pe} neg {ns}..{ne}"));
        }
        let (pos_start, pos_end) = (u32s(golden, "pos_start"), u32s(golden, "pos_end"));
        let (neg_start, neg_end) = (u32s(golden, "neg_start"), u32s(golden, "neg_end"));
        let mut theirs = Strs::default();
        for i in 0..golden.len() {
            theirs.push(&format!("pos {}..{} neg {}..{}", pos_start[i], pos_end[i], neg_start[i], neg_end[i]));
        }
        let rust = Items {
            len: self.len(),
            spans: None,
            context: cx.rust_anchor::<Sentences>(),
            tokens: None,
            units: None,
            labels: Some(Labels::Strs(&ours)),
            values: Some(&self.score),
        };
        let golden_items = Items {
            len: golden.len(),
            spans: None,
            context: anchor_spans(cx, golden),
            tokens: None,
            units: None,
            labels: Some(Labels::Strs(&theirs)),
            values: Some(f64s(golden, "score")),
        };
        compare_items(cx, &rust, &golden_items, Names { label: Some("evidence"), value: Some("score") }, tol)
    }
}

impl From<&Pragmatics> for GoldenLayer {
    fn from(layer: &Pragmatics) -> GoldenLayer {
        let mut pos_start = Vec::with_capacity(layer.len());
        let mut pos_end = Vec::with_capacity(layer.len());
        let mut neg_start = Vec::with_capacity(layer.len());
        let mut neg_end = Vec::with_capacity(layer.len());
        for ev in &layer.evidence {
            let (ps, pe, ns, ne) = ev.map_or((0, 0, 0, 0), |inc| {
                (
                    inc.positive.start.get() as u32,
                    inc.positive.end.get() as u32,
                    inc.negative.start.get() as u32,
                    inc.negative.end.get() as u32,
                )
            });
            pos_start.push(ps);
            pos_end.push(pe);
            neg_start.push(ns);
            neg_end.push(ne);
        }
        GoldenLayer {
            id: Pragmatics::ID.into(),
            anchor: Pragmatics::golden_anchor(),
            columns: vec![
                Column { name: "score".into(), data: ColumnData::F64(layer.score.clone()) },
                Column { name: "pos_start".into(), data: ColumnData::U32(pos_start) },
                Column { name: "pos_end".into(), data: ColumnData::U32(pos_end) },
                Column { name: "neg_start".into(), data: ColumnData::U32(neg_start) },
                Column { name: "neg_end".into(), data: ColumnData::U32(neg_end) },
            ],
        }
    }
}
