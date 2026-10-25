use turbo_calibration::{Calibration, metrics::reliability_selected};
use turbo_langspan::{
    CalibratedSpanDetector, SpanDetector, SpanScratch,
    gold::{GoldDocument, Scores},
};
use turbo_train::calibration::Samples;
use turbo_unicode::category::general_category;

pub struct Rows {
    pub rows: Vec<Vec<f64>>,
    pub gold: Vec<usize>,
    pub selected: Vec<usize>,
    pub notes: Vec<String>,
    pub scores: Scores,
    pub excluded: usize,
}
impl Rows {
    pub fn samples<'a>(&'a self, split: &'a str) -> Samples<'a> {
        Samples { split, probabilities: &self.rows, gold: &self.gold }
    }
}
pub fn collect(detector: &SpanDetector, docs: &[GoldDocument]) -> Rows {
    let labels: Vec<_> = detector.detector().model().supported_languages().collect();
    let mut data =
        Rows { rows: vec![], gold: vec![], selected: vec![], notes: vec![], scores: Scores::default(), excluded: 0 };
    let (mut scratch, mut runs, mut rows) = (SpanScratch::new(), Vec::new(), Vec::new());
    for doc in docs {
        detector.detect_distributions(&doc.text, &mut scratch, &mut runs, &mut rows);
        data.scores.add(doc, &runs.iter().map(|r| (r.span, detector.code(r))).collect::<Vec<_>>());
        for (run, row) in runs.iter().zip(&rows) {
            let Some(selected) = run.outcome else {
                data.excluded += 1;
                continue;
            };
            let mut counts = vec![0usize; labels.len()];
            for (span, code) in &doc.spans {
                let Some(label) = labels.iter().position(|l| *l == code) else { continue };
                let a = span.start.max(run.span.start) as usize;
                let b = span.end.min(run.span.end) as usize;
                if a < b {
                    counts[label] += doc.text[a..b]
                        .chars()
                        .filter(|&c| {
                            let category = general_category(c);
                            category.is_letter() || category.is_mark()
                        })
                        .count();
                }
            }
            // Equal counts choose the earlier model outcome, not the
            // prediction. The target is the dominant gold language.
            let gold = (0..counts.len()).max_by_key(|&i| (counts[i], std::cmp::Reverse(i))).unwrap();
            if counts[gold] == 0 {
                data.excluded += 1;
                continue;
            }
            let selected = selected as usize;
            let selected_rank = 1 + row.iter().filter(|&&p| p > row[selected]).count();
            let gold_rank = 1 + row.iter().filter(|&&p| p > row[gold]).count();
            data.notes.push(format!(
                "{} bytes={}..{} letters={} selected={} gold={} selected_rank={} gold_rank={}",
                doc.id,
                run.span.start,
                run.span.end,
                run.letters,
                labels[selected],
                labels[gold],
                selected_rank,
                gold_rank
            ));
            data.rows.push(row.clone());
            data.gold.push(gold);
            data.selected.push(selected);
        }
    }
    data
}
pub fn transformed(data: &Rows, profile: &Calibration) -> Result<Vec<Vec<f64>>, String> {
    let transform = profile.transform().prepare()?;
    data.rows
        .iter()
        .map(|row| {
            let mut out = vec![0.0; row.len()];
            transform.apply(row, &mut out)?;
            Ok(out)
        })
        .collect()
}
pub fn report(name: &str, data: &Rows, profile: &Calibration, threshold: f64) -> Result<(), String> {
    let rows = transformed(data, profile)?;
    for (kind, probabilities) in [("raw", &data.rows), ("calibrated", &rows)] {
        let m = reliability_selected(probabilities, &data.gold, &data.selected, 10)?;
        let mut accepted = 0;
        let mut correct = 0;
        for ((row, &gold), &selected) in probabilities.iter().zip(&data.gold).zip(&data.selected) {
            if row[selected] >= threshold {
                accepted += 1;
                correct += usize::from(selected == gold);
            }
        }
        println!(
            "{name} {kind}: samples={} excluded={} ECE={:.9} Brier={:.9} accuracy={:.9} cutoff={threshold} accepted={accepted} correct={correct}",
            m.samples, data.excluded, m.ece, m.brier, m.accuracy
        );
    }
    println!("{name}: {}", data.scores);
    if name == "test" {
        for (i, (row, &selected)) in rows.iter().zip(&data.selected).enumerate() {
            if selected != data.gold[i] || row[selected] < threshold {
                println!(
                    "{name} run: {} raw={:.9} calibrated={:.9}",
                    data.notes[i], data.rows[i][selected], row[selected]
                );
            }
        }
    }
    Ok(())
}
pub fn verify(calibrated: &CalibratedSpanDetector, docs: &[GoldDocument]) -> Result<(), String> {
    let (mut scratch, mut raw, mut got, mut rows) = (SpanScratch::new(), vec![], vec![], vec![]);
    for doc in docs {
        calibrated.detector().detect(&doc.text, None, None, &mut scratch, &mut raw);
        calibrated.detect(&doc.text, &mut scratch, &mut got, &mut rows)?;
        if raw.len() != got.len()
            || raw.iter().zip(&got).any(|(a, b)| {
                a.span != b.span || a.outcome != b.outcome || a.script != b.script || a.letters != b.letters
            })
        {
            return Err("calibration changed a boundary or label".into());
        }
    }
    Ok(())
}
