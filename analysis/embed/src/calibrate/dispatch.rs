//! The sentence and chunk vector steps on the device or on the CPU, chosen
//! for each call from a calibration: a [`Dispatch`] holds the step opened
//! on both at one window and cap on rows a run, plans each call
//! (`VectorAnnotator::shape`), and runs all of it on the side whose fitted
//! cost is lower for that call. Each document's layers record the side
//! that made them (`Annotator::annotate_batch_placed`), so a document's
//! vectors are one side's, and which one is in its provenance.
//!
//! A side's cost is fitted to what the calibration measured at the step's
//! window, as a time a run, a row and a token, none below 0: a Hailo-8 pays for every
//! row whatever its length and loads its contexts once a run, and the CPU
//! pays for the tokens. The samples are the whole set of texts, its first
//! sentence alone and its first text alone, at each batch measured.

use std::path::Path;
use std::sync::Arc;

use turbo::status::{Error as CoreError, INVALID_ARGUMENT};
use turbo_analysis::{Analyzer, Error, Execution, Key, ModelRef};
use turbo_analysis_layers::{Sentences, Subwords};

use super::{Calibration, CalibrationOptions, FORCE_VAR, Side, Text, documents, first_sentence, forced};
use crate::encoder::{Device, Precision, default_ranges};
use crate::vectors::{Embedder, Input, Output, Shape};
use crate::window::Which;
use crate::{SubwordAnnotator, VectorAnnotator, VectorOptions};

/// A side's time for a call, in ms: a cost a run, a row and a token.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Cost {
    pub run: f64,
    pub row: f64,
    pub token: f64,
}

impl Cost {
    /// The fit of `samples`, each a call's shape and its time in ms: the
    /// costs, none below 0, that make the sum of the squared relative
    /// errors least, so one sentence counts as much as the whole set. Each
    /// subset of the three costs is solved with the others at 0, and the
    /// best that has none below 0 is kept. None when no subset is solved.
    pub fn fit(samples: &[(Shape, f64)]) -> Option<Cost> {
        let rows: Vec<([f64; 3], f64)> = samples
            .iter()
            .filter(|(_, t)| *t > 0.0)
            .map(|(s, t)| ([s.runs as f64 / t, s.rows as f64 / t, s.tokens as f64 / t], 1.0))
            .collect();
        let mut best: Option<(f64, [f64; 3])> = None;
        for mask in 1..8u32 {
            let used: Vec<usize> = (0..3).filter(|i| mask & (1 << i) != 0).collect();
            let Some(x) = solve(&rows, &used) else { continue };
            if x.iter().any(|&v| v < 0.0) {
                continue;
            }
            let mut full = [0f64; 3];
            for (&i, &v) in used.iter().zip(&x) {
                full[i] = v;
            }
            let err: f64 =
                rows.iter().map(|(a, b)| (a.iter().zip(&full).map(|(p, q)| p * q).sum::<f64>() - b).powi(2)).sum();
            if best.is_none_or(|(e, _)| err < e) {
                best = Some((err, full));
            }
        }
        best.map(|(_, [run, row, token])| Cost { run, row, token })
    }

    pub fn ms(&self, s: Shape) -> f64 {
        self.run * s.runs as f64 + self.row * s.rows as f64 + self.token * s.tokens as f64
    }
}

/// The least squares solution of `a x = b` over the columns `used` of the
/// rows `(a, b)`, through the normal equations with each column scaled to
/// at most 1; None when they are singular.
fn solve(rows: &[([f64; 3], f64)], used: &[usize]) -> Option<Vec<f64>> {
    let n = used.len();
    let scale: Vec<f64> = used.iter().map(|&j| rows.iter().map(|(a, _)| a[j].abs()).fold(0.0, f64::max)).collect();
    if scale.contains(&0.0) {
        return None;
    }
    let mut m = vec![vec![0f64; n]; n];
    let mut v = vec![0f64; n];
    for (a, b) in rows {
        let x: Vec<f64> = used.iter().zip(&scale).map(|(&j, s)| a[j] / s).collect();
        for i in 0..n {
            v[i] += x[i] * b;
            for (mij, xj) in m[i].iter_mut().zip(&x) {
                *mij += x[i] * xj;
            }
        }
    }
    for c in 0..n {
        let p = (c..n).max_by(|&x, &y| m[x][c].abs().total_cmp(&m[y][c].abs()))?;
        if m[p][c].abs() < 1e-9 {
            return None;
        }
        m.swap(c, p);
        v.swap(c, p);
        let (pivot, at) = (m[c].clone(), v[c]);
        for (r, (row, vr)) in m.iter_mut().zip(v.iter_mut()).enumerate() {
            if r != c {
                let f = row[c] / pivot[c];
                for (x, y) in row[c..].iter_mut().zip(&pivot[c..]) {
                    *x -= f * y;
                }
                *vr -= f * at;
            }
        }
    }
    Some((0..n).map(|i| v[i] / m[i][i] / scale[i]).collect())
}

/// The step on the device and on the CPU, and each side's fitted cost.
pub struct Dispatch {
    device: Arc<VectorAnnotator>,
    cpu: Arc<VectorAnnotator>,
    cap: u32,
    /// Whether the device's vectors clear the floor at the step's window.
    eligible: bool,
    device_cost: Cost,
    cpu_cost: Cost,
    /// The calls the calibration timed, by shape, and the side with the
    /// lower median time for each.
    measured: Vec<(Shape, Side)>,
    force: Option<Side>,
}

impl Dispatch {
    /// The step at `base`'s window and cap on rows a run, on the device
    /// and precision of `o` and on the CPU at `Precision::Exact`, choosing
    /// per call from the calibration kept in `dir` (measured first when
    /// there is none, `Calibration::load_or_measure`). `o` must ask for
    /// `base`'s window and cap. The [`FORCE_VAR`] variable sends every call
    /// to one side. Both are checked before anything is measured.
    pub fn open(
        dir: &Path,
        bundle: &Path,
        o: &CalibrationOptions,
        texts: &[Text],
        base: &VectorOptions,
    ) -> Result<Dispatch, CoreError> {
        let force = forced(std::env::var(FORCE_VAR).ok().as_deref())?;
        let (max_seq, cap) = (base.encoder.max_seq, base.encoder.max_batch);
        if !o.windows.contains(&max_seq) || !o.batches.contains(&cap) {
            return Err(CoreError::new(
                INVALID_ARGUMENT,
                format!(
                    "dispatch at {max_seq} tokens a row and {cap} rows a run needs the calibration to ask for that \
                     window and that batch; it asks for windows {:?} and batches {:?}",
                    o.windows, o.batches
                ),
            ));
        }
        let c = Calibration::load_or_measure(dir, bundle, o, texts)?;
        let open = |device: Device, precision: Precision| {
            let mut v = base.clone();
            v.encoder.device = device;
            v.encoder.precision = precision;
            VectorAnnotator::open(bundle, &v).map(Arc::new)
        };
        let (device, cpu) = (open(o.device.clone(), o.precision)?, open(Device::Cpu, Precision::Exact)?);
        let short = first_sentence(texts)?;
        let (mut on_device, mut on_cpu) = (Vec::new(), Vec::new());
        let points: Vec<_> = c.points.iter().filter(|p| p.max_seq == max_seq).collect();
        let eligible = points.iter().all(|p| p.min_cosine >= c.floor);
        for p in points {
            let at = |t: &[Text]| shape_of(bundle, &device, t, p.max_batch);
            let (all, one, doc) = (at(texts)?, at(&short)?, at(&texts[..1])?);
            on_device.extend([(all, p.device_ms), (one, p.short_device_ms), (doc, p.doc_device_ms)]);
            on_cpu.extend([(all, p.cpu_ms), (one, p.short_cpu_ms), (doc, p.doc_cpu_ms)]);
        }
        // A call of the same shape timed in sessions of several caps did the
        // same work in each: its side is the one with the lower median.
        let mut measured: Vec<(Shape, Side)> = Vec::new();
        for (s, _) in &on_device {
            if measured.iter().any(|(m, _)| m == s) {
                continue;
            }
            let median = |samples: &[(Shape, f64)]| {
                let mut t: Vec<f64> = samples.iter().filter(|(m, _)| m == s).map(|&(_, t)| t).collect();
                t.sort_by(f64::total_cmp);
                t[t.len() / 2]
            };
            let device_wins = eligible && median(&on_device) < median(&on_cpu);
            measured.push((*s, if device_wins { Side::Device } else { Side::Cpu }));
        }
        let fit = |samples: &[(Shape, f64)], side: &str| {
            Cost::fit(samples).ok_or_else(|| {
                CoreError::new(
                    INVALID_ARGUMENT,
                    format!("the {side}'s calibration at {max_seq} tokens a row fits no cost"),
                )
            })
        };
        Ok(Dispatch {
            cap,
            eligible,
            device_cost: fit(&on_device, "device")?,
            cpu_cost: fit(&on_cpu, "cpu")?,
            measured,
            force,
            device,
            cpu,
        })
    }

    /// The side a call of shape `s` runs on: the side measured faster when
    /// the calibration timed a call of that shape, else the side whose
    /// fitted cost is lower.
    pub fn side(&self, s: Shape) -> Side {
        if let Some(side) = self.force {
            return side;
        }
        if let Some(&(_, side)) = self.measured.iter().find(|(m, _)| *m == s) {
            return side;
        }
        let device = self.eligible && self.device_cost.ms(s) < self.cpu_cost.ms(s);
        if device { Side::Device } else { Side::Cpu }
    }

    /// Each side's fitted cost: the device's, then the CPU's.
    pub fn costs(&self) -> (Cost, Cost) {
        (self.device_cost, self.cpu_cost)
    }

    /// The step on `side`: the instance this dispatch runs that side's
    /// calls on.
    pub fn on(&self, side: Side) -> &Arc<VectorAnnotator> {
        match side {
            Side::Device => &self.device,
            Side::Cpu => &self.cpu,
        }
    }

    /// The step's cap on rows a run.
    pub fn cap(&self) -> u32 {
        self.cap
    }
}

/// The shape of the sentence vector step over `texts` on `a`, in a session
/// of `cap` rows a run and the default ranges, as the calibration opened
/// it: the texts are split into subwords and planned, not run.
pub fn shape_of(bundle: &Path, a: &VectorAnnotator, texts: &[Text], cap: u32) -> Result<Shape, CoreError> {
    let err = |e: Error| CoreError::new(INVALID_ARGUMENT, e.to_string());
    let words = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .add(SubwordAnnotator::open(bundle)?)
        .build()
        .map_err(err)?;
    let mut docs = documents(texts)?;
    for r in words.run(&mut docs) {
        r.map_err(err)?;
    }
    let inputs: Vec<Input<'_>> = docs
        .iter()
        .filter_map(|d| {
            let (subwords, sentences) = (d.get::<Subwords>()?, d.get::<Sentences>()?);
            Some(Input { subwords, sentences: Some(&sentences.spans), chunks: None, levels: None })
        })
        .collect();
    a.shape_at("calibration", &inputs, cap, default_ranges(cap, a.info().max_seq))
}

impl Embedder for Dispatch {
    fn embed_placed(
        &self,
        annotator: &'static str,
        inputs: &[Input<'_>],
    ) -> (Vec<Result<Output, Error>>, Option<Execution>) {
        // Planned once: the side runs the plan its shape was counted from.
        let planned = self.device.plan(annotator, inputs);
        let a = self.on(self.side(self.device.shape_of_plan(&planned)));
        (a.embed_planned(annotator, inputs, planned), Some(a.execution().clone()))
    }

    fn model(&self) -> &ModelRef {
        self.device.model()
    }

    fn execution(&self) -> &Execution {
        self.device.execution()
    }

    fn overhead(&self) -> u32 {
        self.device.overhead()
    }

    fn config(&self, layout: Which) -> u64 {
        self.device.config(layout)
    }
}
