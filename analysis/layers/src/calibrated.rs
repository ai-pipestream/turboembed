//! Checked calibrated decisions. Small single-row outputs stay on the stack;
//! larger batches use flat score and CSR set columns with no per-row allocations.
use std::{marker::PhantomData, sync::Arc};
use turbo_analysis::{Anchor, Layer, LayerName};
use turbo_calibration::{Abstention, DecisionMetadata, Prediction, Predictor};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibratedRow<'a> {
    pub raw: &'a [f64],
    pub calibrated: &'a [f64],
    pub set: &'a [usize],
    pub abstention: Option<Abstention>,
}
#[derive(Clone, Debug)]
struct Small {
    raw: [f64; 2],
    calibrated: [f64; 2],
    set: [usize; 2],
    count: usize,
    abstention: Option<Abstention>,
    present: bool,
}
impl Default for Small {
    fn default() -> Self {
        Self { raw: [0.0; 2], calibrated: [0.0; 2], set: [0; 2], count: 0, abstention: None, present: false }
    }
}
#[derive(Clone, Debug, Default)]
struct Columns {
    raw: Vec<f64>,
    calibrated: Vec<f64>,
    set: Vec<usize>,
    end: Vec<usize>,
    abstention: Vec<Option<Abstention>>,
}
#[derive(Clone, Debug)]
enum Storage {
    Small(Small),
    Columns(Box<Columns>),
}

#[derive(Debug)]
pub struct Calibrated<L> {
    metadata: Arc<DecisionMetadata>,
    storage: Storage,
    _of: PhantomData<fn() -> L>,
}
impl<L: Layer> Calibrated<L> {
    #[inline]
    pub fn new(predictor: &Predictor) -> Self {
        Self::empty(predictor.shared_metadata())
    }
    fn empty(metadata: Arc<DecisionMetadata>) -> Self {
        let storage = if metadata.identity().labels.len() == 2 {
            Storage::Small(Small::default())
        } else {
            Storage::Columns(Box::default())
        };
        Self { metadata, storage, _of: PhantomData }
    }
    pub fn metadata(&self) -> &DecisionMetadata {
        &self.metadata
    }
    pub fn row(&self, i: usize) -> CalibratedRow<'_> {
        let k = self.metadata.identity().labels.len();
        match &self.storage {
            Storage::Small(s) => {
                assert!(i == 0 && s.present, "calibrated row index");
                CalibratedRow {
                    raw: &s.raw[..k],
                    calibrated: &s.calibrated[..k],
                    set: &s.set[..s.count],
                    abstention: s.abstention,
                }
            }
            Storage::Columns(c) => {
                let start = if i == 0 { 0 } else { c.end[i - 1] };
                CalibratedRow {
                    raw: &c.raw[i * k..(i + 1) * k],
                    calibrated: &c.calibrated[i * k..(i + 1) * k],
                    set: &c.set[start..c.end[i]],
                    abstention: c.abstention[i],
                }
            }
        }
    }
    pub fn rows(&self) -> impl ExactSizeIterator<Item = CalibratedRow<'_>> {
        (0..self.len()).map(|i| self.row(i))
    }
    /// Append a checked prediction without allocating per row. Once columns
    /// have sufficient capacity, only the transform and policy are evaluated.
    #[inline]
    pub fn push(&mut self, predictor: &Predictor, raw: &[f64]) -> Result<(), String> {
        if !std::ptr::eq(self.metadata.as_ref(), predictor.metadata()) && self.metadata.as_ref() != predictor.metadata()
        {
            return Err("calibrated batch/predictor identity mismatch".into());
        }
        let k = self.metadata.identity().labels.len();
        if let Storage::Small(s) = &mut self.storage
            && !s.present
        {
            let (count, reason) = predictor.predict_into(raw, &mut s.calibrated[..k], &mut s.set[..k])?;
            s.raw[..k].copy_from_slice(raw);
            s.count = count;
            s.abstention = reason;
            s.present = true;
            return Ok(());
        }
        self.columns();
        let Storage::Columns(c) = &mut self.storage else { unreachable!() };
        let (a, b) = (c.calibrated.len(), c.set.len());
        c.calibrated.resize(a + k, 0.0);
        c.set.resize(b + k, 0);
        let result = predictor.predict_into(raw, &mut c.calibrated[a..], &mut c.set[b..]);
        match result {
            Ok((count, reason)) => {
                c.raw.extend_from_slice(raw);
                c.set.truncate(b + count);
                c.end.push(c.set.len());
                c.abstention.push(reason);
                Ok(())
            }
            Err(e) => {
                c.calibrated.truncate(a);
                c.set.truncate(b);
                Err(e)
            }
        }
    }
    fn columns(&mut self) {
        if let Storage::Small(s) = &self.storage {
            let k = self.metadata.identity().labels.len();
            let mut c = Columns::default();
            if s.present {
                c.raw.extend_from_slice(&s.raw[..k]);
                c.calibrated.extend_from_slice(&s.calibrated[..k]);
                c.set.extend_from_slice(&s.set[..s.count]);
                c.end.push(s.count);
                c.abstention.push(s.abstention);
            }
            self.storage = Storage::Columns(Box::new(c));
        }
    }
    /// Import externally supplied rows. All policy/identity checks run here;
    /// private immutable columns keep subsequent `Layer::check` constant-time.
    pub fn from_rows(
        metadata: Arc<DecisionMetadata>,
        rows: impl IntoIterator<Item = Prediction>,
    ) -> Result<Self, String> {
        let mut out = Self::empty(metadata);
        let k = out.metadata.identity().labels.len();
        for row in rows {
            row.check_policy(out.metadata.policy(), out.metadata.cutoff())?;
            if row.raw.len() != k {
                return Err("calibrated layer label dimension mismatch".into());
            }
            out.columns();
            let Storage::Columns(c) = &mut out.storage else { unreachable!() };
            c.raw.extend(row.raw);
            c.calibrated.extend(row.calibrated);
            c.set.extend(row.set);
            c.end.push(c.set.len());
            c.abstention.push(row.abstention);
        }
        Ok(out)
    }
}
impl<L: Layer> Layer for Calibrated<L> {
    const ID: &'static str = "turbo:calibrated";
    const OF: Option<&'static LayerName> = Some(L::NAME);
    fn anchor() -> Anchor {
        Anchor::aligned::<L>()
    }
    fn len(&self) -> usize {
        match &self.storage {
            Storage::Small(s) => usize::from(s.present),
            Storage::Columns(c) => c.end.len(),
        }
    }
    fn check(&self) -> Result<(), String> {
        Ok(())
    }
}
impl<L> Clone for Calibrated<L> {
    fn clone(&self) -> Self {
        Self { metadata: self.metadata.clone(), storage: self.storage.clone(), _of: PhantomData }
    }
}
