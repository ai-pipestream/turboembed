use super::*;
use crate::Calibrated;
use std::marker::PhantomData;
use turbo_calibration::{Abstention, Identity, Policy, Prediction, Task};
impl<L: Layer> LayerCodec for Calibrated<L> {
    fn export(&self, _: &ExportCx) -> Result<Vec<u8>, ExportError> {
        self.check().map_err(malformed)?;
        let mut w = Writer::new();
        let i = &self.identity;
        for s in [self.calibration_id.as_ref(), &i.model, &i.revision, &i.sha256, &i.domain] {
            w.str(s)?;
        }
        w.u8(match i.task {
            Task::Categorization => 0,
            Task::TokenTagging => 1,
        });
        w.u64(i.inference_hash);
        w.usize(i.labels.len())?;
        for label in &i.labels {
            w.str(label)?;
        }
        w.f64(self.policy.coverage);
        w.usize(self.policy.max_set_size)?;
        w.f64(self.policy.min_confidence);
        w.f64(self.cutoff);
        w.usize(self.rows.len())?;
        for row in &self.rows {
            put_f64s(&mut w, &row.raw)?;
            put_f64s(&mut w, &row.calibrated)?;
            w.usize(row.set.len())?;
            for &v in &row.set {
                w.usize(v)?;
            }
            w.u8(match row.abstention {
                None => 0,
                Some(Abstention::EmptySet) => 1,
                Some(Abstention::SetTooLarge) => 2,
                Some(Abstention::BelowThreshold) => 3,
            });
        }
        Ok(w.finish())
    }
    fn import(data: &[u8], _: &mut ImportCx, _: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let (calibration_id, model, revision, sha256, domain) =
            (r.string()?, r.string()?, r.string()?, r.string()?, r.string()?);
        let task = match r.u8()? {
            0 => Task::Categorization,
            1 => Task::TokenTagging,
            _ => return Err(malformed("unknown calibrated task")),
        };
        let inference_hash = r.u64()?;
        let mut labels = Vec::new();
        let n = r.usize()?;
        if n > 65536 {
            return Err(malformed("too many calibrated labels"));
        }
        for _ in 0..n {
            labels.push(r.string()?);
        }
        let identity = Identity { model, revision, sha256, domain, task, labels, inference_hash };
        let policy = Policy { coverage: r.f64()?, max_set_size: r.usize()?, min_confidence: r.f64()? };
        let cutoff = r.f64()?;
        let mut rows = Vec::new();
        for _ in 0..r.usize()? {
            let (raw, calibrated) = (get_f64s(&mut r)?, get_f64s(&mut r)?);
            let mut set = Vec::new();
            for _ in 0..r.usize()? {
                set.push(r.usize()?);
            }
            let abstention = match r.u8()? {
                0 => None,
                1 => Some(Abstention::EmptySet),
                2 => Some(Abstention::SetTooLarge),
                3 => Some(Abstention::BelowThreshold),
                _ => return Err(malformed("unknown abstention reason")),
            };
            rows.push(Prediction { raw, calibrated, set, abstention });
        }
        done(&r, "calibrated layer")?;
        let out = Self {
            identity: identity.into(),
            calibration_id: calibration_id.into(),
            policy,
            cutoff,
            rows,
            _of: PhantomData,
        };
        out.check().map_err(malformed)?;
        Ok(out)
    }
}
