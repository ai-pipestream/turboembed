//! The detector as an [`Annotator`]: reads the view, writes `turbo:pii`
//! and `turbo:pii-candidates`, and folds its configuration into one hash.

use turbo_analysis::{Annotator, Error, Refs, View};
use turbo_analysis_layers::{Pii, PiiCandidates};

use super::PiiDetector;
use crate::scan::Extractor;

impl Annotator for PiiDetector {
    const ID: &'static str = "turbo-pii:pii";
    type Reads = (View,);
    type Writes = (Pii, PiiCandidates);

    fn annotate(&self, (view,): Refs<'_, Self>) -> Result<(Pii, PiiCandidates), Error> {
        let text = view.text();
        if u32::try_from(text.len()).is_err() {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!("a text of {} bytes is past the u32 offset range", text.len()),
            });
        }
        Ok(self.layer_with_candidates(text, view.id()))
    }

    fn requires(&self) -> &[turbo_analysis::KeyId] {
        const REQUIRES: &[turbo_analysis::KeyId] = &[turbo_analysis::Key::<View>::DEFAULT.id()];
        REQUIRES
    }

    fn produces(&self) -> &[turbo_analysis::KeyId] {
        const PRODUCES: &[turbo_analysis::KeyId] =
            &[turbo_analysis::Key::<Pii>::DEFAULT.id(), turbo_analysis::Key::<PiiCandidates>::DEFAULT.id()];
        PRODUCES
    }

    /// FNV-1a over each scanner's extractor and kind id bits, plus any
    /// custom-kind table and configured allow lists and deny lists. Bits are
    /// keyed on stable kind ids, so adding a kind leaves the hash of every
    /// existing configuration unchanged.
    fn config_hash(&self) -> u64 {
        let mut h = turbo_analysis::Fnv1a::new();
        for s in &self.scanners {
            let bits = s.kinds().bits();
            h.fold_u64(s.extractor().tag()).fold_u64(bits as u64);
            if bits >> 64 != 0 {
                h.u8(0xb1).fold_u64((bits >> 64) as u64);
            }
            if let Extractor::Custom(name) = s.extractor() {
                h.bytes(name.as_bytes()).u8(0xff);
                for &k in s.custom_kinds() {
                    h.u32(k.code()).u8(0xff);
                }
                h.fold_u64(s.custom_config_hash());
            }
        }
        if self.keep_nested {
            h.fold_u64(1);
        }
        if let Some(min) = self.min_score {
            h.u8(0x5c).f32(min);
        }
        if !self.custom_kinds.is_empty() {
            h.u8(0xc1).usize(self.custom_kinds.len());
            for name in self.custom_kinds.names() {
                h.str(name);
            }
        }
        self.window.fold_hash(&mut h);
        if !self.global_allow.is_empty() {
            h.u8(0xa1);
            self.global_allow.fold_hash(&mut h);
        }
        if !self.kind_allow.is_empty() {
            h.u8(0xa2).usize(self.kind_allow.len());
            for (kind, allow) in &self.kind_allow {
                h.u32(kind.code());
                allow.fold_hash(&mut h);
            }
        }
        if !self.deny_lists.is_empty() {
            h.u8(0xd1).usize(self.deny_lists.len());
            for deny in &self.deny_lists {
                deny.fold_hash(&mut h);
            }
        }
        h.finish()
    }
}
