//! The trained encoder as a bundle carries it: the adapters folded into
//! the base weights, and the tensors named and shaped as the core reads
//! them.

use std::collections::BTreeMap;

use turbo::manifest::TensorRole;
use turbo::status::Result;

use crate::real::Real;

use super::{Encoder, Named, layer, load, slot};

impl<R: Real> Encoder<R> {
    /// The base weights with the adapters folded in, W + s B A, in slot
    /// order: what a bundle carries.
    pub fn merged(&self) -> Vec<Vec<R>> {
        let mut out: Vec<Vec<R>> = self.params[..self.base_slots()].to_vec();
        let Some(a) = self.lora else {
            return out;
        };
        for l in self.lora_from()..self.geo.layers {
            for which in layer::PROJECTIONS {
                let (inp, _) = self.widths(which);
                let (sa, sb) = self.lora_slots(l, which);
                let (ma, mb) = (&self.params[sa], &self.params[sb]);
                for (wo, bo) in out[slot(l, which)].chunks_exact_mut(inp).zip(mb.chunks_exact(a.rank)) {
                    for (&b, ar) in bo.iter().zip(ma.chunks_exact(inp)) {
                        for (w, &av) in wo.iter_mut().zip(ar) {
                            *w += a.scale::<R>() * b * av;
                        }
                    }
                }
            }
        }
        out
    }

    /// The merged weights as f32 tensors named by `names` (a bundle
    /// artifact's `tensor_names`, with `{layer}`), shaped as the core
    /// reads them: what a bundle's weights file holds.
    pub fn named(&self, names: &BTreeMap<TensorRole, String>) -> Result<Vec<Named>> {
        let (h, i) = (self.geo.hidden as u64, self.geo.intermediate as u64);
        let merged = self.merged();
        let mut out = Vec::new();
        for (role, layer, s, shape) in load::roles(self.geo.layers, self.vocab as u64, h, i) {
            if role == TensorRole::AttnLnWeight && layer == Some(0) {
                continue;
            }
            let name = load::name(names, role, layer)?;
            out.push((name, shape, merged[s].iter().map(|v| v.to_f64() as f32).collect()));
        }
        Ok(out)
    }
}
