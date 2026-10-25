use turbo_analysis::{DocPredicate, Document, Key, KeyId, ViewId};

use super::{QualityFlags, TextQuality};

#[derive(Clone, Copy)]
enum Rule {
    Score(f32, QualityFlags),
    Lexical(f32, f32),
}

/// A view-specific quality predicate carrying its input key and configuration.
/// Use ConditionalStep::from_predicate to bind both automatically.
#[derive(Clone, Copy)]
pub struct QualityPredicate {
    key: Key<TextQuality>,
    view: ViewId,
    rule: Rule,
}

pub fn quality_gate(view: ViewId, min_score: f32, forbidden: QualityFlags) -> QualityPredicate {
    quality_gate_on(Key::DEFAULT, view, min_score, forbidden)
}

pub fn quality_gate_on(
    key: Key<TextQuality>,
    view: ViewId,
    min_score: f32,
    forbidden: QualityFlags,
) -> QualityPredicate {
    QualityPredicate { key, view, rule: Rule::Score(min_score, forbidden) }
}

pub fn lexical_quality_gate(view: ViewId, min_density: f32, min_mtld: f32) -> QualityPredicate {
    lexical_quality_gate_on(Key::DEFAULT, view, min_density, min_mtld)
}

pub fn lexical_quality_gate_on(
    key: Key<TextQuality>,
    view: ViewId,
    min_density: f32,
    min_mtld: f32,
) -> QualityPredicate {
    QualityPredicate { key, view, rule: Rule::Lexical(min_density, min_mtld) }
}

impl DocPredicate for QualityPredicate {
    fn test(&self, doc: &Document) -> bool {
        let Some(q) = doc.get_key(&self.key).filter(|q| q.view() == self.view) else {
            return false;
        };
        match self.rule {
            Rule::Score(min, forbidden) => {
                min.is_finite()
                    && (0.0..=1.0).contains(&min)
                    && forbidden.0 & !QualityFlags::ALL.0 == 0
                    && q.metrics.composite_score >= min
                    && !q.metrics.flags.intersects(forbidden)
            }
            Rule::Lexical(density, mtld) => {
                density.is_finite()
                    && (0.0..=1.0).contains(&density)
                    && mtld.is_finite()
                    && mtld >= 0.0
                    && q.metrics.information_density >= density
                    && q.metrics.mtld_lexical_diversity >= mtld
            }
        }
    }
    fn reads(&self) -> Vec<KeyId> {
        vec![self.key.id()]
    }
    fn config_hash(&self) -> u64 {
        let (tag, a, b) = match self.rule {
            Rule::Score(min, flags) => (0u8, min.to_bits(), u32::from(flags.0)),
            Rule::Lexical(density, mtld) => (1, density.to_bits(), mtld.to_bits()),
        };
        b"quality-predicate-v1"
            .iter()
            .copied()
            .chain(self.key.name.bytes())
            .chain([0xff, tag])
            .chain(self.view.0.to_le_bytes())
            .chain(a.to_le_bytes())
            .chain(b.to_le_bytes())
            .fold(0xcbf2_9ce4_8422_2325u64, |h, byte| (h ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3))
    }
}
