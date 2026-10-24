//! The provenance config hash of a [`crate::SchemaAnnotator`]: FNV-1a over
//! a canonical byte encoding of everything besides the model weights that
//! changes what it writes. Strings and lists are length-prefixed, options
//! and enums carry a tag byte, and every float is hashed by its bits after
//! adding `0.0`, so `-0.0` and `0.0` agree and no formatting is involved.

use crate::schema::{Cardinality, Check, ExtractionSchema, Limits, Source};

struct Fnv(u64);

impl Fnv {
    fn bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn u8(&mut self, v: u8) {
        self.bytes(&[v]);
    }

    fn usize(&mut self, v: usize) {
        self.bytes(&(v as u64).to_le_bytes());
    }

    fn f32(&mut self, v: f32) {
        self.bytes(&(v + 0.0).to_bits().to_le_bytes());
    }

    fn str(&mut self, s: &str) {
        self.usize(s.len());
        self.bytes(s.as_bytes());
    }

    fn opt_str(&mut self, s: Option<&String>) {
        match s {
            None => self.u8(0),
            Some(s) => {
                self.u8(1);
                self.str(s);
            }
        }
    }

    fn opt_f32(&mut self, v: Option<f32>) {
        match v {
            None => self.u8(0),
            Some(v) => {
                self.u8(1);
                self.f32(v);
            }
        }
    }

    fn cardinality(&mut self, c: Cardinality) {
        self.u8(match c {
            Cardinality::Single => 0,
            Cardinality::Repeated => 1,
        });
    }

    fn checks(&mut self, checks: &[Check]) {
        let source = |s: Source| match s {
            Source::Glossary => 0u8,
            Source::Pii => 1,
            Source::Numbers => 2,
            Source::Locations => 3,
        };
        self.usize(checks.len());
        for c in checks {
            match *c {
                Check::Require(s) => {
                    self.u8(0);
                    self.u8(source(s));
                }
                Check::Boost(s, b) => {
                    self.u8(1);
                    self.u8(source(s));
                    self.f32(b);
                }
            }
        }
    }
}

/// The hash of `schema` at `threshold`, checked against `limits`, decoded
/// with span heads `max_width` words wide.
pub(crate) fn config_hash(schema: &ExtractionSchema, threshold: f32, limits: &Limits, max_width: usize) -> u64 {
    let mut h = Fnv(0xcbf2_9ce4_8422_2325);
    h.usize(schema.entities.len());
    for e in &schema.entities {
        h.str(&e.name);
        h.opt_str(e.description.as_ref());
        h.opt_f32(e.threshold);
        h.cardinality(e.cardinality);
        h.checks(&e.checks);
    }
    h.usize(schema.classifications.len());
    for t in &schema.classifications {
        h.str(&t.name);
        h.usize(t.labels.len());
        for l in &t.labels {
            h.str(&l.name);
            h.opt_str(l.description.as_ref());
        }
        h.u8(u8::from(t.multi_label));
        h.f32(t.threshold);
        match t.top_k {
            None => h.u8(0),
            Some(k) => {
                h.u8(1);
                h.usize(k);
            }
        }
    }
    h.usize(schema.structures.len());
    for s in &schema.structures {
        h.str(&s.name);
        h.usize(s.fields.len());
        for f in &s.fields {
            h.str(&f.name);
            h.opt_str(f.description.as_ref());
            h.opt_f32(f.threshold);
            h.cardinality(f.cardinality);
            match &f.choices {
                None => h.u8(0),
                Some(choices) => {
                    h.u8(1);
                    h.usize(choices.len());
                    for c in choices {
                        h.str(c);
                    }
                }
            }
            h.checks(&f.checks);
        }
    }
    let Limits {
        max_entity_types,
        max_classification_tasks,
        max_labels_per_task,
        max_structures,
        max_fields_per_structure,
        max_text_bytes,
        max_prompt_subwords,
        max_words,
    } = *limits;
    for v in [
        max_entity_types,
        max_classification_tasks,
        max_labels_per_task,
        max_structures,
        max_fields_per_structure,
        max_text_bytes,
        max_prompt_subwords,
        max_words,
    ] {
        h.usize(v);
    }
    h.f32(threshold);
    h.usize(max_width);
    h.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_span_width_and_every_limit_count() {
        let s = ExtractionSchema::default();
        let base = config_hash(&s, 0.5, &Limits::default(), 8);
        assert_eq!(base, config_hash(&s, 0.5, &Limits::default(), 8));
        assert_ne!(base, config_hash(&s, 0.5, &Limits::default(), 12), "max_width");
        let bumps: [fn(&mut Limits); 8] = [
            |l| l.max_entity_types += 1,
            |l| l.max_classification_tasks += 1,
            |l| l.max_labels_per_task += 1,
            |l| l.max_structures += 1,
            |l| l.max_fields_per_structure += 1,
            |l| l.max_text_bytes += 1,
            |l| l.max_prompt_subwords += 1,
            |l| l.max_words += 1,
        ];
        for (i, bump) in bumps.iter().enumerate() {
            let mut l = Limits::default();
            bump(&mut l);
            assert_ne!(base, config_hash(&s, 0.5, &l, 8), "limit {i}");
        }
    }
}
