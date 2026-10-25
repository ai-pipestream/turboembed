//! The provenance config hash of a [`crate::SchemaAnnotator`]: FNV-1a over
//! a canonical byte encoding of everything besides the model weights that
//! changes what it writes. Strings and lists are length-prefixed, options
//! and enums carry a tag byte, and every float is hashed by its bits after
//! adding `0.0`, so `-0.0` and `0.0` agree and no formatting is involved.

use turbo_analysis::Fnv1a;

use crate::schema::{Cardinality, Check, ExtractionSchema, Limits, Source};

/// Floats are hashed by their bits after adding `0.0`: like
/// [`Fnv1a::f32`] for finite values, but NaN payloads are kept.
fn f32_bits(h: &mut Fnv1a, v: f32) {
    h.u32((v + 0.0).to_bits());
}

fn opt_str(h: &mut Fnv1a, s: Option<&String>) {
    match s {
        None => {
            h.u8(0);
        }
        Some(s) => {
            h.u8(1);
            h.str(s);
        }
    }
}

fn opt_f32(h: &mut Fnv1a, v: Option<f32>) {
    match v {
        None => {
            h.u8(0);
        }
        Some(v) => {
            h.u8(1);
            f32_bits(h, v);
        }
    }
}

fn cardinality(h: &mut Fnv1a, c: Cardinality) {
    h.u8(match c {
        Cardinality::Single => 0,
        Cardinality::Repeated => 1,
    });
}

fn checks(h: &mut Fnv1a, checks: &[Check]) {
    let source = |s: Source| match s {
        Source::Glossary => 0u8,
        Source::Pii => 1,
        Source::Numbers => 2,
        Source::Locations => 3,
    };
    h.usize(checks.len());
    for c in checks {
        match *c {
            Check::Require(s) => {
                h.u8(0);
                h.u8(source(s));
            }
            Check::Boost(s, b) => {
                h.u8(1);
                h.u8(source(s));
                f32_bits(h, b);
            }
        }
    }
}

/// The hash of `schema` at `threshold`, checked against `limits`, decoded
/// with span heads `max_width` words wide.
pub(crate) fn config_hash(schema: &ExtractionSchema, threshold: f32, limits: &Limits, max_width: usize) -> u64 {
    let mut h = Fnv1a::new();
    h.usize(schema.entities.len());
    for e in &schema.entities {
        h.str(&e.name);
        opt_str(&mut h, e.description.as_ref());
        opt_f32(&mut h, e.threshold);
        cardinality(&mut h, e.cardinality);
        checks(&mut h, &e.checks);
    }
    h.usize(schema.classifications.len());
    for t in &schema.classifications {
        h.str(&t.name);
        h.usize(t.labels.len());
        for l in &t.labels {
            h.str(&l.name);
            opt_str(&mut h, l.description.as_ref());
        }
        h.u8(u8::from(t.multi_label));
        f32_bits(&mut h, t.threshold);
        match t.top_k {
            None => {
                h.u8(0);
            }
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
            opt_str(&mut h, f.description.as_ref());
            opt_f32(&mut h, f.threshold);
            cardinality(&mut h, f.cardinality);
            match &f.choices {
                None => {
                    h.u8(0);
                }
                Some(choices) => {
                    h.u8(1);
                    h.usize(choices.len());
                    for c in choices {
                        h.str(c);
                    }
                }
            }
            checks(&mut h, &f.checks);
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
    f32_bits(&mut h, threshold);
    h.usize(max_width);
    h.finish()
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
