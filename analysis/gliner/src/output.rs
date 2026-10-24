use crate::{ExtractionSchema, Model};
use std::sync::Arc;
use turbo_analysis::{
    Anchor, Annotator, Document, Error, Evidence, EvidenceStep, IxRange, Key, Layer, ModelRef, Refs, Span, Spans,
    Support, Tag, TagSet, View,
};
use turbo_analysis_layers::{Entities, Sentences, Tokens};

/// Checkpoint logits and decision threshold aligned to emitted entities.
/// Evidence support points here so thresholds are not mistaken for calibrated scores.
pub struct Decisions {
    pub logits: Vec<f32>,
    pub threshold: f32,
}
impl Layer for Decisions {
    const ID: &'static str = "turbo:gliner-decisions";
    fn anchor() -> Anchor {
        Anchor::aligned::<Entities>()
    }
    fn len(&self) -> usize {
        self.logits.len()
    }
    fn check(&self) -> Result<(), String> {
        if !self.threshold.is_finite()
            || !(0.0..=1.0).contains(&self.threshold)
            || self.logits.iter().any(|x| !x.is_finite())
        {
            return Err("gliner decisions require finite logits and threshold in [0,1]".into());
        }
        Ok(())
    }
}
pub struct GlinerAnnotator {
    model: Arc<Model>,
    schema: ExtractionSchema,
    kinds: Arc<TagSet>,
}
impl GlinerAnnotator {
    pub fn new(model: Arc<Model>, schema: ExtractionSchema) -> Self {
        let labels: Vec<_> = schema.labels().iter().map(String::as_str).collect();
        let kinds = TagSet::new("turbo:gliner-types", &labels);
        Self { model, schema, kinds }
    }
    /// Build evidence after this annotator's two outputs are added to a Document.
    /// Both default and named instances are supported. Thresholds live in the
    /// aligned Decisions layer and are linked by evidence support ranges.
    pub fn evidence(
        doc: &Document,
        entities: Key<Entities>,
        decisions: Key<Decisions>,
    ) -> Result<Evidence<Entities>, Error> {
        let fail = |s: &str| Error::Annotator { annotator: Self::ID, detail: s.into() };
        let d = doc.require(&decisions)?;
        if !doc.made_on(decisions.id()).contains(&entities.id()) {
            return Err(fail("gliner decision/entity instance mismatch"));
        }
        let mut ev = EvidenceStep::new(entities).build(doc)?;
        let index = ev.layers.len() as u32;
        Arc::make_mut(&mut ev.layers).push(turbo_analysis::export::LayerRef {
            layer: Decisions::ID.into(),
            instance: decisions.name.to_owned(),
        });
        let mut supports = Vec::new();
        let mut rows = vec![0];
        for i in 0..d.logits.len() {
            let (a, b) = if ev.support_row.is_empty() {
                (i, i + 1)
            } else {
                (ev.support_row[i] as usize, ev.support_row[i + 1] as usize)
            };
            supports.extend_from_slice(&ev.support[a..b]);
            supports.push(Support { layer: index, start: i as u32, end: i as u32 + 1 });
            rows.push(supports.len() as u32);
        }
        ev.support = supports;
        ev.support_row = rows;
        ev.raw = d.logits.iter().enumerate().map(|(i, &x)| (i as u32, f64::from(x))).collect();
        Ok(ev)
    }
}
impl Annotator for GlinerAnnotator {
    const ID: &'static str = "turbo:gliner";
    const VERSION: &'static str = "3";
    type Reads = (View, Tokens, Option<Sentences>);
    type Writes = (Entities, Decisions);
    fn model(&self) -> Option<ModelRef> {
        Some(self.model.identity().clone())
    }
    fn config_hash(&self) -> u64 {
        let config = serde_json::to_vec(self.model.config()).expect("finite validated config");
        config_hash(
            &config,
            self.schema.labels(),
            self.schema.threshold(),
            self.schema.mode() as u8,
            self.model.identity(),
        )
    }
    fn annotate(&self, (view, tokens, sentences): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let fail = |detail: String| Error::Annotator { annotator: Self::ID, detail };
        let windows = crate::window::pack_windows(&self.model, &self.schema, view, tokens, sentences).map_err(fail)?;
        let (starts, ends) = tokens.spans.columns();
        let mut entities = Entities {
            spans: Spans::new(view.id()),
            tokens: Vec::new(),
            kind: Vec::new(),
            kinds: self.kinds.clone(),
            score: Vec::new(),
        };
        let mut decisions = Decisions { logits: Vec::new(), threshold: self.schema.threshold() };
        for window in windows {
            let candidates =
                crate::decode(self.model.score_packed(&window.packed, &self.schema).map_err(fail)?, &self.schema);
            for c in candidates {
                let start = window.tokens.start + c.start;
                let end = window.tokens.start + c.end;
                entities.spans.push(Span::new(starts[start], ends[end - 1]));
                entities.tokens.push(IxRange::new(start as u32, end as u32));
                entities.kind.push(Tag(c.label as u16));
                entities.score.push(c.score as f64);
                decisions.logits.push(c.logit);
            }
        }
        Ok((entities, decisions))
    }
}

/// FNV-1a over the model config's JSON, the schema's labels, threshold
/// and decode mode, and the model identity. Every variable-length field
/// is length-prefixed, and the threshold's bits are taken after `+ 0.0`,
/// so `-0.0` and `0.0` hash alike.
fn config_hash(config: &[u8], labels: &[String], threshold: f32, mode: u8, model: &ModelRef) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    let mut eat = |bytes: &[u8]| {
        for &b in bytes {
            h = (h ^ u64::from(b)).wrapping_mul(0x100000001b3);
        }
    };
    let mut field = |bytes: &[u8]| {
        eat(&(bytes.len() as u64).to_le_bytes());
        eat(bytes);
    };
    field(config);
    field(&(labels.len() as u64).to_le_bytes());
    for label in labels {
        field(label.as_bytes());
    }
    field(&(threshold + 0.0).to_bits().to_le_bytes());
    field(&[mode]);
    for part in [&model.id, &model.revision, &model.sha256] {
        field(part.as_bytes());
    }
    h
}

#[cfg(test)]
mod hash_tests {
    use super::*;

    #[test]
    fn config_hash_is_fnv_and_folds_negative_zero_threshold() {
        let model = ModelRef { id: "m".into(), revision: "r".into(), sha256: "s".into() };
        let labels = vec!["person".to_string(), "place".to_string()];
        let hash = |t: f32, labels: &[String]| config_hash(b"{}", labels, t, 0, &model);
        assert_eq!(hash(-0.0, &labels), hash(0.0, &labels));
        assert_ne!(hash(0.5, &labels), hash(0.0, &labels));
        assert_ne!(hash(0.5, &labels), hash(0.5, &["personplace".to_string()]));
        assert_eq!(hash(0.5, &labels), hash(0.5, &labels));
    }
}
