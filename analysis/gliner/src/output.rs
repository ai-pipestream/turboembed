use crate::{ExtractionSchema, Model};
use std::{
    hash::{Hash, Hasher},
    sync::Arc,
};
use turbo_analysis::{
    Anchor, Annotator, Document, Error, Evidence, EvidenceStep, IxRange, Key, Layer, ModelRef, Refs, Span, Spans,
    Support, Tag, TagSet, View,
};
use turbo_analysis_layers::{Entities, Tokens};

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
    const VERSION: &'static str = "1";
    type Reads = (View, Tokens);
    type Writes = (Entities, Decisions);
    fn model(&self) -> Option<ModelRef> {
        Some(self.model.identity().clone())
    }
    fn config_hash(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        serde_json::to_vec(self.model.config()).expect("finite validated config").hash(&mut h);
        self.schema.labels().hash(&mut h);
        self.schema.threshold().to_bits().hash(&mut h);
        (self.schema.mode() as u8).hash(&mut h);
        self.model.identity().id.hash(&mut h);
        self.model.identity().revision.hash(&mut h);
        self.model.identity().sha256.hash(&mut h);
        h.finish()
    }
    fn annotate(&self, (view, tokens): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let fail = |detail: String| Error::Annotator { annotator: Self::ID, detail };
        tokens.check().map_err(fail)?;
        if tokens.spans.view() != view.id() || tokens.spans.fragment_columns().is_some() {
            return Err(fail("gliner needs contiguous tokens in its input view".into()));
        }
        let (starts, ends) = tokens.spans.columns();
        if starts.len() > crate::schema::MAX_TOKENS {
            return Err(fail("max_tokens limit exceeded".into()));
        }
        let words = starts
            .iter()
            .zip(ends)
            .map(|(&a, &b)| {
                view.text().get(a as usize..b as usize).ok_or_else(|| fail("gliner token UTF-8 bounds".into()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let candidates = self.model.predict(&words, &self.schema).map_err(fail)?;
        let mut entities = Entities {
            spans: Spans::new(view.id()),
            tokens: Vec::new(),
            kind: Vec::new(),
            kinds: self.kinds.clone(),
            score: Vec::new(),
        };
        let mut decisions = Decisions { logits: Vec::new(), threshold: self.schema.threshold() };
        for c in candidates {
            entities.spans.push(Span::new(starts[c.start], ends[c.end - 1]));
            entities.tokens.push(IxRange::new(c.start as u32, c.end as u32));
            entities.kind.push(Tag(c.label as u16));
            entities.score.push(c.score as f64);
            decisions.logits.push(c.logit);
        }
        Ok((entities, decisions))
    }
}
