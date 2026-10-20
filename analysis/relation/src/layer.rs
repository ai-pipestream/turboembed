//! The extracted relations as a layer: one row per mention, in
//! extraction order (entity layer order, then pattern order),
//! each covering both entity spans. Duplicate rows are kept, as
//! the reference layer keeps duplicate pattern registrations; the
//! companion `Edges<Entities, Entities>` the annotator also
//! writes cannot hold duplicates and keeps each triple once.

use std::sync::Arc;

use turbo_analysis::{Anchor, Ix, Key, KeyId, Layer, Overlap, Spans, Tag, TagSet, TextSpans};
use turbo_analysis_layers::Entities;

/// Typed relations between entities: `opennlp:relations`. Rows
/// are not sorted or deduplicated; `kinds` holds the pattern
/// types in registration order.
#[derive(Clone, Debug)]
pub struct Relations {
    /// The span covering both entity spans, per mention.
    pub spans: Spans<Relations>,
    pub subject: Vec<Ix<Entities>>,
    pub object: Vec<Ix<Entities>>,
    pub kind: Vec<Tag>,
    pub kinds: Arc<TagSet>,
    /// The matching pattern's confidence, per mention.
    pub score: Vec<f64>,
}

impl Relations {
    /// Row `i` as a validated mention.
    pub fn mention(&self, i: Ix<Relations>) -> (Ix<Entities>, Ix<Entities>, &str, f64) {
        let r = i.get();
        (self.subject[r], self.object[r], self.kinds.label(self.kind[r]), self.score[r])
    }
}

impl Layer for Relations {
    const ID: &'static str = "opennlp:relations";
    const OVERLAP: Overlap = Overlap::Any;

    fn anchor() -> Anchor {
        Anchor::Text
    }

    fn len(&self) -> usize {
        self.spans.len()
    }

    fn text_spans(&self) -> Option<TextSpans<'_>> {
        Some(self.spans.text_spans())
    }

    fn indexes(&self) -> Vec<(KeyId, usize)> {
        let bound = self.subject.iter().zip(&self.object).map(|(s, o)| s.get().max(o.get()) + 1).max().unwrap_or(0);
        vec![(Key::<Entities>::DEFAULT.id(), bound)]
    }

    fn check(&self) -> Result<(), String> {
        self.spans.check(Self::OVERLAP)?;
        let n = self.spans.len();
        if self.subject.len() != n || self.object.len() != n || self.kind.len() != n || self.score.len() != n {
            return Err("a subject, an object, a kind and a score for every relation".into());
        }
        if self.kind.iter().any(|k| k.0 as usize >= self.kinds.len()) {
            return Err(format!("a kind outside {}", self.kinds.name));
        }
        if let Some(i) = (0..n).find(|&i| self.subject[i] == self.object[i]) {
            return Err(format!("relation {i} has itself as subject and object"));
        }
        if let Some(s) = self.score.iter().find(|s| !s.is_finite() || **s < 0.0 || **s > 1.0) {
            return Err(format!("a relation score outside [0, 1]: {s}"));
        }
        Ok(())
    }
}
