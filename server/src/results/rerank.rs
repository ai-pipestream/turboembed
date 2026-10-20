//! RerankScoresResult: the turbo-analysis-embed `Reranked` layer (issue
//! #707) with its document-local indices and keys made plain, the model
//! identity riding along, not a bare float.

use std::collections::BTreeSet;

use turbo_analysis::{Layer, ModelRef};
use turbo_analysis_embed::Reranked;

use super::{MAX_RERANK_ITEMS, Wire, mismatched, model_from_proto, model_to_proto, over};
use crate::api::{Failure, TURBO_E_INVALID_ARGUMENT};
use crate::proto::result::RerankScoresResult;

/// The reranked shortlist unbundled from its layer: the source layer's
/// name, its items in relevance order, one finite descending score per
/// item, and the scoring model's identity.
#[derive(Debug)]
pub struct RerankScores {
    pub model: ModelRef,
    /// The layer the items index, its full name (turbo:levels, say).
    pub source: String,
    pub items: Vec<u32>,
    pub scores: Vec<f32>,
}

impl RerankScores {
    /// The layer result the rerank came from.
    pub fn new<L: Layer>(r: &Reranked<L>) -> RerankScores {
        RerankScores {
            model: r.model.clone(),
            source: r.source.id().layer.to_string(),
            items: r.items.iter().map(|i| i.get() as u32).collect(),
            scores: r.scores.clone(),
        }
    }

    /// The layer's own check, re-proved on the plain columns: one finite
    /// score per item, descending, the items unique.
    fn check(&self) -> Result<(), Failure> {
        if self.scores.len() != self.items.len() {
            return Err(mismatched("score", self.scores.len(), self.items.len()));
        }
        if self.scores.iter().any(|s| !s.is_finite()) {
            return Err(Failure::new(TURBO_E_INVALID_ARGUMENT, 0, "one finite relevance score per item"));
        }
        if self.scores.windows(2).any(|w| w[0] < w[1]) {
            return Err(Failure::new(TURBO_E_INVALID_ARGUMENT, 0, "reranked scores must be descending"));
        }
        if self.items.iter().collect::<BTreeSet<_>>().len() != self.items.len() {
            return Err(Failure::new(TURBO_E_INVALID_ARGUMENT, 0, "reranked items must be unique"));
        }
        Ok(())
    }
}

impl Wire for RerankScores {
    type Msg = RerankScoresResult;

    fn to_msg(&self) -> RerankScoresResult {
        RerankScoresResult {
            model: Some(model_to_proto(&self.model)),
            source: self.source.clone(),
            item: self.items.clone(),
            score: self.scores.clone(),
        }
    }

    fn from_msg(msg: &RerankScoresResult) -> Result<Self, Failure> {
        if msg.item.len() as u64 > MAX_RERANK_ITEMS {
            return Err(over("max_rerank_items", "reranked items", msg.item.len() as u64));
        }
        let out = RerankScores {
            model: msg.model.as_ref().map(model_from_proto).unwrap_or(ModelRef {
                id: String::new(),
                revision: String::new(),
                sha256: String::new(),
            }),
            source: msg.source.clone(),
            items: msg.item.clone(),
            scores: msg.score.clone(),
        };
        out.check()?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::TURBO_E_CAPACITY;
    use turbo_analysis::Key;
    use turbo_analysis_layers::Chunks;

    fn model() -> ModelRef {
        ModelRef { id: "org/reranker".into(), revision: "v2".into(), sha256: "f00d".into() }
    }

    /// The layer result a rerank annotator writes, over two chunks.
    fn sample() -> Reranked<Chunks> {
        let r = Reranked {
            source: Key::named("candidates"),
            items: vec![turbo_analysis::Ix::new(7), turbo_analysis::Ix::new(2)],
            scores: vec![9.5, 1.25],
            model: model(),
        };
        r.check().unwrap();
        r
    }

    #[test]
    fn the_wire_round_trip_is_unchanged() {
        let r = sample();
        let m = RerankScores::new(&r);
        assert_eq!(m.source, "turbo:chunks");
        let back = RerankScores::from_msg(&m.to_msg()).unwrap();
        assert_eq!(back.model, r.model);
        assert_eq!(back.items, r.items.iter().map(|i| i.get() as u32).collect::<Vec<_>>());
        assert_eq!(back.scores, r.scores);
    }

    fn refused(mut f: impl FnMut(&mut RerankScoresResult)) -> Failure {
        let mut msg = RerankScores::new(&sample()).to_msg();
        f(&mut msg);
        RerankScores::from_msg(&msg).unwrap_err()
    }

    #[test]
    fn garbage_is_refused_without_panicking() {
        let e = refused(|m| {
            m.score.pop();
        });
        assert!(e.message.contains("score"), "{e:?}");
        let e = refused(|m| {
            m.score[0] = f32::INFINITY;
        });
        assert!(e.message.contains("finite"), "{e:?}");
        let e = refused(|m| {
            m.score.swap(0, 1);
        });
        assert!(e.message.contains("descending"), "{e:?}");
        let e = refused(|m| {
            m.item[1] = 7;
        });
        assert!(e.message.contains("unique"), "{e:?}");
        // Equal scores are allowed; the layer's own check agrees.
        let mut msg = RerankScores::new(&sample()).to_msg();
        msg.score[1] = msg.score[0];
        let back = RerankScores::from_msg(&msg).unwrap();
        assert_eq!(back.scores[0], back.scores[1]);
        // Nothing at all: an empty rerank is a valid abstention.
        let back = RerankScores::from_msg(&RerankScoresResult::default()).unwrap();
        assert!(back.items.is_empty() && back.scores.is_empty());
    }

    #[test]
    fn over_the_item_limit_names_it() {
        let msg = RerankScoresResult {
            item: vec![1; MAX_RERANK_ITEMS as usize + 1],
            score: vec![1.0; MAX_RERANK_ITEMS as usize + 1],
            ..Default::default()
        };
        let e = RerankScores::from_msg(&msg).unwrap_err();
        assert_eq!(e.code, TURBO_E_CAPACITY);
        assert!(e.message.contains("max_rerank_items"), "{e:?}");
    }
}
