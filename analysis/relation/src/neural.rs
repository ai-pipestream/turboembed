use std::sync::Arc;

use turbo_analysis::{Edge, Edges, Ix, Layer, Span, Spans, Tag, TagSet, View};
use turbo_analysis_layers::{Entities, Tokens};

use crate::Relations;

#[derive(Clone, Debug)]
pub struct Config {
    pub labels: Vec<String>,
    pub threshold: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self { labels: Vec::new(), threshold: 0.5 }
    }
}

impl Config {
    pub fn set(&mut self, labels: Vec<String>, threshold: f32) -> Result<(), String> {
        if labels.is_empty() || labels.iter().any(|l| l.trim().is_empty()) {
            return Err("relation labels must not be empty".into());
        }
        if labels.len() > u16::MAX as usize + 1 {
            return Err("too many relation labels".into());
        }
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err(format!("a threshold outside [0, 1]: {threshold}"));
        }
        self.labels = labels;
        self.threshold = threshold;
        Ok(())
    }

    pub fn hash(&self) -> u64 {
        let mut h = turbo_analysis::Fnv1a::new();
        h.u64(self.threshold.to_bits() as u64).usize(self.labels.len());
        for label in &self.labels {
            h.str(label);
        }
        h.finish()
    }
}

type EntityMap = (Entities, Vec<Option<Ix<Entities>>>);

pub fn entities(
    view: &View,
    tokens: &Tokens,
    existing: Option<&Entities>,
    needed: &[Span],
) -> Result<EntityMap, String> {
    if tokens.spans.view() != view.id() || existing.is_some_and(|e| e.spans.view() != view.id()) {
        return Err("tokens or entities are in another view".into());
    }
    tokens.check()?;
    if let Some(e) = existing {
        e.check()?;
    }
    let mut names: Vec<String> = existing
        .map(|e| (0..e.kinds.len()).map(|i| e.kinds.label(Tag(i as u16)).to_owned()).collect())
        .unwrap_or_default();
    if !names.iter().any(|n| n == "entity") {
        names.push("entity".into());
    }
    let kinds = TagSet::new("neural-relation-entities", &names.iter().map(String::as_str).collect::<Vec<_>>());
    let old: Vec<usize> = existing
        .map(|e| {
            (0..e.kinds.len()).map(|i| names.iter().position(|x| x == e.kinds.label(Tag(i as u16))).unwrap()).collect()
        })
        .unwrap_or_default();
    let generic = Tag(names.iter().position(|n| n == "entity").unwrap() as u16);
    let mut out = Entities {
        spans: existing.map(|e| e.spans.clone()).unwrap_or_else(|| Spans::new(view.id())),
        tokens: Vec::new(),
        kind: Vec::new(),
        kinds,
        score: Vec::new(),
    };
    if let Some(e) = existing {
        for (i, _) in e.spans.iter() {
            out.tokens.push(e.tokens[i.get()]);
            out.kind.push(Tag(old[e.kind[i.get()].0 as usize] as u16));
            out.score.push(e.score[i.get()]);
        }
    }
    let mut mapping = Vec::with_capacity(needed.len());
    for &span in needed {
        let found = (0..out.spans.len())
            .find(|&i| !out.spans.is_discontinuous(Ix::new(i as u32)) && out.spans.get(Ix::new(i as u32)) == span);
        let i = match found {
            Some(i) => i,
            None => {
                out.spans.push(span);
                out.tokens.push(tokens.spans.within(span));
                out.kind.push(generic);
                out.score.push(1.0);
                out.spans.len() - 1
            }
        };
        mapping.push(Some(Ix::new(i as u32)));
    }
    Ok((out, mapping))
}

pub fn layers(
    view: &View,
    entities: &Entities,
    labels: &[String],
    found: &[(Ix<Entities>, Ix<Entities>, usize, f32)],
) -> Result<(Relations, Edges<Entities, Entities>), String> {
    let kinds = TagSet::new("relation", &labels.iter().map(String::as_str).collect::<Vec<_>>());
    let mut relations = Relations {
        spans: Spans::new(view.id()),
        subject: Vec::new(),
        object: Vec::new(),
        kind: Vec::new(),
        kinds: Arc::clone(&kinds),
        score: Vec::new(),
    };
    for &(subject, object, label, score) in found {
        if subject == object {
            continue;
        }
        let a = entities.spans.get(subject);
        let b = entities.spans.get(object);
        relations.spans.push(Span::new(a.start.min(b.start), a.end.max(b.end)));
        relations.subject.push(subject);
        relations.object.push(object);
        relations.kind.push(Tag(label as u16));
        relations.score.push(f64::from(score));
    }
    relations.check()?;
    let mut edges: Vec<Edge<Entities, Entities>> = Vec::new();
    for &(subject, object, label, score) in found {
        if subject == object {
            continue;
        }
        let edge = Edge::new(subject, object, Tag(label as u16), f64::from(score));
        if let Some(old) = edges.iter_mut().find(|e| (e.from, e.to, e.kind) == (edge.from, edge.to, edge.kind)) {
            old.score = old.score.max(edge.score);
        } else {
            edges.push(edge);
        }
    }
    Ok((relations, Edges::new(kinds, edges)))
}
