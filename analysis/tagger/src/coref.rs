//! An annotator over a bundle's coreference ranker head: [`CorefTagger`]
//! reads the mentions of the entity layer (as a [`crate::SpanTagger`]
//! writes them from a span classifier head, or any entity layer), links
//! each to its antecedent, and writes the chains as groups of entities
//! and each link as an edge from the mention to its antecedent.
//!
//! The view's tokens are read as one sequence of words, as the sentence
//! tagger reads them, since a chain crosses sentences. Mentions are taken
//! in order of first token then last; entities over the same tokens are
//! one mention, so they share a chain. A mention longer than the head's
//! `span.max_words` is not scored and stays a chain of its own.

use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};

use turbo::bundle::Bundle;
use turbo::head::CorefRanker;
use turbo::head::coref::chains;
use turbo::head::tagger::Work;
use turbo::manifest::HeadKind;
use turbo::status::Error as CoreError;
use turbo_analysis::{
    Annotator, Edge, Edges, Error, Execution, Fnv1a, Groups, Ix, Layer, ModelRef, Refs, TagSet, View,
};
use turbo_analysis_embed::EncoderOptions;
use turbo_analysis_layers::{Entities, Subwords, Tokens};

use crate::annotate::{HeadRunner, find_head, model_ref, refuse};
use crate::words::WordEncoder;

/// The layer id a coreference head names.
pub const COREF_LAYER: &str = "turboembed:coref";

/// Writes coreference chains over the entity layer from a
/// HEAD_COREF_RANKER head.
pub struct CorefTagger {
    head: CorefRanker,
    words: Mutex<WordEncoder>,
    model: ModelRef,
    execution: Execution,
    width: usize,
    kinds: Arc<TagSet>,
}

/// The chains and links of one view.
pub type Resolution = (Groups<Entities>, Edges<Entities, Entities>);

impl CorefTagger {
    /// The head named `name` in the bundle at `dir`, or with no name the
    /// one head whose `layer` is `turboembed:coref`.
    pub fn open(dir: &Path, name: Option<&str>, o: &EncoderOptions) -> Result<CorefTagger, CoreError> {
        let bundle = Bundle::open(dir)?;
        let spec = find_head(&bundle, name, COREF_LAYER, HeadKind::CorefRanker)?;
        let head = CorefRanker::load(&bundle, &spec.name)?;
        let words = WordEncoder::open(dir, o)?;
        if words.hidden() != head.hidden() {
            return Err(refuse(format!(
                "the head reads {} values a word, the encoder gives {}",
                head.hidden(),
                words.hidden()
            )));
        }
        Ok(CorefTagger {
            execution: words.execution().clone(),
            width: words.width(),
            model: model_ref(&bundle),
            kinds: TagSet::new(&spec.name, &["antecedent"]),
            words: Mutex::new(words),
            head,
        })
    }

    pub fn head(&self) -> &CorefRanker {
        &self.head
    }

    /// The chains of `entities` (every entity in one, singletons
    /// included, in order of first member) and the link from each linked
    /// mention's entities to its antecedent's first entity, scored by the
    /// antecedent's softmax probability among the mention's candidates.
    pub fn resolve(
        &self,
        view: &View,
        tokens: &Tokens,
        entities: &Entities,
        subwords: Option<&Subwords>,
    ) -> Result<Resolution, String> {
        let all = Tokens { spans: tokens.spans.clone(), sentence: None };
        let seqs = HeadRunner::sequences(view, &all, None)?;
        let n = tokens.spans.len();
        entities.check().map_err(|e| format!("the entities: {e}"))?;
        if entities.spans.view() != view.id() {
            return Err(format!("entities over {:?}, tokens over {:?}", entities.spans.view(), view.id()));
        }
        if entities.tokens.iter().any(|r| r.end.get() > n || r.start >= r.end) {
            return Err("an entity outside the tokens, or of no tokens".into());
        }
        let max = self.head.spec().span.map_or(0, |s| s.max_words);
        // Each distinct token range in order, and each entity's place.
        let mut ranges: Vec<(u32, u32)> =
            entities.tokens.iter().map(|r| (r.start.get() as u32, r.end.get() as u32 - 1)).collect();
        ranges.sort_unstable();
        ranges.dedup();
        let scored: Vec<(u32, u32)> = ranges.iter().copied().filter(|&(i, j)| j - i < max).collect();
        let mut links: Vec<Option<(u32, f64)>> = vec![None; scored.len()];
        if !scored.is_empty() {
            let pooling = self.head.spec().word_pooling;
            let mut words = self.words.lock().unwrap_or_else(PoisonError::into_inner);
            let features = words.token_features(view, &all, &seqs, subwords, pooling)?;
            drop(words);
            let ids: Vec<u32> = (0..n as u32).collect();
            let (mut g, mut scores, mut best, mut w) = (Vec::new(), Vec::new(), Vec::new(), Work::default());
            self.head.mentions(&features[0], &ids, &scored, &mut g, &mut w).map_err(|e| e.to_string())?;
            self.head.scores(&g, &mut scores).map_err(|e| e.to_string())?;
            self.head.decode(&scores, &mut best).map_err(|e| e.to_string())?;
            let k = self.head.antecedents();
            for (a, b) in best.iter().enumerate() {
                let Some(b) = *b else { continue };
                let row = &scores[a * k..][..k.min(a)];
                let m = row.iter().fold(0f32, |x, &y| x.max(y)) as f64;
                let z = (-m).exp() + row.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>();
                let s = row[a - 1 - b as usize] as f64;
                links[a] = Some((b, (s - m).exp() / z));
            }
        }
        // Chains over the scored mentions; the rest stand alone.
        let antecedents: Vec<Option<u32>> = links.iter().map(|l| l.map(|(b, _)| b)).collect();
        let chain = chains(&antecedents).map_err(|e| e.to_string())?;
        let place = |r: (u32, u32)| scored.binary_search(&r).ok();
        let entity_range = |e: usize| {
            let r = &entities.tokens[e];
            (r.start.get() as u32, r.end.get() as u32 - 1)
        };
        let mut groups: Vec<Vec<Ix<Entities>>> = Vec::new();
        let mut group_of: Vec<Option<usize>> = vec![None; scored.len()];
        let mut alone: Vec<(u32, u32, usize)> = Vec::new();
        let mut order: Vec<usize> = (0..entities.tokens.len()).collect();
        order.sort_by_key(|&e| (entity_range(e), e));
        let mut first_entity: Vec<Option<usize>> = vec![None; scored.len()];
        for e in order {
            let r = entity_range(e);
            let Some(p) = place(r) else {
                // An unscored range: entities over it are one chain.
                match alone.iter().find(|(i, j, _)| (*i, *j) == r) {
                    Some(&(_, _, g)) => groups[g].push(Ix::new(e as u32)),
                    None => {
                        alone.push((r.0, r.1, groups.len()));
                        groups.push(vec![Ix::new(e as u32)]);
                    }
                }
                continue;
            };
            first_entity[p].get_or_insert(e);
            let root = chain[p] as usize;
            let g = *group_of[root].get_or_insert_with(|| {
                groups.push(Vec::new());
                groups.len() - 1
            });
            groups[g].push(Ix::new(e as u32));
        }
        let mut rows = Vec::new();
        for (e, r) in entities.tokens.iter().enumerate() {
            let range = (r.start.get() as u32, r.end.get() as u32 - 1);
            let Some(p) = place(range) else { continue };
            if let Some((b, score)) = links[p] {
                let to = first_entity[b as usize].ok_or("an antecedent with no entity")?;
                rows.push(Edge::new(Ix::new(e as u32), Ix::new(to as u32), turbo_analysis::Tag(0), score));
            }
        }
        Ok((Groups::new(groups), Edges::new(self.kinds.clone(), rows)))
    }
}

impl Annotator for CorefTagger {
    const ID: &'static str = "turbo-tagger-coref";
    type Reads = (View, Tokens, Entities, Option<Subwords>);
    type Writes = (Groups<Entities>, Edges<Entities, Entities>);

    fn annotate(&self, (view, tokens, entities, subwords): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        self.resolve(view, tokens, entities, subwords)
            .map_err(|detail| Error::Annotator { annotator: Self::ID, detail })
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.model.clone())
    }

    fn execution(&self) -> Option<Execution> {
        Some(self.execution.clone())
    }

    fn config_hash(&self) -> u64 {
        Fnv1a::new().str(&self.head.spec().name).u64(self.width as u64).finish()
    }
}
