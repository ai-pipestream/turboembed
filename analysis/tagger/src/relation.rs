//! [`RelationTagger`] writes typed relations between the view's entities
//! from a HEAD_RELATION_CLASSIFIER head: the relation layer and the same
//! triples as entity-to-entity edges, as the rule and zero-shot relation
//! annotators write them.
//!
//! An entity is a mention of the sequence (`HeadRunner::sequences`) that
//! holds both its first and last token, its words those tokens' positions
//! in the sequence; an entity across two sequences is no mention. With
//! `relation.entity_labels`, an entity's kind is its type by name, and an
//! entity of a kind the head does not list is no mention. Each relation
//! covers both entities' spans, so its offsets are theirs.

use std::path::Path;
use std::sync::Arc;

use turbo::bundle::Bundle;
use turbo::head::relation::{Mention, RelationWork};
use turbo::head::{RelationClassifier, sigmoid_f64};
use turbo::manifest::{HeadKind, NO_RELATION};
use turbo::status::Error as CoreError;
use turbo_analysis::{
    Annotator, Edge, Edges, Error, Execution, Ix, Layer as _, ModelRef, Refs, Span, Spans, Tag, TagSet, View,
};
use turbo_analysis_embed::EncoderOptions;
use turbo_analysis_layers::{Entities, Sentences, Subwords, Tokens};
use turbo_relation::Relations;

use crate::annotate::{HeadRunner, find_head};
use crate::segment::Reader;

/// A sequence's entities: each one's row in the entity layer, its type
/// and its words `first..=last` in the sequence.
type Found = Vec<(u32, u32, u32, u32)>;

/// Writes the relation layer and its edges from a relation classifier
/// head; each relation's score is the logistic of its margin over `none`,
/// the pair's threshold, so a pair holding two relations scores each.
pub struct RelationTagger {
    head: RelationClassifier,
    kinds: Arc<TagSet>,
    /// Each label's kind index; None for `none`.
    kind_of: Vec<Option<u16>>,
    read: Reader,
}

impl RelationTagger {
    /// The head named `name` in the bundle at `dir`, or with no name the
    /// one head whose `layer` is the relation layer's.
    pub fn open(dir: &Path, name: Option<&str>, o: &EncoderOptions) -> Result<RelationTagger, CoreError> {
        let bundle = Bundle::open(dir)?;
        let spec = find_head(&bundle, name, Relations::ID, HeadKind::RelationClassifier)?;
        let head = RelationClassifier::load(&bundle, &spec.name)?;
        let read = Reader::open(dir, &bundle, spec, head.hidden(), o)?;
        let mut kind_of = Vec::new();
        let mut types: Vec<&str> = Vec::new();
        for l in head.labels() {
            kind_of.push((l != NO_RELATION).then(|| {
                types.push(l);
                (types.len() - 1) as u16
            }));
        }
        let kinds = TagSet::new(&spec.name, &types);
        Ok(RelationTagger { head, kinds, kind_of, read })
    }

    pub fn head(&self) -> &RelationClassifier {
        &self.head
    }

    pub fn kinds(&self) -> &Arc<TagSet> {
        &self.kinds
    }

    /// Each sequence's mentions, by sequence: the entity behind each, and
    /// the mention itself in the sequence's words.
    fn mentions(&self, tokens: &Tokens, entities: &Entities, seqs: &[Vec<usize>]) -> Vec<Found> {
        let mut at = vec![None; tokens.spans.len()];
        for (s, seq) in seqs.iter().enumerate() {
            for (w, &t) in seq.iter().enumerate() {
                at[t] = Some((s, w as u32));
            }
        }
        let types = self.head.entity_labels();
        let mut out = vec![Vec::new(); seqs.len()];
        for (e, r) in entities.tokens.iter().enumerate() {
            let (first, last) = (r.start.get(), r.end.get());
            if last <= first || last > at.len() {
                continue;
            }
            let (Some((s, a)), Some((t, b))) = (at[first], at[last - 1]) else { continue };
            let kind = if types.is_empty() {
                Some(0)
            } else {
                let kind = entities.kinds.label(entities.kind[e]);
                types.iter().position(|l| l == kind).map(|i| i as u32)
            };
            if let (true, Some(kind)) = (s == t, kind) {
                out[s].push((e as u32, kind, a, b));
            }
        }
        out
    }

    /// The relations the head keeps among the view's entities, in order
    /// of sequence, then head entity, then tail entity.
    pub fn relations(
        &self,
        view: &View,
        tokens: &Tokens,
        entities: &Entities,
        sentences: Option<&Sentences>,
        subwords: Option<&Subwords>,
    ) -> Result<(Relations, Edges<Entities, Entities>), String> {
        if entities.spans.view() != view.id() {
            return Err("the entities are in another view".into());
        }
        entities.check()?;
        let all = HeadRunner::sequences(view, tokens, sentences)?;
        let by_seq = self.mentions(tokens, entities, &all);
        // Only a sequence of two mentions or more has a pair to score.
        let (seqs, found): (Vec<Vec<usize>>, Vec<Found>) =
            all.into_iter().zip(by_seq).filter(|(_, m)| m.len() > 1).unzip();
        let mut out = Relations {
            spans: Spans::new(view.id()),
            subject: Vec::new(),
            object: Vec::new(),
            kind: Vec::new(),
            kinds: self.kinds.clone(),
            score: Vec::new(),
        };
        let features = if seqs.is_empty() {
            Vec::new()
        } else {
            self.read.features(view, tokens, &seqs, subwords, self.head.spec())?
        };
        let (mut scores, mut kept, mut w) = (Vec::new(), Vec::new(), RelationWork::default());
        for ((seq, f), found) in seqs.iter().zip(&features).zip(&found) {
            let words: Vec<u32> = (0..seq.len() as u32).collect();
            // Each row of the entity layer is one entity of one mention.
            let mentions: Vec<Mention> =
                (0..).zip(found).map(|(i, &(_, _, first, last))| Mention { first, last, entity: i }).collect();
            let types: Vec<u32> = found.iter().map(|f| f.1).collect();
            let m = mentions.len();
            self.head.scores(f, &words, &mentions, &types, &mut scores, &mut w).map_err(|e| e.to_string())?;
            self.head.decode(&scores, m, &mut kept).map_err(|e| e.to_string())?;
            for r in &kept {
                let (a, b) = (r.head as usize, r.tail as usize);
                let kind = self.kind_of[r.label as usize].ok_or("a kept relation labelled none")?;
                let (subject, object) = (Ix::new(found[a].0), Ix::new(found[b].0));
                let (x, y) = (entities.spans.get(subject), entities.spans.get(object));
                out.spans.push(Span::new(x.start.min(y.start), x.end.max(y.end)));
                out.subject.push(subject);
                out.object.push(object);
                out.kind.push(Tag(kind));
                // The label's odds against the pair's threshold, `none`.
                out.score.push(sigmoid_f64(f64::from(r.margin)));
            }
        }
        out.check()?;
        let edges = (0..out.subject.len()).map(|i| Edge::new(out.subject[i], out.object[i], out.kind[i], out.score[i]));
        let edges = Edges::new(Arc::clone(&self.kinds), edges.collect::<Vec<_>>());
        Ok((out, edges))
    }
}

impl Annotator for RelationTagger {
    const ID: &'static str = "turbo-tagger-relations";
    type Reads = (View, Tokens, Entities, Option<Subwords>, Option<Sentences>);
    type Writes = (Relations, Edges<Entities, Entities>);

    fn annotate(&self, (view, tokens, entities, subwords, sentences): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        self.relations(view, tokens, entities, sentences, subwords)
            .map_err(|detail| Error::Annotator { annotator: Self::ID, detail })
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.read.model.clone())
    }

    fn execution(&self) -> Option<Execution> {
        Some(self.read.execution.clone())
    }

    fn config_hash(&self) -> u64 {
        self.read.config(self.head.spec())
    }
}
