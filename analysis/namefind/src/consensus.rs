//! Consensus over several recognizers' entities: the dictionary, regex,
//! MaxEnt, CRF, neural and gazetteer finders run side by side, each into
//! its own named instance of `opennlp:entities`, and
//! [`ConsensusEntityMerger`] writes one `opennlp:entities` from them with
//! an `Evidence<Entities>` record per entity that names every
//! recognizer that voted for it.
//!
//! The candidates are taken best first: the highest score, the earlier
//! recognizer winning a tie (under [`EntityMergePolicy::PriorityFallback`],
//! the earliest recognizer's first). Each one not yet taken is a group's
//! entity, and every candidate of its type not yet taken whose token range
//! overlaps it (for [`EntityMergePolicy::RawUnion`], is the same range)
//! votes for it, so a vote always overlaps the entity it is for. Each input
//! instance contributes at most one quorum vote. Priority fallback keeps
//! every first-recognizer entity and checks each later candidate against
//! retained winners before it can consume votes. The group's score is the
//! highest score in it. Its
//! evidence record holds, for each vote in recognizer order, the voter's
//! own span in the source text and a support range that points at the
//! voter's entity in the voter's instance, where its own score is; the
//! record's producer is the winning voter's and its raw score the
//! highest score of the voters that ran a model, since the finders that
//! run none write 0 for a score they do not have.

use std::any::Any;
use std::collections::HashSet;
use std::num::NonZeroU32;
use std::sync::Arc;

use turbo_analysis::annotate::{AnchorUse, Step};
use turbo_analysis::export::LayerRef;
use turbo_analysis::{
    Document, Error, Evidence, FilterRule, Ix, Key, KeyId, Layer, Producer, Provenance, Span, Spans, Status, Support,
};
use turbo_analysis_layers::{Entities, Tokens};

use crate::merge::union_kinds;

/// How the candidates of several recognizers become one layer.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum EntityMergePolicy {
    /// Every distinct candidate, a range and type the recognizers agree
    /// on once with all their votes. Overlaps stay.
    RawUnion,
    /// The first recognizer's entities, then each later recognizer's
    /// where no entity kept so far overlaps it.
    PriorityFallback,
    /// The groups at least `min_votes` recognizers voted for, with
    /// overlaps between groups of different types resolved by
    /// `overlap_rule`.
    Consensus { min_votes: NonZeroU32, overlap_rule: FilterRule },
}

/// One recognizer's entity that voted for a merged entity: its index in
/// the inputs and in that input's layer.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Vote {
    pub input: u32,
    pub entity: u32,
}

/// The merged layer and its votes: entity i's are
/// `votes[vote_row[i]..vote_row[i + 1]]`, in recognizer order, and
/// `winner[i]` is the vote its span, type and score come from.
pub struct Consensus {
    pub entities: Entities,
    pub vote_row: Vec<u32>,
    pub votes: Vec<Vote>,
    pub winner: Vec<Vote>,
}

impl Consensus {
    pub fn votes_of(&self, i: usize) -> &[Vote] {
        &self.votes[self.vote_row[i] as usize..self.vote_row[i + 1] as usize]
    }
}

/// A candidate: its vote, token range, type label and score.
struct Candidate<'a> {
    vote: Vote,
    start: usize,
    end: usize,
    label: &'a str,
    score: f64,
}

/// A score ordered by `f64::total_cmp`, so NaN, which `Entities::check`
/// refuses, cannot break a sort.
#[derive(Copy, Clone, PartialEq)]
struct Score(f64);

impl Eq for Score {}

impl PartialOrd for Score {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Score {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Token ranges that share a token, or the same empty range.
fn overlaps(a: &Candidate<'_>, b: &Candidate<'_>) -> bool {
    (a.start < b.end && b.start < a.end) || (a.start, a.end) == (b.start, b.end)
}

/// The entities of `layers`, in recognizer priority order, merged by
/// `policy`, in text order: by first token, then longest first. Every
/// layer must be in one view and made on the same tokens.
pub fn consensus(layers: &[&Entities], policy: EntityMergePolicy) -> Result<Consensus, String> {
    let kinds = union_kinds(layers)?;
    let view = layers[0].spans.view();
    let mut candidates: Vec<Candidate<'_>> = Vec::new();
    for (l, layer) in layers.iter().enumerate() {
        for e in 0..layer.spans.len() {
            let r = layer.tokens[e];
            candidates.push(Candidate {
                vote: Vote { input: l as u32, entity: e as u32 },
                start: r.start.get(),
                end: r.end.get(),
                label: layer.kinds.label(layer.kind[e]),
                score: layer.score[e],
            });
        }
    }
    // The order winners are taken in: the highest score, the earlier
    // recognizer and then the earlier entity winning a tie; under
    // PriorityFallback the earliest recognizer first.
    let priority = matches!(policy, EntityMergePolicy::PriorityFallback);
    let mut order: Vec<usize> = (0..candidates.len()).collect();
    order.sort_by_key(|&c| {
        let c = &candidates[c];
        (if priority { c.vote.input } else { 0 }, std::cmp::Reverse(Score(c.score)), c.vote.input, c.vote.entity)
    });
    let mut kept = group(&candidates, &order, matches!(policy, EntityMergePolicy::RawUnion), priority);
    match policy {
        EntityMergePolicy::RawUnion => {}
        // Priority eligibility is checked before grouping so a rejected
        // winner cannot consume candidates that are still usable.
        EntityMergePolicy::PriorityFallback => {}
        EntityMergePolicy::Consensus { min_votes, overlap_rule } => {
            let voters = |g: &[usize]| {
                let mut inputs: Vec<u32> = g.iter().map(|&c| candidates[c].vote.input).collect();
                inputs.sort_unstable();
                inputs.dedup();
                inputs.len()
            };
            kept.retain(|(g, _)| voters(g) >= min_votes.get() as usize);
            let mut spans: Spans<Entities> = Spans::with_capacity(view, kept.len());
            for (_, w) in &kept {
                let v = candidates[*w].vote;
                spans.push(layers[v.input as usize].spans.get(Ix::new(v.entity)));
            }
            let scores: Vec<f64> = kept.iter().map(|(_, w)| candidates[*w].score).collect();
            let mut keep = vec![false; kept.len()];
            for ix in spans.filter(Some(&scores), overlap_rule) {
                keep[ix.get()] = true;
            }
            let mut k = keep.iter();
            kept.retain(|_| *k.next().expect("one flag per group"));
        }
    }
    kept.sort_by_key(|(_, w)| (candidates[*w].start, std::cmp::Reverse(candidates[*w].end), candidates[*w].vote.input));

    let mut out = Consensus {
        entities: Entities {
            spans: Spans::with_capacity(view, kept.len()),
            tokens: Vec::with_capacity(kept.len()),
            kind: Vec::with_capacity(kept.len()),
            kinds: Arc::clone(&kinds),
            score: Vec::with_capacity(kept.len()),
        },
        vote_row: vec![0],
        votes: Vec::new(),
        winner: Vec::with_capacity(kept.len()),
    };
    for (g, w) in kept {
        let c = &candidates[w];
        let layer = layers[c.vote.input as usize];
        let ix = Ix::new(c.vote.entity);
        if layer.spans.is_discontinuous(ix) {
            let frags: Vec<Span> = layer.spans.fragments(ix).collect();
            out.entities.spans.push_fragments(&frags);
        } else {
            out.entities.spans.push(layer.spans.get(ix));
        }
        out.entities.tokens.push(layer.tokens[c.vote.entity as usize]);
        out.entities.kind.push(kinds.tag(c.label).expect("every label is in the union"));
        out.entities.score.push(g.iter().map(|&i| candidates[i].score).fold(c.score, f64::max));
        let mut votes: Vec<Vote> = g.iter().map(|&i| candidates[i].vote).collect();
        votes.sort_by_key(|v| (v.input, v.entity));
        out.votes.extend(votes);
        out.vote_row.push(out.votes.len() as u32);
        out.winner.push(c.vote);
    }
    Ok(out)
}

/// The groups and their winners: each candidate in `order` not yet in a
/// group wins one, with every candidate of its type not yet in one whose
/// range overlaps the winner's (or, when `exact`, is the same range). A
/// group's members are in candidate order. In priority mode, a later
/// winner must not overlap a retained winner, and first-recognizer
/// candidates remain separate winners rather than voting for one another.
fn group(candidates: &[Candidate<'_>], order: &[usize], exact: bool, priority: bool) -> Vec<(Vec<usize>, usize)> {
    // By type and first token, so the candidates that may overlap a
    // winner are a short run: those of its type that start within the
    // type's longest range before it and no later than its end.
    let mut sorted: Vec<usize> = (0..candidates.len()).collect();
    sorted.sort_by_key(|&c| (candidates[c].label, candidates[c].start, candidates[c].end, c));
    // Each type's longest range, in tokens.
    let mut reach: Vec<(&str, usize)> = Vec::new();
    for &c in &sorted {
        let (label, len) = (candidates[c].label, candidates[c].end - candidates[c].start);
        match reach.last_mut() {
            Some((l, n)) if *l == label => *n = (*n).max(len),
            _ => reach.push((label, len)),
        }
    }
    let mut taken = vec![false; candidates.len()];
    let mut groups: Vec<(Vec<usize>, usize)> = Vec::new();
    for &w in order {
        if taken[w] {
            continue;
        }
        let win = &candidates[w];
        if priority && win.vote.input != 0 && groups.iter().any(|(_, kept)| overlaps(win, &candidates[*kept])) {
            continue;
        }
        taken[w] = true;
        let longest = reach.iter().find(|&&(l, _)| l == win.label).map_or(0, |&(_, n)| n);
        let from = win.start.saturating_sub(longest);
        let at = sorted.partition_point(|&c| (candidates[c].label, candidates[c].start) < (win.label, from));
        let mut members = vec![w];
        for &c in &sorted[at..] {
            let x = &candidates[c];
            if x.label != win.label || x.start > win.end {
                break;
            }
            let votes = if exact { (x.start, x.end) == (win.start, win.end) } else { overlaps(x, win) };
            // Every candidate from the first recognizer must survive,
            // including its nested or overlapping entities.
            if priority && x.vote.input == 0 {
                continue;
            }
            if !taken[c] && votes {
                taken[c] = true;
                members.push(c);
            }
        }
        members.sort_unstable();
        groups.push((members, w));
    }
    groups
}

/// Writes the consensus of `inputs`, in priority order, to `output`, and
/// its evidence to the `Evidence<Entities>` instance of the same name.
pub struct ConsensusEntityMerger {
    pub inputs: Vec<Key<Entities>>,
    pub output: Key<Entities>,
    /// The tokens every input was made on.
    pub tokens: Key<Tokens>,
    pub policy: EntityMergePolicy,
}

impl ConsensusEntityMerger {
    pub const ID: &'static str = "turbo-namefind:consensus";
    /// Winner-centered grouping with priority eligibility before voting.
    pub const VERSION: &'static str = "1";

    pub fn new(inputs: Vec<Key<Entities>>, output: Key<Entities>, policy: EntityMergePolicy) -> Self {
        ConsensusEntityMerger { inputs, output, tokens: Key::DEFAULT, policy }
    }

    pub fn evidence(&self) -> Key<Evidence<Entities>> {
        Key::named(self.output.name)
    }

    /// The merged layer and evidence for one document. Refuses missing
    /// selected tokens, repeated input instances, and inputs whose resolved
    /// token binding differs from the selected instance.
    pub fn build(&self, doc: &Document) -> Result<(Entities, Evidence<Entities>), Error> {
        let fail = |detail: String| Error::Annotator { annotator: Self::ID, detail };
        doc.require(&self.tokens)?;
        let mut seen = HashSet::with_capacity(self.inputs.len());
        for key in &self.inputs {
            if !seen.insert(key.id()) {
                return Err(fail(format!("duplicate recognizer input {}", key.id())));
            }
        }
        let layers: Vec<&Entities> = self.inputs.iter().map(|k| doc.require(k)).collect::<Result<_, _>>()?;
        let mut producers: Vec<Producer> = Vec::with_capacity(layers.len());
        let mut ran_model: Vec<bool> = Vec::with_capacity(layers.len());
        for k in &self.inputs {
            let p = doc.provenance(k.id()).ok_or(Error::Missing(k.id()))?;
            if !doc.made_on(k.id()).contains(&self.tokens.id()) {
                return Err(fail(format!("{} was made on other tokens than {}", k.id(), self.tokens.id())));
            }
            producers.push(Producer { annotator: p.annotator.to_owned(), model: p.model.clone() });
            ran_model.push(p.model.is_some());
        }
        let merged = consensus(&layers, self.policy).map_err(fail)?;
        let view = merged.entities.spans.view();
        let mut ev = Evidence::<Entities>::default();
        ev.producers = Arc::new(producers);
        ev.layers = Arc::new(
            self.inputs.iter().map(|k| LayerRef { layer: Entities::ID.into(), instance: k.name.into() }).collect(),
        );
        ev.span_row = vec![0];
        for i in 0..merged.winner.len() {
            ev.producer.push(merged.winner[i].input);
            let mut raw: Option<f64> = None;
            for v in merged.votes_of(i) {
                let layer = layers[v.input as usize];
                let ix = Ix::new(v.entity);
                ev.spans.extend(layer.spans.fragments(ix).map(|s| doc.to_source(view, s)));
                ev.support.push(Support { layer: v.input, start: v.entity, end: v.entity + 1 });
                if ran_model[v.input as usize] {
                    let s = layer.score[v.entity as usize];
                    raw = Some(raw.map_or(s, |r: f64| r.max(s)));
                }
            }
            ev.span_row.push(ev.spans.len() as u32);
            if let Some(r) = raw.filter(|r| r.is_finite()) {
                ev.raw.push((i as u32, r));
            }
        }
        ev.support_row = merged.vote_row;
        Ok((merged.entities, ev))
    }

    fn provenance(&self, doc: &Document) -> Provenance {
        let mut reads: Vec<KeyId> = self.inputs.iter().map(Key::id).collect();
        if doc.has(self.tokens.id()) {
            reads.push(self.tokens.id());
        }
        Provenance { reads, version: Self::VERSION, config: Step::config_hash(self), ..Provenance::by(Self::ID) }
    }
}

type Out = Vec<Result<(Entities, Evidence<Entities>), Error>>;

impl Step for ConsensusEntityMerger {
    fn id(&self) -> &'static str {
        Self::ID
    }

    fn reads(&self) -> Vec<KeyId> {
        self.inputs.iter().map(Key::id).chain([self.tokens.id()]).collect()
    }

    fn required(&self) -> Vec<bool> {
        vec![true; self.inputs.len() + 1]
    }

    fn writes(&self) -> Vec<KeyId> {
        vec![self.output.id(), self.evidence().id()]
    }

    fn anchors(&self) -> Vec<AnchorUse> {
        vec![AnchorUse::None, AnchorUse::To(self.output.id())]
    }

    fn config_hash(&self) -> u64 {
        let policy = match self.policy {
            EntityMergePolicy::RawUnion => 1,
            EntityMergePolicy::PriorityFallback => 2,
            EntityMergePolicy::Consensus { min_votes, overlap_rule } => {
                let rule = match overlap_rule {
                    FilterRule::LongestFirst => 1,
                    FilterRule::HighestScore => 2,
                    FilterRule::LeftmostLongest => 3,
                };
                (u64::from(min_votes.get()) << 8) | (rule << 4) | 3
            }
        };
        // Revision 1 uses the bit above the 32-bit quorum and 8-bit policy.
        (1u64 << 40) | policy
    }

    fn compute(&self, docs: &[Document]) -> Box<dyn Any + Send> {
        let out: Out = docs
            .iter()
            .map(|d| {
                for k in self.reads() {
                    match d.status(k) {
                        Some(Status::Complete) => {}
                        Some(Status::Skipped(why)) => return Err(Error::Skipped(format!("{k} was skipped: {why}"))),
                        Some(Status::Failed(e)) => return Err(Error::Skipped(format!("{k} failed: {e}"))),
                        None => return Err(Error::Missing(k)),
                    }
                }
                self.build(d)
            })
            .collect();
        Box::new(out)
    }

    fn store(&self, out: Box<dyn Any + Send>, docs: &mut [Document], status: &mut [Result<(), Error>]) {
        let out = *out.downcast::<Out>().expect("this step's own output");
        for (i, r) in out.into_iter().enumerate() {
            let p = self.provenance(&docs[i]);
            let failure = match r {
                Ok((entities, ev)) => docs[i].add(&self.output, entities, p.clone()).and_then(|()| {
                    let p = Provenance { reads: vec![self.output.id()], ..p.clone() };
                    docs[i].add(&self.evidence(), ev, p)
                }),
                Err(Error::Skipped(why)) => {
                    self.writes().into_iter().try_for_each(|k| docs[i].skip(k, why.clone(), p.clone()))
                }
                Err(e) => Err(e),
            };
            let Err(e) = failure else { continue };
            for k in self.writes() {
                if !docs[i].has(k) {
                    let _ = docs[i].fail(k, e.clone(), p.clone());
                }
            }
            if status[i].is_ok() {
                status[i] = Err(e);
            }
        }
    }
}
