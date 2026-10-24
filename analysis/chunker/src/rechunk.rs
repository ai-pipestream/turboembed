//! Incremental re-chunking after a document edit, driven by the
//! same [`Alignment`] that maps spans between two views of a text:
//! the parent side is the old text, the child side the edited one.
//!
//! The chunker already cuts each top-level section as an
//! independent run, so a re-chunking can carry the old rows of a run
//! forward whenever everything the optimizer reads for that run is
//! unchanged: the sentences map across the alignment span for span,
//! the token lengths, boundary costs and reasons and the atoms the
//! hard limits make are all equal. Then the DP over the run is the
//! same computation as last time, so its rows are transplanted with
//! spans mapped forward (`span_to_child`) instead of recomputed.
//! Any run that fails one of the checks is cut again by the same
//! code a full chunking uses, so a re-chunking always equals a full
//! [`Chunker`] chunking of the new inputs, however the edit fell.
//! The upstream layers of the edited text (sentences, subwords,
//! boundaries, structure) are inputs, as they are for a chunking;
//! only the chunker's own DP is skipped, run by run.

use std::ops::Range;

use turbo_analysis::{Alignment, Annotator, Error, Ix, Span, Spans};
use turbo_analysis_layers::{Boundaries, Chunks, Cut, Sentences, Structure, Subwords};

use crate::{Budget, Chunker, Lengths, Row, RunCx, chunk_cost, gaps, limits, no_term, run_atoms, saturate};

/// The previous chunking of a document and the layers it was cut
/// from, everything [`Chunker::rechunk`] reads on the parent (old)
/// side of the edit alignment. `chunks` is this same chunker's
/// output on those layers (same budget): a carried row is also
/// checked against the token count and cost this budget gives it,
/// and a mismatch re-chunks the run instead.
pub struct Previous<'a> {
    pub sentences: &'a Sentences,
    pub subwords: &'a Subwords,
    pub bounds: &'a Boundaries,
    pub structure: Option<&'a Structure>,
    pub chunks: &'a Chunks,
}

/// What a re-chunking did: how many section runs were carried
/// forward against how many were cut again, how many level 0 chunks
/// that carried, and the sentence windows the edits touch on both
/// sides ([`affected_sentences`]).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RechunkStats {
    pub reused_runs: usize,
    pub rechunked_runs: usize,
    pub reused_chunks: usize,
    pub affected_old: Range<usize>,
    pub affected_new: Range<usize>,
}

/// The windows of sentences an edit touches, on the old (parent)
/// side and on the new (child) side: from the first sentence
/// holding a byte of an edit (or holding an insertion point inside
/// it) to one past the last. Empty at 0 when no edit touches a
/// sentence. Sentences outside the windows map across `edit` span
/// for span with their text unchanged, so their runs are the
/// candidates a re-chunking carries forward.
pub fn affected_sentences(edit: &Alignment, old: &Sentences, new: &Sentences) -> (Range<usize>, Range<usize>) {
    let window = |sentences: &Sentences, parent_side: bool| {
        let (starts, ends) = sentences.spans.columns();
        let (mut lo, mut hi) = (usize::MAX, 0);
        for e in edit.edits() {
            let s = if parent_side { e.parent } else { e.child };
            if s.is_empty() {
                // An insertion or deletion point inside one
                // sentence: the last sentence starting before it.
                let at = starts.partition_point(|&x| x < s.start);
                if let Some(i) = at.checked_sub(1).filter(|&i| s.start < ends[i]) {
                    lo = lo.min(i);
                    hi = hi.max(i + 1);
                }
                continue;
            }
            // The sentences holding a byte of `s`: from the first
            // ending after it starts to the last starting before
            // it ends.
            let first = ends.partition_point(|&x| x <= s.start);
            for i in first..starts.partition_point(|&x| x < s.end) {
                lo = lo.min(i);
                hi = hi.max(i + 1);
            }
        }
        if lo == usize::MAX { 0..0 } else { lo..hi }
    };
    (window(old, true), window(new, false))
}

/// One side's chunking inputs, prepared the way `Chunker::chunk`
/// prepares them.
struct Side<'a> {
    /// Whether its document has structure, so its runs carry a
    /// parent row.
    parented: bool,
    sentences: &'a Sentences,
    bounds: &'a Boundaries,
    lengths: Lengths,
    limits: Vec<limits::Limit>,
    runs: Vec<Range<usize>>,
}

impl<'a> Side<'a> {
    fn new(
        id: &'static str,
        budget: Budget,
        sentences: &'a Sentences,
        subwords: &'a Subwords,
        bounds: &'a Boundaries,
        structure: Option<&'a Structure>,
    ) -> Result<Side<'a>, Error> {
        let view = sentences.spans.view();
        if subwords.spans.view() != view || structure.is_some_and(|s| s.spans.view() != view) {
            return Err(Error::Annotator { annotator: id, detail: "its inputs are in different views".into() });
        }
        let n = sentences.spans.len();
        if bounds.cost.len() != n || bounds.reason.len() != n {
            return Err(Error::Annotator {
                annotator: id,
                detail: format!(
                    "{} boundary costs and {} reasons for {n} sentences",
                    bounds.cost.len(),
                    bounds.reason.len()
                ),
            });
        }
        let lengths = Lengths::new(&sentences.spans, &subwords.spans);
        let size = |r: Range<usize>| u64::from(lengths.run(r.start, r.end)) + u64::from(budget.overhead);
        let gaps = gaps::gaps(&sentences.spans, structure);
        let limits = limits::limits(&gaps.units, n, |u| size(u.clone()) <= u64::from(budget.max));
        let runs = match structure {
            Some(st) => gaps::sections(sentences, st),
            None if n == 0 => Vec::new(),
            None => std::iter::once(0..n).collect(),
        };
        Ok(Side { parented: structure.is_some(), sentences, bounds, lengths, limits, runs })
    }

    fn span(&self, sentences: Range<usize>) -> Span {
        let (first, last) = (
            self.sentences.spans.get(Ix::new(sentences.start as u32)),
            self.sentences.spans.get(Ix::new(sentences.end as u32 - 1)),
        );
        Span::new(first.start, last.end)
    }

    /// The atoms of one run, relative to its start, so two sides'
    /// atoms compare directly.
    fn atoms(&self, run: &Range<usize>, budget: Budget) -> Vec<(usize, Option<Cut>)> {
        let size = |r: Range<usize>| u64::from(self.lengths.run(r.start, r.end)) + u64::from(budget.overhead);
        run_atoms(&self.limits, run.clone(), &size, budget).into_iter().map(|(i, c)| (i - run.start, c)).collect()
    }
}

impl Chunker {
    /// The chunks of an edited document, reusing `previous` where
    /// the edit left a whole section run's optimizer inputs
    /// unchanged. `edit` aligns the old text (parent) to the new
    /// one (child); the new side's sentences, subwords, boundary
    /// costs and structure are the edited document's own layers, in
    /// its own view. The result is always what chunking the new
    /// inputs from scratch with this chunker gives; the returned
    /// stats say which runs were carried forward. Carried rows keep
    /// their cost, cut reason and token count, with spans mapped
    /// forward across `edit`, so their [`crate::ContentHashes`]
    /// also survive wherever the text itself did.
    ///
    /// This is the plain [`Chunker`]'s re-chunking (no chunk term);
    /// a cohesion term over whole chunks is not carried.
    pub fn rechunk(
        &self,
        edit: &Alignment,
        previous: Previous<'_>,
        sentences: &Sentences,
        subwords: &Subwords,
        bounds: &Boundaries,
        structure: Option<&Structure>,
    ) -> Result<(Chunks, RechunkStats), Error> {
        let id = <Chunker as Annotator>::ID;
        if previous.chunks.spans.view() != previous.sentences.spans.view() {
            return Err(Error::Annotator { annotator: id, detail: "the previous chunks are in another view".into() });
        }
        let b = self.budget;
        let old = Side::new(id, b, previous.sentences, previous.subwords, previous.bounds, previous.structure)?;
        let new = Side::new(id, b, sentences, subwords, bounds, structure)?;
        let (affected_old, affected_new) = affected_sentences(edit, old.sentences, new.sentences);
        let mut stats = RechunkStats { affected_old, affected_new, ..RechunkStats::default() };
        let mut chunks = Chunks {
            spans: Spans::new(new.sentences.spans.view()),
            level: Vec::new(),
            parent: Vec::new(),
            sentences: Vec::new(),
            cut: Vec::new(),
            cost: Vec::new(),
            model_tokens: Vec::new(),
            over_budget: Vec::new(),
            prompt_overhead: b.overhead,
        };
        let new_cx = RunCx {
            sentences: new.sentences,
            subwords,
            lengths: &new.lengths,
            bounds: new.bounds,
            limits: &new.limits,
            clauses: None,
        };
        for run in &new.runs {
            let carried = old_run_for(edit, &old, &new, run)
                .filter(|old_run| clean(edit, &old, &new, old_run, run, b))
                .and_then(|old_run| plan(edit, previous.chunks, &old, &new, &old_run, run, b));
            match carried {
                Some(rows) => {
                    stats.reused_runs += 1;
                    stats.reused_chunks += rows.iter().filter(|r| r.level == 0).count();
                    transplant(&rows, &mut chunks);
                }
                None => {
                    stats.rechunked_runs += 1;
                    self.push_run(id, &new_cx, run.clone(), structure.is_some(), no_term, &mut chunks)?;
                }
            }
        }
        Ok((chunks, stats))
    }
}

/// The old run a new run may correspond to: the old sentence the new
/// run's first sentence maps back to (`span_to_parent`), when it
/// starts an old run of the same number of sentences.
fn old_run_for(edit: &Alignment, old: &Side<'_>, new: &Side<'_>, run: &Range<usize>) -> Option<Range<usize>> {
    let first = new.sentences.spans.get(Ix::new(run.start as u32));
    let back = edit.span_to_parent(first);
    let (starts, ends) = old.sentences.spans.columns();
    let at = starts.binary_search(&back.start).ok()?;
    if ends[at] != back.end {
        return None;
    }
    old.runs.iter().find(|r| r.start == at && r.len() == run.len()).cloned()
}

/// Whether the optimizer would read exactly the same inputs for
/// both runs: every sentence maps forward span for span, the token
/// lengths relative to the run's start agree, the boundary costs
/// and reasons agree, and the hard limits make the same atoms. The
/// opening cut also reads whether the run starts the document.
fn clean(
    edit: &Alignment,
    old: &Side<'_>,
    new: &Side<'_>,
    old_run: &Range<usize>,
    new_run: &Range<usize>,
    b: Budget,
) -> bool {
    if old.parented != new.parented || (old_run.start == 0) != (new_run.start == 0) {
        return false;
    }
    let spans_map = (0..old_run.len()).all(|k| {
        let o = old.sentences.spans.get(Ix::new((old_run.start + k) as u32));
        let n = new.sentences.spans.get(Ix::new((new_run.start + k) as u32));
        edit.span_to_child(o) == n
    });
    if !spans_map {
        return false;
    }
    let relative = |side: &Side<'_>, run: &Range<usize>| {
        let base = side.lengths.start[run.start];
        (run.clone()).map(|i| (side.lengths.start[i] - base, side.lengths.end[i] - base)).collect::<Vec<_>>()
    };
    relative(old, old_run) == relative(new, new_run)
        && old.bounds.cost[old_run.clone()] == new.bounds.cost[new_run.clone()]
        && old.bounds.reason[old_run.clone()] == new.bounds.reason[new_run.clone()]
        && old.atoms(old_run, b) == new.atoms(new_run, b)
}

/// One carried row, validated and mapped to the new side but for
/// its parent, which names the old row until `transplant` assigns
/// the new rows.
struct Carried {
    span: Span,
    level: u8,
    parent: Option<usize>,
    sentences: Range<usize>,
    cut: Cut,
    cost: i64,
    tokens: u32,
    over_budget: bool,
}

/// The old rows of `old_run` as rows of `new_run`, or nothing when
/// the old chunks do not hold the run exactly (one parent row over
/// level 0 rows tiling it, as the chunker writes) or a mapped span
/// is not the span the new sentences give.
fn plan(
    edit: &Alignment,
    chunks: &Chunks,
    old: &Side<'_>,
    new: &Side<'_>,
    old_run: &Range<usize>,
    new_run: &Range<usize>,
    b: Budget,
) -> Option<Vec<Carried>> {
    let shift = |r: &Range<usize>| (r.start - old_run.start + new_run.start)..(r.end - old_run.start + new_run.start);
    let mut rows = Vec::new();
    let (mut cursor, mut parents) = (old_run.start, 0usize);
    for i in 0..chunks.spans.len() {
        let r = chunks.sentences[i].start.get()..chunks.sentences[i].end.get();
        if r.start < old_run.start || old_run.end < r.end {
            continue;
        }
        if chunks.level[i] == 0 {
            if r.start != cursor || r.start == r.end {
                return None;
            }
            cursor = r.end;
        } else {
            if !rows.is_empty() || r != *old_run {
                return None;
            }
            parents += 1;
        }
        let sentences = shift(&r);
        let span = edit.span_to_child(chunks.spans.get(Ix::new(i as u32)));
        if span != new.span(sentences.clone()) {
            return None;
        }
        // The row must be this budget's row: the token count over
        // its whole span and, at level 0, the exact cost the
        // optimizer charges it (its closing boundary, free at the
        // run's end, plus its size penalty).
        let size = u64::from(old.lengths.run(r.start, r.end)) + u64::from(b.overhead);
        if u64::from(chunks.model_tokens[i]) != size || chunks.over_budget[i] != (size > u64::from(b.max)) {
            return None;
        }
        if chunks.level[i] == 0 {
            let closing = if r.end == old_run.end { 0 } else { old.bounds.cost[r.end - 1] };
            if chunks.cost[i] != saturate(chunk_cost(closing, size, b, 0)) {
                return None;
            }
        } else if chunks.cost[i] != 0 {
            return None;
        }
        rows.push(Carried {
            span,
            level: chunks.level[i],
            parent: chunks.parent[i].map(|p| p.get()),
            sentences,
            cut: chunks.cut[i],
            cost: chunks.cost[i],
            tokens: chunks.model_tokens[i],
            over_budget: chunks.over_budget[i],
        });
    }
    if cursor != old_run.end || parents != usize::from(new.parented) {
        return None;
    }
    // Every level 0 row's parent is the run's parent row (or none),
    // as the chunker writes it: nothing points outside the run.
    let old_parent = (0..chunks.spans.len()).find(|&i| {
        let r = chunks.sentences[i].start.get()..chunks.sentences[i].end.get();
        chunks.level[i] > 0 && r == *old_run
    });
    if rows.iter().filter(|r| r.level == 0).any(|r| r.parent != old_parent) {
        return None;
    }
    Some(rows)
}

/// Appends the carried rows, remapping each parent to the new row
/// its old row became (parents always come before their children).
fn transplant(rows: &[Carried], chunks: &mut Chunks) {
    // The parent of a level 0 row is the run's parent row, the only
    // row above level 0, which `plan` checked and which is pushed
    // first.
    let mut parent_row: Option<Ix<Chunks>> = None;
    for r in rows {
        let parent = if r.level == 0 { parent_row } else { None };
        let row = Row { span: r.span, level: r.level, parent, cut: r.cut, cost: r.cost, tokens: r.tokens };
        let ix = row.push(chunks, r.sentences.clone(), r.over_budget);
        if r.level > 0 {
            parent_row = Some(ix);
        }
    }
}
