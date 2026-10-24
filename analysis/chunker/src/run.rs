//! Converting optimized section runs into chunk rows and optional traces.
use super::*;

impl Chunker {
    /// Appends one run's rows to `chunks`: its parent row first when
    /// `with_parent` (the run is a top-level section), then the chunks
    /// the optimizer cuts from it. The one place a run becomes rows,
    /// shared by a full chunking and by the dirty runs of a
    /// re-chunking (`crate::rechunk`).
    pub(crate) fn push_run<T: Fn(usize, usize) -> i64>(
        &self,
        id: &'static str,
        cx: &RunCx<'_>,
        run: Range<usize>,
        with_parent: bool,
        term: T,
        chunks: &mut Chunks,
    ) -> Result<(), Error> {
        self.push_run_with_trace(id, cx, run, with_parent, term, chunks, None)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn push_run_with_trace<T: Fn(usize, usize) -> i64>(
        &self,
        id: &'static str,
        cx: &RunCx<'_>,
        run: Range<usize>,
        with_parent: bool,
        term: T,
        chunks: &mut Chunks,
        mut trace: Option<&mut Vec<Option<DecisionMargin>>>,
    ) -> Result<(), Error> {
        let (b, sentences, lengths, bounds) = (self.budget, cx.sentences, cx.lengths, cx.bounds);
        let size = |r: Range<usize>| u64::from(lengths.run(r.start, r.end)) + u64::from(b.overhead);
        let span = |r: &Range<usize>| {
            let (first, last) =
                (sentences.spans.get(Ix::new(r.start as u32)), sentences.spans.get(Ix::new(r.end as u32 - 1)));
            Span::new(first.start, last.end)
        };
        let tokens = |t: u64| {
            u32::try_from(t).map_err(|_| Error::Annotator {
                annotator: id,
                detail: format!("a chunk of {t} tokens, more than a u32 holds"),
            })
        };
        let opening = if run.start == 0 { Cut::DocumentStart } else { Cut::Section };
        let parent = if with_parent {
            let t = size(run.clone());
            let row = Row { span: span(&run), level: 1, parent: None, cut: opening, cost: 0, tokens: tokens(t)? };
            {
                if let Some(trace) = trace.as_deref_mut() {
                    trace.push(None);
                }
                Some(row.push(chunks, run.clone(), t > u64::from(b.max)))
            }
        } else {
            None
        };
        // The optimizer's units are the atoms: sentence `at[k].0` up
        // to `at[k + 1].0`.
        let at = run_atoms(cx.limits, run.clone(), &size, b);
        let cut: Vec<i32> = at[1..].iter().map(|&(e, _)| bounds.cost[e - 1]).collect();
        let sizes = |i: usize, j: usize| lengths.run(at[i].0, at[j].0);
        let terms = |i: usize, j: usize| term(at[i].0, at[j].0);
        let solved = if trace.is_some() {
            optimizer::solve::<true>(&cut, sizes, b, terms)
        } else {
            optimizer::solve::<false>(&cut, sizes, b, terms)
        };
        for (index, s) in solved.segments.into_iter().enumerate() {
            let r = at[s.start].0..at[s.end].0;
            let cut = match at[s.start] {
                (i, _) if i == run.start => opening,
                (_, Some(forced)) => forced,
                (i, None) => bounds.reason[i - 1],
            };
            let over_budget = s.tokens > u64::from(b.max);
            if over_budget
                && r.len() == 1
                && let Some(clauses) = cx.clauses
                && clauses::push_clause_rows(id, cx, r.start, cut, parent, b, clauses, chunks)?
            {
                if let Some(trace) = trace.as_deref_mut() {
                    trace.resize(chunks.spans.len(), None);
                }
                continue;
            }
            let row = Row { span: span(&r), level: 0, parent, cut, cost: saturate(s.cost), tokens: tokens(s.tokens)? };
            row.push(chunks, r, over_budget);
            if let Some(trace) = trace.as_deref_mut() {
                trace.push(Some(solved.decisions[index]));
            }
        }
        Ok(())
    }
}
