use super::*;

/// What a call makes the encoder do (`VectorAnnotator::shape`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Shape {
    pub runs: u64,
    pub rows: u64,
    /// Tokens computed: each run's rows times its longest row.
    pub tokens: u64,
}

/// A call planned and not yet run (`VectorAnnotator::plan`).
pub struct Planned {
    plan: Plan,
    out: Vec<Result<Output, Error>>,
}
impl VectorAnnotator {
    /// The vectors of a batch of documents, a result for each. A document
    /// whose layers do not fit together fails alone; a run that fails
    /// fails every document of the batch. `annotator` names the step in
    /// errors.
    pub fn embed(&self, annotator: &'static str, inputs: &[Input<'_>]) -> Vec<Result<Output, Error>> {
        let mut out: Vec<Result<Output, Error>> = inputs.iter().map(|i| self.output_for(annotator, i)).collect();
        let mut guard = self.lock();
        self.plan_batch(annotator, inputs, &mut guard.plan, &mut out);
        self.finish(annotator, guard, inputs, out)
    }

    /// A call of `inputs` planned and not run, for `shape_of_plan` to count
    /// and `embed_planned` to run without planning it again.
    pub fn plan(&self, annotator: &'static str, inputs: &[Input<'_>]) -> Planned {
        let mut out: Vec<Result<Output, Error>> = inputs.iter().map(|i| self.output_for(annotator, i)).collect();
        let mut plan = Plan::new(self.width());
        self.plan_batch(annotator, inputs, &mut plan, &mut out);
        Planned { plan, out }
    }

    /// `embed` of the call `planned` was made for, the same `inputs`, on
    /// this step, which must lay out windows as the one that planned it
    /// did: the same bundle, window and prompt.
    pub fn embed_planned(
        &self,
        annotator: &'static str,
        inputs: &[Input<'_>],
        planned: Planned,
    ) -> Vec<Result<Output, Error>> {
        let Planned { mut plan, out } = planned;
        let mut guard = self.lock();
        std::mem::swap(&mut guard.plan, &mut plan);
        self.finish(annotator, guard, inputs, out)
    }

    /// Runs the plan in `guard` and pools, splits and normalizes into `out`.
    fn finish(
        &self,
        annotator: &'static str,
        mut guard: std::sync::MutexGuard<'_, State>,
        inputs: &[Input<'_>],
        mut out: Vec<Result<Output, Error>>,
    ) -> Vec<Result<Output, Error>> {
        let st = &mut *guard;
        if let Err(e) = self.run(st, inputs, &mut out) {
            let e = Error::Annotator { annotator, detail: e.to_string() };
            for o in out.iter_mut().filter(|o| o.is_ok()) {
                *o = Err(e.clone());
            }
            return out;
        }
        let dim = self.dim as usize;
        for s in &st.plan.split {
            if let Ok(o) = &mut out[s.target.doc as usize] {
                let inv = 1.0 / s.pieces as f32;
                for x in row_of(o, s.target, dim) {
                    *x *= inv;
                }
            }
        }
        drop(guard);
        if self.l2 {
            for o in out.iter_mut().flatten() {
                let rows = [o.sentences.as_mut().map(|v| &mut v.data), o.chunks.as_mut().map(|v| &mut v.data)];
                let levels = o.levels.iter_mut().flatten().map(|v| &mut v.data);
                for data in rows.into_iter().flatten().chain(levels) {
                    for v in data.chunks_exact_mut(dim) {
                        normalize(v);
                    }
                }
            }
        }
        out
    }

    /// The runs, rows and tokens a call of `inputs` makes in this step's
    /// session: packed as `embed` packs them (`pack`), each run's rows as
    /// long as its longest. The call is planned, not run.
    pub fn shape(&self, annotator: &'static str, inputs: &[Input<'_>]) -> Shape {
        self.shape_of_plan(&self.plan(annotator, inputs))
    }

    /// `shape` of a planned call.
    pub fn shape_of_plan(&self, planned: &Planned) -> Shape {
        self.count(&planned.plan, self.max_batch, self.max_ranges)
    }

    /// `shape` in a session of `max_batch` rows and `max_ranges` ranges a
    /// run, each at least 1 (a session resolves 0 to its default before it
    /// runs); an error for 0.
    pub fn shape_at(
        &self,
        annotator: &'static str,
        inputs: &[Input<'_>],
        max_batch: u32,
        max_ranges: u32,
    ) -> Result<Shape, CoreError> {
        if max_batch == 0 || max_ranges == 0 {
            return Err(CoreError::new(
                INVALID_ARGUMENT,
                format!("shape of {max_batch} rows and {max_ranges} ranges a run: each must be at least 1"),
            ));
        }
        Ok(self.count(&self.plan(annotator, inputs).plan, max_batch, max_ranges))
    }

    /// The shape `plan` packs into at these limits, each at least 1.
    fn count(&self, plan: &Plan, max_batch: u32, max_ranges: u32) -> Shape {
        let ends = (self.head.len() + self.tail.len()) as u64;
        let mut shape = Shape { runs: 0, rows: 0, tokens: 0 };
        let (mut rows, mut longest) = (0u64, 0u64);
        let counted: Result<(), std::convert::Infallible> = pack(plan, max_batch, max_ranges, |step| {
            match step {
                Pack::Row { window, .. } => {
                    let w = &plan.windows[window];
                    rows += 1;
                    longest = longest.max((w.end - w.start) as u64 + ends);
                }
                Pack::Flush => {
                    shape.runs += 1;
                    shape.rows += rows;
                    shape.tokens += rows * longest;
                    (rows, longest) = (0, 0);
                }
            }
            Ok(())
        });
        let Ok(()) = counted;
        shape
    }

    /// Lays out the windows of every document of `inputs` whose layers fit
    /// together, failing a document alone when its items do not place.
    fn plan_batch(
        &self,
        annotator: &'static str,
        inputs: &[Input<'_>],
        plan: &mut Plan,
        out: &mut [Result<Output, Error>],
    ) {
        plan.clear();
        for (d, (input, slot)) in inputs.iter().zip(out.iter_mut()).enumerate() {
            let Ok(o) = slot else { continue };
            let (starts, ends) = input.subwords.spans.columns();
            let d = d as u32;
            // The levels are placed scope by scope, before anything else
            // is queued for the document.
            if let (Some(l), Some(v)) = (input.levels, o.levels.as_mut())
                && let Err(e) = crate::levels::plan(plan, d, l, starts, ends, v)
            {
                *slot = Err(Error::Annotator { annotator, detail: e.to_string() });
                continue;
            }
            // The finest layer given lays out the windows.
            if let (Some(s), Some(v)) = (input.sentences, o.sentences.as_mut()) {
                for (i, span) in s.iter() {
                    let r = pieces_of(starts, ends, span.start, span.end);
                    if r.is_empty() {
                        v.empty.push(i);
                    }
                    plan.layout(Target { doc: d, which: Which::Sentences, item: i.get() as u32 }, r);
                }
            }
            if let (Some(c), Some(v)) = (input.chunks, o.chunks.as_mut()) {
                let lays_out = input.sentences.is_none();
                for (i, span) in c.iter() {
                    let r = pieces_of(starts, ends, span.start, span.end);
                    if r.is_empty() {
                        v.empty.push(i);
                    }
                    let t = Target { doc: d, which: Which::Chunks, item: i.get() as u32 };
                    if lays_out { plan.layout(t, r) } else { plan.item(t, r) }
                }
            }
            if let Err(e) = plan.place(d, starts.len() as u32) {
                *slot = Err(Error::Annotator { annotator, detail: e.to_string() });
            }
        }
    }

    /// Pools one vector per span in `spans` from `subwords` through the
    /// encoder's stage 2 range write (`turbo_embed_write_ranges`, `#263`),
    /// returning `Vectors<L>` anchored to any layer `L` (sentences, chunks,
    /// opinions, aspects, or arbitrary spans).
    ///
    /// When `sentences` is `Some`, the encoder windows are laid out by the
    /// sentences and each span in `spans` is pooled over the exact subword
    /// piece range (`pieces_of`) overlapping that span inside its enclosing
    /// sentence window. When `sentences` is `None`, the spans lay out the
    /// windows directly.
    pub fn embed_spans<L: Layer>(
        &self,
        annotator: &'static str,
        subwords: &Subwords,
        sentences: Option<&Spans<Sentences>>,
        view: ViewId,
        spans: &[Span],
    ) -> Result<Vectors<L>, Error> {
        let mut target_spans = Spans::<Chunks>::new(view);
        for &span in spans {
            target_spans.push(span);
        }
        let mut out =
            self.embed(annotator, &[Input { subwords, sentences, chunks: Some(&target_spans), levels: None }]);
        let c = out.remove(0)?.chunks.ok_or_else(|| Error::Annotator {
            annotator,
            detail: "VectorAnnotator did not return span vectors".into(),
        })?;
        let empty = c.empty.into_iter().map(|ix| Ix::<L>::new(ix.get() as u32)).collect();
        Ok(Vectors::new(c.dim, c.data, empty, c.model))
    }

    /// Extracts one contextual vector per subword piece in `subwords`
    /// through the encoder's stage 2 range write (`turbo_embed_write_ranges`,
    /// `#263`, `#709`), selecting for each piece the window where it is most
    /// interior and applying the bundle's optional projection head before
    /// L2 normalization, returning `Vectors<Subwords>`.
    pub fn embed_subwords(
        &self,
        annotator: &'static str,
        subwords: &Subwords,
        sentences: Option<&Spans<Sentences>>,
    ) -> Result<Vectors<Subwords>, Error> {
        Ok(self.embed_subword_tokens(annotator, subwords, sentences, self.stride)?.into_subword_vectors())
    }

    /// Zero vectors for every item of `i`, or why its layers do not fit
    /// together: subwords of another model, a layer in another view, a
    /// piece without its id, pieces out of text order, or a span that
    /// ends before it starts.
    fn output_for(&self, annotator: &'static str, i: &Input<'_>) -> Result<Output, Error> {
        let fail = |detail: String| Err(Error::Annotator { annotator, detail });
        fn has_gaps<L>(spans: &Spans<L>) -> bool {
            spans.iter().any(|(ix, _)| spans.is_discontinuous(ix))
        }
        if has_gaps(&i.subwords.spans)
            || i.sentences.is_some_and(has_gaps)
            || i.chunks.is_some_and(has_gaps)
            || i.levels.is_some_and(|l| has_gaps(&l.spans))
        {
            return fail("discontinuous spans are not supported by vector window pooling".into());
        }
        let (starts, ends) = i.subwords.spans.columns();
        if let Some(k) = starts.iter().zip(ends).position(|(s, e)| s > e) {
            return fail(format!("subword {k} ends before it starts: {}..{}", starts[k], ends[k]));
        }
        // The subwords' own check: an id per piece, starts and ends in
        // text order, which the binary searches over them rely on.
        if let Err(e) = i.subwords.check() {
            return fail(format!("the subwords: {e}"));
        }
        // The levels' own check: a parent for every unit, before it and
        // around it, which placing each unit in its parent relies on.
        if let Some(Err(e)) = i.levels.map(|l| l.check()) {
            return fail(format!("the levels: {e}"));
        }
        let items = [
            i.sentences.map(|s| ("sentence", s.columns())),
            i.chunks.map(|c| ("chunk", c.columns())),
            i.levels.map(|l| ("unit", l.spans.columns())),
        ];
        for (what, (starts, ends)) in items.into_iter().flatten() {
            if let Some(k) = starts.iter().zip(ends).position(|(s, e)| s > e) {
                return fail(format!("{what} {k} ends before it starts: {}..{}", starts[k], ends[k]));
            }
        }
        if i.subwords.model != self.model {
            return fail(format!(
                "the subwords are {}'s pieces (revision {}), the vectors {}'s (revision {})",
                i.subwords.model.id, i.subwords.model.revision, self.model.id, self.model.revision
            ));
        }
        let view = i.subwords.spans.view();
        let views = [
            i.sentences.map(|s| ("sentences", s.view())),
            i.chunks.map(|c| ("chunks", c.view())),
            i.levels.map(|l| ("levels", l.view())),
        ];
        if let Some((what, v)) = views.into_iter().flatten().find(|(_, v)| *v != view) {
            return fail(format!("the {what} are in view {}, the subwords in view {}", v.0, view.0));
        }
        let zeros = |n: usize| vec![0.0f32; n * self.dim as usize];
        Ok(Output {
            sentences: i.sentences.map(|s| Vectors::new(self.dim, zeros(s.len()), Vec::new(), self.model.clone())),
            chunks: i.chunks.map(|c| Vectors::new(self.dim, zeros(c.len()), Vec::new(), self.model.clone())),
            levels: i.levels.map(|l| {
                VectorKind::ALL.map(|_| Vectors::new(self.dim, zeros(l.len()), Vec::new(), self.model.clone()))
            }),
        })
    }

    /// Every window of the plan through the encoder, in as few runs as the
    /// session's rows and ranges allow. A window whose ranges do not all
    /// fit in the run being built goes again in the next.
    fn run(&self, st: &mut State, inputs: &[Input<'_>], out: &mut [Result<Output, Error>]) -> Result<(), CoreError> {
        let State { plan, run } = st;
        run.rows.clear();
        run.ranges.clear();
        run.targets.clear();
        run.done = Shape::default();
        let dim = self.dim as usize;
        let head = self.head.len() as u32;
        pack(plan, self.max_batch, self.max_ranges, |step| match step {
            Pack::Flush => run.flush(dim, &plan.placed, out),
            Pack::Row { window, placed } => {
                let win = &plan.windows[window];
                let ids = &inputs[win.doc as usize].subwords.ids[win.start as usize..win.end as usize];
                let row = run.rows.push(&[&self.head, ids, &self.tail]);
                for (k, x) in plan.placed[placed.clone()].iter().enumerate() {
                    run.ranges.push(turbo_token_range { row, start: head + x.start, end: head + x.end });
                    run.targets.push((placed.start + k) as u32);
                }
                Ok(())
            }
        })
    }
}

/// One step of packing a plan into runs.
enum Pack {
    /// Run the rows built so far.
    Flush,
    /// A row of the window, holding these of the plan's placements.
    Row { window: usize, placed: std::ops::Range<usize> },
}

/// The rows and runs a plan goes through the encoder in, in order: a row
/// for each window with placements, again for those that do not fit in
/// the run being built, and a flush when a run holds `max_batch` rows or
/// `max_ranges` ranges, and after the last row. `embed` runs by it and
/// `shape` counts by it, so the two never differ.
fn pack<E>(plan: &Plan, max_batch: u32, max_ranges: u32, mut f: impl FnMut(Pack) -> Result<(), E>) -> Result<(), E> {
    let (mut rows, mut ranges) = (0u32, 0u32);
    let mut p = 0usize;
    for w in 0..plan.windows.len() {
        let end = p + plan.placed[p..].iter().take_while(|x| x.window as usize == w).count();
        while p < end {
            if rows == max_batch || ranges == max_ranges {
                f(Pack::Flush)?;
                (rows, ranges) = (0, 0);
            }
            let take = (end - p).min((max_ranges - ranges) as usize);
            f(Pack::Row { window: w, placed: p..p + take })?;
            rows += 1;
            ranges += take as u32;
            p += take;
        }
    }
    if rows > 0 {
        f(Pack::Flush)?;
    }
    Ok(())
}
