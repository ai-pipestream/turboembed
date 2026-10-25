//! Raw window states retained for heads whose spans depend on BIO predictions.
use super::*;
use crate::window::place_tokens_in_windows;

pub(crate) struct CachedWindows {
    windows: Vec<crate::window::Window>,
    states: Vec<Vec<f32>>,
    tokens: Vec<crate::window::Window>,
}

impl VectorAnnotator {
    #[cfg(test)]
    pub(crate) fn run_counts(&self) -> (usize, usize) {
        self.lock().run.encoder.run_counts()
    }

    /// Encode the union of the sentence-pooling and token-selection windows.
    /// Each distinct window runs once, regardless of the number of later heads.
    pub(crate) fn cache_windows(
        &self,
        caller: &'static str,
        inputs: &[(&Subwords, &Spans<Sentences>)],
    ) -> Vec<Result<CachedWindows, Error>> {
        let fail = |e: String| Error::Annotator { annotator: caller, detail: e };
        let width = self.width();
        let mut out: Vec<_> = inputs
            .iter()
            .map(|&(sub, sentences)| {
                self.output_for(
                    caller,
                    &Input { subwords: sub, sentences: Some(sentences), chunks: None, levels: None },
                )?;
                let mut plan = Plan::new(width);
                layout(&mut plan, sub, sentences);
                plan.place(0, sub.len() as u32).map_err(|e| fail(e.to_string()))?;
                let mut windows = plan.windows.clone();
                plan.clear();
                layout(&mut plan, sub, sentences);
                plan.place_tokens(0, sub.len() as u32, self.stride).map_err(|e| fail(e.to_string()))?;
                let tokens = plan.windows;
                for &w in &tokens {
                    if !windows.contains(&w) {
                        windows.push(w);
                    }
                }
                let states =
                    windows.iter().map(|w| vec![0.0; (w.end - w.start) as usize * self.dim as usize]).collect();
                Ok(CachedWindows { windows, states, tokens })
            })
            .collect();
        let mut guard = self.lock();
        let run = &mut guard.run;
        let result = (|| {
            run.encoder.reserve_token_ranges()?;
            run.rows.clear();
            run.ranges.clear();
            let mut locations = Vec::new();
            for (d, doc) in out.iter().enumerate() {
                let Ok(doc) = doc else { continue };
                for (w, window) in doc.windows.iter().enumerate() {
                    locations.push((d, w, *window));
                }
            }
            for batch in locations.chunks(self.max_batch as usize) {
                for &(d, _, w) in batch {
                    let ids = &inputs[d].0.ids[w.start as usize..w.end as usize];
                    let row = run.rows.push(&[&self.head, ids, &self.tail]);
                    for p in 0..ids.len() as u32 {
                        let start = self.head.len() as u32 + p;
                        run.ranges.push(turbo_token_range { row, start, end: start + 1 });
                    }
                }
                run.encoder.run_ranges(&run.rows, &run.ranges, &mut run.flat)?;
                let mut offset = 0;
                for &(d, w, _) in batch {
                    let states = &mut out[d].as_mut().expect("validated document").states[w];
                    let len = states.len();
                    states.copy_from_slice(&run.flat[offset..offset + len]);
                    offset += states.len();
                }
                run.rows.clear();
                run.ranges.clear();
            }
            Ok::<_, CoreError>(())
        })();
        if let Err(e) = result {
            for slot in out.iter_mut().filter(|x| x.is_ok()) {
                *slot = Err(fail(e.to_string()));
            }
        }
        out
    }

    pub(crate) fn cached_tokens(
        &self,
        caller: &'static str,
        sub: &Subwords,
        cached: &CachedWindows,
    ) -> Result<Vectors<Subwords>, Error> {
        let dim = self.dim as usize;
        let mut states = vec![0.0; sub.len() * dim];
        for p in place_tokens_in_windows(0, sub.len() as u32, &cached.tokens) {
            let win = cached.tokens[p.window as usize];
            let w = cached.windows.iter().position(|&w| w == win).expect("token window retained");
            states[p.piece as usize * dim..(p.piece as usize + 1) * dim]
                .copy_from_slice(&cached.states[w][p.offset as usize * dim..(p.offset as usize + 1) * dim]);
        }
        let data = if let Some(proj) = &self.projection {
            proj.project(&states, self.l2).map_err(|e| Error::Annotator { annotator: caller, detail: e.to_string() })?
        } else {
            if self.l2 {
                for row in states.chunks_exact_mut(dim) {
                    normalize(row);
                }
            }
            states
        };
        Ok(Vectors::new(self.token_dim(), data, Vec::new(), self.model.clone()))
    }

    pub(crate) fn cached_spans<L: Layer>(
        &self,
        caller: &'static str,
        sub: &Subwords,
        sentences: &Spans<Sentences>,
        cached: &CachedWindows,
        spans: &[Span],
    ) -> Result<Vectors<L>, Error> {
        let fail = |detail: String| Error::Annotator { annotator: caller, detail };
        let mut plan = Plan::new(self.width());
        layout(&mut plan, sub, sentences);
        let (starts, ends) = sub.spans.columns();
        let mut empty = Vec::new();
        for (i, s) in spans.iter().enumerate() {
            let r = pieces_of(starts, ends, s.start, s.end);
            if r.is_empty() {
                empty.push(Ix::new(i as u32));
            }
            plan.item(Target { doc: 0, which: Which::Chunks, item: i as u32 }, r);
        }
        plan.place(0, sub.len() as u32).map_err(|e| fail(e.to_string()))?;
        let dim = self.dim as usize;
        let mut data = vec![0.0; spans.len() * dim];
        for x in &plan.placed {
            if x.target.which != Which::Chunks {
                continue;
            }
            let w = cached
                .windows
                .iter()
                .position(|&w| w == plan.windows[x.window as usize])
                .ok_or_else(|| fail("candidate span is outside the cached sentence windows".into()))?;
            let dst = &mut data[x.target.item as usize * dim..(x.target.item as usize + 1) * dim];
            let mut mean = vec![0.0; dim];
            for row in cached.states[w][x.start as usize * dim..x.end as usize * dim].chunks_exact(dim) {
                for (a, &b) in mean.iter_mut().zip(row) {
                    *a += b;
                }
            }
            let inv = 1.0 / (x.end - x.start) as f32;
            for (d, v) in dst.iter_mut().zip(mean) {
                let v = v * inv;
                if x.whole {
                    *d = v;
                } else {
                    *d += v * (x.end - x.start) as f32;
                }
            }
        }
        for s in plan.split.iter().filter(|s| s.target.which == Which::Chunks) {
            let inv = 1.0 / s.pieces as f32;
            for v in &mut data[s.target.item as usize * dim..(s.target.item as usize + 1) * dim] {
                *v *= inv;
            }
        }
        if self.l2 {
            for row in data.chunks_exact_mut(dim) {
                normalize(row);
            }
        }
        Ok(Vectors::new(self.dim, data, empty, self.model.clone()))
    }
}

fn layout(plan: &mut Plan, sub: &Subwords, sentences: &Spans<Sentences>) {
    let (starts, ends) = sub.spans.columns();
    for (i, s) in sentences.iter() {
        plan.layout(
            Target { doc: 0, which: Which::Sentences, item: i.get() as u32 },
            pieces_of(starts, ends, s.start, s.end),
        );
    }
}

#[cfg(test)]
impl CachedWindows {
    pub(super) fn window_count(&self) -> usize {
        self.windows.len()
    }
}
