//! Per-token contextual vector extraction with source spans and window
//! provenance (`#709`).

use turbo::manifest::{PromptRole, Truncation};
use turbo::status::Error as CoreError;
use turbo::tokenizer::Encode;
use turbo::turbo_token_range;
use turbo_analysis::{Error, Ix, Layer, ModelRef, Span, Spans, ViewId};
use turbo_analysis_layers::{Sentences, Subwords, Vectors};

use crate::encoder::normalize;
use crate::window::{Placed, Target, Which, Window, pieces_of};

use super::{State, VectorAnnotator};

/// Unpooled (or linearly projected) per-token contextual vectors for one
/// document or text, alongside each token's source byte span and the
/// encoder window index from which its hidden state was extracted.
///
/// Special tokens (`[CLS]`, `[SEP]`, `<s>`, `</s>`, prompt prefix tokens,
/// and padding) are excluded so every row `i in 0..self.len()` corresponds
/// to subword piece `i` and a real byte span `self.spans[i]` in the input
/// text.
#[derive(Clone, Debug, PartialEq)]
pub struct ContextualTokens {
    /// Output vector width (`projection.dim` when a projection head is
    /// active, else `architecture.hidden`).
    pub dim: u32,
    /// Row-major token vectors `[total_tokens, dim]`.
    pub data: Vec<f32>,
    /// Subword token IDs `[total_tokens]` in text order.
    pub ids: Vec<i32>,
    /// Source byte spans `[total_tokens]` in text order.
    pub spans: Vec<Span>,
    /// Window index (`0..self.windows.len()`) from which each token's
    /// vector was taken (`[total_tokens]`).
    pub window_indices: Vec<u32>,
    /// Distance of each token to the nearest boundary of its chosen window
    /// (`[total_tokens]`).
    pub interior_distances: Vec<u32>,
    /// Encoder windows (`[start_piece, end_piece)`) executed for this input.
    pub windows: Vec<Window>,
    /// Model provenance.
    pub model: ModelRef,
}

impl ContextualTokens {
    /// Number of tokens in the output.
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    /// Whether the token sequence is empty.
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// The contextual vector of token `index` (`0..self.len()`).
    pub fn vector(&self, index: usize) -> &[f32] {
        let d = self.dim as usize;
        &self.data[index * d..(index + 1) * d]
    }

    /// Converts this token embedding output into a `Vectors<Subwords>` layer.
    pub fn into_subword_vectors(self) -> Vectors<Subwords> {
        Vectors::new(self.dim, self.data, Vec::new(), self.model)
    }

    /// Clones the token vectors into a `Vectors<Subwords>` layer.
    pub fn to_subword_vectors(&self) -> Vectors<Subwords> {
        Vectors::new(self.dim, self.data.clone(), Vec::new(), self.model.clone())
    }
}

impl VectorAnnotator {
    /// Tokenizes `text` and extracts one contextual vector per subword token
    /// (excluding special tokens and padding) using `self.stride()`, returning
    /// the `[total_tokens, out_dim]` matrix alongside source byte spans and
    /// window indices.
    pub fn embed_text_tokens(&self, text: &str) -> Result<ContextualTokens, Error> {
        self.embed_text_tokens_with_stride(text, self.stride)
    }

    /// Tokenizes `text` and extracts one contextual vector per subword token
    /// using the given sliding-window `stride` (`0` uses the default half-window
    /// stride).
    pub fn embed_text_tokens_with_stride(&self, text: &str, stride: u32) -> Result<ContextualTokens, Error> {
        let opts = Encode {
            add_special_tokens: false,
            truncation: Truncation::None,
            max_tokens: u32::MAX,
            prompt: PromptRole::None,
        };
        let (mut ids, mut raw_spans) = (Vec::new(), Vec::new());
        self.tokenizer
            .encode_spans(text, opts, &mut ids, &mut raw_spans)
            .map_err(|e| Error::Annotator { annotator: "turbo:subword-vectors", detail: e.to_string() })?;
        let mut spans = Spans::with_capacity(ViewId::SOURCE, raw_spans.len());
        for &[start, end] in &raw_spans {
            spans.push(Span::new(start, end));
        }
        let subwords = Subwords { spans, ids, model: self.model.clone(), token: None };
        self.embed_subword_tokens("turbo:subword-vectors", &subwords, None, stride)
    }

    /// Extracts per-token contextual vectors, source spans, and most-interior
    /// window indices for `subwords`.
    pub fn embed_subword_tokens(
        &self,
        annotator: &'static str,
        subwords: &Subwords,
        sentences: Option<&Spans<Sentences>>,
        stride: u32,
    ) -> Result<ContextualTokens, Error> {
        self.embed_subword_tokens_batch(annotator, &[(subwords, sentences)], stride).remove(0)
    }

    /// Batched extraction of per-token contextual vectors, source spans, and
    /// most-interior window indices across multiple documents.
    pub fn embed_subword_tokens_batch(
        &self,
        annotator: &'static str,
        inputs: &[(&Subwords, Option<&Spans<Sentences>>)],
        stride: u32,
    ) -> Vec<Result<ContextualTokens, Error>> {
        let width = self.width();
        if stride > width {
            let err = Error::Annotator {
                annotator,
                detail: format!("token window stride {stride} exceeds window width {width}"),
            };
            return vec![Err(err); inputs.len()];
        }

        let hidden_dim = self.dim as usize;
        let out_dim = self.token_dim();
        let mut raw_states: Vec<Result<Vec<f32>, Error>> =
            inputs.iter().map(|&(sub, sents)| self.validate_token_input(annotator, sub, sents, hidden_dim)).collect();

        #[derive(Clone)]
        struct DocTokenMeta {
            window_indices: Vec<u32>,
            interior_distances: Vec<u32>,
            windows: Vec<Window>,
        }

        let mut doc_meta: Vec<Option<DocTokenMeta>> = vec![None; inputs.len()];

        let mut guard = self.lock();
        let st = &mut *guard;
        st.plan.clear();

        for (d, (&(sub, sents), slot)) in inputs.iter().zip(&mut raw_states).enumerate() {
            if slot.is_err() {
                continue;
            }
            let (starts, ends) = sub.spans.columns();
            let doc_id = d as u32;
            if let Some(s) = sents {
                for (i, span) in s.iter() {
                    let r = pieces_of(starts, ends, span.start, span.end);
                    st.plan.layout(Target { doc: doc_id, which: Which::Sentences, item: i.get() as u32 }, r);
                }
            }
            let win_offset = st.plan.windows.len() as u32;
            let placements = match st.plan.place_tokens(doc_id, sub.len() as u32, stride) {
                Ok(p) => p,
                Err(e) => {
                    *slot = Err(Error::Annotator { annotator, detail: e.to_string() });
                    continue;
                }
            };
            let windows: Vec<Window> = st.plan.windows[win_offset as usize..]
                .iter()
                .map(|w| Window { doc: 0, start: w.start, end: w.end })
                .collect();
            let mut window_indices = Vec::with_capacity(placements.len());
            let mut interior_distances = Vec::with_capacity(placements.len());
            for tp in placements {
                window_indices.push(tp.window - win_offset);
                interior_distances.push(tp.interior_distance);
                st.plan.placed.push(Placed {
                    window: tp.window,
                    start: tp.offset,
                    end: tp.offset + 1,
                    target: Target { doc: doc_id, which: Which::Sentences, item: tp.piece },
                    whole: true,
                });
            }
            doc_meta[d] = Some(DocTokenMeta { window_indices, interior_distances, windows });
        }

        st.plan.placed.sort_by_key(|x| x.window);

        if let Err(e) = self.run_tokens(st, inputs, &mut raw_states) {
            let err = Error::Annotator { annotator, detail: e.to_string() };
            drop(guard);
            return raw_states.into_iter().map(|r| r.and(Err(err.clone()))).collect();
        }
        drop(guard);

        raw_states
            .into_iter()
            .zip(inputs)
            .zip(doc_meta)
            .map(|((res, &(sub, _)), meta)| {
                let mut states = res?;
                let meta = meta.expect("metadata present for valid document");
                let data = if let Some(proj) = &self.projection {
                    proj.project(&states, self.l2).map_err(|e| Error::Annotator { annotator, detail: e.to_string() })?
                } else {
                    if self.l2 {
                        for row in states.chunks_exact_mut(hidden_dim) {
                            normalize(row);
                        }
                    }
                    states
                };
                let spans: Vec<Span> = (0..sub.len()).map(|i| sub.spans.get(Ix::new(i as u32))).collect();
                Ok(ContextualTokens {
                    dim: out_dim,
                    data,
                    ids: sub.ids.clone(),
                    spans,
                    window_indices: meta.window_indices,
                    interior_distances: meta.interior_distances,
                    windows: meta.windows,
                    model: self.model.clone(),
                })
            })
            .collect()
    }

    fn validate_token_input(
        &self,
        annotator: &'static str,
        subwords: &Subwords,
        sentences: Option<&Spans<Sentences>>,
        hidden_dim: usize,
    ) -> Result<Vec<f32>, Error> {
        let fail = |detail: String| Err(Error::Annotator { annotator, detail });
        fn has_gaps<L>(spans: &Spans<L>) -> bool {
            spans.iter().any(|(ix, _)| spans.is_discontinuous(ix))
        }
        if has_gaps(&subwords.spans) || sentences.is_some_and(has_gaps) {
            return fail("discontinuous spans are not supported by vector window pooling".into());
        }
        let (starts, ends) = subwords.spans.columns();
        if let Some(k) = starts.iter().zip(ends).position(|(s, e)| s > e) {
            return fail(format!("subword {k} ends before it starts: {}..{}", starts[k], ends[k]));
        }
        if let Err(e) = subwords.check() {
            return fail(format!("the subwords: {e}"));
        }
        if let Some(s) = sentences {
            let (s_starts, s_ends) = s.columns();
            if let Some(k) = s_starts.iter().zip(s_ends).position(|(a, b)| a > b) {
                return fail(format!("sentence {k} ends before it starts: {}..{}", s_starts[k], s_ends[k]));
            }
            if s.view() != subwords.spans.view() {
                return fail(format!(
                    "the sentences are in view {}, the subwords in view {}",
                    s.view().0,
                    subwords.spans.view().0
                ));
            }
        }
        if subwords.model != self.model {
            return fail(format!(
                "the subwords are {}'s pieces (revision {}), the vectors {}'s (revision {})",
                subwords.model.id, subwords.model.revision, self.model.id, self.model.revision
            ));
        }
        Ok(vec![0.0f32; subwords.len() * hidden_dim])
    }

    fn run_tokens(
        &self,
        st: &mut State,
        inputs: &[(&Subwords, Option<&Spans<Sentences>>)],
        out: &mut [Result<Vec<f32>, Error>],
    ) -> Result<(), CoreError> {
        let State { plan, run } = st;
        run.rows.clear();
        run.ranges.clear();
        run.targets.clear();
        let dim = self.dim as usize;
        let head = self.head.len() as u32;
        let mut p = 0usize;
        for (w, win) in plan.windows.iter().enumerate() {
            let end = p + plan.placed[p..].iter().take_while(|x| x.window as usize == w).count();
            if p == end {
                continue;
            }
            let ids = &inputs[win.doc as usize].0.ids[win.start as usize..win.end as usize];
            while p < end {
                if run.rows.len() == self.max_batch || run.ranges.len() as u32 == self.max_ranges {
                    flush_tokens(run, dim, &plan.placed, out)?;
                }
                let row = run.rows.push(&[&self.head, ids, &self.tail]);
                let take = (end - p).min(self.max_ranges as usize - run.ranges.len());
                for (k, x) in plan.placed[p..p + take].iter().enumerate() {
                    run.ranges.push(turbo_token_range { row, start: head + x.start, end: head + x.end });
                    run.targets.push((p + k) as u32);
                }
                p += take;
            }
        }
        if !run.rows.is_empty() {
            flush_tokens(run, dim, &plan.placed, out)?;
        }
        Ok(())
    }
}

fn flush_tokens(
    run: &mut super::Run,
    dim: usize,
    placed: &[Placed],
    out: &mut [Result<Vec<f32>, Error>],
) -> Result<(), CoreError> {
    run.encoder.run_ranges(&run.rows, &run.ranges, &mut run.flat)?;
    for (v, &t) in run.flat.chunks_exact(dim).zip(&run.targets) {
        let x = placed[t as usize];
        let Ok(states) = &mut out[x.target.doc as usize] else { continue };
        let idx = x.target.item as usize;
        states[idx * dim..(idx + 1) * dim].copy_from_slice(v);
    }
    run.rows.clear();
    run.ranges.clear();
    run.targets.clear();
    Ok(())
}
