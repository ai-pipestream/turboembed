use super::*;

/// One document's existing text, token, and subword layers.
pub struct Input<'a> {
    pub view: ViewId,
    pub text: &'a str,
    pub regions: Vec<(usize, usize)>,
    pub tokens: Option<&'a Tokens>,
    pub subwords: Option<&'a Subwords>,
}

impl NameFinderDl {
    pub(super) fn find_layers(&self, inputs: &[Input<'_>]) -> Result<Vec<Vec<Candidate>>, String> {
        let mut found: Vec<Vec<Candidate>> = vec![Vec::new(); inputs.len()];
        let mut run = self.run.lock().unwrap_or_else(PoisonError::into_inner);
        run.rows.clear();
        run.ranges.clear();
        run.pieces.clear();
        run.chunk = Chunk::default();
        let mut ids = Vec::new();
        let mut spans = Vec::new();
        let bare = Encode {
            add_special_tokens: false,
            truncation: Truncation::None,
            max_tokens: u32::MAX,
            prompt: PromptRole::None,
        };
        for (d, input) in inputs.iter().enumerate() {
            if let Some(subwords) = input.subwords {
                turbo_analysis::Layer::check(subwords).map_err(|e| format!("invalid subwords: {e}"))?;
                if subwords.spans.view() != input.view {
                    return Err("the subwords are in another view".into());
                }
                if subwords.model != self.model {
                    return Err("the subwords are from another model".into());
                }
            }
            let (text, regions) = (input.text, &input.regions);
            for &(lo, hi) in regions {
                let region = text.get(lo..hi).ok_or_else(|| format!("sentence {lo}..{hi} is not in the text"))?;
                // WordSlice input separates these controls, while whole-text
                // WordPiece can delete them and join the neighboring words.
                // Such a layer is not compatible even with the same model.
                let subwords = input.subwords.filter(|_| {
                    !region.chars().any(|c| c.is_control() && c.is_whitespace() && !matches!(c, '\t' | '\n' | '\r'))
                });
                let fallback = decode::words(region);
                let reused = input.tokens.map(|tokens| {
                    (0..tokens.spans.len())
                        .flat_map(|i| tokens.spans.fragments(turbo_analysis::Ix::new(i as u32)))
                        .filter(|s| lo as u32 <= s.start && s.end <= hi as u32)
                        .map(|s| (s.start as usize - lo, s.end as usize - lo))
                        .collect::<Vec<_>>()
                });
                let words = reused.filter(|words| *words == fallback).unwrap_or(fallback);
                let (mut region_ids, mut region_spans) = (Vec::new(), Vec::new());
                if subwords.is_none() {
                    for &(start, end) in &words {
                        let at = region_ids.len();
                        self.tokenizer
                            .encode_spans(&region[start..end], bare, &mut region_ids, &mut region_spans)
                            .map_err(|e| e.to_string())?;
                        for span in &mut region_spans[at..] {
                            span[0] += (lo + start) as u32;
                            span[1] += (lo + start) as u32;
                        }
                    }
                }
                let pre = decode::name_words(region);
                for (a, b) in decode::chunk_ranges(words.len(), self.split, self.overlap)? {
                    ids.clear();
                    spans.clear();
                    if !words.is_empty() {
                        let (ws, we) = (lo + words[a].0, lo + words[b - 1].1);
                        if let Some(subwords) = subwords {
                            for (i, (_, span)) in subwords.spans.iter().enumerate() {
                                if span.start < we as u32 && (ws as u32) < span.end {
                                    ids.push(subwords.ids[i]);
                                    spans.push([span.start, span.end]);
                                }
                            }
                        } else {
                            for (i, &[start, end]) in region_spans.iter().enumerate() {
                                if start < we as u32 && (ws as u32) < end {
                                    ids.push(region_ids[i]);
                                    spans.push([start, end]);
                                }
                            }
                        }
                    }
                    let locate = |s: &[u32; 2]| -> Result<Option<(usize, usize)>, String> {
                        let (a, b) = (s[0] as usize, s[1] as usize);
                        if !(text.is_char_boundary(a) && text.is_char_boundary(b) && b <= hi) {
                            return Err(format!("piece {}..{} is not text characters", s[0], s[1]));
                        }
                        Ok(Some((a, b)))
                    };
                    let parts = decode::row_ranges(ids.len(), self.width);
                    let count = parts.len();
                    for (k, &(first, end, own_start, own_end)) in parts.iter().enumerate() {
                        let part = &ids[first..end];
                        let rows = self.cls.len() + part.len() + self.sep.len();
                        if run.rows.len() as usize == self.max_batch || run.ranges.len() + rows > self.max_ranges {
                            self.flush(&mut run, &mut found)?;
                        }
                        let row = run.rows.push(&[&self.cls, part, &self.sep]);
                        let mut bytes: Positions = vec![None; self.cls.len()];
                        let piece = &spans[first..first + part.len()];
                        for s in piece {
                            bytes.push(locate(s)?);
                        }
                        bytes.resize(rows, None);
                        let mut words = vec![None; self.cls.len()];
                        words.extend(
                            piece
                                .iter()
                                .map(|s| Some(pre.partition_point(|w| lo + w.0 <= s[0] as usize).saturating_sub(1))),
                        );
                        words.resize(rows, None);
                        for p in 0..rows as u32 {
                            run.ranges.push(turbo_token_range { row, start: p, end: p + 1 });
                        }
                        let c = self.cls.len();
                        let lo = if k == 0 { 0 } else { c + own_start - first };
                        let hi = if k + 1 == count { rows } else { c + own_end - first };
                        run.pieces.push(Row { doc: d, keep: lo..hi, last: k + 1 == count, bytes, words });
                    }
                }
            }
        }
        self.flush(&mut run, &mut found)?;
        Ok(inputs.iter().zip(found).map(|(input, c)| decode::resolve(input.text, c)).collect())
    }
}
