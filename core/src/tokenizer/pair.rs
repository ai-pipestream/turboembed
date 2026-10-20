use super::*;
use crate::manifest::PairTemplate;

/// One joint encoder input, with its passage token interval.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PairWindow {
    pub ids: Vec<i32>,
    pub types: Vec<i32>,
    pub passage: Range<usize>,
}

impl Tokenizer {
    /// Right-truncate the query first, then cover the passage with windows.
    /// Zero limits select the bundle settings; zero stride uses disjoint windows.
    pub fn encode_pair(
        &self,
        query: &str,
        passage: &str,
        query_tokens: u32,
        max_tokens: u32,
        stride: u32,
    ) -> Result<Vec<PairWindow>> {
        let r = self
            .rerank
            .as_ref()
            .ok_or_else(|| Error::new(crate::status::UNSUPPORTED_TASK, "tokenizer has no rerank pair template"))?;
        let max_tokens = if max_tokens == 0 { r.max_seq } else { max_tokens };
        let query_tokens = if query_tokens == 0 { r.query_tokens } else { query_tokens };
        let specials = r.pair_template.specials();
        if max_tokens > r.max_seq || max_tokens <= specials + 1 || query_tokens >= max_tokens - specials {
            return Err(Error::new(INVALID_ARGUMENT, "pair budgets must leave a passage token within max_seq"));
        }
        let opts = Encode {
            add_special_tokens: false,
            truncation: Truncation::Right,
            max_tokens: query_tokens,
            prompt: PromptRole::None,
        };
        let query = self.encode(query, opts)?;
        let passage = self.encode(passage, Encode { truncation: Truncation::None, max_tokens: u32::MAX, ..opts })?;
        let room = max_tokens as usize - specials as usize - query.len();
        let step = if stride == 0 { room } else { stride as usize };
        if step == 0 || step > room {
            return Err(Error::new(INVALID_ARGUMENT, "pair stride must be within passage capacity"));
        }
        let (head, tail) = self.template_ends();
        if head.len() != 1 || tail.len() != 1 {
            return Err(invalid("rerank pair template requires one start and end token"));
        }
        let mut result = Vec::new();
        let mut start = 0usize;
        loop {
            let end = start.saturating_add(room).min(passage.len());
            let mut ids = Vec::with_capacity(max_tokens as usize);
            ids.push(head[0]);
            ids.extend_from_slice(&query);
            ids.push(tail[0]);
            if r.pair_template == PairTemplate::Roberta {
                ids.push(tail[0]);
            }
            let boundary = ids.len();
            ids.extend_from_slice(&passage[start..end]);
            ids.push(tail[0]);
            let mut types = vec![0; ids.len()];
            if r.pair_template == PairTemplate::Bert {
                types[boundary..].fill(1);
            }
            result.push(PairWindow { ids, types, passage: start..end });
            if end == passage.len() {
                break;
            }
            start += step;
        }
        Ok(result)
    }

    pub(super) fn check_pair_reference(&self, bundle: &Bundle) -> Result<()> {
        let m = &bundle.manifest;
        let file = &m.reference.file;
        let bytes = bundle.read_verified(file)?;
        let st = safetensors::File::parse(file, &bytes)?;
        let ids = st.get("ids", Dtype::I32, 2)?;
        let types = st.get("types", Dtype::I32, 2)?;
        let lengths = st.get("lengths", Dtype::I32, 1)?;
        let scores = st.get("scores", Dtype::F32, 2)?;
        let n = m.reference.cases.len() as u64;
        if ids.shape[0] != n || types.shape != ids.shape || lengths.shape != [n] || scores.shape != [n, 1] {
            return Err(invalid("reference: rerank ids/types/lengths/scores shapes disagree"));
        }
        if scores.f32s().iter().any(|v| !v.is_finite()) {
            return Err(invalid("reference.scores: non-finite relevance logit"));
        }
        let width = ids.shape[1] as usize;
        let ids = ids.i32s();
        let types = types.i32s();
        let lengths = lengths.i32s();
        for (i, case) in m.reference.cases.iter().enumerate() {
            let passage = case
                .passage
                .as_ref()
                .ok_or_else(|| invalid(format!("reference.cases[{i}].passage: required for rerank")))?;
            let windows = self.encode_pair(&case.text, passage, 0, 0, 0)?;
            let w = &windows[0];
            if lengths[i] < 0 || lengths[i] as usize != w.ids.len() || w.ids.len() > width {
                return Err(invalid(format!("reference.cases[{i}]: pair length mismatch")));
            }
            let range = i * width..i * width + w.ids.len();
            if ids[range.clone()] != w.ids || types[range] != w.types {
                return Err(invalid(format!("reference.cases[{i}]: pair ids or segment types mismatch")));
            }
        }
        Ok(())
    }
}
