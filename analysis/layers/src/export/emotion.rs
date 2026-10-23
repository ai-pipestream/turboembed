//! The `turbo:emotion` codec: the token, sentence and chunk references,
//! the targets (tag 0 a sentence, 1 a chunk, then the index), the scores
//! (28 per item), then the evidence labels, spans and ends. Reading
//! checks every reference, index, count and score.

use super::*;
use crate::{EMOTION_DIMENSIONS, EmotionDimension, EmotionLayer, EmotionTarget};

impl LayerCodec for EmotionLayer {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_ref(&mut w, &cx.resolve(&self.tokens_key))?;
        put_ref(&mut w, &cx.resolve(&self.sentences_key))?;
        put_ref(&mut w, &cx.resolve(&self.chunks_key))?;
        w.usize(self.target.len())?;
        for t in &self.target {
            let (tag, i) = match t {
                EmotionTarget::Sentence(i) => (0, i.get()),
                EmotionTarget::Chunk(i) => (1, i.get()),
            };
            w.u8(tag);
            w.u32(i as u32);
        }
        put_f64s(&mut w, &self.scores)?;
        w.usize(self.evidence_label.len())?;
        for l in &self.evidence_label {
            w.u8(*l as u8);
        }
        w.usize(self.evidence_tokens.len())?;
        for t in &self.evidence_tokens {
            w.u32(t.start.get() as u32);
            w.u32(t.end.get() as u32);
        }
        put_u32s(&mut w, &self.evidence_end)?;
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let tokens_ref = get_ref(&mut r)?;
        want_target::<Tokens>(&tokens_ref, cx)?;
        let sentences_ref = get_ref(&mut r)?;
        want_target::<Sentences>(&sentences_ref, cx)?;
        let chunks_ref = get_ref(&mut r)?;
        want_target::<Chunks>(&chunks_ref, cx)?;
        let tokens_key = cx.key::<Tokens>(&tokens_ref.instance.clone());
        let sentences_key = cx.key::<Sentences>(&sentences_ref.instance.clone());
        let chunks_key = cx.key::<Chunks>(&chunks_ref.instance.clone());
        let mut target = Vec::new();
        for _ in 0..r.usize()? {
            let tag = r.u8()?;
            let index = r.u32()?;
            target.push(match tag {
                0 => {
                    cx.require_node(&NodeRef { layer: sentences_ref.clone(), index })?;
                    EmotionTarget::Sentence(Ix::new(index))
                }
                1 => {
                    cx.require_node(&NodeRef { layer: chunks_ref.clone(), index })?;
                    EmotionTarget::Chunk(Ix::new(index))
                }
                tag => return Err(malformed(format!("an emotion target tag of {tag}"))),
            });
        }
        let scores = get_f64s(&mut r)?;
        let mut evidence_label = Vec::new();
        for _ in 0..r.usize()? {
            let l = EmotionDimension::from_index(r.u8()? as usize)
                .ok_or_else(|| malformed("an emotion label outside the table"))?;
            evidence_label.push(l);
        }
        let mut evidence_tokens = Vec::new();
        for _ in 0..r.usize()? {
            evidence_tokens.push(get_checked_range(&mut r, cx, &tokens_ref)?);
        }
        let evidence_end = get_u32s(&mut r)?;
        let n = target.len();
        want_len(
            scores.len(),
            n.checked_mul(EMOTION_DIMENSIONS).ok_or_else(|| malformed("too many items"))?,
            "scores",
        )?;
        want_len(evidence_tokens.len(), evidence_label.len(), "evidence spans")?;
        want_len(evidence_end.len(), n, "evidence ends")?;
        if evidence_end.windows(2).any(|w| w[0] > w[1])
            || evidence_end.last().copied().unwrap_or(0) as usize != evidence_label.len()
        {
            return Err(malformed("evidence ends do not partition the evidence"));
        }
        done(&r, "emotion")?;
        Ok(EmotionLayer {
            target,
            scores,
            evidence_label,
            evidence_tokens,
            evidence_end,
            tokens_key,
            sentences_key,
            chunks_key,
        })
    }
}
