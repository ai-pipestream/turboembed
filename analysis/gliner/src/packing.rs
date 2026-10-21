use crate::{
    Config, ExtractionSchema,
    schema::{MAX_TOKEN_BYTES, MAX_TOKENS},
};
use turbo::sentencepiece::{PieceType, Scratch, SentencePiece};

/// Checked again by `score_packed`, including positions, marker IDs and bounds.
/// This lower-level API permits independently tokenized numerical fixtures.
#[derive(Clone, Debug)]
pub struct PackedInput {
    pub ids: Vec<u32>,
    pub prompts: Vec<usize>,
    pub words: Vec<usize>,
}
impl PackedInput {
    pub(crate) fn check(&self, c: &Config, ec: &turbo::deberta::Config, labels: usize) -> Result<(), String> {
        if self.ids.len() < 3 || self.ids.len() > ec.max_sequence {
            return Err("max_sequence limit exceeded".into());
        }
        if self.words.is_empty() || self.words.len() > MAX_TOKENS {
            return Err("max_tokens limit: 1..=384".into());
        }
        if self.prompts.len() != labels || self.ids[0] != c.cls_token || self.ids.last() != Some(&c.eos_token) {
            return Err("gliner prompt layout mismatch".into());
        }
        let mut prev = 0;
        for &p in &self.prompts {
            if p <= prev || self.ids.get(p) != Some(&c.entity_token) {
                return Err("gliner prompt position mismatch".into());
            }
            prev = p;
        }
        let separators: Vec<_> =
            self.ids.iter().enumerate().filter_map(|(i, &v)| (v == c.separator_token).then_some(i)).collect();
        if separators.len() != 1 || separators[0] <= prev {
            return Err("gliner separator layout mismatch".into());
        }
        prev = separators[0];
        for &p in &self.words {
            if p <= prev || p >= self.ids.len() - 1 {
                return Err("gliner word position mismatch".into());
            }
            prev = p;
        }
        if self.ids.iter().filter(|&&x| x == c.entity_token).count() != labels {
            return Err("gliner unexpected entity marker".into());
        }
        Ok(())
    }
}
pub(crate) fn pack(
    sp: &SentencePiece,
    c: &Config,
    ec: &turbo::deberta::Config,
    words: &[&str],
    schema: &ExtractionSchema,
) -> Result<PackedInput, String> {
    if words.is_empty() || words.len() > MAX_TOKENS {
        return Err("max_tokens limit: 1..=384".into());
    }
    let mut out = PackedInput { ids: vec![c.cls_token], prompts: Vec::new(), words: Vec::new() };
    let mut scratch = Scratch::default();
    let mut pieces = Vec::new();
    let mut append = |text: &str, ids: &mut Vec<u32>| -> Result<(), String> {
        pieces.clear();
        sp.encode_ids(text, &mut scratch, &mut pieces);
        if pieces.is_empty() {
            // Preserve a position for nonempty text erased by normalization.
            pieces.push(sp.unk_id());
        }
        // HF DebertaV2TokenizerFast disables SentencePiece byte fallback and
        // fuses adjacent unknown pieces. Preserve that checkpoint frontend.
        let mut unknown = false;
        for &id in &pieces {
            let byte = sp.kind(id) == Some(PieceType::Byte);
            if !byte || !unknown {
                ids.push(if byte { sp.unk_id() } else { id });
            }
            unknown = byte;
            if ids.len() + 1 > ec.max_sequence {
                return Err("max_sequence limit exceeded".into());
            }
        }
        Ok(())
    };
    for label in schema.labels() {
        out.prompts.push(out.ids.len());
        out.ids.push(c.entity_token);
        append(label, &mut out.ids)?;
    }
    out.ids.push(c.separator_token);
    for word in words {
        if word.trim().is_empty() || word.len() > MAX_TOKEN_BYTES {
            return Err("max_token_bytes limit: nonempty, <=4096 bytes".into());
        }
        if ["<<ENT>>", "<<SEP>>", "[FLERT]", "[CLS]", "[SEP]", "[MASK]", "[PAD]", "[UNK]"]
            .iter()
            .any(|s| word.contains(s))
        {
            return Err("reserved prompt token in text".into());
        }
        out.words.push(out.ids.len());
        append(word, &mut out.ids)?;
    }
    out.ids.push(c.eos_token);
    out.check(c, ec, schema.labels().len())?;
    Ok(out)
}
