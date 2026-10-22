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
/// Reused normalization workspace for prompt labels and document words.
pub(crate) struct Packer<'a> {
    sp: &'a SentencePiece,
    scratch: Scratch,
    pieces: Vec<u32>,
}
impl<'a> Packer<'a> {
    pub(crate) fn new(sp: &'a SentencePiece) -> Self {
        Self { sp, scratch: Scratch::default(), pieces: Vec::new() }
    }
    fn encode(&mut self, text: &str) -> &[u32] {
        self.pieces.clear();
        self.sp.encode_ids(text, &mut self.scratch, &mut self.pieces);
        if self.pieces.is_empty() {
            self.pieces.push(self.sp.unk_id());
        }
        // Match HF fast-tokenizer byte fallback: fuse a byte run to one UNK.
        let mut used = 0;
        let mut unknown = false;
        for i in 0..self.pieces.len() {
            let id = self.pieces[i];
            let byte = self.sp.kind(id) == Some(PieceType::Byte);
            if !byte || !unknown {
                self.pieces[used] = if byte { self.sp.unk_id() } else { id };
                used += 1;
            }
            unknown = byte;
        }
        self.pieces.truncate(used);
        &self.pieces
    }
    pub(crate) fn word(&mut self, word: &str) -> Result<&[u32], String> {
        if word.trim().is_empty() || word.len() > MAX_TOKEN_BYTES {
            return Err("max_token_bytes limit: nonempty, <=4096 bytes".into());
        }
        if ["<<ENT>>", "<<SEP>>", "[FLERT]", "[CLS]", "[SEP]", "[MASK]", "[PAD]", "[UNK]"]
            .iter()
            .any(|s| word.contains(s))
        {
            return Err("reserved prompt token in text".into());
        }
        Ok(self.encode(word))
    }
    pub(crate) fn prefix(
        &mut self,
        c: &Config,
        max_sequence: usize,
        schema: &ExtractionSchema,
    ) -> Result<PackedInput, String> {
        let mut out = PackedInput { ids: vec![c.cls_token], prompts: Vec::new(), words: Vec::new() };
        for label in schema.labels() {
            out.prompts.push(out.ids.len());
            out.ids.push(c.entity_token);
            out.ids.extend_from_slice(self.encode(label));
        }
        out.ids.push(c.separator_token);
        // Reserve an EOS and at least one document piece.
        if out.ids.len() + 2 > max_sequence {
            return Err("max_sequence limit: schema prompt leaves no room for a token".into());
        }
        Ok(out)
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
    let mut packer = Packer::new(sp);
    let mut out = packer.prefix(c, ec.max_sequence, schema)?;
    for word in words {
        let pieces = packer.word(word)?;
        if out.ids.len() + pieces.len() + 1 > ec.max_sequence {
            return Err("max_sequence limit exceeded".into());
        }
        out.words.push(out.ids.len());
        out.ids.extend_from_slice(pieces);
    }
    out.ids.push(c.eos_token);
    out.check(c, ec, schema.labels().len())?;
    Ok(out)
}
