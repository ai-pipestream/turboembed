use super::*;
impl Manifest {
    pub(super) fn validate_tokenizer(&self, e: &Embed, listed_path: &dyn Fn(&str, &str) -> Result<()>) -> Result<()> {
        let t = &self.tokenizer;
        listed_path("tokenizer.file", &t.file)?;
        match (&t.wordpiece, &t.unigram, &t.bpe, &t.sentencepiece) {
            (Some(w), None, None, None) => {
                required("tokenizer.wordpiece.continuing_prefix", &w.continuing_prefix)?;
                positive("tokenizer.wordpiece.max_chars_per_word", w.max_chars_per_word)?;
                if t.normalizer.is_none() {
                    return Err(invalid("manifest.json: tokenizer.normalizer: required with wordpiece"));
                }
            }
            (None, Some(u), None, None) => {
                if u.metaspace.chars().count() != 1 {
                    return Err(invalid(format!(
                        "manifest.json: tokenizer.unigram.metaspace: {:?} is not one character",
                        u.metaspace
                    )));
                }
                if t.normalizer.is_some() {
                    return Err(invalid(
                        "manifest.json: tokenizer.normalizer: a unigram tokenizer normalizes with its file's map, not BertNormalizer",
                    ));
                }
            }
            (None, None, Some(_), None) => {
                if t.normalizer.is_some() {
                    return Err(invalid(
                        "manifest.json: tokenizer.normalizer: a bpe tokenizer reads the text as it is, not BertNormalizer",
                    ));
                }
            }
            (None, None, None, Some(_)) => {
                if t.normalizer.is_some() {
                    return Err(invalid(
                        "manifest.json: tokenizer.normalizer: a sentencepiece tokenizer normalizes as its model says, not BertNormalizer",
                    ));
                }
                if let Some(i) = t.special_tokens.iter().position(|s| s.lstrip) {
                    return Err(invalid(format!(
                        "manifest.json: tokenizer.special_tokens[{i}].lstrip: a sentencepiece tokenizer matches no special token in the text"
                    )));
                }
            }
            _ => return Err(invalid("manifest.json: tokenizer: one of wordpiece, unigram, bpe and sentencepiece")),
        }
        let mut roles = HashSet::new();
        let mut contents = HashSet::new();
        for (i, s) in t.special_tokens.iter().enumerate() {
            if !roles.insert(s.role) {
                return Err(invalid(format!("manifest.json: tokenizer.special_tokens[{i}].role: listed twice")));
            }
            required(&format!("tokenizer.special_tokens[{i}].content"), &s.content)?;
            if !contents.insert(s.content.as_str()) {
                return Err(invalid(format!("manifest.json: tokenizer.special_tokens[{i}].content: listed twice")));
            }
            if (s.single_word || s.rstrip) && !s.normalized {
                return Err(invalid(format!(
                    "manifest.json: tokenizer.special_tokens[{i}]: single_word and rstrip are read with normalized"
                )));
            }
            if t.unigram.is_some() && s.normalized && (s.single_word || s.rstrip) {
                return Err(invalid(
                    "manifest.json: tokenizer.special_tokens: unigram single_word/rstrip are unsupported",
                ));
            }
            // A bpe tokenizer has no normalizer: a normalized token is
            // matched in the text as it is, like the others.
            if s.normalized && t.wordpiece.is_none() && t.bpe.is_none() && t.unigram.is_none() {
                return Err(invalid(format!(
                    "manifest.json: tokenizer.special_tokens[{i}].normalized: requires wordpiece, bpe, or unigram"
                )));
            }
            if (s.single_word || s.rstrip) && t.wordpiece.is_none() {
                return Err(invalid(format!(
                    "manifest.json: tokenizer.special_tokens[{i}]: single_word and rstrip are for a wordpiece tokenizer only"
                )));
            }
            if s.id > i32::MAX as u32 {
                return Err(invalid(format!("manifest.json: tokenizer.special_tokens[{i}].id: over INT32_MAX")));
            }
        }
        if !roles.contains(&SpecialRole::Unk) {
            return Err(invalid("manifest.json: tokenizer.special_tokens: the tokenizer needs a SPECIAL_UNK token"));
        }
        let texts = t.template.iter().filter(|s| *s == "$TEXT").count();
        if texts != 1 {
            return Err(invalid(format!("manifest.json: tokenizer.template: has {texts} \"$TEXT\" entries, not 1")));
        }
        for (i, s) in t.template.iter().enumerate() {
            if s != "$TEXT" && !contents.contains(s.as_str()) {
                return Err(invalid(format!("manifest.json: tokenizer.template[{i}]: {s:?} is not a special token")));
            }
        }
        // TRUNCATE_NONE is a caller's choice, not a model's: a bundle names
        // the side it cuts, and its reference carries a case that is cut.
        if t.truncation == Truncation::None {
            return Err(invalid(
                "manifest.json: tokenizer.truncation: TRUNCATE_NONE is a caller option; a bundle says TRUNCATE_RIGHT or TRUNCATE_LEFT",
            ));
        }
        let specials = t.template.len() as u32 - 1;
        if e.max_seq <= specials {
            return Err(invalid(format!(
                "manifest.json: embed.max_seq: {} leaves no room for text beside {specials} special tokens",
                e.max_seq
            )));
        }

        Ok(())
    }
}
