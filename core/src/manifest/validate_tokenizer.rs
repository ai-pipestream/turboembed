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
        for (i, s) in t.special_tokens.iter().enumerate() {
            if !roles.insert(s.role) {
                return Err(invalid(format!("manifest.json: tokenizer.special_tokens[{i}].role: listed twice")));
            }
        }
        // Special tokens and added tokens are matched in the text alike.
        let specials =
            t.special_tokens.iter().enumerate().map(|(i, s)| {
                (format!("special_tokens[{i}]"), &s.content, s.id, s.normalized, s.single_word || s.rstrip)
            });
        let added =
            t.added_tokens.iter().enumerate().map(|(i, a)| {
                (format!("added_tokens[{i}]"), &a.content, a.id, a.normalized, a.single_word || a.rstrip)
            });
        if t.sentencepiece.is_some() && !t.added_tokens.is_empty() {
            return Err(invalid(
                "manifest.json: tokenizer.added_tokens: a sentencepiece tokenizer matches no added token in the text",
            ));
        }
        let mut contents = HashSet::new();
        let mut ids = HashSet::new();
        for (at, content, id, normalized, word_or_rstrip) in specials.chain(added) {
            required(&format!("tokenizer.{at}.content"), content)?;
            if !contents.insert(content.as_str()) {
                return Err(invalid(format!("manifest.json: tokenizer.{at}.content: listed twice")));
            }
            if !ids.insert(id) {
                return Err(invalid(format!("manifest.json: tokenizer.{at}.id: {id} is another token's")));
            }
            if t.unigram.is_some() && normalized && word_or_rstrip {
                return Err(invalid(format!(
                    "manifest.json: tokenizer.{at}: unigram single_word/rstrip are unsupported on a normalized token"
                )));
            }
            if normalized && t.sentencepiece.is_some() {
                return Err(invalid(format!(
                    "manifest.json: tokenizer.{at}.normalized: requires wordpiece, bpe, or unigram"
                )));
            }
            if word_or_rstrip && t.sentencepiece.is_some() {
                return Err(invalid(format!(
                    "manifest.json: tokenizer.{at}: a sentencepiece tokenizer matches no token in the text"
                )));
            }
            if id > i32::MAX as u32 {
                return Err(invalid(format!("manifest.json: tokenizer.{at}.id: over INT32_MAX")));
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
            if s != "$TEXT" && !t.special_tokens.iter().any(|x| x.content == *s) {
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
