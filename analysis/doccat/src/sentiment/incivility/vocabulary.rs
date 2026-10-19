use super::*;

impl IncivilityLexicon {
    /// Builds an empty [`IncivilityLexicon`].
    pub fn empty() -> Self {
        let mut lex = IncivilityLexicon { entries: Vec::new(), first_token: Vec::new(), sha256: String::new() };
        lex.recompute_digest();
        lex
    }

    /// Loads the default English incivility phrase table covering the six
    /// Oasis / HKBU dimensions.
    pub fn bundled() -> Self {
        let mut entries = BTreeMap::new();
        for &(phrase, label, weight) in DEFAULT_INCIVILITY_PHRASES {
            let toks: Vec<String> = phrase.split_whitespace().map(|s| s.to_ascii_lowercase()).collect();
            if !toks.is_empty() {
                entries.insert((toks, label), weight.clamp(0.0, 1.0));
            }
        }
        let entries = entries.into_iter().map(|((toks, label), weight)| (toks, label, weight)).collect();
        let mut lex = IncivilityLexicon { entries, first_token: Vec::new(), sha256: String::new() };
        lex.rebuild_index();
        lex.recompute_digest();
        lex
    }

    /// Inserts or updates a phrase rule (`phrase` split on ASCII whitespace and
    /// lowercased) for `label` with `weight` clamped to `0.0..=1.0`, and
    /// recomputes the canonical SHA-256 digest. Empty phrases and non-finite
    /// weights leave the lexicon unchanged.
    pub fn insert(&mut self, phrase: &str, label: IncivilityLabel, weight: f64) {
        if !weight.is_finite() {
            return;
        }
        let toks: Vec<String> = phrase
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'').to_ascii_lowercase())
            .filter(|w| !w.is_empty())
            .collect();
        if !toks.is_empty() {
            let at = self
                .entries
                .binary_search_by(|(t, l, _)| (t.as_slice(), l).cmp(&(toks.as_slice(), &label)))
                .unwrap_or_else(|i| i);
            if at < self.entries.len() && self.entries[at].0 == toks && self.entries[at].1 == label {
                self.entries[at].2 = weight.clamp(0.0, 1.0);
            } else {
                self.entries.insert(at, (toks, label, weight.clamp(0.0, 1.0)));
            }
            self.rebuild_index();
            self.recompute_digest();
        }
    }

    /// Number of `(phrase, label)` entries in the lexicon.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the lexicon has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Canonical SHA-256 hex digest of the lexicon's entries.
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    /// Groups the entries by their first token; entries share their canonical
    /// order, so one token's patterns are always a contiguous run.
    fn rebuild_index(&mut self) {
        self.first_token.clear();
        for (i, (toks, _, _)) in self.entries.iter().enumerate() {
            if let Some(last) = self.first_token.last_mut()
                && last.0 == toks[0]
            {
                last.2 = i as u32 + 1;
            } else {
                self.first_token.push((toks[0].clone(), i as u32, i as u32 + 1));
            }
        }
    }

    fn recompute_digest(&mut self) {
        let mut canonical = String::new();
        for (toks, label, weight) in &self.entries {
            canonical.push_str(&toks.join(" "));
            canonical.push('\t');
            canonical.push_str(label.name());
            canonical.push('\t');
            canonical.push_str(&format!("{:016x}\n", weight.to_bits()));
        }
        self.sha256 = sha256(canonical.as_bytes()).to_hex();
    }
}
