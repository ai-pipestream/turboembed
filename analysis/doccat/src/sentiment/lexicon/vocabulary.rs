use super::*;

/// Sentiment lexicon mapping lowercase token strings to prior valence scores.
#[derive(Clone, Debug)]
pub struct VaderLexicon {
    entries: HashMap<String, f64>,
    sha256_hex: String,
}

static BUNDLED_LEXICON: OnceLock<Arc<VaderLexicon>> = OnceLock::new();

impl VaderLexicon {
    /// Returns the bundled MIT-licensed VADER lexicon (`vader_lexicon.txt`).
    pub fn bundled() -> Arc<VaderLexicon> {
        BUNDLED_LEXICON
            .get_or_init(|| {
                Arc::new(VaderLexicon::from_tsv(BUNDLED_LEXICON_TSV).expect("bundled vader_lexicon.txt is valid TSV"))
            })
            .clone()
    }

    /// Parses a tab-separated lexicon (`word\tvalence`, ignoring empty lines
    /// and `#` comment lines). Extra columns, such as the standard deviation
    /// and raw ratings in the upstream file, are ignored.
    pub fn from_tsv(tsv: &str) -> Result<VaderLexicon, Error> {
        let invalid = |line_no: usize, detail: String| Error::Annotator {
            annotator: LexiconSentiment::ID,
            detail: format!("lexicon line {}: {detail}", line_no + 1),
        };
        let mut entries = HashMap::new();
        for (line_no, raw_line) in tsv.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut fields = line.split('\t');
            let Some(word) = fields.next() else {
                return Err(invalid(line_no, "missing word column".into()));
            };
            let Some(measure_str) = fields.next() else {
                return Err(invalid(line_no, format!("missing valence column for {word:?}")));
            };
            let measure: f64 = measure_str
                .trim()
                .parse()
                .map_err(|e| invalid(line_no, format!("invalid valence {measure_str:?}: {e}")))?;
            if !measure.is_finite() {
                return Err(invalid(line_no, format!("non-finite valence for {word:?}")));
            }
            entries.insert(word.to_string(), measure);
        }
        let mut lexicon = VaderLexicon { entries, sha256_hex: String::new() };
        lexicon.recompute_digest();
        Ok(lexicon)
    }

    /// Recomputes the digest over the entries in canonical form (sorted
    /// `word\tvalence` lines), so two tables with the same entries share a
    /// digest whatever their comments or column layout.
    fn recompute_digest(&mut self) {
        let mut words: Vec<&String> = self.entries.keys().collect();
        words.sort_unstable();
        let mut canonical = String::new();
        for word in words {
            canonical.push_str(word);
            canonical.push('\t');
            canonical.push_str(&self.entries[word].to_string());
            canonical.push('\n');
        }
        self.sha256_hex = sha256(canonical.as_bytes()).to_hex();
    }

    /// Looks up the prior valence of `word_lower`.
    #[inline]
    pub fn get(&self, word_lower: &str) -> Option<f64> {
        self.entries.get(word_lower).copied()
    }

    /// Returns `true` if `word_lower` is present in the lexicon.
    #[inline]
    pub fn contains(&self, word_lower: &str) -> bool {
        self.entries.contains_key(word_lower)
    }

    /// Inserts or updates a lowercase lexicon entry and recomputes the digest.
    /// Blank words and non-finite valences leave the lexicon unchanged.
    pub fn insert(&mut self, word_lower: impl Into<String>, valence: f64) {
        let word = word_lower.into();
        if word.trim().is_empty() || !valence.is_finite() {
            return;
        }
        self.entries.insert(word, valence);
        self.recompute_digest();
    }

    /// Number of entries in the lexicon.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the lexicon has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// SHA-256 hex digest of the entries in canonical form (sorted
    /// `word\tvalence` lines).
    pub fn sha256_hex(&self) -> &str {
        &self.sha256_hex
    }
}
