//! The stemmer interface: OpenNLP's `opennlp.tools.stemmer.Stemmer`.

/// Turns a word into its stem.
///
/// The Java stemmers work on UTF-16 `char`s, and a length test or a step
/// of one character counts code units, so the one required method takes
/// and gives UTF-16 code units. [`Stemmer::stem`] is the convenience over
/// `&str`.
///
/// Every stemmer here holds no state between words, so one value can be
/// shared by any number of threads. OpenNLP's `SharingStemmer` and
/// `DelegatingStemmer`, which give each thread its own Java stemmer, have
/// nothing to do here and are not ported.
pub trait Stemmer: Send + Sync {
    /// Writes the stem of `word` to `out`, replacing what it held. Reusing
    /// `out` across words saves an allocation per word.
    fn stem_utf16(&self, word: &[u16], out: &mut Vec<u16>);

    /// The stem of `word`.
    ///
    /// A lone surrogate in the stem, which needs a stemmer to split a
    /// surrogate pair of `word` (none was seen to), would become U+FFFD.
    fn stem(&self, word: &str) -> String {
        let units: Vec<u16> = word.encode_utf16().collect();
        let mut out = Vec::with_capacity(units.len());
        self.stem_utf16(&units, &mut out);
        String::from_utf16_lossy(&out)
    }

    /// A deterministic configuration hash identifying this stemmer, or `0`
    /// for custom external implementations.
    fn config_hash(&self) -> u64 {
        0
    }
}
