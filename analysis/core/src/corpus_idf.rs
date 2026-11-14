//! Compact collection document-frequency (`DF`) table and smooth BM25
//! inverse-document-frequency (`IDF`) weights (`#2346`).
//!
//! Downstream search shards periodically export `(term, df, doc_count)`
//! snapshots to `TurboVocabularyService`. `CorpusIdf` stores those counts in a
//! deterministic sorted table (`TCIDF001` binary format) so `turbo-chunker`
//! (`IdfLexicalBoundaries`) and `turbo-keywords` (`YakeExtractor`,
//! `TextRankExtractor`) weight terms by true collection rarity rather than
//! single-document frequency alone.

use std::collections::BTreeMap;

use crate::hash::Fnv1a;

/// Binary magic header for serialized [`CorpusIdf`] snapshots.
pub const CORPUS_IDF_MAGIC: &[u8; 8] = b"TCIDF001";

/// Immutable collection document-frequency snapshot and smooth BM25 IDF lookup.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CorpusIdf {
    doc_count: u64,
    terms: Vec<(Box<str>, u64)>,
    hash: u64,
}

impl CorpusIdf {
    /// Builds a [`CorpusIdf`] from `doc_count` and `(term, df)` pairs.
    ///
    /// Duplicate terms sum their document frequencies (clamped to `doc_count`),
    /// and zero-DF or empty terms are omitted.
    pub fn from_entries<S: AsRef<str>>(
        doc_count: u64,
        entries: impl IntoIterator<Item = (S, u64)>,
    ) -> Result<Self, String> {
        if doc_count == 0 {
            return Err("CorpusIdf doc_count must be at least 1".into());
        }
        let mut merged: BTreeMap<String, u64> = BTreeMap::new();
        for (term, df) in entries {
            let t = term.as_ref();
            if t.is_empty() || df == 0 {
                continue;
            }
            if t.len() > u16::MAX as usize {
                return Err(format!("term length {} exceeds u16::MAX", t.len()));
            }
            let slot = merged.entry(t.to_owned()).or_insert(0);
            *slot = slot.saturating_add(df).min(doc_count);
        }
        let terms: Vec<(Box<str>, u64)> = merged.into_iter().map(|(k, v)| (k.into_boxed_str(), v)).collect();
        let mut h = Fnv1a::new();
        h.u64(doc_count).u64(terms.len() as u64);
        for (t, df) in &terms {
            h.bytes(t.as_bytes()).u8(0).u64(*df);
        }
        Ok(Self { doc_count, terms, hash: h.finish() })
    }

    /// Total documents in the collection snapshot (`N`).
    pub fn doc_count(&self) -> u64 {
        self.doc_count
    }

    /// Number of distinct non-zero DF terms recorded in the snapshot.
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    /// True when the snapshot holds zero explicit term entries.
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// Deterministic configuration hash over `doc_count` and sorted `(term, df)` entries.
    pub fn config_hash(&self) -> u64 {
        self.hash
    }

    /// Sorted `(term, df)` entries.
    pub fn entries(&self) -> &[(Box<str>, u64)] {
        &self.terms
    }

    /// Looks up the recorded document frequency of `term` (`0` when unseen).
    pub fn df(&self, term: &str) -> u64 {
        match self.terms.binary_search_by(|(probe, _)| probe.as_ref().cmp(term)) {
            Ok(ix) => self.terms[ix].1,
            Err(_) => 0,
        }
    }

    /// Smooth BM25 / Robertson-Sparck Jones IDF:
    /// `ln(1.0 + (N - df + 0.5) / (df + 0.5))`, strictly positive (`> 0.0`) for
    /// all `0 <= df <= N`.
    pub fn idf(&self, term: &str) -> f64 {
        let n = self.doc_count as f64;
        let df = self.df(term).min(self.doc_count) as f64;
        (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
    }

    /// Minimum possible IDF (for a ubiquitous term appearing in all `N` docs):
    /// `ln(1.0 + 0.5 / (N + 0.5))`.
    pub fn min_idf(&self) -> f64 {
        let n = self.doc_count as f64;
        (1.0 + 0.5 / (n + 0.5)).ln()
    }

    /// Relative collection rarity factor `>= 1.0` (`1.0 + idf(term)`), suitable
    /// for scaling positive salience weights or dividing lower-is-better YAKE
    /// term scores without collapsing ubiquitous terms to zero.
    pub fn rarity_boost(&self, term: &str) -> f64 {
        1.0 + self.idf(term)
    }

    /// Encodes this snapshot into deterministic `TCIDF001` binary bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(24 + self.terms.len() * 16);
        out.extend_from_slice(CORPUS_IDF_MAGIC);
        out.extend_from_slice(&self.doc_count.to_le_bytes());
        out.extend_from_slice(&(self.terms.len() as u64).to_le_bytes());
        for (term, df) in &self.terms {
            out.extend_from_slice(&(term.len() as u16).to_le_bytes());
            out.extend_from_slice(term.as_bytes());
            out.extend_from_slice(&df.to_le_bytes());
        }
        out
    }

    /// Decodes a `TCIDF001` binary snapshot, verifying sorted uniqueness and bounds.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 24 {
            return Err("TCIDF001 buffer truncated before 24-byte header".into());
        }
        if &bytes[..8] != CORPUS_IDF_MAGIC {
            return Err("invalid TCIDF001 magic header".into());
        }
        let doc_count = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        if doc_count == 0 {
            return Err("TCIDF001 doc_count must be at least 1".into());
        }
        let count = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
        let mut at = 24usize;
        let mut terms = Vec::with_capacity(count.min(65536));
        let mut prev: Option<&str> = None;
        for _ in 0..count {
            let end_len = at.checked_add(2).ok_or_else(|| "TCIDF001 offset overflow".to_string())?;
            if end_len > bytes.len() {
                return Err("TCIDF001 truncated reading term length".into());
            }
            let tlen = u16::from_le_bytes(bytes[at..end_len].try_into().unwrap()) as usize;
            if tlen == 0 {
                return Err("TCIDF001 contains empty term".into());
            }
            at = end_len;
            let end_str = at.checked_add(tlen).ok_or_else(|| "TCIDF001 term overflow".to_string())?;
            let end_df = end_str.checked_add(8).ok_or_else(|| "TCIDF001 df overflow".to_string())?;
            if end_df > bytes.len() {
                return Err("TCIDF001 truncated reading term entry".into());
            }
            let s = std::str::from_utf8(&bytes[at..end_str]).map_err(|_| "TCIDF001 term is not valid UTF-8")?;
            if let Some(p) = prev
                && s <= p
            {
                return Err("TCIDF001 terms must be strictly sorted".into());
            }
            let df = u64::from_le_bytes(bytes[end_str..end_df].try_into().unwrap());
            if df == 0 || df > doc_count {
                return Err(format!("TCIDF001 df {df} out of 1..={doc_count}"));
            }
            terms.push((s.to_owned().into_boxed_str(), df));
            prev = Some(terms.last().unwrap().0.as_ref());
            at = end_df;
        }
        if at != bytes.len() {
            return Err("TCIDF001 has trailing bytes".into());
        }
        let mut h = Fnv1a::new();
        h.u64(doc_count).u64(terms.len() as u64);
        for (t, df) in &terms {
            h.bytes(t.as_bytes()).u8(0).u64(*df);
        }
        Ok(Self { doc_count, terms, hash: h.finish() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_binary_codec_and_orders_idf_monotonically() {
        let table = CorpusIdf::from_entries(1000, [("the", 980), ("search", 120), ("zirconia", 2)]).unwrap();
        let encoded = table.to_bytes();
        let decoded = CorpusIdf::from_bytes(&encoded).unwrap();
        assert_eq!(table, decoded);
        assert!(decoded.idf("unseen") > decoded.idf("zirconia"));
        assert!(decoded.idf("zirconia") > decoded.idf("search"));
        assert!(decoded.idf("search") > decoded.idf("the"));
        assert!(decoded.idf("the") > 0.0);
    }
}
