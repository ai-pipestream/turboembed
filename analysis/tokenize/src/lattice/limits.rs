//! The bounds a dictionary is read within.

/// Bounds on what a dictionary may declare, so a crafted file cannot force
/// an outsized allocation. The defaults are OpenNLP's
/// `ResourceLimits.MAX_ENTRIES` and `ResourceLimits.MAX_MATRIX_CELLS`, which
/// the JVM reads from system properties; here they are a parameter.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DictionaryLimits {
    /// The most entries the lexicon files may list together, and the most
    /// `unk.def` may list; also the largest matrix dimension.
    pub max_entries: usize,
    /// The most cells the connection matrix may have.
    pub max_matrix_cells: usize,
}

impl Default for DictionaryLimits {
    fn default() -> DictionaryLimits {
        DictionaryLimits { max_entries: 10_000_000, max_matrix_cells: 134_217_728 }
    }
}
