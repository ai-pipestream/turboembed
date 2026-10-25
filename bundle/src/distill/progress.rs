//! What a distillation reports while it runs, for the command line or a
//! caller that streams it on.

/// One step of a distillation, in the order they happen.
#[derive(Debug, Clone, PartialEq)]
pub enum Progress {
    /// Which entries are embedded: `kept` of the base's `vocab_size`, and
    /// `added` learned terms, with `refused` terms left out (`Staged`
    /// says why).
    Kept { kept: usize, vocab_size: usize, added: usize, refused: usize },
    /// Rows through the base encoder so far, of `total`.
    Embedded { done: usize, total: usize },
    /// The rows reduced to `pca_dims`, keeping this share of the variance.
    Reduced { pca_dims: usize, explained_variance: f64 },
    /// The quality texts' nearest-neighbour agreement with their groups,
    /// base and static, and the rank correlation of the two models'
    /// similarities.
    Measured { base_top1: f64, static_top1: f64, similarity_spearman: f64 },
}

/// The command line's report: each step on its own line.
pub fn print(p: Progress) {
    match p {
        Progress::Kept { kept, vocab_size, added, refused } => {
            println!("distill: {kept} of {vocab_size} entries kept, {added} learned terms added, {refused} refused")
        }
        Progress::Embedded { done, total } => println!("distill: {done} of {total} entries embedded"),
        Progress::Reduced { pca_dims, explained_variance } => {
            println!("distill: reduced to {pca_dims} dims, explained variance {explained_variance:.4}")
        }
        Progress::Measured { base_top1, static_top1, similarity_spearman } => println!(
            "distill: quality base_top1 {base_top1}, static_top1 {static_top1}, similarity_spearman {similarity_spearman}"
        ),
    }
}
