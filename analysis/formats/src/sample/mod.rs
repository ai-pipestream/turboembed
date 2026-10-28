//! The typed training and evaluation samples.
//!
//! Each type holds the parts of one sample: a token, tag or lemma is a
//! [`Field`](crate::Field) of the source text, and a sample made of text
//! of its own (a token or sentence sample) keeps the source span of each
//! token or sentence beside its span in that text.
//!
//! The samples borrow the source text, so the text outlives them; a
//! reader holds the text and gives out samples that point into it.

pub mod chunk;
pub mod dependency;
pub mod document;
pub mod lemma;
pub mod name;
pub mod pos;
pub mod sentence;
pub mod token;

pub use chunk::{ChunkSample, Phrase};
pub use dependency::{DependencySample, DependencyTree, Head, TreeError};
pub use document::{DocumentSample, LanguageSample};
pub use lemma::LemmaSample;
pub use name::{DEFAULT_NAME_TYPE, Name, NameSample};
pub use pos::PosSample;
pub use sentence::SentenceSample;
pub use token::{SPLIT_MARKER, SplitMarker, TokenSample};

/// The name format's line for a sample, for reader tests: an empty line
/// first when the sample clears adaptive data, then the tokens separated
/// by spaces with `<START:type>` and `<END>` around the names.
#[cfg(test)]
pub(crate) fn name_sample_line(sample: &NameSample<'_>) -> String {
    let mut out = String::new();
    if sample.clear_adaptive_data() {
        out.push('\n');
    }
    let tokens = sample.tokens();
    for (i, token) in tokens.iter().enumerate() {
        let i = i as u32;
        for name in sample.names() {
            if name.start == i {
                match name.kind {
                    None => out.push_str("<START> "),
                    Some(k) => out.push_str(&format!("<START:{k}> ")),
                }
            }
            if name.end == i {
                out.push_str("<END> ");
            }
        }
        out.push_str(token.as_str());
        out.push(' ');
    }
    if tokens.len() > 1 {
        out.pop();
    }
    for name in sample.names() {
        if name.end as usize == tokens.len() {
            out.push_str(" <END>");
        }
    }
    out
}
