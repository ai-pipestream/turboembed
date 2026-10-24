//! A transformer name finder on turboembed's encoder backends, as OpenNLP's
//! `NameFinderDL` finds names with a token-classification model.

pub mod decode;
pub mod finder;
pub mod place;

pub use finder::{NameFinderDl, NameFinderDlOptions};
