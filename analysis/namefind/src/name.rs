//! What a name finder returns, and the trait the finders share.

use std::sync::Arc;

use turbo_analysis_layers::{Tag, TagSet};

use crate::error::Error;

/// A name: tokens `[start, end)` of the tokens a finder was given, and
/// its entity type. OpenNLP's `Span` over token indices.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Name {
    pub start: u32,
    pub end: u32,
    pub kind: Tag,
}

/// `opennlp.tools.namefind.TokenNameFinder` without adaptive data, which
/// neither of these finders keeps.
pub trait TokenNameFinder: Send + Sync {
    /// The entity types its names can have; [`Name::kind`] indexes it.
    fn kinds(&self) -> &Arc<TagSet>;

    /// `find(String[] tokens)`: appends the names in `tokens` to `out`, in
    /// the order the Java finder returns them.
    fn find(&self, tokens: &[&str], out: &mut Vec<Name>) -> Result<(), Error>;
}

/// A tag set from entity type names, in order, refusing a repeat.
pub(crate) fn kinds<'a>(names: impl Iterator<Item = &'a str>) -> Result<Arc<TagSet>, Error> {
    let names: Vec<&str> = names.collect();
    if names.len() > u16::MAX as usize + 1 {
        return Err(Error::TooManyTypes);
    }
    let set = TagSet::new("entity types", &names);
    // A repeated name maps to its last position, so an earlier one does
    // not map back to itself.
    if let Some((_, repeat)) = names.iter().enumerate().find(|&(i, n)| set.tag(n) != Some(Tag(i as u16))) {
        return Err(Error::DuplicateType((*repeat).into()));
    }
    Ok(set)
}
