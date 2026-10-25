use std::sync::Arc;
use turbo_analysis::TagSet;
use turbo_sequence::codec::name_type;

pub(crate) fn from_labels<'a>(name: &str, labels: impl Iterator<Item = &'a str>) -> Arc<TagSet> {
    let mut kinds: Vec<_> = labels.filter(|s| *s != "other").map(|s| name_type(s).unwrap_or("default")).collect();
    kinds.sort_unstable();
    kinds.dedup();
    TagSet::new(name, &kinds)
}
