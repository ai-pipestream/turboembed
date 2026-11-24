//! What the matcher tests share.

use turbo_glossary_layers::Glossary;

/// A glossary of `(id, term)` entries.
pub(crate) fn glossary(entries: &[(&str, &str)]) -> Glossary {
    let mut g = Glossary::new();
    for (id, term) in entries {
        g.push(id, term).unwrap();
    }
    g
}
