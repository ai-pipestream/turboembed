//! The ids of the shared layers, in one table, and the reader-side map
//! from the ids graphs carried before the shared layers moved into the
//! `turbo:` namespace.
//!
//! Every shared layer's [`Layer::ID`](crate::Layer::ID) is one of the
//! constants here, so an id is spelled once. Readers of stored data (the
//! document graph decoder, codecs that read layer references out of a
//! body, golden files, step names on a request) pass each layer name
//! through [`canonical_layer_name`], which rewrites a legacy id, alone
//! or nested in a generic name such as `turbo:categories<...>`, to its
//! current id and leaves every other name as it is. Writers only ever
//! write the current ids.

use std::borrow::Cow;

/// Sentence spans.
pub const SENTENCES: &str = "turbo:sentences";
/// Token spans.
pub const TOKENS: &str = "turbo:tokens";
/// Model subword pieces.
pub const SUBWORDS: &str = "turbo:subwords";
/// Named entities.
pub const ENTITIES: &str = "turbo:entities";
/// Part-of-speech tags.
pub const POS: &str = "turbo:pos";
/// Lemmas.
pub const LEMMAS: &str = "turbo:lemmas";
/// Document language.
pub const LANGUAGE: &str = "turbo:language";
/// Vectors aligned to another layer.
pub const EMBEDDINGS: &str = "turbo:embeddings";
/// Dependency trees.
pub const DEPENDENCIES: &str = "turbo:dependencies";
/// Phrase chunks (noun and verb groups). Retrieval chunks are `turbo:chunks`.
pub const PHRASE_CHUNKS: &str = "turbo:phrase-chunks";
/// Per-token chunk tags.
pub const CHUNK_TAGS: &str = "turbo:chunk-tags";
/// Constituency phrases.
pub const PHRASES: &str = "turbo:phrases";
/// Coreference mentions.
pub const MENTIONS: &str = "turbo:mentions";
/// Speaker turns.
pub const SPEAKERS: &str = "turbo:speakers";
/// Typed relations between entities.
pub const RELATIONS: &str = "turbo:relations";

/// The namespace the shared layers used before `turbo:`.
const LEGACY_NAMESPACE: &str = "opennlp:";

/// Each legacy id's name inside [`LEGACY_NAMESPACE`], with the id that
/// replaced it.
const LEGACY: &[(&str, &str)] = &[
    ("sentences", SENTENCES),
    ("tokens", TOKENS),
    ("subwords", SUBWORDS),
    ("entities", ENTITIES),
    ("pos", POS),
    ("lemmas", LEMMAS),
    ("language", LANGUAGE),
    ("embeddings", EMBEDDINGS),
    ("dependencies", DEPENDENCIES),
    ("chunks", PHRASE_CHUNKS),
    ("chunk-tags", CHUNK_TAGS),
    ("phrases", PHRASES),
    ("mentions", MENTIONS),
    ("speakers", SPEAKERS),
    ("relations", RELATIONS),
];

/// Every legacy id with the id that replaced it, for tools that check
/// sources or data for ids that should no longer be written.
pub fn legacy_ids() -> impl Iterator<Item = (String, &'static str)> {
    LEGACY.iter().map(|&(old, new)| (format!("{LEGACY_NAMESPACE}{old}"), new))
}

/// The current id for one bare legacy id, if `id` is one.
fn current(id: &str) -> Option<&'static str> {
    let rest = id.strip_prefix(LEGACY_NAMESPACE)?;
    LEGACY.iter().find(|&&(old, _)| old == rest).map(|&(_, new)| new)
}

/// Whether `c` ends one id inside a full layer name: the generic
/// brackets and separators, or the `@` of a `layer@instance` reference.
fn ends_id(c: char) -> bool {
    matches!(c, '<' | '>' | ',' | '@') || c.is_whitespace()
}

/// `name` with every legacy id in it replaced by the current id. A full
/// generic name is rewritten part by part (`turbo:categories<old>`), and
/// so is a `layer@instance` reference. Names with no legacy id, which is
/// every name this build writes, come back borrowed.
pub fn canonical_layer_name(name: &str) -> Cow<'_, str> {
    if !name.contains(LEGACY_NAMESPACE) {
        return Cow::Borrowed(name);
    }
    let mut out = String::with_capacity(name.len());
    let mut rest = name;
    while let Some(at) = rest.find(LEGACY_NAMESPACE) {
        // Only an id that starts a part: after the start, a bracket or a separator.
        let starts_part = out.is_empty() && at == 0
            || rest[..at].chars().next_back().or_else(|| out.chars().next_back()).is_some_and(ends_id);
        out.push_str(&rest[..at]);
        let tail = &rest[at..];
        let end =
            tail.char_indices().skip(LEGACY_NAMESPACE.len()).find(|&(_, c)| ends_id(c)).map_or(tail.len(), |(i, _)| i);
        let id = &tail[..end];
        match current(id).filter(|_| starts_part) {
            Some(new) => out.push_str(new),
            None => out.push_str(id),
        }
        rest = &tail[end..];
    }
    out.push_str(rest);
    Cow::Owned(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_names_pass_through_borrowed() {
        for name in [TOKENS, "turbo:categories<turbo:sentences>", "acme:invoice", "", "turbo:chunks"] {
            assert!(matches!(canonical_layer_name(name), Cow::Borrowed(n) if n == name));
        }
    }

    #[test]
    fn legacy_ids_map_alone_nested_and_with_an_instance() {
        let old = |s: &str| format!("{LEGACY_NAMESPACE}{s}");
        assert_eq!(canonical_layer_name(&old("tokens")), TOKENS);
        assert_eq!(canonical_layer_name(&old("chunks")), PHRASE_CHUNKS);
        assert_eq!(canonical_layer_name(&old("chunk-tags")), CHUNK_TAGS);
        let nested = format!("turbo:edges<{}, {}>", old("entities"), old("tokens"));
        assert_eq!(canonical_layer_name(&nested), "turbo:edges<turbo:entities, turbo:tokens>");
        let deeper = format!("turbo:groups<turbo:edges<{}, {}>>", old("mentions"), old("mentions"));
        assert_eq!(canonical_layer_name(&deeper), "turbo:groups<turbo:edges<turbo:mentions, turbo:mentions>>");
        let generic = format!("{}<turbo:words>", old("dependencies"));
        assert_eq!(canonical_layer_name(&generic), "turbo:dependencies<turbo:words>");
        assert_eq!(canonical_layer_name(&format!("{}@gold", old("pos"))), "turbo:pos@gold");
    }

    #[test]
    fn unknown_and_embedded_legacy_text_is_left_alone() {
        let old = |s: &str| format!("{LEGACY_NAMESPACE}{s}");
        // Not a shared layer id.
        assert_eq!(canonical_layer_name(&old("doccat")), old("doccat"));
        assert_eq!(canonical_layer_name(&old("tokens-extra")), old("tokens-extra"));
        // Inside another id rather than starting a part.
        let inside = format!("acme{}", old("tokens"));
        assert_eq!(canonical_layer_name(&inside), inside);
        // Multibyte text around an id stays byte for byte.
        let wide = format!("é<{}>ü", old("tokens"));
        assert_eq!(canonical_layer_name(&wide), "é<turbo:tokens>ü");
    }

    #[test]
    fn every_legacy_id_maps_to_a_distinct_current_id() {
        let all: Vec<_> = legacy_ids().collect();
        for (i, (old, new)) in all.iter().enumerate() {
            assert_eq!(canonical_layer_name(old), *new);
            assert!(new.starts_with("turbo:"));
            assert!(all[i + 1..].iter().all(|(_, other)| other != new), "{new} twice");
        }
    }
}
