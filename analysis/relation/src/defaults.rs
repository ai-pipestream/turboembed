//! The default English pattern libraries. Labels are Universal
//! Dependencies v2, because that is what the shipped dependency
//! parsers are trained on; a model trained on another scheme
//! needs its own set. The library is named for the language
//! rather than the scheme because the label set is the parser's
//! property, not the library's.

use crate::pattern::RelationPattern;

fn shape(relation_type: &str, path: &str) -> RelationPattern {
    RelationPattern::new(relation_type, path).expect("a shipped default pattern is valid")
}

/// The six shape rules of the reference design: no triggers, so
/// they match on dependency shape alone and a user narrows them
/// with triggers (which is what the discovery report is for).
/// One honest limitation: `prepositional-attachment` cannot
/// distinguish `of` from `in`, because a pattern's trigger
/// constrains the pivot's form only and the preposition is a
/// `case` dependent of the object, not the pivot.
pub fn english_defaults() -> Vec<RelationPattern> {
    vec![
        shape("subject-object", "<nsubj >obj"),
        shape("passive-agent", "<nsubj:pass >obl"),
        shape("apposition", ">appos"),
        shape("possession", "<nmod:poss"),
        shape("prepositional-attachment", ">nmod"),
        shape("attribute", "<nsubj"),
    ]
}

fn semantic(
    relation_type: &str,
    path: &str,
    pivot_lemma: &str,
    subject_kind: &str,
    object_kind: &str,
) -> RelationPattern {
    RelationPattern::new(relation_type, path)
        .expect("a shipped default pattern is valid")
        .pivot_lemma(pivot_lemma)
        .expect("a shipped default lemma is valid")
        .subject_kind(subject_kind)
        .expect("a shipped default kind is valid")
        .object_kind(object_kind)
        .expect("a shipped default kind is valid")
}

/// The semantic starting set: common English relations as a
/// dependency shape, the pivot verb's lemma, and the entity kinds
/// on both ends (the `person`, `organization` and `location`
/// kind labels the entity layers in this workspace use). These
/// are starting points to narrow against a corpus with the
/// discovery report, not a trained extractor: the verb choices
/// are the canonical head lemmas, so `work for` via `>obl` with
/// pivot `work`, passive acquisitions via `nsubj:pass` plus the
/// agent `obl`, and so on.
pub fn english_semantic_defaults() -> Vec<RelationPattern> {
    vec![
        semantic("works_for", "<nsubj >obl", "work", "person", "organization"),
        semantic("located_in", "<nsubj >obl", "locate", "organization", "location"),
        semantic("founded_by", "<nsubj:pass >obl", "found", "organization", "person"),
        semantic("headquartered_in", "<nsubj:pass >obl", "headquarter", "organization", "location"),
        semantic("member_of", "<nsubj >obl", "join", "person", "organization"),
        semantic("acquired_by", "<nsubj:pass >obl", "acquire", "organization", "organization"),
        semantic("born_in", "<nsubj:pass >obl", "bear", "person", "location"),
        semantic("spouse_of", "<nsubj >obl", "marry", "person", "person"),
    ]
}
