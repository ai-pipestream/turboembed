//! Resolves a [`RawLexicon`]'s id references and builds the [`InMemoryWordNetLexicon`].

use std::collections::{BTreeMap, HashMap, HashSet};

use super::raw::{RawEntry, RawLexicon, RawSense, RawSynset};
use super::{OTHER_RELATION, malformed};
use crate::error::WordNetError;
use crate::folding::{fold, split_on_spaces};
use crate::lexicon::{InMemoryWordNetLexicon, LemmaKey, Synset};
use crate::pos::WordNetPos;
use crate::relation::WordNetRelation;

/// Exact 59 synset-only relation names in WN-LMF 1.4.
fn is_synset_only_rel_type(rel_type: &str) -> bool {
    matches!(
        rel_type,
        "attribute"
            | "be_in_state"
            | "causes"
            | "classified_by"
            | "classifies"
            | "co_agent_instrument"
            | "co_agent_patient"
            | "co_agent_result"
            | "co_instrument_agent"
            | "co_instrument_patient"
            | "co_instrument_result"
            | "co_patient_agent"
            | "co_patient_instrument"
            | "co_result_agent"
            | "co_result_instrument"
            | "co_role"
            | "direction"
            | "entails"
            | "eq_synonym"
            | "holo_location"
            | "holo_member"
            | "holo_part"
            | "holo_portion"
            | "holo_substance"
            | "holonym"
            | "hypernym"
            | "hyponym"
            | "in_manner"
            | "instance_hypernym"
            | "instance_hyponym"
            | "involved"
            | "involved_agent"
            | "involved_direction"
            | "involved_instrument"
            | "involved_location"
            | "involved_patient"
            | "involved_result"
            | "involved_source_direction"
            | "involved_target_direction"
            | "ir_synonym"
            | "is_caused_by"
            | "is_entailed_by"
            | "is_subevent_of"
            | "manner_of"
            | "mero_location"
            | "mero_member"
            | "mero_part"
            | "mero_portion"
            | "mero_substance"
            | "meronym"
            | "patient"
            | "restricted_by"
            | "restricts"
            | "role"
            | "source_direction"
            | "state_of"
            | "subevent"
            | "target_direction"
    )
}

/// Exact 21 sense-only relation names in WN-LMF 1.4.
fn is_sense_only_rel_type(rel_type: &str) -> bool {
    matches!(
        rel_type,
        "body_part"
            | "by_means_of"
            | "derivation"
            | "destination"
            | "event"
            | "has_metaphor"
            | "has_metonym"
            | "material"
            | "metaphor"
            | "metonym"
            | "participle"
            | "pertainym"
            | "property"
            | "secondary_aspect_ip"
            | "secondary_aspect_pi"
            | "simple_aspect_ip"
            | "simple_aspect_pi"
            | "state"
            | "undergoer"
            | "uses"
            | "vehicle"
    )
}

pub(super) struct Materializer<'a> {
    lexicon: &'a RawLexicon,
    resource_name: &'a str,
}

impl<'a> Materializer<'a> {
    pub(super) fn new(lexicon: &'a RawLexicon, resource_name: &'a str) -> Self {
        Self { lexicon, resource_name }
    }

    fn fail(&self, line: Option<usize>, message: impl AsRef<str>) -> WordNetError {
        malformed(self.resource_name, line, message)
    }

    pub(super) fn materialize(&self) -> Result<InMemoryWordNetLexicon, WordNetError> {
        let senses_by_synset = self.senses_by_synset()?;
        let mut sense_order: HashMap<LemmaKey, Vec<String>> = HashMap::new();
        for sense_id in &self.lexicon.sense_order {
            let sense = &self.lexicon.senses[sense_id];
            let entry = self.entry(sense)?;
            let key = LemmaKey::of(&entry.lemma, entry.pos);
            let order = sense_order.entry(key).or_default();
            if !order.contains(&sense.synset_id) {
                order.push(sense.synset_id.clone());
            }
        }

        let mut synsets_by_id: HashMap<String, Synset> = HashMap::with_capacity(self.lexicon.synsets.len());
        for synset_id in &self.lexicon.synset_order {
            let raw = &self.lexicon.synsets[synset_id];
            let senses = senses_by_synset.get(raw.id.as_str()).map(Vec::as_slice).unwrap_or(&[]);
            let relations = self.resolve_relations(raw, senses)?;
            let lemmas = self.member_lemmas(raw, senses)?;
            synsets_by_id
                .insert(raw.id.clone(), Synset::new(raw.id.clone(), raw.pos, lemmas, raw.gloss.clone(), relations)?);
        }
        InMemoryWordNetLexicon::new(synsets_by_id, sense_order)
    }

    pub(super) fn validate(&self) -> Result<(), WordNetError> {
        let senses_by_synset = self.senses_by_synset()?;
        for synset_id in &self.lexicon.synset_order {
            let raw = &self.lexicon.synsets[synset_id];
            let senses = senses_by_synset.get(raw.id.as_str()).map(Vec::as_slice).unwrap_or(&[]);
            self.resolve_relations(raw, senses)?;
            self.member_lemmas(raw, senses)?;
        }
        Ok(())
    }

    fn senses_by_synset(&self) -> Result<HashMap<&str, Vec<&RawSense>>, WordNetError> {
        let mut out: HashMap<&str, Vec<&RawSense>> = HashMap::new();
        for sense_id in &self.lexicon.sense_order {
            let sense = &self.lexicon.senses[sense_id];
            let Some(synset) = self.lexicon.synsets.get(&sense.synset_id) else {
                return Err(self.fail(
                    Some(sense.line),
                    format!(
                        "Sense {} at line {} references undeclared synset {}",
                        sense.id, sense.line, sense.synset_id
                    ),
                ));
            };
            let entry = self.entry(sense)?;
            if entry.pos != synset.pos {
                return Err(self.fail(
                    Some(sense.line),
                    format!(
                        "Sense {} at line {} has part of speech {:?} ({}) but synset {} has {:?} ({})",
                        sense.id,
                        sense.line,
                        entry.pos,
                        pos_upper_name(entry.pos),
                        synset.id,
                        synset.pos,
                        pos_upper_name(synset.pos)
                    ),
                ));
            }
            out.entry(synset.id.as_str()).or_default().push(sense);
        }
        Ok(out)
    }

    fn entry(&self, sense: &RawSense) -> Result<&RawEntry, WordNetError> {
        self.lexicon.entries.get(&sense.entry_id).ok_or_else(|| {
            self.fail(
                Some(sense.line),
                format!("Sense {} belongs to unresolved lexical entry {}", sense.id, sense.entry_id),
            )
        })
    }

    fn resolve_relations(
        &self,
        raw: &RawSynset,
        senses: &[&RawSense],
    ) -> Result<BTreeMap<WordNetRelation, Vec<String>>, WordNetError> {
        let mut combined: Vec<(&str, &str, usize, bool)> = Vec::with_capacity(raw.relations.len());
        for rel in &raw.relations {
            combined.push((&rel.rel_type, &rel.target, rel.line, false));
        }
        for sense in senses {
            for rel in &sense.relations {
                if rel.rel_type == OTHER_RELATION {
                    continue;
                }
                if rel.target == sense.id {
                    return Err(self.fail(
                        Some(rel.line),
                        format!(
                            "SenseRelation {} at line {} from sense {} targets its own sense",
                            rel.rel_type, rel.line, sense.id
                        ),
                    ));
                }
                let Some(target_sense) = self.lexicon.senses.get(&rel.target) else {
                    return Err(self.fail(
                        Some(rel.line),
                        format!(
                            "SenseRelation at line {} from sense {} references undeclared sense {}",
                            rel.line, sense.id, rel.target
                        ),
                    ));
                };
                combined.push((&rel.rel_type, &target_sense.synset_id, rel.line, true));
            }
        }

        let mut typed: BTreeMap<WordNetRelation, Vec<String>> = BTreeMap::new();
        for (rel_type, target_id, line, sense_relation) in combined {
            let rel_enum = self.parse_relation(rel_type, raw.pos, line, sense_relation)?;
            // A sense relation between two senses of one synset projects onto that synset and is
            // kept; a synset relation naming its own synset is malformed.
            if !sense_relation && target_id == raw.id {
                return Err(self.fail(
                    Some(line),
                    format!("Relation {rel_type} at line {line} on synset {} targets its own synset", raw.id),
                ));
            }
            let Some(target) = self.lexicon.synsets.get(target_id) else {
                return Err(self.fail(
                    Some(line),
                    format!(
                        "Relation {rel_type} at line {line} on synset {} references undeclared synset {target_id}",
                        raw.id
                    ),
                ));
            };
            let list = typed.entry(rel_enum).or_default();
            if !list.contains(&target.id) {
                list.push(target.id.clone());
            }
        }
        Ok(typed)
    }

    fn member_lemmas(&self, raw: &RawSynset, senses: &[&RawSense]) -> Result<Vec<String>, WordNetError> {
        // An absent, empty, or blank `members` attribute defaults to the synset's senses.
        let mut member_ids: Vec<&str> = raw.members.as_deref().map(split_on_spaces).unwrap_or_default();
        if member_ids.is_empty() {
            member_ids.extend(senses.iter().map(|sense| sense.entry_id.as_str()));
        }
        for extra in &raw.extra_members {
            member_ids.push(extra);
        }

        let mut lemmas: Vec<String> = Vec::with_capacity(member_ids.len());
        let mut folded_lemmas: HashSet<String> = HashSet::with_capacity(member_ids.len());
        for member_id in member_ids {
            let sense = self.lexicon.senses.get(member_id);
            if let Some(s) = sense
                && s.synset_id != raw.id
            {
                return Err(self.fail(
                    Some(raw.line),
                    format!(
                        "Synset {} at line {} lists member sense {} assigned to synset {}",
                        raw.id, raw.line, member_id, s.synset_id
                    ),
                ));
            }
            let entry_id = sense.map_or(member_id, |s| s.entry_id.as_str());
            let Some(entry) = self.lexicon.entries.get(entry_id) else {
                return Err(self.fail(
                    Some(raw.line),
                    format!(
                        "Synset {} at line {} lists undeclared member sense or entry {}",
                        raw.id, raw.line, member_id
                    ),
                ));
            };
            if sense.is_none() && !senses.iter().any(|s| s.entry_id == entry.id) {
                return Err(self.fail(
                    Some(raw.line),
                    format!(
                        "Synset {} at line {} lists member entry {} with no sense assigned to {}",
                        raw.id, raw.line, member_id, raw.id
                    ),
                ));
            }
            if raw.pos != entry.pos {
                return Err(self.fail(
                    Some(raw.line),
                    format!(
                        "Synset {} at line {} has part of speech {} but member entry {} has {}",
                        raw.id,
                        raw.line,
                        pos_upper_name(raw.pos),
                        entry.id,
                        pos_upper_name(entry.pos)
                    ),
                ));
            }
            if !lemmas.contains(&entry.lemma) {
                lemmas.push(entry.lemma.clone());
                folded_lemmas.insert(fold(&entry.lemma));
            }
        }

        for sense in senses {
            let entry = self.entry(sense)?;
            if !folded_lemmas.contains(&fold(&entry.lemma)) {
                return Err(self.fail(
                    Some(sense.line),
                    format!(
                        "Sense {} at line {} has lemma {} missing from members of synset {}",
                        sense.id, sense.line, entry.lemma, raw.id
                    ),
                ));
            }
        }

        Ok(lemmas)
    }

    fn parse_relation(
        &self,
        rel_type: &str,
        source_pos: WordNetPos,
        line: usize,
        sense_relation: bool,
    ) -> Result<WordNetRelation, WordNetError> {
        if sense_relation && is_synset_only_rel_type(rel_type) {
            return Err(
                self.fail(Some(line), format!("Relation type {rel_type} is not legal on SenseRelation at line {line}"))
            );
        }
        if !sense_relation && is_sense_only_rel_type(rel_type) {
            return Err(self
                .fail(Some(line), format!("Relation type {rel_type} is not legal on SynsetRelation at line {line}")));
        }
        if rel_type == "similar" {
            return Ok(if !sense_relation && source_pos == WordNetPos::Verb {
                WordNetRelation::VerbGroup
            } else {
                WordNetRelation::SimilarTo
            });
        }
        WordNetRelation::from_lmf_name(rel_type)
            .ok_or_else(|| self.fail(Some(line), format!("Unknown relation type {rel_type} at line {line}")))
    }
}

fn pos_upper_name(pos: WordNetPos) -> &'static str {
    match pos {
        WordNetPos::Noun => "NOUN",
        WordNetPos::Verb => "VERB",
        WordNetPos::Adjective => "ADJECTIVE",
        WordNetPos::Adverb => "ADVERB",
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::super::{WnLmfReader, cat_lexicon, wrap_lexicon};
    use super::*;
    use crate::lexicon::LexicalKnowledgeBase;

    fn read(doc: &str) -> Result<InMemoryWordNetLexicon, WordNetError> {
        WnLmfReader::read_named(Cursor::new(doc.as_bytes()), "test.xml")
    }

    fn assert_err_contains(result: Result<impl std::fmt::Debug, WordNetError>, fragment: &str) {
        let err = result.expect_err("expected a parse error");
        assert!(err.to_string().contains(fragment), "expected '{fragment}' in '{err}'");
    }

    const TWO_CATS: &str = "<LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>\
         <Sense id=\"t-cat-n-1\" synset=\"t-1\"/></LexicalEntry>\
         <LexicalEntry id=\"t-feline-n\"><Lemma writtenForm=\"feline\" partOfSpeech=\"n\"/>\
         <Sense id=\"t-feline-n-1\" synset=\"t-1\"/></LexicalEntry>";

    #[test]
    fn a_synset_relation_to_its_own_synset_is_rejected() {
        assert_err_contains(
            read(&cat_lexicon("", "<SynsetRelation relType=\"hypernym\" target=\"t-1\"/>")),
            "Relation hypernym at line 4 on synset t-1 targets its own synset",
        );
    }

    #[test]
    fn a_sense_relation_to_its_own_sense_is_rejected() {
        let doc = wrap_lexicon(
            "<LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>\
             <Sense id=\"t-cat-n-1\" synset=\"t-1\"><SenseRelation relType=\"antonym\" target=\"t-cat-n-1\"/></Sense>\
             </LexicalEntry><Synset id=\"t-1\" partOfSpeech=\"n\"/>",
        );
        assert_err_contains(read(&doc), "SenseRelation antonym at line 4 from sense t-cat-n-1 targets its own sense");
    }

    #[test]
    fn a_sense_relation_between_two_senses_of_one_synset_is_kept() {
        let doc = wrap_lexicon(&format!(
            "{}<Synset id=\"t-1\" partOfSpeech=\"n\"/>",
            TWO_CATS.replace(
                "<Sense id=\"t-cat-n-1\" synset=\"t-1\"/>",
                "<Sense id=\"t-cat-n-1\" synset=\"t-1\"><SenseRelation relType=\"derivation\" target=\"t-feline-n-1\"/></Sense>",
            )
        ));
        let kb = read(&doc).unwrap();
        assert_eq!(&["t-1"], kb.related("t-1", WordNetRelation::DerivationallyRelated));
    }

    #[test]
    fn members_default_to_the_senses_when_the_attribute_is_absent_empty_or_blank() {
        for attrs in ["", " members=\"\"", " members=\" \"", " members=\"&#32;&#x20;\""] {
            let kb = read(&wrap_lexicon(&format!("{TWO_CATS}<Synset id=\"t-1\" partOfSpeech=\"n\"{attrs}/>")))
                .unwrap_or_else(|e| panic!("members attrs {attrs:?}: {e}"));
            assert_eq!(&["cat", "feline"], kb.synset("t-1").unwrap().lemmas(), "{attrs:?}");
        }
    }

    #[test]
    fn explicit_members_set_the_lemma_order_and_may_name_entries_or_senses() {
        for members in ["t-feline-n-1 t-cat-n-1", "t-feline-n t-cat-n", "t-feline-n-1 t-cat-n t-feline-n"] {
            let kb = read(&wrap_lexicon(&format!(
                "{TWO_CATS}<Synset id=\"t-1\" partOfSpeech=\"n\" members=\"{members}\"/>"
            )))
            .unwrap_or_else(|e| panic!("{members}: {e}"));
            assert_eq!(&["feline", "cat"], kb.synset("t-1").unwrap().lemmas(), "{members}");
        }
    }

    #[test]
    fn dangling_and_mismatched_references_are_rejected() {
        assert_err_contains(
            read(&wrap_lexicon(&format!("{TWO_CATS}<Synset id=\"t-1\" partOfSpeech=\"n\" members=\"t-cat-n-1\"/>"))),
            "Sense t-feline-n-1 at line 4 has lemma feline missing from members of synset t-1",
        );
        assert_err_contains(
            read(&wrap_lexicon(&format!("{TWO_CATS}<Synset id=\"t-1\" partOfSpeech=\"n\" members=\"t-ghost\"/>"))),
            "lists undeclared member sense or entry t-ghost",
        );
        assert_err_contains(
            read(&cat_lexicon("", "<SynsetRelation relType=\"hypernym\" target=\"t-ghost\"/>")),
            "references undeclared synset t-ghost",
        );
        assert_err_contains(
            read(&cat_lexicon("", "").replace("synset=\"t-1\"", "synset=\"t-ghost\"")),
            "Sense t-cat-n-1 at line 4 references undeclared synset t-ghost",
        );
        assert_err_contains(
            read(&cat_lexicon("", "<SynsetRelation relType=\"derivation\" target=\"t-1\"/>")),
            "Relation type derivation is not legal on SynsetRelation",
        );
        assert_err_contains(
            read(&wrap_lexicon(&format!(
                "{TWO_CATS}<Synset id=\"t-1\" partOfSpeech=\"n\"/><Synset id=\"t-2\" partOfSpeech=\"n\" members=\"t-cat-n\"/>"
            ))),
            "Synset t-2 at line 4 lists member entry t-cat-n with no sense assigned to t-2",
        );
    }

    #[test]
    fn relation_cycles_between_distinct_synsets_are_kept() {
        let doc = wrap_lexicon(
            "<LexicalEntry id=\"t-a-n\"><Lemma writtenForm=\"a\" partOfSpeech=\"n\"/>\
             <Sense id=\"t-a-n-1\" synset=\"t-1\"/><Sense id=\"t-a-n-2\" synset=\"t-2\"/></LexicalEntry>\
             <Synset id=\"t-1\" partOfSpeech=\"n\"><SynsetRelation relType=\"hypernym\" target=\"t-2\"/></Synset>\
             <Synset id=\"t-2\" partOfSpeech=\"n\"><SynsetRelation relType=\"hypernym\" target=\"t-1\"/></Synset>",
        );
        let kb = read(&doc).unwrap();
        assert_eq!(&["t-2"], kb.related("t-1", WordNetRelation::Hypernym));
        assert_eq!(&["t-1"], kb.related("t-2", WordNetRelation::Hypernym));
    }
}
