//! Composes a `<LexiconExtension>` onto the base lexicon its `<Extends>` names.

#[cfg(test)]
mod tests;

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::BufReader;

use super::materialize::Materializer;
use super::raw::{RawExternalSynset, RawLexicon, RawSenseRelation, RawSynset, parse_raw};
use super::{
    MAX_EXTENSION_DEPTH, OTHER_RELATION, WnLmfDependency, WnLmfResolver, WnLmfSource, malformed,
    reject_same_document_base,
};
use crate::error::WordNetError;
use crate::folding::split_on_spaces;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ResolutionKey {
    id: String,
    version: String,
}

impl std::fmt::Display for ResolutionKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.id, self.version)
    }
}

#[derive(Debug, Clone)]
pub(super) struct ResolvedLexicon {
    pub(super) lexicon: RawLexicon,
    extension_path: Vec<ResolutionKey>,
}

pub(super) struct Composition<'a, Res: WnLmfResolver> {
    resolver: &'a Res,
    cache: HashMap<ResolutionKey, ResolvedLexicon>,
    stack: Vec<ResolutionKey>,
}

impl<'a, Res: WnLmfResolver> Composition<'a, Res> {
    pub(super) fn new(resolver: &'a Res) -> Self {
        Self { resolver, cache: HashMap::new(), stack: Vec::new() }
    }

    pub(super) fn compose(
        &mut self,
        extension: &RawLexicon,
        resource_name: &str,
    ) -> Result<ResolvedLexicon, WordNetError> {
        let key = ResolutionKey { id: extension.id.clone(), version: extension.version.clone() };
        self.validate_path(std::slice::from_ref(&key), resource_name)?;
        self.stack.push(key.clone());
        let res = (|| {
            let ext_ref = extension.extends_ref.as_ref().unwrap();
            let base = self.base(ext_ref, resource_name)?;
            let mut extension_path = Vec::with_capacity(base.extension_path.len() + 1);
            extension_path.push(key.clone());
            extension_path.extend(base.extension_path);
            let merged = self.merge(&base.lexicon, extension, resource_name)?;
            Ok(ResolvedLexicon { lexicon: merged, extension_path })
        })();
        self.stack.pop();
        res
    }

    fn base(&mut self, reference: &WnLmfDependency, resource_name: &str) -> Result<ResolvedLexicon, WordNetError> {
        let key = ResolutionKey { id: reference.id().to_string(), version: reference.version().to_string() };
        // A base already being composed further up the chain is a cycle; resolving it again would
        // only rediscover that after a resolver call.
        if self.stack.contains(&key) {
            return Err(self.cycle(&[key], resource_name));
        }
        if let Some(cached) = self.cache.get(&key).cloned() {
            self.validate_path(&cached.extension_path, resource_name)?;
            return Ok(cached);
        }

        let source = self.resolver.resolve(reference)?.ok_or_else(|| {
            WordNetError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Unresolved base lexicon '{}' version '{}'", reference.id(), reference.version()),
            ))
        })?;
        let source_name = source.name().to_string();
        let resolved_lexicons = match source {
            WnLmfSource::NamedBytes(_, bytes) | WnLmfSource::Bytes(bytes) => {
                parse_raw(std::io::Cursor::new(bytes), &source_name, true)?
            }
            WnLmfSource::File(path) => {
                let file = File::open(&path)?;
                parse_raw(BufReader::new(file), &source_name, true)?
            }
            WnLmfSource::NamedReader(_, r) | WnLmfSource::Reader(r) => parse_raw(r, &source_name, true)?,
        };

        reject_same_document_base(&resolved_lexicons, &source_name)?;
        let matched = resolved_lexicons
            .into_iter()
            .find(|lex| lex.id == reference.id() && lex.version == reference.version())
            .ok_or_else(|| {
                malformed(
                    resource_name,
                    None,
                    format!(
                        "Resolved resource {source_name} does not contain lexicon {} version {}",
                        reference.id(),
                        reference.version()
                    ),
                )
            })?;

        let result = if matched.extension {
            self.compose(&matched, &source_name)?
        } else {
            ResolvedLexicon { lexicon: matched, extension_path: Vec::new() }
        };
        Materializer::new(&result.lexicon, &source_name).validate()?;
        self.cache.insert(key, result.clone());
        Ok(result)
    }

    fn validate_path(&self, extension_path: &[ResolutionKey], resource_name: &str) -> Result<(), WordNetError> {
        for (i, key) in extension_path.iter().enumerate() {
            if self.stack.contains(key) {
                return Err(self.cycle(&extension_path[..=i], resource_name));
            }
            if self.stack.len() + i >= MAX_EXTENSION_DEPTH {
                return Err(malformed(
                    resource_name,
                    None,
                    format!("Extension chain at {key} exceeds the maximum composition depth of {MAX_EXTENSION_DEPTH}"),
                ));
            }
        }
        Ok(())
    }

    /// Reports the composition stack followed by `path` as a dependency cycle.
    fn cycle(&self, path: &[ResolutionKey], resource_name: &str) -> WordNetError {
        let chain: Vec<String> = self.stack.iter().chain(path).map(ToString::to_string).collect();
        malformed(resource_name, None, format!("Extension dependency cycle: {}", chain.join(" -> ")))
    }

    fn merge(
        &self,
        base: &RawLexicon,
        extension: &RawLexicon,
        resource_name: &str,
    ) -> Result<RawLexicon, WordNetError> {
        for entry_id in &extension.entry_order {
            let entry = &extension.entries[entry_id];
            self.require_new_id(base, extension, &entry.id, "lexical entry", entry.line, resource_name)?;
        }
        for sense_id in &extension.sense_order {
            let sense = &extension.senses[sense_id];
            self.require_new_id(base, extension, &sense.id, "sense", sense.line, resource_name)?;
        }
        for synset_id in &extension.synset_order {
            let synset = &extension.synsets[synset_id];
            self.require_new_id(base, extension, &synset.id, "synset", synset.line, resource_name)?;
        }

        let mut external_synsets: HashMap<String, &RawExternalSynset> = HashMap::new();
        for ext_syn in &extension.external_synsets {
            if !base.synsets.contains_key(&ext_syn.id) {
                return Err(malformed(
                    resource_name,
                    Some(ext_syn.line),
                    format!(
                        "ExternalSynset {} at line {} does not resolve to a synset in base {} version {}{}",
                        ext_syn.id,
                        ext_syn.line,
                        base.id,
                        base.version,
                        kind_hint(base, &ext_syn.id)
                    ),
                ));
            }
            external_synsets.insert(ext_syn.id.clone(), ext_syn);
        }

        let mut declared_external_entries: HashSet<String> = HashSet::new();
        let mut declared_external_senses: HashSet<String> = HashSet::new();
        let mut added_sense_relations: Vec<(String, Vec<RawSenseRelation>)> = Vec::new();
        for ext_entry in &extension.external_entries {
            if !base.entries.contains_key(&ext_entry.id) {
                return Err(malformed(
                    resource_name,
                    Some(ext_entry.line),
                    format!(
                        "ExternalLexicalEntry {} at line {} does not resolve to a lexical entry in base {} version {}{}",
                        ext_entry.id,
                        ext_entry.line,
                        base.id,
                        base.version,
                        kind_hint(base, &ext_entry.id)
                    ),
                ));
            }
            declared_external_entries.insert(ext_entry.id.clone());
            for ext_sense in &ext_entry.external_senses {
                let Some(base_sense) = base.senses.get(&ext_sense.id) else {
                    return Err(malformed(
                        resource_name,
                        Some(ext_sense.line),
                        format!(
                            "ExternalSense {} at line {} does not resolve to a sense in base {} version {}{}",
                            ext_sense.id,
                            ext_sense.line,
                            base.id,
                            base.version,
                            kind_hint(base, &ext_sense.id)
                        ),
                    ));
                };
                if base_sense.entry_id != ext_entry.id {
                    return Err(malformed(
                        resource_name,
                        Some(ext_sense.line),
                        format!(
                            "ExternalSense {} at line {} is not a sense of lexical entry {}; it belongs to {}",
                            ext_sense.id, ext_sense.line, ext_entry.id, base_sense.entry_id
                        ),
                    ));
                }
                declared_external_senses.insert(ext_sense.id.clone());
                if !ext_sense.relations.is_empty() {
                    added_sense_relations.push((ext_sense.id.clone(), ext_sense.relations.clone()));
                }
            }
        }

        for sense_id in &extension.sense_order {
            let sense = &extension.senses[sense_id];
            self.require_synset_target(
                extension,
                &external_synsets,
                &sense.synset_id,
                &format!("Sense {}", sense.id),
                sense.line,
                resource_name,
            )?;
            self.require_sense_targets(
                extension,
                &declared_external_senses,
                &sense.id,
                &sense.relations,
                resource_name,
            )?;
        }
        for synset_id in &extension.synset_order {
            let synset = &extension.synsets[synset_id];
            self.require_member_targets(
                extension,
                synset,
                &declared_external_entries,
                &declared_external_senses,
                resource_name,
            )?;
            for rel in &synset.relations {
                self.require_synset_target(
                    extension,
                    &external_synsets,
                    &rel.target,
                    &format!("Relation {} on synset {}", rel.rel_type, synset.id),
                    rel.line,
                    resource_name,
                )?;
            }
        }
        for ext_syn in &extension.external_synsets {
            for rel in &ext_syn.relations {
                self.require_synset_target(
                    extension,
                    &external_synsets,
                    &rel.target,
                    &format!("Relation {} on ExternalSynset {}", rel.rel_type, ext_syn.id),
                    rel.line,
                    resource_name,
                )?;
            }
        }
        for (source_id, rels) in &added_sense_relations {
            self.require_sense_targets(extension, &declared_external_senses, source_id, rels, resource_name)?;
        }

        let mut entries = base.entries.clone();
        let mut entry_order = base.entry_order.clone();
        for id in &extension.entry_order {
            entry_order.push(id.clone());
            entries.insert(id.clone(), extension.entries[id].clone());
        }

        let mut senses = base.senses.clone();
        let mut sense_order = base.sense_order.clone();
        for (sense_id, added) in added_sense_relations {
            let base_sense = senses.get_mut(&sense_id).unwrap();
            base_sense.relations.extend(added);
        }
        for id in &extension.sense_order {
            sense_order.push(id.clone());
            senses.insert(id.clone(), extension.senses[id].clone());
        }

        let mut extra_members: HashMap<String, Vec<String>> = HashMap::new();
        for id in &extension.sense_order {
            let sense = &extension.senses[id];
            if external_synsets.contains_key(&sense.synset_id) {
                extra_members.entry(sense.synset_id.clone()).or_default().push(sense.entry_id.clone());
            }
        }

        let mut synsets = base.synsets.clone();
        let mut synset_order = base.synset_order.clone();
        for ext_syn in &extension.external_synsets {
            let base_synset = synsets.get_mut(&ext_syn.id).unwrap();
            base_synset.relations.extend(ext_syn.relations.iter().cloned());
            if base_synset.gloss.is_empty()
                && let Some(ref def) = ext_syn.definition
            {
                base_synset.gloss = def.clone();
            }
            if let Some(extras) = extra_members.get(&ext_syn.id) {
                base_synset.extra_members.extend(extras.iter().cloned());
            }
        }
        for id in &extension.synset_order {
            synset_order.push(id.clone());
            synsets.insert(id.clone(), extension.synsets[id].clone());
        }

        Ok(RawLexicon {
            extension: true,
            id: extension.id.clone(),
            label: extension.label.clone(),
            language: extension.language.clone(),
            version: extension.version.clone(),
            metadata: extension.metadata.clone(),
            requires: extension.requires.clone(),
            extends_ref: extension.extends_ref.clone(),
            entries,
            entry_order,
            senses,
            sense_order,
            synsets,
            synset_order,
            external_entries: Vec::new(),
            external_synsets: Vec::new(),
            line: extension.line,
        })
    }

    fn require_new_id(
        &self,
        base: &RawLexicon,
        extension: &RawLexicon,
        id: &str,
        kind: &str,
        line: usize,
        resource_name: &str,
    ) -> Result<(), WordNetError> {
        if base.entries.contains_key(id) || base.senses.contains_key(id) || base.synsets.contains_key(id) {
            return Err(malformed(
                resource_name,
                Some(line),
                format!(
                    "New {kind} id {id} at line {line} in extension {} collides with content of base {} version {}",
                    extension.id, base.id, base.version
                ),
            ));
        }
        Ok(())
    }

    fn require_synset_target(
        &self,
        extension: &RawLexicon,
        external_synsets: &HashMap<String, &RawExternalSynset>,
        target: &str,
        what: &str,
        line: usize,
        resource_name: &str,
    ) -> Result<(), WordNetError> {
        if !extension.synsets.contains_key(target) && !external_synsets.contains_key(target) {
            return Err(malformed(
                resource_name,
                Some(line),
                format!(
                    "{what} at line {line} references synset {target}, which is neither declared in the extension nor as an ExternalSynset"
                ),
            ));
        }
        Ok(())
    }

    fn require_sense_targets(
        &self,
        extension: &RawLexicon,
        declared_external_senses: &HashSet<String>,
        source_id: &str,
        relations: &[RawSenseRelation],
        resource_name: &str,
    ) -> Result<(), WordNetError> {
        for rel in relations {
            if rel.rel_type == OTHER_RELATION {
                continue;
            }
            if !extension.senses.contains_key(&rel.target) && !declared_external_senses.contains(&rel.target) {
                return Err(malformed(
                    resource_name,
                    Some(rel.line),
                    format!(
                        "SenseRelation {} at line {} from sense {source_id} references sense {}, which is neither declared in the extension nor as an ExternalSense",
                        rel.rel_type, rel.line, rel.target
                    ),
                ));
            }
        }
        Ok(())
    }

    fn require_member_targets(
        &self,
        extension: &RawLexicon,
        synset: &RawSynset,
        external_entries: &HashSet<String>,
        external_senses: &HashSet<String>,
        resource_name: &str,
    ) -> Result<(), WordNetError> {
        let Some(ref members) = synset.members else {
            return Ok(());
        };
        for member in split_on_spaces(members) {
            if !extension.entries.contains_key(member)
                && !external_entries.contains(member)
                && !extension.senses.contains_key(member)
                && !external_senses.contains(member)
            {
                return Err(malformed(
                    resource_name,
                    Some(synset.line),
                    format!(
                        "Synset {} at line {} lists member {member}, which needs a sense or entry declaration in the extension, or an ExternalSense or ExternalLexicalEntry",
                        synset.id, synset.line
                    ),
                ));
            }
        }
        Ok(())
    }
}

fn kind_hint(base: &RawLexicon, id: &str) -> &'static str {
    if base.entries.contains_key(id) {
        "; it names a lexical entry"
    } else if base.senses.contains_key(id) {
        "; it names a sense"
    } else if base.synsets.contains_key(id) {
        "; it names a synset"
    } else {
        ""
    }
}
