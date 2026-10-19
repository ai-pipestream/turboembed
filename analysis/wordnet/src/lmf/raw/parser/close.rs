use super::super::{RawEntry, RawExternalEntry, RawExternalSense, RawExternalSynset, RawSense, RawSynset};
use super::RawParser;
use crate::error::WordNetError;

impl RawParser<'_> {
    pub(super) fn end_definition(&mut self) {
        let Some(definition) = self.definition.take() else {
            return;
        };
        if let Some(ref mut synset) = self.synset {
            synset.definitions.push(definition);
        } else if let Some(ref mut external_synset) = self.external_synset
            && external_synset.definition.is_none()
        {
            external_synset.definition = Some(definition);
        }
    }

    pub(super) fn end_lexical_entry(&mut self) -> Result<(), WordNetError> {
        let entry = self.entry.take().expect("LexicalEntry end matches a handled start");
        let Some((lemma, pos)) = entry.lemma else {
            return Err(self.fail(entry.line, format!("LexicalEntry {} has no Lemma", entry.id)));
        };
        let lexicon = self.lexicon_mut();
        lexicon.entry_order.push(entry.id.clone());
        lexicon.entries.insert(entry.id.clone(), RawEntry { id: entry.id, lemma, pos, line: entry.line });
        Ok(())
    }

    pub(super) fn end_sense(&mut self) {
        let Some(sense) = self.sense.take() else {
            return;
        };
        let lexicon = self.lexicon_mut();
        lexicon.sense_order.push(sense.id.clone());
        lexicon.senses.insert(
            sense.id.clone(),
            RawSense {
                id: sense.id,
                entry_id: sense.entry_id,
                synset_id: sense.synset_id,
                relations: sense.relations,
                line: sense.line,
            },
        );
    }

    pub(super) fn end_external_sense(&mut self) {
        if let (Some(external_entry), Some(external_sense)) = (self.external_entry.as_mut(), self.external_sense.take())
        {
            external_entry.external_senses.push(RawExternalSense {
                id: external_sense.id,
                relations: external_sense.relations,
                line: external_sense.line,
            });
        }
    }

    pub(super) fn end_external_lexical_entry(&mut self) {
        let Some(external_entry) = self.external_entry.take() else {
            return;
        };
        self.lexicon_mut().external_entries.push(RawExternalEntry {
            id: external_entry.id,
            external_senses: external_entry.external_senses,
            line: external_entry.line,
        });
    }

    pub(super) fn end_synset(&mut self) {
        let Some(synset) = self.synset.take() else {
            return;
        };
        let lexicon = self.lexicon_mut();
        lexicon.synset_order.push(synset.id.clone());
        lexicon.synsets.insert(
            synset.id.clone(),
            RawSynset {
                id: synset.id,
                pos: synset.pos,
                members: synset.members,
                extra_members: Vec::new(),
                gloss: synset.definitions.join("; "),
                relations: synset.relations,
                line: synset.line,
            },
        );
    }

    pub(super) fn end_external_synset(&mut self) {
        let Some(external_synset) = self.external_synset.take() else {
            return;
        };
        self.lexicon_mut().external_synsets.push(RawExternalSynset {
            id: external_synset.id,
            definition: external_synset.definition,
            relations: external_synset.relations,
            line: external_synset.line,
        });
    }

    pub(super) fn end_lexicon(&mut self, line: usize) -> Result<(), WordNetError> {
        let Some(lexicon) = self.lexicon.take() else {
            return Ok(());
        };
        if lexicon.extension && lexicon.extends_ref.is_none() {
            return Err(self.fail(line, format!("LexiconExtension {} is missing its Extends element", lexicon.id)));
        }
        self.lexicons.push(lexicon);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use crate::lmf::raw::parse_raw;

    #[test]
    fn external_synset_keeps_first_definition() {
        let doc = "<LexicalResource>\
                   <LexiconExtension id=\"x\" label=\"x\" language=\"en\" version=\"1\">\
                   <Extends ref=\"b\" version=\"1\"/>\
                   <ExternalSynset id=\"b-2\">\
                   <Definition>a canine</Definition>\
                   <Definition>a domestic dog</Definition>\
                   </ExternalSynset>\
                   </LexiconExtension>\
                   </LexicalResource>";
        let lexicons = parse_raw(Cursor::new(doc.as_bytes()), "ext.xml", true).unwrap();
        assert_eq!(lexicons[0].external_synsets[0].definition.as_deref(), Some("a canine"));
    }

    #[test]
    fn foreign_namespace_root_lexical_resource_is_rejected() {
        let doc = "<o:LexicalResource xmlns:o=\"urn:other\">\
                   <Lexicon id=\"a\" label=\"a\" language=\"en\" version=\"1\"/>\
                   </o:LexicalResource>";
        let err = parse_raw(Cursor::new(doc.as_bytes()), "foreign.xml", true).unwrap_err();
        assert!(err.to_string().contains("Expected root element <LexicalResource>"), "{err}");
    }
}
