//! Every WN-LMF relation the reader maps.

use super::*;

// ============================================================================
// WnLmfRelationCoverageTest
// ============================================================================

#[test]
fn test_every_legal_synset_and_sense_relation() {
    let common = [
        ("agent", WordNetRelation::Agent),
        ("also", WordNetRelation::AlsoSee),
        ("anto_converse", WordNetRelation::AntoConverse),
        ("anto_gradable", WordNetRelation::AntoGradable),
        ("anto_simple", WordNetRelation::AntoSimple),
        ("antonym", WordNetRelation::Antonym),
        ("augmentative", WordNetRelation::Augmentative),
        ("diminutive", WordNetRelation::Diminutive),
        ("domain_region", WordNetRelation::DomainRegion),
        ("domain_topic", WordNetRelation::DomainTopic),
        ("exemplifies", WordNetRelation::DomainUsage),
        ("feminine", WordNetRelation::Feminine),
        ("has_augmentative", WordNetRelation::HasAugmentative),
        ("has_diminutive", WordNetRelation::HasDiminutive),
        ("has_domain_region", WordNetRelation::MemberOfDomainRegion),
        ("has_domain_topic", WordNetRelation::MemberOfDomainTopic),
        ("has_feminine", WordNetRelation::HasFeminine),
        ("has_masculine", WordNetRelation::HasMasculine),
        ("has_young", WordNetRelation::HasYoung),
        ("instrument", WordNetRelation::Instrument),
        ("is_exemplified_by", WordNetRelation::MemberOfDomainUsage),
        ("location", WordNetRelation::Location),
        ("masculine", WordNetRelation::Masculine),
        ("result", WordNetRelation::Result),
        ("young", WordNetRelation::Young),
    ];

    let synset_only = [
        ("attribute", WordNetRelation::Attribute),
        ("be_in_state", WordNetRelation::BeInState),
        ("causes", WordNetRelation::Cause),
        ("classified_by", WordNetRelation::ClassifiedBy),
        ("classifies", WordNetRelation::Classifies),
        ("co_agent_instrument", WordNetRelation::CoAgentInstrument),
        ("co_agent_patient", WordNetRelation::CoAgentPatient),
        ("co_agent_result", WordNetRelation::CoAgentResult),
        ("co_instrument_agent", WordNetRelation::CoInstrumentAgent),
        ("co_instrument_patient", WordNetRelation::CoInstrumentPatient),
        ("co_instrument_result", WordNetRelation::CoInstrumentResult),
        ("co_patient_agent", WordNetRelation::CoPatientAgent),
        ("co_patient_instrument", WordNetRelation::CoPatientInstrument),
        ("co_result_agent", WordNetRelation::CoResultAgent),
        ("co_result_instrument", WordNetRelation::CoResultInstrument),
        ("co_role", WordNetRelation::CoRole),
        ("direction", WordNetRelation::Direction),
        ("entails", WordNetRelation::Entailment),
        ("eq_synonym", WordNetRelation::EqSynonym),
        ("holo_location", WordNetRelation::LocationHolonym),
        ("holo_member", WordNetRelation::MemberHolonym),
        ("holo_part", WordNetRelation::PartHolonym),
        ("holo_portion", WordNetRelation::PortionHolonym),
        ("holo_substance", WordNetRelation::SubstanceHolonym),
        ("holonym", WordNetRelation::Holonym),
        ("hypernym", WordNetRelation::Hypernym),
        ("hyponym", WordNetRelation::Hyponym),
        ("in_manner", WordNetRelation::InManner),
        ("instance_hypernym", WordNetRelation::InstanceHypernym),
        ("instance_hyponym", WordNetRelation::InstanceHyponym),
        ("involved", WordNetRelation::Involved),
        ("involved_agent", WordNetRelation::InvolvedAgent),
        ("involved_direction", WordNetRelation::InvolvedDirection),
        ("involved_instrument", WordNetRelation::InvolvedInstrument),
        ("involved_location", WordNetRelation::InvolvedLocation),
        ("involved_patient", WordNetRelation::InvolvedPatient),
        ("involved_result", WordNetRelation::InvolvedResult),
        ("involved_source_direction", WordNetRelation::InvolvedSourceDirection),
        ("involved_target_direction", WordNetRelation::InvolvedTargetDirection),
        ("ir_synonym", WordNetRelation::IrSynonym),
        ("is_caused_by", WordNetRelation::CausedBy),
        ("is_entailed_by", WordNetRelation::EntailedBy),
        ("is_subevent_of", WordNetRelation::IsSubeventOf),
        ("manner_of", WordNetRelation::MannerOf),
        ("mero_location", WordNetRelation::LocationMeronym),
        ("mero_member", WordNetRelation::MemberMeronym),
        ("mero_part", WordNetRelation::PartMeronym),
        ("mero_portion", WordNetRelation::PortionMeronym),
        ("mero_substance", WordNetRelation::SubstanceMeronym),
        ("meronym", WordNetRelation::Meronym),
        ("patient", WordNetRelation::Patient),
        ("restricted_by", WordNetRelation::RestrictedBy),
        ("restricts", WordNetRelation::Restricts),
        ("role", WordNetRelation::Role),
        ("similar", WordNetRelation::SimilarTo),
        ("source_direction", WordNetRelation::SourceDirection),
        ("state_of", WordNetRelation::StateOf),
        ("subevent", WordNetRelation::Subevent),
        ("target_direction", WordNetRelation::TargetDirection),
    ];

    let sense_only = [
        ("body_part", WordNetRelation::BodyPart),
        ("by_means_of", WordNetRelation::ByMeansOf),
        ("derivation", WordNetRelation::DerivationallyRelated),
        ("destination", WordNetRelation::Destination),
        ("event", WordNetRelation::Event),
        ("has_metaphor", WordNetRelation::HasMetaphor),
        ("has_metonym", WordNetRelation::HasMetonym),
        ("material", WordNetRelation::Material),
        ("metaphor", WordNetRelation::Metaphor),
        ("metonym", WordNetRelation::Metonym),
        ("participle", WordNetRelation::Participle),
        ("pertainym", WordNetRelation::Pertainym),
        ("property", WordNetRelation::Property),
        ("secondary_aspect_ip", WordNetRelation::SecondaryAspectIp),
        ("secondary_aspect_pi", WordNetRelation::SecondaryAspectPi),
        ("similar", WordNetRelation::SimilarTo),
        ("simple_aspect_ip", WordNetRelation::SimpleAspectIp),
        ("simple_aspect_pi", WordNetRelation::SimpleAspectPi),
        ("state", WordNetRelation::State),
        ("undergoer", WordNetRelation::Undergoer),
        ("uses", WordNetRelation::Uses),
        ("vehicle", WordNetRelation::Vehicle),
    ];

    assert_eq!(84, common.len() + synset_only.len());
    assert_eq!(47, common.len() + sense_only.len());

    for (rel_type, expected) in common.iter().chain(synset_only.iter()) {
        let doc = wrap_lmf(&format!(
            "<LexicalEntry id=\"t-source-entry-n\">\
             <Lemma writtenForm=\"source\" partOfSpeech=\"n\"/>\
             <Sense id=\"t-source-sense-n\" synset=\"t-source-n\"/></LexicalEntry>\
             <LexicalEntry id=\"t-target-entry-n\">\
             <Lemma writtenForm=\"target\" partOfSpeech=\"n\"/>\
             <Sense id=\"t-target-sense-n\" synset=\"t-target-n\"/></LexicalEntry>\
             <Synset id=\"t-source-n\" partOfSpeech=\"n\">\
             <SynsetRelation relType=\"{rel_type}\" target=\"t-target-n\"/></Synset>\
             <Synset id=\"t-target-n\" partOfSpeech=\"n\"/>"
        ));
        let lex = parse_lmf(&doc).unwrap();
        assert_eq!(&["t-target-n"], lex.related("t-source-n", *expected));
    }

    for (rel_type, expected) in common.iter().chain(sense_only.iter()) {
        let doc = wrap_lmf(&format!(
            "<LexicalEntry id=\"t-source-entry-v\">\
             <Lemma writtenForm=\"source\" partOfSpeech=\"v\"/>\
             <Sense id=\"t-source-sense-v\" synset=\"t-source-v\">\
             <SenseRelation relType=\"{rel_type}\" target=\"t-target-sense-v\"/></Sense></LexicalEntry>\
             <LexicalEntry id=\"t-target-entry-v\">\
             <Lemma writtenForm=\"target\" partOfSpeech=\"v\"/>\
             <Sense id=\"t-target-sense-v\" synset=\"t-target-v\"/></LexicalEntry>\
             <Synset id=\"t-source-v\" partOfSpeech=\"v\"/>\
             <Synset id=\"t-target-v\" partOfSpeech=\"v\"/>"
        ));
        let lex = parse_lmf(&doc).unwrap();
        assert_eq!(&["t-target-v"], lex.related("t-source-v", *expected));
    }
}
