//! The WN-LMF reader and the bundled resource.

use super::*;

// ============================================================================
// WnLmfReaderTest
// ============================================================================

#[test]
fn test_lmf_lookup_returns_synset_with_all_components() {
    let lexicon = lmf_fixture();
    let senses = lexicon.lookup("dog", WordNetPos::Noun);
    assert_eq!(1, senses.len());
    let dog = senses[0];
    assert_eq!("mini-n1", dog.id());
    assert_eq!(WordNetPos::Noun, dog.pos());
    assert_eq!(&["dog", "domestic dog"], dog.lemmas());
    assert_eq!("a domesticated canid", dog.gloss());
    assert_eq!(&["mini-n2"], dog.related(WordNetRelation::Hypernym));
}

#[test]
fn test_lmf_lookup_folds_case_and_underscore() {
    let lexicon = lmf_fixture();
    assert_eq!("mini-n1", lexicon.lookup("Domestic_Dog", WordNetPos::Noun)[0].id());
    assert_eq!("mini-n1", lexicon.lookup("DOG", WordNetPos::Noun)[0].id());
}

#[test]
fn test_lmf_lookup_keeps_sense_order() {
    let lexicon = lmf_fixture();
    let ids: Vec<&str> = lexicon.lookup("run", WordNetPos::Noun).iter().map(|s| s.id()).collect();
    assert_eq!(vec!["mini-n5", "mini-n9"], ids);
}

#[test]
fn test_lmf_preserves_multiple_definitions() {
    let lexicon = parse_lmf(&wrap_lmf(
        "<LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>\
         <Sense id=\"t-cat-n-1\" synset=\"t-1\"/></LexicalEntry>\
         <Synset id=\"t-1\" partOfSpeech=\"n\">\
         <Definition>a feline</Definition>\
         <Definition>a domesticated cat</Definition></Synset>",
    ))
    .unwrap();
    assert_eq!("a feline; a domesticated cat", lexicon.lookup("cat", WordNetPos::Noun)[0].gloss());
}

#[test]
fn test_lmf_normalizes_line_ends_in_text() {
    let document = wrap_lmf(
        "<LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>\
         <Sense id=\"t-cat-n-1\" synset=\"t-1\"/></LexicalEntry>\
         <Synset id=\"t-1\" partOfSpeech=\"n\">\
         <Definition>a small\r\ndomesticated\rfeline</Definition></Synset>",
    );
    let lexicon = parse_lmf(&document).unwrap();
    assert_eq!("a small\ndomesticated\nfeline", lexicon.lookup("cat", WordNetPos::Noun)[0].gloss());
}

#[test]
fn test_lmf_lookup_is_pos_scoped() {
    let lexicon = lmf_fixture();
    assert_eq!(1, lexicon.lookup("run", WordNetPos::Verb).len());
    assert!(lexicon.lookup("dog", WordNetPos::Verb).is_empty());
    assert!(!lexicon.contains("walk", WordNetPos::Noun));
    assert!(lexicon.contains("walk", WordNetPos::Verb));
}

#[test]
fn test_lmf_relation_navigation() {
    let lexicon = lmf_fixture();
    assert_eq!(&["mini-n1"], lexicon.related("mini-n2", WordNetRelation::Hyponym));
    assert_eq!(&["mini-v1", "mini-v2"], lexicon.related("mini-v4", WordNetRelation::Hyponym));
    assert_eq!(&["mini-v4"], lexicon.related("mini-v1", WordNetRelation::Hypernym));
}

#[test]
fn test_lmf_sense_relations_are_represented_at_synset_level() {
    let lexicon = lmf_fixture();
    assert_eq!(&["mini-a2"], lexicon.related("mini-a1", WordNetRelation::Antonym));
    assert_eq!(&["mini-a1"], lexicon.related("mini-a2", WordNetRelation::Antonym));
    assert_eq!(&["mini-v1"], lexicon.related("mini-n5", WordNetRelation::DerivationallyRelated));
    assert_eq!(&["mini-n5"], lexicon.related("mini-v1", WordNetRelation::DerivationallyRelated));
}

#[test]
fn test_lmf_satellite_normalizes_to_adjective() {
    let lexicon = lmf_fixture();
    let senses = lexicon.lookup("large", WordNetPos::Adjective);
    assert_eq!(1, senses.len());
    assert_eq!(WordNetPos::Adjective, senses[0].pos());
    assert_eq!(&["mini-a4"], lexicon.related("mini-a3", WordNetRelation::SimilarTo));
    assert_eq!(&["mini-a3"], lexicon.related("mini-a4", WordNetRelation::SimilarTo));
}

#[test]
fn test_lmf_similar_on_verb_synset_maps_to_verb_group() {
    let lexicon = parse_lmf(&wrap_lmf(
        "<LexicalEntry id=\"t-sing-v\"><Lemma writtenForm=\"sing\" partOfSpeech=\"v\"/>\
         <Sense id=\"t-sing-v-1\" synset=\"t-v1\"/></LexicalEntry>\
         <LexicalEntry id=\"t-chant-v\"><Lemma writtenForm=\"chant\" partOfSpeech=\"v\"/>\
         <Sense id=\"t-chant-v-1\" synset=\"t-v2\"/></LexicalEntry>\
         <Synset id=\"t-v1\" partOfSpeech=\"v\">\
         <Definition>produce musical tones</Definition>\
         <SynsetRelation relType=\"similar\" target=\"t-v2\"/></Synset>\
         <Synset id=\"t-v2\" partOfSpeech=\"v\">\
         <Definition>sing monotonously</Definition></Synset>",
    ))
    .unwrap();
    assert_eq!(&["t-v2"], lexicon.related("t-v1", WordNetRelation::VerbGroup));
    assert!(lexicon.related("t-v1", WordNetRelation::SimilarTo).is_empty());
}

#[test]
fn test_lmf_skips_doctype_and_rejects_xxe() {
    let doc = "<?xml version=\"1.0\"?>\n\
        <!DOCTYPE LexicalResource SYSTEM \"http://example.invalid/WN-LMF-1.3.dtd\">\n\
        <LexicalResource><Lexicon id=\"t\" label=\"t\" language=\"en\" version=\"1\">\
        <LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>\
        <Sense id=\"t-cat-n-1\" synset=\"t-1\"/></LexicalEntry>\
        <Synset id=\"t-1\" partOfSpeech=\"n\"><Definition>a feline</Definition></Synset>\
        </Lexicon></LexicalResource>";
    let lexicon = parse_lmf(doc).unwrap();
    assert_eq!("a feline", lexicon.lookup("cat", WordNetPos::Noun)[0].gloss());

    let xxe_doc = "<?xml version=\"1.0\"?>\n\
        <!DOCTYPE LexicalResource [<!ENTITY xxe SYSTEM \"file:///etc/passwd\">]>\n\
        <LexicalResource><Lexicon id=\"t\" label=\"t\" language=\"en\" version=\"1\">\
        <LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"&xxe;\" partOfSpeech=\"n\"/>\
        <Sense id=\"t-cat-n-1\" synset=\"t-1\"/></LexicalEntry>\
        <Synset id=\"t-1\" partOfSpeech=\"n\"><Definition>a feline</Definition></Synset>\
        </Lexicon></LexicalResource>";
    assert!(parse_lmf(xxe_doc).is_err());
}

#[test]
fn test_lmf_rejects_structurally_invalid_documents() {
    let cases: &[(&str, &[&str])] = &[
        (
            "<LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>\
             <Sense id=\"t-cat-n-1\"/></LexicalEntry>",
            &["synset"],
        ),
        (
            "<LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>\
             <Sense id=\"t-cat-n-1\" synset=\"t-9\"/></LexicalEntry>",
            &["t-9"],
        ),
        (
            "<LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>\
             <Sense id=\"t-cat-n-1\" synset=\"t-1\"/></LexicalEntry>\
             <Synset id=\"t-1\" partOfSpeech=\"n\"><Definition>a feline</Definition>\
             <SynsetRelation relType=\"hypernym\" target=\"t-9\"/></Synset>",
            &["t-9"],
        ),
        (
            "<LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>\
             <Sense id=\"t-cat-n-1\" synset=\"t-1\"/></LexicalEntry>\
             <Synset id=\"t-1\" partOfSpeech=\"n\"><Definition>a feline</Definition>\
             <SynsetRelation relType=\"quasi_synonym\" target=\"t-1\"/></Synset>",
            &["quasi_synonym"],
        ),
        (
            "<LexicalEntry id=\"t-cat-x\"><Lemma writtenForm=\"cat\" partOfSpeech=\"x\"/>\
             <Sense id=\"t-cat-x-1\" synset=\"t-1\"/></LexicalEntry>\
             <Synset id=\"t-1\" partOfSpeech=\"n\"><Definition>a feline</Definition></Synset>",
            &["x"],
        ),
        (
            "<LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>\
             <Sense id=\"t-cat-n-1\" synset=\"t-1\"/></LexicalEntry>\
             <Synset id=\"t-1\" partOfSpeech=\"n\"><Definition>a feline</Definition></Synset>\
             <Synset id=\"t-1\" partOfSpeech=\"n\"><Definition>a repeat</Definition></Synset>",
            &["Duplicate synset id t-1"],
        ),
        (
            "<LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>\
             <Sense id=\"t-cat-n-1\" synset=\"t-1\"/></LexicalEntry>\
             <LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"dog\" partOfSpeech=\"n\"/>\
             <Sense id=\"t-dog-n-1\" synset=\"t-1\"/></LexicalEntry>\
             <Synset id=\"t-1\" partOfSpeech=\"n\"><Definition>a feline</Definition></Synset>",
            &["Duplicate lexical entry id t-cat-n"],
        ),
        (
            "<LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>\
             <Sense id=\"t-cat-n-1\" synset=\"t-1\"/>\
             <Sense id=\"t-cat-n-1\" synset=\"t-2\"/></LexicalEntry>\
             <Synset id=\"t-1\" partOfSpeech=\"n\"/>\
             <Synset id=\"t-2\" partOfSpeech=\"n\"/>",
            &["Duplicate sense id t-cat-n-1"],
        ),
        (
            "<LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>\
             <Sense id=\"t-cat-n-1\" synset=\"t-1\"/></LexicalEntry>\
             <Synset id=\"t-1\" partOfSpeech=\"v\"><Definition>a feline</Definition></Synset>",
            &["t-cat-n", "Verb", "Noun"],
        ),
        (
            "<LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>\
             <Sense id=\"t-cat-n-1\" synset=\"t-1\"/></LexicalEntry>\
             <LexicalEntry id=\"t-dog-n\"><Lemma writtenForm=\"dog\" partOfSpeech=\"n\"/>\
             <Sense id=\"t-dog-n-1\" synset=\"t-2\"/></LexicalEntry>\
             <Synset id=\"t-1\" partOfSpeech=\"n\" members=\"t-dog-n-1\"/>\
             <Synset id=\"t-2\" partOfSpeech=\"n\" members=\"t-dog-n-1\"/>",
            &["t-dog-n-1", "t-1", "t-2"],
        ),
        (
            "<LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>\
             <Sense id=\"t-cat-n-1\" synset=\"t-1\">\
             <SenseRelation relType=\"antonym\" target=\"t-ghost-1\"/></Sense>\
             </LexicalEntry>\
             <Synset id=\"t-1\" partOfSpeech=\"n\"/>",
            &["t-ghost-1"],
        ),
        ("<Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>", &["Lemma outside a LexicalEntry"]),
        (
            "<LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>\
             <Lemma writtenForm=\"dog\" partOfSpeech=\"n\"/></LexicalEntry>",
            &["Duplicate Lemma in LexicalEntry t-cat-n"],
        ),
        (
            "<LexicalEntry id=\"t-outer-n\"><Lemma writtenForm=\"outer\" partOfSpeech=\"n\"/>\
             <LexicalEntry id=\"t-inner-n\"><Lemma writtenForm=\"inner\" partOfSpeech=\"n\"/>\
             </LexicalEntry></LexicalEntry>",
            &["Nested LexicalEntry inside t-outer-n"],
        ),
        (
            "<LexicalEntry id=\"t-cat-n\"><Sense id=\"t-cat-n-1\" synset=\"t-1\"/>\
             <Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/></LexicalEntry>\
             <Synset id=\"t-1\" partOfSpeech=\"n\"/>",
            &["Sense before its entry's Lemma"],
        ),
        (
            "<LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>\
             <Sense id=\"t-cat-n-1\" synset=\"t-1\">\
             <Sense id=\"t-cat-n-2\" synset=\"t-1\"/></Sense></LexicalEntry>\
             <Synset id=\"t-1\" partOfSpeech=\"n\"/>",
            &["Nested Sense inside t-cat-n-1"],
        ),
        (
            "<Synset id=\"t-1\" partOfSpeech=\"n\">\
             <Synset id=\"t-2\" partOfSpeech=\"n\"/></Synset>",
            &["Nested Synset inside t-1"],
        ),
        ("<Definition>orphaned definition</Definition>", &["Definition outside a Synset"]),
        (
            "<LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>\
             <Sense id=\"t-cat-n-1\" synset=\"t-1\"/>\
             <SenseRelation relType=\"antonym\" target=\"t-cat-n-1\"/></LexicalEntry>\
             <Synset id=\"t-1\" partOfSpeech=\"n\"/>",
            &["SenseRelation outside a Sense"],
        ),
        ("<SynsetRelation relType=\"hypernym\" target=\"t-1\"/>", &["SynsetRelation outside a Synset"]),
    ];

    for (body, fragments) in cases {
        let err = parse_lmf(&wrap_lmf(body)).unwrap_err();
        let msg = err.to_string();
        for fragment in *fragments {
            assert!(msg.contains(fragment), "expected '{fragment}' in '{msg}' for body: {body}");
        }
    }
}

#[test]
fn test_lmf_skips_other_relation_type_on_sense_and_synset_relations() {
    let lex1 = parse_lmf(&wrap_lmf(
        "<LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>\
         <Sense id=\"t-cat-n-1\" synset=\"t-1\">\
         <SenseRelation relType=\"other\" target=\"t-cat-n-1\"/></Sense></LexicalEntry>\
         <Synset id=\"t-1\" partOfSpeech=\"n\"><Definition>a feline</Definition></Synset>",
    ))
    .unwrap();
    assert!(lex1.synset("t-1").unwrap().relations().is_empty());

    let lex2 = parse_lmf(&wrap_lmf(
        "<LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>\
         <Sense id=\"t-cat-n-1\" synset=\"t-1\"/></LexicalEntry>\
         <Synset id=\"t-1\" partOfSpeech=\"n\"><Definition>a feline</Definition>\
         <SynsetRelation relType=\"other\" target=\"t-1\"/></Synset>",
    ))
    .unwrap();
    assert!(lex2.synset("t-1").unwrap().relations().is_empty());
}

// ============================================================================
// WnLmfResourceTest
// ============================================================================

#[test]
fn test_lmf_reads_multilingual_resource_without_merging_lexicons() {
    let path = fixtures_dir().join("omw-multilingual.xml");
    let resource = WnLmfReader::read_resource(&path).unwrap();
    let ids: Vec<&str> = resource.lexicons().iter().map(WnLmfLexicon::id).collect();
    assert_eq!(vec!["omw-it", "omw-es", "omw-sv"], ids);

    let italian = resource.lexicon("omw-it").unwrap();
    assert_eq!("it", italian.language());
    assert_eq!("2.0", italian.version());
    assert_eq!(
        Some(&"https://creativecommons.org/licenses/by/3.0/".to_string()),
        italian.metadata().get(&QName::local("license"))
    );
    assert_eq!(
        Some(&"Open Multilingual Wordnet".to_string()),
        italian.metadata().get(&QName::new("https://globalwordnet.github.io/schemas/dc/", "publisher"))
    );
    assert_eq!(&[WnLmfDependency::new("omw-en", "2.0", None).unwrap()], italian.dependencies());
    assert_eq!("omw-it-02084071-n", italian.knowledge_base().lookup("cane", WordNetPos::Noun)[0].id());
    assert_eq!(
        "omw-es-02084071-n",
        resource.lexicon("omw-es").unwrap().knowledge_base().lookup("perro", WordNetPos::Noun)[0].id()
    );
    assert_eq!(
        "omw-sv-02084071-n",
        resource.lexicon("omw-sv").unwrap().knowledge_base().lookup("hund", WordNetPos::Noun)[0].id()
    );
    assert!(italian.knowledge_base().lookup("perro", WordNetPos::Noun).is_empty());
    assert!(resource.lexicon("missing").is_none());

    // Single-lexicon convenience rejects multi-lexicon resource
    let err = WnLmfReader::read(&path).unwrap_err();
    assert!(err.to_string().contains("contains 3 lexicons"), "{err}");
}
