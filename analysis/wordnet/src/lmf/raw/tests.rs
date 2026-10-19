use std::io::{Cursor, Read};

use super::super::{EXT_NAMESPACE, WnLmfDependency, cat_lexicon, wrap_lexicon};
use super::*;
use crate::error::WordNetError;
use crate::pos::WordNetPos;

fn raw(doc: &str) -> Result<Vec<RawLexicon>, WordNetError> {
    parse_raw(Cursor::new(doc.as_bytes()), "test.xml", true)
}

fn assert_err_contains(result: Result<impl std::fmt::Debug, WordNetError>, fragment: &str) {
    let err = result.expect_err("expected a parse error");
    assert!(err.to_string().contains(fragment), "expected '{fragment}' in '{err}'");
}

/// Hands the document to `parse_raw` one byte per `read` call, splitting every multibyte character.
struct OneByteReader<'a>(&'a [u8]);

impl Read for OneByteReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let Some((&first, rest)) = self.0.split_first() else { return Ok(0) };
        if buf.is_empty() {
            return Ok(0);
        }
        buf[0] = first;
        self.0 = rest;
        Ok(1)
    }
}

#[test]
fn records_entries_senses_and_synsets_in_document_order() {
    let lexicons = raw(&cat_lexicon(
        " members=\"t-cat-n-1\"",
        "<Definition>a feline</Definition><Definition>a pet</Definition>\
         <SynsetRelation relType=\"other\" target=\"t-1\"/><SynsetRelation relType=\"hypernym\" target=\"t-2\"/>",
    ))
    .unwrap();
    assert_eq!(1, lexicons.len());
    let lexicon = &lexicons[0];
    assert_eq!(("t", "t", "en", "1"), (&*lexicon.id, &*lexicon.label, &*lexicon.language, &*lexicon.version));
    assert_eq!(vec!["t-cat-n"], lexicon.entry_order);
    assert_eq!(("cat", WordNetPos::Noun), (&*lexicon.entries["t-cat-n"].lemma, lexicon.entries["t-cat-n"].pos));
    assert_eq!("t-cat-n", lexicon.senses["t-cat-n-1"].entry_id);
    let synset = &lexicon.synsets["t-1"];
    assert_eq!("a feline; a pet", synset.gloss);
    assert_eq!(Some("t-cat-n-1"), synset.members.as_deref());
    assert_eq!(1, synset.relations.len(), "relType other is dropped while parsing");
    assert_eq!(("hypernym", "t-2"), (&*synset.relations[0].rel_type, &*synset.relations[0].target));
}

#[test]
fn unmodelled_elements_are_skipped_with_their_whole_subtree() {
    let lexicons = raw(&wrap_lexicon(
        "<LexicalEntry id=\"t-run-v\"><Lemma writtenForm=\"run\" partOfSpeech=\"v\">\
         <Pronunciation>r\u{28c}n</Pronunciation><Tag category=\"x\">y</Tag></Lemma>\
         <Form writtenForm=\"ran\"/>\
         <Sense id=\"t-run-v-1\" synset=\"t-1\"><Example>they <Synset id=\"bogus\"/> run</Example>\
         <Count>3</Count></Sense>\
         <SyntacticBehaviour subcategorizationFrame=\"Somebody ----s\"/></LexicalEntry>\
         <Synset id=\"t-1\" partOfSpeech=\"v\"><ILIDefinition>move fast</ILIDefinition>\
         <Definition>move fast</Definition><Example>run home</Example></Synset>\
         <SyntacticBehaviour id=\"t-frame-1\" subcategorizationFrame=\"Something ----s\"/>",
    ))
    .unwrap();
    let lexicon = &lexicons[0];
    assert_eq!(vec!["t-1"], lexicon.synset_order, "a Synset nested in an unmodelled Example is skipped");
    assert_eq!("move fast", lexicon.synsets["t-1"].gloss);
    assert_eq!("run", lexicon.entries["t-run-v"].lemma);
}

#[test]
fn foreign_namespace_elements_are_not_mistaken_for_wn_lmf_elements() {
    let lexicons = raw(&cat_lexicon(
        "",
        "<o:Definition xmlns:o=\"urn:example:other\">not a gloss</o:Definition>\
         <Definition>a feline</Definition>\
         <o:SynsetRelation xmlns:o=\"urn:example:other\" relType=\"hypernym\" target=\"t-9\"/>",
    ))
    .unwrap();
    let synset = &lexicons[0].synsets["t-1"];
    assert_eq!("a feline", synset.gloss);
    assert!(synset.relations.is_empty(), "{:?}", synset.relations);

    let lexicons = raw(&wrap_lexicon(
        "<x:LexicalEntry xmlns:x=\"urn:example:other\" id=\"t-dog-n\">\
         <x:Lemma writtenForm=\"dog\" partOfSpeech=\"n\"/></x:LexicalEntry>",
    ))
    .unwrap();
    assert!(lexicons[0].entries.is_empty(), "{:?}", lexicons[0].entry_order);
}

#[test]
fn extension_namespace_qualified_elements_are_wn_lmf_elements() {
    let doc = format!(
        "<LexicalResource xmlns:ext=\"{EXT_NAMESPACE}\">\
         <ext:LexiconExtension id=\"x\" label=\"x\" language=\"en\" version=\"1\">\
         <ext:Extends ref=\"base\" version=\"2\"/>\
         <ext:ExternalSynset id=\"base-1\"><SynsetRelation relType=\"hypernym\" target=\"x-1\"/></ext:ExternalSynset>\
         </ext:LexiconExtension></LexicalResource>"
    );
    let lexicons = raw(&doc).unwrap();
    assert!(lexicons[0].extension);
    assert_eq!(Some("base"), lexicons[0].extends_ref.as_ref().map(WnLmfDependency::id));
    assert_eq!("base-1", lexicons[0].external_synsets[0].id);
}

#[test]
fn ids_are_unique_across_every_kind_and_every_lexicon_in_a_document() {
    assert_err_contains(
        raw(&cat_lexicon("", "").replace("<Synset id=\"t-1\"", "<Synset id=\"t-cat-n-1\"")),
        "Duplicate synset id t-cat-n-1",
    );
    assert_err_contains(raw(&cat_lexicon("", "").replace("id=\"t-1\"", "id=\"t\"")), "Duplicate synset id t");
    let two = "<LexicalResource>\
         <Lexicon id=\"a\" label=\"a\" language=\"en\" version=\"1\"><Synset id=\"s\" partOfSpeech=\"n\"/></Lexicon>\
         <Lexicon id=\"b\" label=\"b\" language=\"en\" version=\"1\"><Synset id=\"s\" partOfSpeech=\"n\"/></Lexicon>\
         </LexicalResource>";
    assert_err_contains(raw(two), "Duplicate synset id s");
    assert_err_contains(raw(&two.replace("id=\"b\"", "id=\"a\"")), "Duplicate lexicon id a");
}

#[test]
fn lexicons_in_one_resource_keep_separate_tables() {
    let doc = "<LexicalResource>\
         <Lexicon id=\"a\" label=\"a\" language=\"en\" version=\"1\"><Synset id=\"a-1\" partOfSpeech=\"n\"/></Lexicon>\
         <Lexicon id=\"b\" label=\"b\" language=\"de\" version=\"2\"><Synset id=\"b-1\" partOfSpeech=\"v\"/></Lexicon>\
         </LexicalResource>";
    let lexicons = raw(doc).unwrap();
    assert_eq!(vec![vec!["a-1"], vec!["b-1"]], lexicons.iter().map(|l| l.synset_order.clone()).collect::<Vec<_>>());
}

#[test]
fn a_second_lexical_resource_after_the_first_is_rejected() {
    let doc = "<LexicalResource><Lexicon id=\"a\" label=\"a\" language=\"en\" version=\"1\"/></LexicalResource>\
               <LexicalResource><Lexicon id=\"b\" label=\"b\" language=\"en\" version=\"1\"/></LexicalResource>";
    assert_err_contains(raw(doc), "second root element <LexicalResource>");
}

#[test]
fn multibyte_characters_split_across_reads_are_reassembled() {
    let doc = cat_lexicon("", "<Definition>\u{e9}t\u{e9} \u{65e5}\u{672c} \u{1F431}</Definition>")
        .replace("writtenForm=\"cat\"", "writtenForm=\"ch\u{e2}t\"");
    let lexicons = parse_raw(OneByteReader(doc.as_bytes()), "test.xml", false).unwrap();
    assert_eq!("ch\u{e2}t", lexicons[0].entries["t-cat-n"].lemma);
    assert_eq!("\u{e9}t\u{e9} \u{65e5}\u{672c} \u{1F431}", lexicons[0].synsets["t-1"].gloss);
}

#[test]
fn invalid_utf8_is_reported_as_a_malformed_document_naming_the_resource() {
    let mut bytes = cat_lexicon("", "").into_bytes();
    let at = bytes.iter().position(|&b| b == b'c').unwrap();
    bytes[at] = 0xC3;
    let err = parse_raw(Cursor::new(bytes), "bad-utf8.xml", false).unwrap_err();
    assert!(matches!(err, WordNetError::Parse(_)), "{err:?}");
    assert!(err.to_string().contains("bad-utf8.xml"), "{err}");
    assert!(err.to_string().contains("UTF-8"), "{err}");
}

#[test]
fn extension_elements_need_a_resolver_and_an_extends() {
    let ext = "<LexicalResource><LexiconExtension id=\"x\" label=\"x\" language=\"en\" version=\"1\">\
               </LexiconExtension></LexicalResource>";
    assert_err_contains(parse_raw(Cursor::new(ext.as_bytes()), "test.xml", false), "without a resolver");
    assert_err_contains(raw(ext), "LexiconExtension x is missing its Extends element");
    assert_err_contains(raw(&wrap_lexicon("<ExternalSynset id=\"s\"/>")), "outside a LexiconExtension");
    assert_err_contains(raw(&wrap_lexicon("<Extends ref=\"b\" version=\"1\"/>")), "outside a LexiconExtension");
}
