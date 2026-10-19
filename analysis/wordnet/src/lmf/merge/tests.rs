use std::cell::Cell;
use std::io::Cursor;

use super::super::{WnLmfReader, WnLmfResource};
use super::*;
use crate::lexicon::LexicalKnowledgeBase;
use crate::pos::WordNetPos;
use crate::relation::WordNetRelation;

const BASE: &str = "<LexicalResource><Lexicon id=\"b\" label=\"b\" language=\"en\" version=\"1\">\
     <LexicalEntry id=\"b-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>\
     <Sense id=\"b-cat-n-1\" synset=\"b-1\"/></LexicalEntry>\
     <LexicalEntry id=\"b-dog-n\"><Lemma writtenForm=\"dog\" partOfSpeech=\"n\"/>\
     <Sense id=\"b-dog-n-1\" synset=\"b-2\"/></LexicalEntry>\
     <Synset id=\"b-1\" partOfSpeech=\"n\"><Definition>a feline</Definition></Synset>\
     <Synset id=\"b-2\" partOfSpeech=\"n\"/>\
     </Lexicon></LexicalResource>";

fn extension(body: &str) -> String {
    format!(
        "<LexicalResource>\n<LexiconExtension id=\"x\" label=\"x\" language=\"en\" version=\"1\">\n\
         <Extends ref=\"b\" version=\"1\"/>\n{body}\n</LexiconExtension>\n</LexicalResource>"
    )
}

/// Reads `doc` with a resolver serving [`BASE`] for `b 1`, and returns the result with the resolver call count.
fn compose(doc: &str) -> (Result<WnLmfResource, WordNetError>, usize) {
    let calls = Cell::new(0usize);
    let resolver = |dependency: &WnLmfDependency| {
        calls.set(calls.get() + 1);
        Ok((dependency.id() == "b" && dependency.version() == "1")
            .then(|| WnLmfSource::NamedBytes("base.xml".to_string(), BASE.as_bytes().to_vec())))
    };
    let result = WnLmfReader::read_resource_named_with_resolver(Cursor::new(doc.as_bytes()), "ext.xml", &resolver);
    (result, calls.get())
}

fn assert_err_contains(result: Result<impl std::fmt::Debug, WordNetError>, fragment: &str) {
    let err = result.expect_err("expected a composition error");
    assert!(err.to_string().contains(fragment), "expected '{fragment}' in '{err}'");
}

#[test]
fn extension_adds_entries_senses_members_and_relations_to_the_base() {
    let (resource, calls) = compose(&extension(
        "<ExternalLexicalEntry id=\"b-cat-n\">\
         <ExternalSense id=\"b-cat-n-1\"><SenseRelation relType=\"antonym\" target=\"b-dog-n-1\"/></ExternalSense>\
         </ExternalLexicalEntry>\
         <ExternalLexicalEntry id=\"b-dog-n\"><ExternalSense id=\"b-dog-n-1\"/></ExternalLexicalEntry>\
         <LexicalEntry id=\"x-kitty-n\"><Lemma writtenForm=\"kitty\" partOfSpeech=\"n\"/>\
         <Sense id=\"x-kitty-n-1\" synset=\"b-1\"/></LexicalEntry>\
         <ExternalSynset id=\"b-1\"><SynsetRelation relType=\"hypernym\" target=\"x-1\"/></ExternalSynset>\
         <ExternalSynset id=\"b-2\"><Definition>a canine</Definition></ExternalSynset>\
         <Synset id=\"x-1\" partOfSpeech=\"n\"><Definition>a pet</Definition></Synset>",
    ));
    assert_eq!(1, calls);
    let lexicon = resource.unwrap().into_lexicons().remove(0);
    assert_eq!(Some("b"), lexicon.extension_of().map(WnLmfDependency::id));
    let kb = lexicon.knowledge_base();
    let cat = kb.synset("b-1").unwrap();
    assert_eq!(&["cat", "kitty"], cat.lemmas());
    assert_eq!("a feline", cat.gloss());
    assert_eq!(&["x-1"], cat.related(WordNetRelation::Hypernym));
    assert_eq!(&["b-2"], cat.related(WordNetRelation::Antonym));
    assert_eq!("a canine", kb.synset("b-2").unwrap().gloss(), "an empty base gloss takes the extension definition");
    assert_eq!(1, kb.lookup("kitty", WordNetPos::Noun).len());
}

#[test]
fn new_ids_must_not_collide_with_base_content() {
    for (body, kind, id) in [
        ("<Synset id=\"b-cat-n\" partOfSpeech=\"n\"/>", "synset", "b-cat-n"),
        (
            "<LexicalEntry id=\"b-1\"><Lemma writtenForm=\"one\" partOfSpeech=\"n\"/></LexicalEntry>",
            "lexical entry",
            "b-1",
        ),
        (
            "<LexicalEntry id=\"x-a-n\"><Lemma writtenForm=\"a\" partOfSpeech=\"n\"/>\
             <Sense id=\"b-dog-n-1\" synset=\"x-1\"/></LexicalEntry><Synset id=\"x-1\" partOfSpeech=\"n\"/>",
            "sense",
            "b-dog-n-1",
        ),
    ] {
        assert_err_contains(compose(&extension(body)).0, &format!("New {kind} id {id} at line 4 in extension x"));
    }
}

#[test]
fn external_declarations_must_resolve_to_the_right_kind_in_the_base() {
    assert_err_contains(
        compose(&extension("<ExternalSynset id=\"b-cat-n\"/>")).0,
        "ExternalSynset b-cat-n at line 4 does not resolve to a synset in base b version 1; it names a lexical entry",
    );
    assert_err_contains(
        compose(&extension("<ExternalLexicalEntry id=\"b-1\"/>")).0,
        "does not resolve to a lexical entry in base b version 1; it names a synset",
    );
    assert_err_contains(
        compose(&extension(
            "<ExternalLexicalEntry id=\"b-cat-n\"><ExternalSense id=\"b-ghost\"/></ExternalLexicalEntry>",
        ))
        .0,
        "ExternalSense b-ghost at line 4 does not resolve to a sense in base b version 1",
    );
    assert_err_contains(
        compose(&extension(
            "<ExternalLexicalEntry id=\"b-cat-n\"><ExternalSense id=\"b-dog-n-1\"/></ExternalLexicalEntry>",
        ))
        .0,
        "ExternalSense b-dog-n-1 at line 4 is not a sense of lexical entry b-cat-n; it belongs to b-dog-n",
    );
}

#[test]
fn references_into_the_base_need_an_external_declaration() {
    assert_err_contains(
        compose(&extension(
            "<Synset id=\"x-1\" partOfSpeech=\"n\"><SynsetRelation relType=\"hypernym\" target=\"b-1\"/></Synset>",
        ))
        .0,
        "references synset b-1, which is neither declared in the extension nor as an ExternalSynset",
    );
    assert_err_contains(
        compose(&extension(
            "<LexicalEntry id=\"x-a-n\"><Lemma writtenForm=\"a\" partOfSpeech=\"n\"/>\
             <Sense id=\"x-a-n-1\" synset=\"x-1\"><SenseRelation relType=\"antonym\" target=\"b-cat-n-1\"/></Sense>\
             </LexicalEntry><Synset id=\"x-1\" partOfSpeech=\"n\"/>",
        ))
        .0,
        "references sense b-cat-n-1, which is neither declared in the extension nor as an ExternalSense",
    );
    assert_err_contains(
        compose(&extension("<Synset id=\"x-1\" partOfSpeech=\"n\" members=\"b-cat-n\"/>")).0,
        "Synset x-1 at line 4 lists member b-cat-n, which needs a sense or entry declaration",
    );
    let (res, _) = compose(&extension(
        "<ExternalLexicalEntry id=\"b-cat-n\"/><Synset id=\"x-1\" partOfSpeech=\"n\" members=\"b-cat-n\"/>",
    ));
    let kb = res.unwrap().into_lexicons().remove(0).into_knowledge_base();
    assert_eq!(&["cat"], kb.synset("x-1").unwrap().lemmas());
}

#[test]
fn an_extension_that_extends_itself_is_a_cycle_found_without_resolving_anything() {
    let doc = "<LexicalResource><LexiconExtension id=\"x\" label=\"x\" language=\"en\" version=\"1\">\
               <Extends ref=\"x\" version=\"1\"/></LexiconExtension></LexicalResource>";
    let (result, calls) = compose(doc);
    assert_err_contains(result, "Extension dependency cycle: x 1 -> x 1");
    assert_eq!(0, calls, "the resolver must not be asked for the extension being composed");
}

#[test]
fn an_unresolved_base_is_a_not_found_error() {
    let doc = extension("").replace("ref=\"b\"", "ref=\"missing\"");
    let err = compose(&doc).0.unwrap_err();
    assert!(matches!(err, WordNetError::Io(ref e) if e.kind() == std::io::ErrorKind::NotFound), "{err:?}");
    assert!(err.to_string().contains("Unresolved base lexicon 'missing' version '1'"), "{err}");
}

#[test]
fn a_resolved_document_without_the_requested_version_is_rejected() {
    let doc = extension("").replace("version=\"1\"/>", "version=\"2\"/>");
    let calls = Cell::new(0usize);
    let resolver = |_: &WnLmfDependency| {
        calls.set(calls.get() + 1);
        Ok(Some(WnLmfSource::NamedBytes("base.xml".to_string(), BASE.as_bytes().to_vec())))
    };
    let result = WnLmfReader::read_resource_named_with_resolver(Cursor::new(doc.as_bytes()), "ext.xml", &resolver);
    assert_err_contains(result, "Resolved resource base.xml does not contain lexicon b version 2");
}
