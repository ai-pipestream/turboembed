use turbo_analysis::{Document, Ix, Key, Layer, Provenance, Span, Spans, Tag, TagSet, ViewId};
use turbo_analysis_layers::{Sentences, Tokens, WordDependencies, Words};
use turbo_formats::FormatError;
use turbo_formats::conllu::{
    ConlluWordReader, SyntacticSentence, misc_spaces_in_token, misc_trailing_space, write_conllu_sentences,
    write_document_conllu,
};

#[test]
fn conllu_with_multiword_tokens_space_after_and_spaces_in_token_round_trips_byte_identically() {
    let src = "\
# newdoc id = ancora-es-1
# sent_id = 1
# text = Vamos al mar del norte, ¿no?
1\tVamos\tir\tVERB\t_\tMood=Ind|Number=Plur|Person=1|Tense=Pres|VerbForm=Fin\t0\troot\t_\t_
2-3\tal\t_\t_\t_\t_\t_\t_\t_\t_
2\ta\ta\tADP\t_\t_\t4\tcase\t_\t_
3\tel\tel\tDET\t_\tDefinite=Def|Gender=Masc|Number=Sing|PronType=Art\t4\tdet\t_\t_
4\tmar\tmar\tNOUN\t_\tGender=Com|Number=Sing\t1\tobl\t_\t_
5-6\tdel\t_\t_\t_\t_\t_\t_\t_\t_
5\tde\tde\tADP\t_\t_\t7\tcase\t_\t_
6\tel\tel\tDET\t_\tDefinite=Def|Gender=Masc|Number=Sing|PronType=Art\t7\tdet\t_\t_
7\tnorte\tnorte\tNOUN\t_\tGender=Masc|Number=Sing\t4\tnmod\t_\tSpaceAfter=No
8\t,\t,\tPUNCT\t_\t_\t10\tpunct\t_\t_
9\t¿\t¿\tPUNCT\t_\tPunctSide=Ini\t10\tpunct\t_\tSpaceAfter=No
10\tno\tno\tADV\t_\tPolarity=Neg\t1\tadvmod\t_\tSpaceAfter=No
11\t?\t?\tPUNCT\t_\tPunctSide=Fin\t1\tpunct\t_\tSpacesAfter=\\n

# sent_id = 2
# text = We can't wait.
1\tWe\twe\tPRON\tPRP\tCase=Nom|Number=Plur|Person=1|PronType=Prs\t4\tnsubj\t_\t_
2-3\tcan't\t_\t_\t_\t_\t_\t_\t_\t_
2\tca\tcan\tAUX\tMD\tVerbForm=Fin\t4\taux\t_\t_
3\tn't\tnot\tPART\tRB\tPolarity=Neg\t4\tadvmod\t_\t_
4\twait\twait\tVERB\tVB\tVerbForm=Inf\t0\troot\t_\tSpaceAfter=No
5\t.\t.\tPUNCT\t.\t_\t4\tpunct\t_\t_

";

    let sentences: Vec<SyntacticSentence<'_>> = ConlluWordReader::new(src).unwrap().map(Result::unwrap).collect();
    assert_eq!(sentences.len(), 2);

    // Byte-identical round-trip of the entire CoNLL-U file.
    let written = write_conllu_sentences(&sentences);
    assert_eq!(written, src);

    // Check MISC helpers.
    assert_eq!(misc_spaces_in_token("SpacesInToken=ca\\sn't").as_deref(), Some("ca n't"));
    assert_eq!(misc_trailing_space("SpaceAfter=No"), "");
    assert_eq!(misc_trailing_space("SpacesAfter=\\n"), "\n");

    // Convert sentence 1 (Spanish fusional MWTs `al` and `del`) into a Document.
    let doc1 = sentences[0].to_document().unwrap();
    let tok1 = doc1.get::<Tokens>().unwrap();
    let words1 = doc1.get::<Words>().unwrap();
    assert_eq!(tok1.len(), 9);
    assert_eq!(words1.len(), 11);
    assert!(words1.form.is_some());
    assert!(words1.subspan.is_none());

    let forms1: Vec<&str> = words1.iter(doc1.source(), tok1).map(|w| w.form).collect();
    assert_eq!(forms1, ["Vamos", "a", "el", "mar", "de", "el", "norte", ",", "¿", "no", "?"]);

    // Convert sentence 2 (English substring MWT `can't` -> `ca` + `n't`) into a Document.
    let doc2 = sentences[1].to_document().unwrap();
    let tok2 = doc2.get::<Tokens>().unwrap();
    let words2 = doc2.get::<Words>().unwrap();
    assert_eq!(tok2.len(), 4);
    assert_eq!(words2.len(), 5);
    assert!(words2.form.is_none());
    assert!(words2.subspan.is_some());
    assert_eq!(words2.subspan_of(Ix::new(1)), Some(Span::new(3, 5)));
    assert_eq!(words2.subspan_of(Ix::new(2)), Some(Span::new(5, 8)));

    // Write Document 1 back to CoNLL-U and verify multi-word ranges and SpaceAfter=No.
    let doc1_conllu = write_document_conllu(&doc1).unwrap();
    assert!(doc1_conllu.contains("2-3\tal\t_\t_\t_\t_\t_\t_\t_\t_\n"));
    assert!(doc1_conllu.contains("5-6\tdel\t_\t_\t_\t_\t_\t_\t_\t_\n"));
    assert!(doc1_conllu.contains("7\tnorte\tnorte\tNOUN\t_\tGender=Masc|Number=Sing\t4\tnmod\t_\tSpaceAfter=No\n"));
}

#[test]
fn a_head_outside_the_sentence_is_refused_rather_than_read_as_a_root() {
    let src = "\
# text = Go now
1\tGo\tgo\tVERB\t_\t_\t0\troot\t_\t_
2\tnow\tnow\tADV\t_\t_\t9\tadvmod\t_\t_

# text = Stay
1\tStay\tstay\tVERB\t_\t_\t1\troot\t_\t_

# text = Stay
1\tStay\tstay\tVERB\t_\t_\tx\troot\t_\t_

";
    let sentences: Vec<SyntacticSentence<'_>> = ConlluWordReader::new(src).unwrap().map(Result::unwrap).collect();
    for s in &sentences {
        let Err(err) = s.to_document() else { panic!("a bad head was accepted") };
        assert!(matches!(err, FormatError::ConlluHead { .. }), "{err}");
    }
}

#[test]
fn a_document_writes_the_gaps_between_its_tokens_and_refuses_a_head_across_sentences() {
    // Two sentences; the gap after "a" is two spaces, the gap after
    // "b" a line break with a pipe, and "c" abuts ".".
    let text = "a  b|\nc. d";
    let mut doc = Document::new(text);
    let mut sents = Spans::<Sentences>::new(ViewId::SOURCE);
    sents.push(Span::new(0, 8));
    sents.push(Span::new(9, 10));
    doc.add(&Key::DEFAULT, Sentences { spans: sents }, Provenance::by("test")).unwrap();
    let mut toks = Spans::<Tokens>::new(ViewId::SOURCE);
    for (a, b) in [(0, 1), (3, 4), (6, 7), (7, 8), (9, 10)] {
        toks.push(Span::new(a, b));
    }
    let sentence = vec![Ix::new(0), Ix::new(0), Ix::new(0), Ix::new(0), Ix::new(1)];
    doc.add(&Key::DEFAULT, Tokens { spans: toks, sentence: Some(sentence) }, Provenance::by("test")).unwrap();
    let written = write_document_conllu(&doc).unwrap();
    assert_eq!(
        written,
        "\
# sent_id = 1
# text = a  b| c.
1\ta\t_\t_\t_\t_\t_\t_\t_\tSpacesAfter=\\s\\s
2\tb\t_\t_\t_\t_\t_\t_\t_\tSpacesAfter=\\p\\n
3\tc\t_\t_\t_\t_\t_\t_\t_\tSpaceAfter=No
4\t.\t_\t_\t_\t_\t_\t_\t_\t_

# sent_id = 2
# text = d
1\td\t_\t_\t_\t_\t_\t_\t_\t_

"
    );
    // Read back, the spacing rebuilds the text of each sentence.
    let read: Vec<SyntacticSentence<'_>> = ConlluWordReader::new(&written).unwrap().map(Result::unwrap).collect();
    assert_eq!(read[0].reconstruct_surface_text(), "a  b|\nc.");

    let relations = TagSet::new("deprel", &["dep"]);
    let deps = WordDependencies {
        head: vec![None, Some(Ix::new(0)), None, None, Some(Ix::new(0))],
        relation: vec![Tag(0); 5],
        relations,
    };
    doc.add(&Key::DEFAULT, Words::from_tokens(doc.get::<Tokens>().unwrap()), Provenance::by("test")).unwrap();
    doc.add(&Key::DEFAULT, deps, Provenance::by("test")).unwrap();
    let err = write_document_conllu(&doc).unwrap_err();
    assert!(matches!(err, FormatError::HeadOutsideSentence { word: 4, head: 0 }), "{err}");
}
