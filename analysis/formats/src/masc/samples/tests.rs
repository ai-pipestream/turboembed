use super::*;
use crate::masc::document::MascFiles;

/// A document of three sentences, `One two.`, `Three four.` and
/// `Five six.`, whose middle sentence has no Penn tokens.
fn corpus_with_an_empty_sentence() -> MascCorpus {
    let text = "One two. Three four. Five six.";
    let quarks = [(0, 3), (4, 7), (7, 8), (9, 14), (15, 19), (19, 20), (21, 25), (26, 29), (29, 30)];
    let mut seg = String::from("<graph>");
    for (i, (a, b)) in quarks.iter().enumerate() {
        seg.push_str(&format!("<region xml:id=\"seg-r{i}\" anchors=\"{a} {b}\"/>"));
    }
    seg.push_str("</graph>");
    let mut s = String::from("<graph>");
    for (i, (a, b)) in [(0, 8), (9, 20), (21, 30)].iter().enumerate() {
        s.push_str(&format!(
            "<region xml:id=\"s-r{i}\" anchors=\"{a} {b}\"/><node xml:id=\"s-n{i}\"><link targets=\"s-r{i}\"/></node>"
        ));
    }
    s.push_str("</graph>");
    let mut penn = String::from("<graph>");
    for i in [0, 1, 2, 6, 7, 8] {
        penn.push_str(&format!(
            "<node xml:id=\"penn-n{i}\"><link targets=\"seg-r{i}\"/></node>\
             <a label=\"tok\" ref=\"penn-n{i}\"><fs><f name=\"msd\" value=\"NN\"/></fs></a>"
        ));
    }
    penn.push_str("</graph>");
    let files = MascFiles {
        text: text.as_bytes(),
        segments: Some(seg.as_bytes()),
        sentences: Some(s.as_bytes()),
        penn: Some(penn.as_bytes()),
        named_entities: None,
    };
    let doc = MascDocument::parse("/corpus/doc.hdr", files).unwrap();
    // The same document twice, so the sentence after the empty one in
    // the second document shows where the read stops.
    MascCorpus::from_documents(vec![doc.clone(), doc])
}

fn texts(reader: MascTokenReader<'_>) -> Vec<(usize, String)> {
    reader.map(|s| s.unwrap()).map(|s| (s.document, s.sample.text().to_owned())).collect()
}

#[test]
fn an_empty_sentence_ends_the_corpus_as_in_opennlp_or_is_skipped_alone() {
    let corpus = corpus_with_an_empty_sentence();
    let opennlp = MascTokenReader::new(&corpus).unwrap();
    assert_eq!(texts(opennlp), [(0, "One two.".to_owned())]);

    let mut reader = MascTokenReader::with_empty_sentences(&corpus, EmptySentence::Skipped).unwrap();
    let mut read = Vec::new();
    while let Some(s) = reader.read().unwrap() {
        read.push((s.document, s.sample.text().to_owned()));
    }
    assert_eq!(
        read,
        [(0, "One two.".into()), (0, "Five six.".into()), (1, "One two.".into()), (1, "Five six.".into())]
    );
    assert_eq!(reader.skipped(), 2);
    reader.reset();
    assert_eq!(reader.skipped(), 0);
    assert_eq!(texts(reader).len(), 4);
}

#[test]
fn sentence_reader_with_zero_per_sample_returns_none_even_on_empty_corpus() {
    let empty = MascCorpus::from_documents(Vec::new());
    let mut reader = MascSentenceReader::new(&empty, 0);
    assert!(reader.read().unwrap().is_none());

    let corpus = corpus_with_an_empty_sentence();
    let mut reader = MascSentenceReader::new(&corpus, 0);
    assert!(reader.read().unwrap().is_none());
}
