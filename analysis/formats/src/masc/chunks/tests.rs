use super::*;
use crate::masc::MascFiles;

/// A document of `sentences` (words separated by spaces, each a quark
/// and a Penn token numbered from 0 across the document, tagged `NN`
/// unless the word is `?`, which has no tag), with these chunk files.
fn document(sentences: &[&str], noun: Option<&str>, verb: Option<&str>) -> Result<MascDocument, MascError> {
    let (mut text, mut seg, mut s, mut penn) = (String::new(), String::new(), String::new(), String::new());
    let mut q = 0;
    for sentence in sentences {
        if !text.is_empty() {
            text.push_str("  ");
        }
        let start = text.len();
        for (k, word) in sentence.split(' ').enumerate() {
            if k > 0 {
                text.push(' ');
            }
            let a = text.len();
            text.push_str(word);
            seg.push_str(&format!(r#"<region xml:id="seg-r{q}" anchors="{a} {}"/>"#, text.len()));
            let msd = if word == "?" { String::new() } else { r#"<f name="msd" value="NN"/>"#.to_owned() };
            penn.push_str(&format!(
                r#"<node xml:id="penn-n{q}"><link targets="seg-r{q}"/></node><a ref="penn-n{q}"><fs>{msd}</fs></a>"#
            ));
            q += 1;
        }
        s.push_str(&format!(r#"<region anchors="{start} {}"/>"#, text.len()));
    }
    let graph = |inner: &str| format!("<graph>{inner}</graph>");
    let (seg, s, penn) = (graph(&seg), graph(&s), graph(&penn));
    let mut doc = MascDocument::parse(
        "t.hdr",
        MascFiles {
            text: text.as_bytes(),
            segments: Some(seg.as_bytes()),
            sentences: Some(s.as_bytes()),
            penn: Some(penn.as_bytes()),
            named_entities: None,
        },
    )?;
    doc.add_chunks(MascChunkFiles { noun: noun.map(str::as_bytes), verb: verb.map(str::as_bytes) })?;
    Ok(doc)
}

/// A chunk file of `kind` with a chunk per list of Penn token
/// numbers, its edges written last to first as MASC writes them.
fn chunks(kind: MascChunkKind, lists: &[&[i32]]) -> String {
    let p = &kind.node_prefix()[..2];
    let mut xml = String::from("<graph>");
    for (n, _) in lists.iter().enumerate() {
        xml.push_str(&format!(r#"<node xml:id="{p}-n{n}"/><a label="{}" ref="{p}-n{n}"/>"#, kind.label()));
    }
    for (n, tokens) in lists.iter().enumerate().rev() {
        for t in tokens.iter().rev() {
            xml.push_str(&format!(r#"<edge from="{p}-n{n}" to="penn-n{t}"/>"#));
        }
    }
    xml.push_str("</graph>");
    xml
}

fn spans(chunks: &[MascChunk]) -> Vec<(MascChunkKind, i32, Vec<u32>)> {
    chunks.iter().map(|c| (c.kind, c.id, c.tokens().to_vec())).collect()
}

use MascChunkKind::{Noun, Verb};

#[test]
fn chunks_are_placed_in_their_sentence_as_places_of_its_tokens() {
    let nc = chunks(Noun, &[&[0, 1], &[3], &[5, 6, 6]]);
    let vc = chunks(Verb, &[&[2], &[7]]);
    let doc = document(&["the dog barks at cats", "a cat runs ."], Some(&nc), Some(&vc)).unwrap();
    assert!(doc.has_chunks());
    assert!(doc.warnings().is_empty(), "{:?}", doc.warnings());
    let [first, second] = doc.sentences() else { panic!("two sentences") };
    assert_eq!(spans(first.chunks()), [(Noun, 0, vec![0, 1]), (Verb, 0, vec![2]), (Noun, 1, vec![3])]);
    // the second sentence's tokens are 5 to 8; an edge given twice is
    // one token
    assert_eq!(spans(second.chunks()), [(Noun, 2, vec![0, 1]), (Verb, 1, vec![2])]);
    assert!(first.partial_chunks().is_empty() && second.partial_chunks().is_empty());
    let c = &first.chunks()[0];
    assert_eq!((c.start(), c.end(), c.is_contiguous()), (0, 2, true));
}

#[test]
fn a_chunk_without_tokens_outside_the_sentences_or_across_two_is_left_out() {
    // chunk 0 has no edge (MASC links two noun chunks to a region of
    // the text instead), chunk 1 names a token that is not there,
    // chunk 2 crosses from the first sentence into the second
    let nc = r#"<graph><node xml:id="nc-n0"><link targets="nc-r0"/></node><a label="nchunk" ref="nc-n0"/>
        <edge from="nc-n1" to="penn-n1"/><edge from="nc-n1" to="penn-n40"/>
        <edge from="nc-n2" to="penn-n2"/><edge from="nc-n2" to="penn-n3"/></graph>"#;
    let doc = document(&["a b c", "d e"], Some(nc), None).unwrap();
    assert_eq!(
        doc.warnings(),
        [
            MascWarning::ChunkWithoutTokens { kind: Noun, chunk: 0 },
            MascWarning::ChunkOutsideSentences { kind: Noun, chunk: 1, token: 40 },
            MascWarning::ChunkCrossesSentences { kind: Noun, chunk: 2 },
        ]
    );
    let [first, second] = doc.sentences() else { panic!("two sentences") };
    assert!(first.chunks().is_empty() && second.chunks().is_empty());
    // each sentence lists the chunks it has part of, with its tokens
    assert_eq!(spans(first.partial_chunks()), [(Noun, 1, vec![1]), (Noun, 2, vec![2])]);
    assert_eq!(spans(second.partial_chunks()), [(Noun, 2, vec![0])]);
}

#[test]
fn a_chunk_file_with_another_label_or_ids_is_refused() {
    let refused = |nc: &str| {
        let err = document(&["a b"], Some(nc), None).unwrap_err();
        assert!(matches!(err, MascError::Annotation { file: AnnotationFile::NounChunks, .. }), "{err:?}");
        assert_eq!(err.java_class(), None);
        assert_eq!(err.to_string(), "Could not parse the noun chunk annotation file");
        assert!(err.skips_document());
    };
    refused(r#"<graph><node xml:id="nc-n0"/><a label="vchunk" ref="nc-n0"/></graph>"#);
    refused(r#"<graph><node xml:id="nc-n0"/><a ref="nc-n0"/></graph>"#);
    refused(r#"<graph><node xml:id="vc-n0"/></graph>"#);
    refused(r#"<graph><edge from="nc-n0" to="seg-r0"/></graph>"#);
    refused(r#"<graph><edge from="nc-nx" to="penn-n0"/></graph>"#);
    refused("<graph>");
    let err = document(&["a b"], None, Some(r#"<graph><a label="nchunk" ref="vc-n0"/></graph>"#)).unwrap_err();
    assert_eq!(err.to_string(), "Could not parse the verb chunk annotation file");

    // A failed second call to `add_chunks` clears previously placed chunks.
    let mut doc = document(&["a b"], Some(&chunks(Noun, &[&[0]])), None).unwrap();
    assert!(doc.has_chunks());
    assert!(doc.add_chunks(MascChunkFiles { noun: Some(b"<graph>"), verb: None }).is_err());
    assert!(!doc.has_chunks());
    assert!(doc.sentences()[0].chunks().is_empty());
}

#[test]
fn a_document_without_penn_tokens_or_chunk_files_has_no_chunks() {
    let seg = br#"<graph><region xml:id="seg-r0" anchors="0 1"/></graph>"#;
    let s = br#"<graph><region anchors="0 1"/></graph>"#;
    let files = MascFiles { text: b"a", segments: Some(seg), sentences: Some(s), penn: None, named_entities: None };
    let mut doc = MascDocument::parse("t.hdr", files).unwrap();
    let nc = chunks(Noun, &[&[0]]);
    doc.add_chunks(MascChunkFiles { noun: Some(nc.as_bytes()), verb: None }).unwrap();
    assert!(!doc.has_chunks());
    assert_eq!(doc.warnings(), [MascWarning::ChunksWithoutTokens]);
    let doc = document(&["a b"], None, None).unwrap();
    assert!(!doc.has_chunks());
    assert!(matches!(MascChunkReader::new(&MascCorpus::from_documents(vec![doc])), Err(MascError::NoChunks)));
    assert_eq!(MascError::NoChunks.java_class(), None);
}

/// The chunk tags of each sentence the reader reads, or why it
/// skipped it.
fn read(doc: MascDocument) -> Vec<Result<Vec<String>, ChunkSkip>> {
    let corpus = MascCorpus::from_documents(vec![doc]);
    MascChunkReader::new(&corpus)
        .unwrap()
        .map(|r| match r.unwrap() {
            MascChunkRead::Sample(s) => Ok(s.sample.chunks().iter().map(|c| c.as_str().to_owned()).collect()),
            MascChunkRead::Skipped { reason, .. } => Err(reason),
        })
        .collect()
}

fn tags(tags: &str) -> Result<Vec<String>, ChunkSkip> {
    Ok(tags.split(' ').map(str::to_owned).collect())
}

#[test]
fn the_reader_writes_b_and_i_tags_of_np_and_vp_and_o_elsewhere() {
    let nc = chunks(Noun, &[&[0, 1], &[5, 6], &[7]]);
    let vc = chunks(Verb, &[&[2, 3], &[8]]);
    let doc = document(&["the dog has barked at the cats", "it ran"], Some(&nc), Some(&vc)).unwrap();
    let corpus = MascCorpus::from_documents(vec![doc]);
    let samples: Vec<_> = MascChunkReader::new(&corpus).unwrap().map(Result::unwrap).collect();
    let MascChunkRead::Sample(first) = &samples[0] else { panic!("a sample") };
    let doc = &corpus.documents()[0];
    assert_eq!((first.document, first.sentence), (0, 0));
    let s = &first.sample;
    let words: Vec<&str> = s.tokens().iter().map(|t| t.as_str()).collect();
    assert_eq!(words, ["the", "dog", "has", "barked", "at", "the", "cats"]);
    let chunk_tags: Vec<&str> = s.chunks().iter().map(|t| t.as_str()).collect();
    assert_eq!(chunk_tags, ["B-NP", "I-NP", "B-VP", "I-VP", "O", "B-NP", "I-NP"]);
    // every field has the span of its token, whose bytes it is
    for ((token, tag), chunk) in s.tokens().iter().zip(s.tags()).zip(s.chunks()) {
        assert_eq!(&doc.text()[token.span.range()], token.as_str());
        assert_eq!((tag.as_str(), tag.span, chunk.span), ("NN", token.span, token.span));
    }
    assert_eq!(&doc.text()[s.source().range()], "the dog has barked at the cats");
    assert_eq!(read(doc.clone())[1], tags("B-NP B-VP"));
}

#[test]
fn a_sentence_whose_chunks_tags_cannot_write_is_skipped() {
    let doc = |nc: &[&[i32]], vc: &[&[i32]]| {
        document(&["a b c", "d e"], Some(&chunks(Noun, nc)), Some(&chunks(Verb, vc))).unwrap()
    };
    // part of a chunk that crosses into the next sentence
    assert_eq!(read(doc(&[&[2, 3]], &[])), [Err(ChunkSkip::PartialChunk), Err(ChunkSkip::PartialChunk)]);
    // a noun chunk inside a verb chunk, and two chunks of other kinds
    // over the same tokens
    assert_eq!(read(doc(&[&[1]], &[&[0, 1], &[4]]))[0], Err(ChunkSkip::OverlappingChunks));
    assert_eq!(read(doc(&[&[4]], &[&[4]]))[1], Err(ChunkSkip::OverlappingChunks));
    assert_eq!(read(doc(&[&[0, 1], &[1, 2]], &[]))[0], Err(ChunkSkip::OverlappingChunks));
    // two chunks of one kind over the same tokens are one chunk
    assert_eq!(read(doc(&[&[0, 1], &[0, 1]], &[&[3], &[3]])), [tags("B-NP I-NP O"), tags("B-VP O")]);
    // a chunk whose tokens are not next to each other
    assert_eq!(read(doc(&[&[0, 2]], &[]))[0], Err(ChunkSkip::ChunkWithGap));
    // a token without a tag
    let untagged = document(&["a ? b", "c"], Some(&chunks(Noun, &[&[3]])), None).unwrap();
    assert_eq!(read(untagged), [Err(ChunkSkip::NullTag), tags("B-NP")]);
}

#[test]
fn out_of_bounds_token_anchors_error_even_when_sentence_has_null_tag_or_partial_chunk() {
    let seg = br#"<graph><region xml:id="seg-r0" anchors="0 50"/></graph>"#;
    let s = br#"<graph><region anchors="0 50"/></graph>"#;
    let penn = br#"<graph><node xml:id="penn-n0"><link targets="seg-r0"/></node><a ref="penn-n0"><fs/></a></graph>"#;
    let mut doc = MascDocument::parse(
        "t.hdr",
        MascFiles { text: b"a", segments: Some(seg), sentences: Some(s), penn: Some(penn), named_entities: None },
    )
    .unwrap();
    let nc = chunks(Noun, &[&[0]]);
    doc.add_chunks(MascChunkFiles { noun: Some(nc.as_bytes()), verb: None }).unwrap();
    let corpus = MascCorpus::from_documents(vec![doc]);
    let res = MascChunkReader::new(&corpus).unwrap().next().unwrap();
    assert!(matches!(res, Err(MascError::Substring { begin: 0, end: 50, len: 1 })), "{res:?}");
}

#[test]
fn a_sentence_of_no_tokens_is_skipped() {
    // the second sentence's quark has no Penn token
    let seg = br#"<graph><region xml:id="seg-r0" anchors="0 1"/><region xml:id="seg-r1" anchors="3 4"/></graph>"#;
    let s = br#"<graph><region anchors="0 1"/><region anchors="3 4"/></graph>"#;
    let penn = br#"<graph><node xml:id="penn-n0"><link targets="seg-r0"/></node><a ref="penn-n0"><fs><f name="msd" value="DT"/></fs></a></graph>"#;
    let files =
        MascFiles { text: b"a  b", segments: Some(seg), sentences: Some(s), penn: Some(penn), named_entities: None };
    let mut doc = MascDocument::parse("t.hdr", files).unwrap();
    let nc = chunks(Noun, &[&[0]]);
    doc.add_chunks(MascChunkFiles { noun: Some(nc.as_bytes()), verb: None }).unwrap();
    assert_eq!(read(doc), [tags("B-NP"), Err(ChunkSkip::NoTokens)]);
    assert_eq!(ChunkSkip::NoTokens.to_string(), "a sentence of no tokens");
}

#[test]
fn a_corpus_read_with_chunks_reads_the_chunk_files_its_headers_name() {
    use crate::masc::{ChunkFiles, read_document_with};
    let dir = std::env::temp_dir().join(format!("turbo-formats-masc-chunks-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let files = [
        ("d.txt", "a b".to_owned()),
        (
            "d-seg.xml",
            r#"<graph><region xml:id="seg-r0" anchors="0 1"/><region xml:id="seg-r1" anchors="2 3"/></graph>"#
                .to_owned(),
        ),
        ("d-s.xml", r#"<graph><region anchors="0 3"/></graph>"#.to_owned()),
        (
            "d-penn.xml",
            (0..2)
                .map(|q| {
                    format!(
                        r#"<node xml:id="penn-n{q}"><link targets="seg-r{q}"/></node><a ref="penn-n{q}"><fs><f name="msd" value="NN"/></fs></a>"#
                    )
                })
                .collect::<String>(),
        ),
        ("d-nc.xml", chunks(Noun, &[&[0]])),
        ("d-vc.xml", chunks(Verb, &[&[1]])),
    ];
    for (name, content) in &files {
        let content = if name.ends_with("penn.xml") { format!("<graph>{content}</graph>") } else { content.clone() };
        std::fs::write(dir.join(name), content).unwrap();
    }
    let header = |ids: &[(&str, &str)]| {
        let annotations: String =
            ids.iter().map(|(id, loc)| format!(r#"<annotation f.id="{id}" loc="{loc}"/>"#)).collect();
        format!(r#"<documentHeader><primaryData f.id="f.text" loc="d.txt"/>{annotations}</documentHeader>"#)
    };
    let base = [("f.seg", "d-seg.xml"), ("f.s", "d-s.xml"), ("f.penn", "d-penn.xml")];
    let all = [base.as_slice(), &[("f.nc", "d-nc.xml"), ("f.vc", "d-vc.xml")]].concat();
    std::fs::write(dir.join("d.hdr"), header(&all)).unwrap();
    // as OpenNLP, the plain read leaves the chunk files alone
    let plain = MascCorpus::read(&dir).unwrap();
    assert!(!plain.documents()[0].has_chunks());
    assert!(plain.documents()[0].sentences()[0].chunks().is_empty());
    let corpus = MascCorpus::read_with_chunks(&dir).unwrap();
    let read: Vec<_> = MascChunkReader::new(&corpus).unwrap().map(Result::unwrap).collect();
    let [MascChunkRead::Sample(s)] = &read[..] else { panic!("one sample: {read:?}") };
    assert_eq!(s.sample.chunks().iter().map(|c| c.as_str()).collect::<Vec<_>>(), ["B-NP", "B-VP"]);
    // a header that names only the noun chunks
    std::fs::write(dir.join("d.hdr"), header(&[base.as_slice(), &[("f.nc", "d-nc.xml")]].concat())).unwrap();
    let doc = read_document_with(&std::path::absolute(dir.join("d.hdr")).unwrap(), ChunkFiles::Read).unwrap();
    assert_eq!(spans(doc.sentences()[0].chunks()), [(Noun, 0, vec![0])]);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn the_kinds_name_their_files_labels_and_phrases() {
    assert_eq!(MascChunkKind::ALL.map(MascChunkKind::label), ["nchunk", "vchunk"]);
    assert_eq!(MascChunkKind::ALL.map(MascChunkKind::phrase), ["NP", "VP"]);
    assert_eq!(MascChunkKind::ALL.map(|k| k.file().id()), ["f.nc", "f.vc"]);
}
