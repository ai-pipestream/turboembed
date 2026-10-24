use turbo_analysis::{Annotator, Document, Error, Span, Spans, Tag, TagSet, View};
use turbo_analysis_layers::{PosTags, Tokens};
use turbo_morfologik::{
    BaseFormEncoding, DictionaryCharset, MorfologikDictionary, MorfologikDictionaryBuilder, MorfologikDictionaryReader,
    MorfologikLemmatizer, PoliMorfDictionaryReader, serialize_fsa5,
};

const INFO: &str = "fsa.dict.separator=+\nfsa.dict.encoding=UTF-8\nfsa.dict.encoder=NONE\n";

#[test]
fn annotators_refuse_invalid_pos_ids_in_direct_calls() {
    let bytes = MorfologikDictionaryBuilder::compile_bytes(b"dog+dogs+NNS\n", INFO.as_bytes()).unwrap();
    let compiled = MorfologikLemmatizer::from_bytes(&bytes, INFO.as_bytes()).unwrap();
    let expanded = MorfologikDictionaryReader::read(&bytes, INFO.as_bytes()).unwrap();
    let doc = Document::new("dogs");
    let view = doc.get::<View>().unwrap();
    let mut spans = Spans::new(view.id());
    spans.push(Span::new(0, 4));
    let tokens = Tokens { spans, sentence: None };
    for id in [1, 123, u16::MAX] {
        let tags = PosTags { tagset: TagSet::new("pos", &["NNS"]), tags: vec![Tag(id)], prob: None };
        for result in [compiled.annotate((view, &tokens, &tags)), expanded.annotate((view, &tokens, &tags))] {
            assert!(matches!(result, Err(Error::Annotator { ref detail, .. }) if detail.contains("outside tag set")));
        }
    }
    let tags = PosTags { tagset: TagSet::new("pos", &["NNS"]), tags: vec![Tag(0)], prob: None };
    assert_eq!(compiled.annotate((view, &tokens, &tags)).unwrap().0.lemma.get(0), "dog");
    assert_eq!(expanded.annotate((view, &tokens, &tags)).unwrap().0.lemma.get(0), "dog");
}

#[test]
fn compiled_and_expanded_identities_include_metadata() {
    let bytes = MorfologikDictionaryBuilder::compile_bytes(b"dog+dogs+NNS\n", INFO.as_bytes()).unwrap();
    let compiled = MorfologikDictionary::from_bytes(&bytes, INFO.as_bytes()).unwrap();
    let expanded = MorfologikDictionaryReader::read(&bytes, INFO.as_bytes()).unwrap();
    let conversion = format!("{INFO}fsa.dict.input-conversion=cats dogs\n");
    let other = MorfologikDictionary::from_bytes(&bytes, conversion.as_bytes()).unwrap();
    assert_eq!(compiled.first_lemma("cats", "NNS"), None);
    assert_eq!(other.first_lemma("cats", "NNS"), Some("dog".into()));
    assert_ne!(compiled.model_ref().sha256, other.model_ref().sha256);
    let other = MorfologikDictionaryReader::read(&bytes, conversion.as_bytes()).unwrap();
    assert_eq!(other.lemma("cats", "NNS"), "dog");
    assert_ne!(expanded.model_ref().sha256, other.model_ref().sha256);
    for info in [INFO.replace("=+", "=|"), INFO.replace("UTF-8", "ISO-8859-1"), INFO.replace("NONE", "SUFFIX")] {
        let other = MorfologikDictionary::from_bytes(&bytes, info.as_bytes()).unwrap();
        assert_ne!(compiled.model_ref().sha256, other.model_ref().sha256);
    }
    assert_eq!(compiled.model_ref(), MorfologikDictionary::from_bytes(&bytes, INFO.as_bytes()).unwrap().model_ref());
}

#[test]
fn reader_options_and_edits_change_identity() {
    let bytes = serialize_fsa5(&[], b'+');
    let read =
        |sep, encoder, charset| MorfologikDictionaryReader::read_with_options(&bytes, sep, encoder, charset).unwrap();
    let base = read(b'+', BaseFormEncoding::None, DictionaryCharset::Utf8);
    for other in [
        read(b'|', BaseFormEncoding::None, DictionaryCharset::Utf8),
        read(b'+', BaseFormEncoding::Suffix, DictionaryCharset::Utf8),
        read(b'+', BaseFormEncoding::None, DictionaryCharset::Iso8859_1),
    ] {
        assert_ne!(base.model_ref().sha256, other.model_ref().sha256);
    }
    let mut dict = PoliMorfDictionaryReader::read(b"dogs\tdog\tNNS\n").unwrap();
    let latin = PoliMorfDictionaryReader::read_with_charset(b"dogs\tdog\tNNS\n", DictionaryCharset::Iso8859_1).unwrap();
    assert_ne!(dict.model_ref().sha256, latin.model_ref().sha256);
    let before = dict.model_ref().sha256.clone();
    dict.put("dogs", "NNS", vec!["hound".into()]);
    assert_ne!(before, dict.model_ref().sha256);
    assert_eq!(dict.lemma("dogs", "NNS"), "hound");
    assert_eq!(dict.len(), 1);
    dict.put("dogs", "NN", vec!["canine".into()]);
    assert_eq!(dict.len(), 2);
    assert_eq!(dict.lemma("dogs", "NN"), "canine");
}

#[test]
fn first_match_agrees_with_full_lookup_for_encoders_and_charsets() {
    for charset in [DictionaryCharset::Utf8, DictionaryCharset::Iso8859_1, DictionaryCharset::Windows1252] {
        for encoder in ["NONE", "SUFFIX", "PREFIX", "INFIX"] {
            let info = format!(
                "fsa.dict.separator=+\nfsa.dict.encoding={}\nfsa.dict.encoder={encoder}\n",
                charset.canonical_name()
            );
            let table = charset.encode("café+cafés+N\ncafés+cafés+N\ncat+cats+N\ncast+cats+V\n").unwrap();
            let bytes = MorfologikDictionaryBuilder::compile_bytes(&table, info.as_bytes()).unwrap();
            let dict = MorfologikDictionary::from_bytes(&bytes, info.as_bytes()).unwrap();
            for word in ["cafés", "cats", "missing", "cats+", ""] {
                for tag in ["N", "V", "missing", ""] {
                    let expected =
                        dict.lookup(word).into_iter().find(|d| d.tag.as_deref() == Some(tag)).map(|d| d.stem);
                    assert_eq!(dict.first_lemma(word, tag), expected, "{charset:?} {encoder} {word} {tag}");
                }
            }
        }
    }
}
