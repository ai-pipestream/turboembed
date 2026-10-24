//! `DictionaryLemmatizer`: lemmas looked up by word and tag in a tab
//! separated dictionary.

use std::collections::HashMap;
use std::path::Path;

use turbo_analysis::{Annotator, Error, ModelRef, Refs, View};
use turbo_analysis_layers::{Lemmas, PosTags, Strs, Tokens};
use turbo_maxent::sha256;
use turbo_unicode::java::lowercase_cow;

use crate::error::DictionaryError;
use crate::maxent::words;

/// What `lemmatize` returns for a word and tag the dictionary lacks.
pub const NOT_FOUND: &str = "O";

/// A dictionary of `word<tab>tag<tab>lemma` lines, where the lemma field
/// may hold several lemmas separated by `#`, and the lemmatizer that looks
/// the lowercased word and its tag up in it.
///
/// Lines are cut as `BufferedReader.readLine` cuts them and fields as
/// `StringUtil.split` does: trailing empty fields are dropped, so a line
/// needs three fields up to its lemma and anything after the lemma is
/// ignored. A line with fewer is refused, as `DictionaryLemmatizer` throws
/// on it. A word listed twice with one tag keeps its last lemmas. The
/// word is stored as written, while a lookup lowercases the token with
/// `String.toLowerCase(Locale.ROOT)`, so an entry with a capital letter
/// is never found.
#[derive(Debug)]
pub struct DictionaryLemmatizer {
    entries: HashMap<String, HashMap<String, Vec<String>>>,
    model_ref: ModelRef,
}

impl DictionaryLemmatizer {
    /// Reads a dictionary file from disk; `name` goes into the model
    /// reference, as a model's language does.
    pub fn read(path: impl AsRef<Path>, name: &str) -> Result<DictionaryLemmatizer, Error> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .map_err(|e| Error::Annotator { annotator: Self::ID, detail: format!("{}: {e}", path.display()) })?;
        DictionaryLemmatizer::parse(&bytes, name)
            .map_err(|e| Error::Annotator { annotator: Self::ID, detail: format!("{}: {e}", path.display()) })
    }

    /// Reads a dictionary from its bytes, decoded as UTF-8 with malformed
    /// sequences replaced, as `InputStreamReader` decodes them.
    pub fn parse(bytes: &[u8], name: &str) -> Result<DictionaryLemmatizer, DictionaryError> {
        let text = String::from_utf8_lossy(bytes);
        let mut entries: HashMap<String, HashMap<String, Vec<String>>> = HashMap::new();
        for (n, line) in java_lines(&text).enumerate() {
            let fields = java_split(line, '\t');
            if fields.len() < 3 {
                return Err(DictionaryError { line: n + 1, fields: fields.len() });
            }
            let lemmas = java_split(fields[2], '#').into_iter().map(str::to_string).collect();
            entries.entry(fields[0].to_owned()).or_default().insert(fields[1].to_owned(), lemmas);
        }
        let model_ref = ModelRef {
            id: format!("opennlp:DictionaryLemmatizer:{name}"),
            revision: String::new(),
            sha256: sha256(bytes).to_hex(),
        };
        Ok(DictionaryLemmatizer { entries, model_ref })
    }

    /// How many word and tag pairs the dictionary lists.
    pub fn len(&self) -> usize {
        self.entries.values().map(HashMap::len).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Every lemma listed for the lowercased word and the tag, if any
    /// (`getAllLemmas` without its `O`).
    pub fn lemmas(&self, word: &str, tag: &str) -> Option<&[String]> {
        self.entries.get(lowercase_cow(word).as_ref())?.get(tag).map(Vec::as_slice)
    }

    /// `lemmatize(String, String)`: the first lemma listed for the
    /// lowercased word and the tag, or [`NOT_FOUND`]. An entry whose lemma
    /// field was empty (`#` alone) lists no lemma and gives [`NOT_FOUND`]
    /// as well.
    pub fn lemma<'s>(&'s self, word: &str, tag: &str) -> &'s str {
        self.lemmas(word, tag).and_then(|l| l.first()).map_or(NOT_FOUND, String::as_str)
    }

    /// The dictionary as provenance records it.
    pub fn model_ref(&self) -> &ModelRef {
        &self.model_ref
    }
}

/// `BufferedReader.readLine`: lines ended by LF, CR or CR LF, the last
/// one without an end if the text has one.
fn java_lines(text: &str) -> impl Iterator<Item = &str> {
    let mut rest = text;
    std::iter::from_fn(move || {
        if rest.is_empty() {
            return None;
        }
        let end = rest.find(['\n', '\r']).unwrap_or(rest.len());
        let line = &rest[..end];
        rest = &rest[end..];
        if rest.starts_with("\r\n") {
            rest = &rest[2..];
        } else if !rest.is_empty() {
            rest = &rest[1..];
        }
        Some(line)
    })
}

/// `StringUtil.split(input, separator)`: the fields around every
/// separator, with the trailing empty ones dropped; an empty input is one
/// empty field.
fn java_split(input: &str, separator: char) -> Vec<&str> {
    let mut fields: Vec<&str> = input.split(separator).collect();
    if input.is_empty() {
        return fields;
    }
    while fields.last().is_some_and(|f| f.is_empty()) {
        fields.pop();
    }
    fields
}

impl Annotator for DictionaryLemmatizer {
    const ID: &'static str = "turbo-lemmatizer:dictionary";
    const VERSION: &'static str = "2";
    type Reads = (View, Tokens, PosTags);
    type Writes = (Lemmas,);

    /// `lemmatize(String[], String[])` over the tokens with the tags'
    /// labels: a lemma per token, [`NOT_FOUND`] where the dictionary has
    /// none. Sentences do not matter to a lookup.
    fn annotate(&self, (view, tokens, tags): Refs<'_, Self>) -> Result<(Lemmas,), Error> {
        let words = words(Self::ID, view, tokens, tags)?;
        let mut lemmas = Strs::default();
        for (word, &tag) in words.iter().zip(&tags.tags) {
            lemmas.push(self.lemma(word, tags.tagset.label(tag)));
        }
        Ok((Lemmas { lemma: lemmas },))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.model_ref.clone())
    }
}

#[cfg(test)]
mod tests {
    use turbo_analysis::{Document, Span, Spans};
    use turbo_analysis_layers::{Tag, TagSet};

    use super::*;

    fn dictionary(text: &str) -> DictionaryLemmatizer {
        DictionaryLemmatizer::parse(text.as_bytes(), "test").unwrap()
    }

    #[test]
    fn looks_the_lowercased_word_and_the_tag_up() {
        let d = dictionary("dogs\tNNS\tdog\nbarking\tVBG\tbark\n");
        assert_eq!(d.len(), 2);
        assert_eq!(d.lemma("dogs", "NNS"), "dog");
        assert_eq!(d.lemma("DOGS", "NNS"), "dog");
        assert_eq!(d.lemma("dogs", "NN"), "O");
        assert_eq!(d.lemma("cats", "NNS"), "O");
        assert_eq!(d.lemmas("dogs", "NNS").unwrap(), ["dog"]);
        assert_eq!(d.lemmas("cats", "NNS"), None);
    }

    #[test]
    fn several_lemmas_are_separated_by_hash_and_the_first_is_the_lemma() {
        let d = dictionary("bears\tNNS\tbear#bears\nleft\tVBD\tleave#left#\nx\tN\t#");
        assert_eq!(d.lemmas("bears", "NNS").unwrap(), ["bear", "bears"]);
        assert_eq!(d.lemma("bears", "NNS"), "bear");
        // Trailing empty fields are dropped.
        assert_eq!(d.lemmas("left", "VBD").unwrap(), ["leave", "left"]);
        // A lemma field of only separators lists nothing.
        assert_eq!(d.lemmas("x", "N").unwrap(), Vec::<String>::new());
        assert_eq!(d.lemma("x", "N"), "O");
    }

    #[test]
    fn the_last_line_of_a_word_and_tag_wins_and_extra_fields_are_ignored() {
        let d = dictionary("went\tVERB\twalk\r\nwent\tVERB\tgo\textra\tfields\r\npower\tNOUN\tpower\t\t\n");
        assert_eq!(d.len(), 2);
        assert_eq!(d.lemma("went", "VERB"), "go");
        assert_eq!(d.lemma("power", "NOUN"), "power");
    }

    #[test]
    fn an_entry_with_a_capital_letter_is_never_found() {
        let d = dictionary("The\tDET\tthe\nu.s.\tPROPN\tUnited States");
        assert_eq!(d.lemma("The", "DET"), "O");
        assert_eq!(d.lemma("the", "DET"), "O");
        assert_eq!(d.lemma("U.S.", "PROPN"), "United States");
    }

    #[test]
    fn lowercasing_is_javas() {
        let d = dictionary("σοφος\tADJ\tσοφος\ni\u{307}s\tX\tdotted");
        assert_eq!(d.lemma("ΣΟΦΟΣ", "ADJ"), "σοφος");
        assert_eq!(d.lemma("İs", "X"), "dotted");
    }

    #[test]
    fn a_short_line_is_refused() {
        let e = DictionaryLemmatizer::parse(b"dogs\tNNS\tdog\nbad\tline\t\n", "t").unwrap_err();
        assert_eq!(e, DictionaryError { line: 2, fields: 2 });
        let e = DictionaryLemmatizer::parse(b"\n", "t").unwrap_err();
        assert_eq!(e, DictionaryError { line: 1, fields: 1 });
        assert!(DictionaryLemmatizer::parse(b"", "t").unwrap().is_empty());
    }

    #[test]
    fn java_lines_cut_at_every_line_end() {
        let lines: Vec<&str> = java_lines("a\nb\r\nc\rd\n\ne").collect();
        assert_eq!(lines, ["a", "b", "c", "d", "", "e"]);
        assert_eq!(java_lines("a\n").collect::<Vec<_>>(), ["a"]);
        assert_eq!(java_split("", '\t'), [""]);
        assert_eq!(java_split("\t", '\t'), Vec::<&str>::new());
        assert_eq!(java_split("a\t\tb\t", '\t'), ["a", "", "b"]);
    }

    #[test]
    fn annotate_rejects_out_of_bounds_token_spans_and_tags() {
        let d = dictionary("dogs\tNNS\tdog\n");
        let doc = Document::new("é");
        let view = doc.get::<View>().unwrap();

        let mut bad_span = Spans::new(view.id());
        bad_span.push(Span::new(0, 1));
        let tokens = Tokens { spans: bad_span, sentence: None };
        let tags = PosTags { tagset: TagSet::new("ptb", &["NNS"]), tags: vec![Tag(0)], prob: None };
        assert!(d.annotate((view, &tokens, &tags)).is_err());

        let mut ok_span = Spans::new(view.id());
        ok_span.push(Span::new(0, 2));
        let tokens = Tokens { spans: ok_span, sentence: None };
        let bad_tags = PosTags { tagset: TagSet::new("ptb", &["NNS"]), tags: vec![Tag(5)], prob: None };
        assert!(d.annotate((view, &tokens, &bad_tags)).is_err());
    }

    #[test]
    fn pipeline_records_validation_version() {
        use turbo_analysis::{Analyzer, Key, Provenance};
        let mut doc = Document::new("dogs");
        let mut spans = Spans::new(doc.get::<View>().unwrap().id());
        spans.push(Span::new(0, 4));
        doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("test")).unwrap();
        let tags = PosTags { tagset: TagSet::new("ptb", &["NNS"]), tags: vec![Tag(0)], prob: None };
        doc.add(&Key::DEFAULT, tags, Provenance::by("test")).unwrap();
        let analyzer = Analyzer::builder()
            .given(Key::<Tokens>::DEFAULT)
            .given(Key::<PosTags>::DEFAULT)
            .add(dictionary("dogs\tNNS\tdog\n"))
            .build()
            .unwrap();
        assert_eq!(analyzer.run(std::slice::from_mut(&mut doc)), [Ok(())]);
        let provenance = doc.provenance(Key::<Lemmas>::DEFAULT.id()).unwrap();
        assert_eq!(provenance.annotator, DictionaryLemmatizer::ID);
        assert_eq!(provenance.version, "2");
    }

    #[test]
    fn annotate_rejects_invalid_token_spans_without_panicking() {
        use turbo_analysis::{Document, Span, Spans, Tag, TagSet, ViewId};

        let d = dictionary("café\tNN\tcafé\n");
        let doc = Document::new("café");
        let view = doc.get::<View>().unwrap();

        let tags = PosTags { tagset: TagSet::new("test", &["NN"]), tags: vec![Tag(0)], prob: None };

        // Out-of-bounds span (end > text.len()).
        let mut oob_spans = Spans::new(ViewId::SOURCE);
        oob_spans.push(Span::new(0, 99));
        let oob_tokens = Tokens { spans: oob_spans, sentence: None };
        let err = d.annotate((view, &oob_tokens, &tags)).err().unwrap();
        assert!(err.to_string().contains("not a valid slice"), "unexpected error: {err}");

        // Mid-codepoint UTF-8 boundary ('é' is bytes 3..5 in "café").
        let mut mid_spans = Spans::new(ViewId::SOURCE);
        mid_spans.push(Span::new(0, 4));
        let mid_tokens = Tokens { spans: mid_spans, sentence: None };
        let err = d.annotate((view, &mid_tokens, &tags)).err().unwrap();
        assert!(err.to_string().contains("not a valid slice"), "unexpected error: {err}");
    }
}
