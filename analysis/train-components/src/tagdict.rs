//! `POSDictionary`: the tag dictionary a part-of-speech model carries as
//! `tags.tagdict`, as a training run builds it and as `serialize` writes
//! it into the model file.
//!
//! The dictionary maps each word to its tags in deterministic insertion
//! order. `POSDictionary.create(InputStream)` puts every entry of the XML
//! into the dictionary, lowercasing each word (`StringUtil.toLowerCase`, per
//! code point) when the dictionary is not case sensitive. `serialize` writes
//! the entries in insertion order, each with its tags joined by single spaces
//! in a `tags` attribute, through the same serializer as the abbreviation
//! dictionary.
//!
//! The lookup side (`getTags`) that the tagger's sequence validator uses
//! is `turbo_sequence::postag::TagDictionary`, which reads the same XML.

use turbo_maxent::{Entries, JavaString};
use turbo_unicode::java::{lowercase_code_points as lowercase, lowercase_code_points_changes};

use crate::dictionary::{XmlEntry, write_entries_xml};
use crate::error::{Error, Result};

/// A tag dictionary: the words in the order the JVM's map first saw them
/// (lowercased when the dictionary is not case sensitive), each with the
/// tags it was last given.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PosDictionary {
    case_sensitive: bool,
    words: Vec<(String, Vec<String>)>,
}

impl PosDictionary {
    /// `new POSDictionary(caseSensitive)` followed by `put(word, tags)` for
    /// each entry: a word put twice keeps its place and takes the later
    /// tags; in a dictionary that is not case sensitive the word is
    /// lowercased first, so words that differ only in case are one entry.
    pub fn new(case_sensitive: bool, entries: impl IntoIterator<Item = (String, Vec<String>)>) -> PosDictionary {
        let mut d = PosDictionary { case_sensitive, words: Vec::new() };
        for (word, tags) in entries {
            let word = if case_sensitive { word } else { lowercase(&word) };
            d.put(word, tags);
        }
        d
    }

    /// `POSDictionary.create(InputStream)`. An entry of more than one token
    /// is refused as the JVM refuses it (`InvalidFormatException`,
    /// [`Error::TagDictionaryEntry`]), and so is an entry without a `tags`
    /// attribute (`StringUtil.split(null)` throws). The tags are the
    /// attribute split at single spaces as `StringUtil.split` splits it:
    /// an empty field between two spaces is an empty tag, trailing empty
    /// fields are dropped, and an empty attribute is one empty tag.
    pub fn read(xml: &[u8]) -> Result<PosDictionary> {
        let entries = Entries::read(xml).map_err(Error::DictionaryXml)?;
        let case_sensitive = entries.is_case_sensitive();
        let mut dict = PosDictionary { case_sensitive, words: Vec::new() };
        for entry in entries.iter() {
            let [word] = entry.tokens.as_slice() else {
                let shown: Vec<&str> = entry.tokens.iter().map(|t| &**t).collect();
                return Err(Error::TagDictionaryEntry(format!(
                    "Each entry must have exactly one token! [{}]",
                    shown.join(",")
                )));
            };
            let Some(list) = entry.attribute("tags") else {
                return Err(Error::TagDictionaryEntry(format!("the entry {word:?} has no tags attribute")));
            };
            let key = if case_sensitive { word.to_string() } else { lowercase(word) };
            dict.put(key, java_split(list, ' ').into_iter().map(str::to_owned).collect());
        }
        Ok(dict)
    }

    /// `HashMap.put` on the word as given.
    fn put(&mut self, word: String, tags: Vec<String>) {
        match self.words.iter_mut().find(|(w, _)| *w == word) {
            Some(entry) => entry.1 = tags,
            None => self.words.push((word, tags)),
        }
    }

    /// `isCaseSensitive`.
    pub fn is_case_sensitive(&self) -> bool {
        self.case_sensitive
    }

    pub fn len(&self) -> usize {
        self.words.len()
    }

    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    /// The words as the map holds them, in the order it first saw them,
    /// each with its tags.
    pub fn words(&self) -> impl Iterator<Item = (&str, &[String])> {
        self.words.iter().map(|(w, t)| (w.as_str(), t.as_slice()))
    }

    /// `getTags`: the tags of `word`, lowercased first when the dictionary
    /// is not case sensitive.
    pub fn tags(&self, word: &str) -> Option<&[String]> {
        let lowered;
        let key = if self.case_sensitive || !lowercase_code_points_changes(word) {
            word
        } else {
            lowered = lowercase(word);
            &lowered
        };
        self.words.iter().find(|(w, _)| w == key).map(|(_, t)| t.as_slice())
    }

    /// Every tag the dictionary uses, each once, in deterministic first-seen
    /// order across the words.
    pub fn all_tags(&self) -> Result<Vec<&str>> {
        let mut seen: Vec<&str> = Vec::new();
        for i in self.java_order()? {
            for t in &self.words[i].1 {
                if !seen.contains(&t.as_str()) {
                    seen.push(t);
                }
            }
        }
        Ok(seen)
    }

    /// `validatePOSDictionary`: the tags the model has no outcome for, in
    /// [`all_tags`](Self::all_tags) order. `POSModel`'s constructor refuses a
    /// dictionary with any: an `InvalidFormatException` that
    /// `BaseModel.checkArtifactMap` wraps in an `IllegalArgumentException`.
    pub fn tags_unknown_to(&self, outcomes: &[JavaString]) -> Result<Vec<String>> {
        Ok(self.all_tags()?.into_iter().filter(|t| !outcomes.iter().any(|o| o == t)).map(str::to_owned).collect())
    }

    /// The entries in deterministic first-seen insertion order.
    pub fn java_order(&self) -> Result<Vec<usize>> {
        Ok((0..self.words.len()).collect())
    }

    /// `POSDictionary.serialize`: the bytes of the `tags.tagdict` entry.
    pub fn serialize(&self) -> Result<Vec<u8>> {
        let order = self.java_order()?;
        write_entries_xml(
            self.case_sensitive,
            order.iter().map(|&i| {
                let (word, tags) = &self.words[i];
                XmlEntry { tokens: std::slice::from_ref(word), tags: Some(tags) }
            }),
        )
    }
}

/// `StringUtil.split(input, separator)`: the fields between separators
/// with trailing empty fields removed; an empty input is one empty field.
fn java_split(input: &str, separator: char) -> Vec<&str> {
    if input.is_empty() {
        return vec![""];
    }
    let mut fields: Vec<&str> = input.split(separator).collect();
    while fields.last().is_some_and(|f| f.is_empty()) {
        fields.pop();
    }
    fields
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tags(list: &[&str]) -> Vec<String> {
        list.iter().map(|t| t.to_string()).collect()
    }

    fn outcomes(list: &[&str]) -> Vec<JavaString> {
        list.iter().map(|t| JavaString::new(t)).collect()
    }

    /// The fixture `analysis/goldens/data/fixtures/tags.tagdict`.
    const FIXTURE: &[u8] = b"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<dictionary case_sensitive=\"false\">\n\
        <entry tags=\"DT\"><token>the</token></entry>\n<entry tags=\"NN VB VBP\"><token>Run</token></entry>\n\
        <entry tags=\"VBD VBN\"><token>fell</token></entry>\n<entry tags=\"NNP\"><token>Paris</token></entry>\n\
        </dictionary>\n";

    #[test]
    fn reading_lowercases_and_serializes_in_insertion_order() {
        let d = PosDictionary::read(FIXTURE).unwrap();
        assert!(!d.is_case_sensitive());
        assert_eq!(d.len(), 4);
        assert_eq!(d.tags("RUN").unwrap(), ["NN", "VB", "VBP"]);
        assert_eq!(d.tags("run").unwrap(), ["NN", "VB", "VBP"]);
        assert_eq!(d.tags("paris").unwrap(), ["NNP"]);
        assert_eq!(d.tags("London"), None);
        assert_eq!(d.all_tags().unwrap(), ["DT", "NN", "VB", "VBP", "VBD", "VBN", "NNP"]);
        assert_eq!(d.tags_unknown_to(&outcomes(&["DT", "NN", "VB", "VBP", "VBD", "VBN"])).unwrap(), ["NNP"]);
        assert_eq!(d.tags_unknown_to(&outcomes(&["DT", "NNP"])).unwrap(), ["NN", "VB", "VBP", "VBD", "VBN"]);
        let xml = String::from_utf8(d.serialize().unwrap()).unwrap();
        assert_eq!(
            xml,
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?><dictionary case_sensitive=\"false\">\n\
             \x20   <entry tags=\"DT\">\n        <token>the</token>\n    </entry>\n\
             \x20   <entry tags=\"NN VB VBP\">\n        <token>run</token>\n    </entry>\n\
             \x20   <entry tags=\"VBD VBN\">\n        <token>fell</token>\n    </entry>\n\
             \x20   <entry tags=\"NNP\">\n        <token>paris</token>\n    </entry>\n</dictionary>\n"
        );
        // The entry reads back as the tagger's dictionary.
        let lookup = turbo_sequence::postag::TagDictionary::read(xml.as_bytes()).unwrap();
        assert_eq!(lookup.tags("Run").unwrap(), ["NN", "VB", "VBP"]);
    }

    #[test]
    fn put_keeps_the_place_of_a_word_and_takes_its_later_tags() {
        let d = PosDictionary::new(
            false,
            [
                ("Dog".to_owned(), tags(&["NN"])),
                ("cat".to_owned(), tags(&["NN"])),
                ("DOG".to_owned(), tags(&["NN", "VB"])),
            ],
        );
        let words: Vec<(&str, &[String])> = d.words().collect();
        assert_eq!(words.len(), 2);
        assert_eq!(words[0], ("dog", &tags(&["NN", "VB"])[..]));
        let d = PosDictionary::new(true, [("Dog".to_owned(), tags(&["NN"])), ("dog".to_owned(), tags(&["VB"]))]);
        assert_eq!(d.len(), 2);
        assert_eq!(d.tags("Dog").unwrap(), ["NN"]);
        assert_eq!(d.tags("DOG"), None);
        assert!(PosDictionary::new(true, Vec::new()).is_empty());
    }

    #[test]
    fn case_variants_preserve_first_normalized_position_and_last_tags() {
        let xml_ci = b"<dictionary case_sensitive=\"false\">\
            <entry tags=\"FIRST\"><token>word</token></entry>\
            <entry tags=\"KEEP\"><token>Other</token></entry>\
            <entry tags=\"MIDDLE\"><token>WORD</token></entry>\
            <entry tags=\"LAST\"><token>word</token></entry>\
            </dictionary>";
        let from_read = PosDictionary::read(xml_ci).unwrap();
        let from_new = PosDictionary::new(
            false,
            [
                ("word".to_owned(), tags(&["FIRST"])),
                ("Other".to_owned(), tags(&["KEEP"])),
                ("WORD".to_owned(), tags(&["MIDDLE"])),
                ("word".to_owned(), tags(&["LAST"])),
            ],
        );
        assert_eq!(from_read, from_new);
        let words_ci: Vec<(&str, &[String])> = from_read.words().collect();
        assert_eq!(words_ci, [("word", &tags(&["LAST"])[..]), ("other", &tags(&["KEEP"])[..])]);
        assert_eq!(from_read.tags("word").unwrap(), ["LAST"]);
        assert_eq!(from_read.tags("WORD").unwrap(), ["LAST"]);

        let serialized_ci = from_read.serialize().unwrap();
        let roundtrip_ci = PosDictionary::read(&serialized_ci).unwrap();
        assert_eq!(roundtrip_ci, from_read);
        let lookup_ci = turbo_sequence::postag::TagDictionary::read(&serialized_ci).unwrap();
        assert_eq!(lookup_ci.tags("word").unwrap(), ["LAST"]);
        assert_eq!(lookup_ci.tags("WORD").unwrap(), ["LAST"]);

        // Case-sensitive counterpart: "word" and "WORD" remain distinct entries,
        // each keeping its first position and taking its own last tags.
        let xml_cs = b"<dictionary case_sensitive=\"true\">\
            <entry tags=\"FIRST\"><token>word</token></entry>\
            <entry tags=\"KEEP\"><token>Other</token></entry>\
            <entry tags=\"MIDDLE\"><token>WORD</token></entry>\
            <entry tags=\"LAST\"><token>word</token></entry>\
            </dictionary>";
        let cs_read = PosDictionary::read(xml_cs).unwrap();
        let cs_new = PosDictionary::new(
            true,
            [
                ("word".to_owned(), tags(&["FIRST"])),
                ("Other".to_owned(), tags(&["KEEP"])),
                ("WORD".to_owned(), tags(&["MIDDLE"])),
                ("word".to_owned(), tags(&["LAST"])),
            ],
        );
        assert_eq!(cs_read, cs_new);
        let words_cs: Vec<(&str, &[String])> = cs_read.words().collect();
        assert_eq!(
            words_cs,
            [("word", &tags(&["LAST"])[..]), ("Other", &tags(&["KEEP"])[..]), ("WORD", &tags(&["MIDDLE"])[..]),]
        );
        let serialized_cs = cs_read.serialize().unwrap();
        let roundtrip_cs = PosDictionary::read(&serialized_cs).unwrap();
        assert_eq!(roundtrip_cs, cs_read);
        let lookup_cs = turbo_sequence::postag::TagDictionary::read(&serialized_cs).unwrap();
        assert_eq!(lookup_cs.tags("word").unwrap(), ["LAST"]);
        assert_eq!(lookup_cs.tags("WORD").unwrap(), ["MIDDLE"]);
    }

    #[test]
    fn tags_split_as_string_util_splits_and_bad_entries_are_refused() {
        let xml = b"<dictionary case_sensitive=\"true\"><entry tags=\"A  B \"><token>x</token></entry>\
            <entry tags=\"\"><token>y</token></entry></dictionary>";
        let d = PosDictionary::read(xml).unwrap();
        assert_eq!(d.tags("x").unwrap(), ["A", "", "B"]);
        assert_eq!(d.tags("y").unwrap(), [""]);
        assert_eq!(java_split(" ", ' '), Vec::<&str>::new());
        let two = b"<dictionary><entry tags=\"A\"><token>a</token><token>b</token></entry></dictionary>";
        assert!(matches!(
            PosDictionary::read(two),
            Err(Error::TagDictionaryEntry(m)) if m == "Each entry must have exactly one token! [a,b]"
        ));
        let none = b"<dictionary><entry><token>a</token></entry></dictionary>";
        assert!(matches!(PosDictionary::read(none), Err(Error::TagDictionaryEntry(_))));
    }

    #[test]
    fn an_empty_dictionary_is_one_element_and_attributes_are_escaped() {
        let d = PosDictionary::new(true, Vec::new());
        assert_eq!(
            String::from_utf8(d.serialize().unwrap()).unwrap(),
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?><dictionary case_sensitive=\"true\"/>\n"
        );
        let d = PosDictionary::new(true, [("a&b".to_owned(), tags(&["x\"y", "<z>"]))]);
        let xml = String::from_utf8(d.serialize().unwrap()).unwrap();
        assert!(xml.contains("<entry tags=\"x&quot;y &lt;z&gt;\">\n        <token>a&amp;b</token>"), "{xml}");
    }

    #[test]
    fn a_lookup_lowercases_per_code_point_as_java_does() {
        let d = PosDictionary::new(false, [("Straße".to_owned(), vec!["NN".to_owned()])]);
        assert_eq!(d.words().next().unwrap().0, "straße");
        assert_eq!(d.tags("STRAßE").unwrap(), ["NN"]);
        // `Character.toLowerCase` maps a code point to one code point, so
        // the full case folding of "SS" to "ß" is not applied.
        assert_eq!(d.tags("STRASSE"), None);
        assert_eq!(d.tags("strasse"), None);
        let sensitive = PosDictionary::new(true, [("Run".to_owned(), vec!["VB".to_owned()])]);
        assert_eq!(sensitive.tags("Run").unwrap(), ["VB"]);
        assert_eq!(sensitive.tags("run"), None);
    }
}
