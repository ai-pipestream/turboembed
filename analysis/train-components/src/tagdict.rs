//! `POSDictionary`: the tag dictionary a part-of-speech model carries as
//! `tags.tagdict`, as a training run builds it and as `serialize` writes
//! it into the model file.
//!
//! The dictionary is a `HashMap<String, String[]>` from a word to its
//! tags. `POSDictionary.create(InputStream)` puts every entry of the XML
//! into the map as written, and then, when the dictionary is not case
//! sensitive, walks that map and puts each word lowercased
//! (`StringUtil.toLowerCase`, per code point) into a second map, so the
//! insertion order of the final map is the first map's iteration order,
//! not the document order. `put(word, tags)` lowercases the word first
//! when the dictionary is not case sensitive. `serialize` writes the
//! entries in the final map's iteration order, each with its tags joined
//! by single spaces in a `tags` attribute, through the same serializer as
//! the abbreviation dictionary. Both map orders are simulated
//! ([`crate::java_hash`]), so the entry written here is the JVM's byte for
//! byte (`tests/fixtures.rs` checks `pos-fixture.bin`).
//!
//! The lookup side (`getTags`) that the tagger's sequence validator uses
//! is `turbo_sequence::postag::TagDictionary`, which reads the same XML.

use turbo_maxent::Entries;
use turbo_unicode::java::to_lower_char;

use crate::dictionary::{XmlEntry, write_entries_xml};
use crate::error::{Error, Result};
use crate::java_hash::{hash_map_order, str_hash_code};

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
        let mut as_written = PosDictionary { case_sensitive: true, words: Vec::new() };
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
            as_written.put(word.to_string(), java_split(list, ' ').into_iter().map(str::to_owned).collect());
        }
        if entries.is_case_sensitive() {
            return Ok(as_written);
        }
        let mut lowered = PosDictionary { case_sensitive: false, words: Vec::new() };
        for i in as_written.java_order()? {
            let (word, tags) = &as_written.words[i];
            lowered.put(lowercase(word), tags.clone());
        }
        Ok(lowered)
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
        let key = if self.case_sensitive {
            word
        } else {
            lowered = lowercase(word);
            &lowered
        };
        self.words.iter().find(|(w, _)| w == key).map(|(_, t)| t.as_slice())
    }

    /// Every tag the dictionary uses, each once, in the order
    /// `validatePOSDictionary` meets them: a `HashSet<String>` filled from
    /// the words in the dictionary's order, iterated in hash order.
    pub fn all_tags(&self) -> Result<Vec<&str>> {
        let mut seen: Vec<&str> = Vec::new();
        for i in self.java_order()? {
            for t in &self.words[i].1 {
                if !seen.contains(&t.as_str()) {
                    seen.push(t);
                }
            }
        }
        let hashes: Vec<i32> = seen.iter().map(|t| str_hash_code(t)).collect();
        Ok(hash_map_order(&hashes)?.into_iter().map(|i| seen[i]).collect())
    }

    /// `validatePOSDictionary`: the tags the model has no outcome for, in
    /// [`all_tags`](Self::all_tags) order, which is the order the JVM's
    /// message lists them in. `POSModel`'s constructor refuses a dictionary
    /// with any (`InvalidFormatException`).
    pub fn tags_unknown_to(&self, outcomes: &[String]) -> Result<Vec<String>> {
        Ok(self.all_tags()?.into_iter().filter(|t| !outcomes.iter().any(|o| o == t)).map(str::to_owned).collect())
    }

    /// The entries in the order `iterator()` and `serialize` give them:
    /// the map's, by `String.hashCode` of the words.
    pub fn java_order(&self) -> Result<Vec<usize>> {
        let hashes: Vec<i32> = self.words.iter().map(|(w, _)| str_hash_code(w)).collect();
        hash_map_order(&hashes)
    }

    /// `POSDictionary.serialize`: the bytes of the `tags.tagdict` entry.
    pub fn serialize(&self) -> Result<Vec<u8>> {
        // Each entry's one token and its `tags` attribute, the tags joined
        // with spaces as `DictionaryEntryPersistor` joins them.
        let order = self.java_order()?;
        let tokens: Vec<[String; 1]> = order.iter().map(|&i| [self.words[i].0.clone()]).collect();
        let attributes: Vec<[(String, String); 1]> =
            order.iter().map(|&i| [("tags".to_owned(), self.words[i].1.join(" "))]).collect();
        write_entries_xml(
            self.case_sensitive,
            tokens.iter().zip(&attributes).map(|(tokens, attributes)| XmlEntry { tokens, attributes }),
        )
    }
}

/// `StringUtil.toLowerCase`: `Character.toLowerCase` of each code point.
fn lowercase(word: &str) -> String {
    word.chars().map(to_lower_char).collect()
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

    /// The fixture `analysis/sequence/golden/resources/tags.tagdict`.
    const FIXTURE: &[u8] = b"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<dictionary case_sensitive=\"false\">\n\
        <entry tags=\"DT\"><token>the</token></entry>\n<entry tags=\"NN VB VBP\"><token>Run</token></entry>\n\
        <entry tags=\"VBD VBN\"><token>fell</token></entry>\n<entry tags=\"NNP\"><token>Paris</token></entry>\n\
        </dictionary>\n";

    #[test]
    fn reading_lowercases_in_the_first_maps_order_and_serializes_in_the_seconds() {
        let d = PosDictionary::read(FIXTURE).unwrap();
        assert!(!d.is_case_sensitive());
        assert_eq!(d.len(), 4);
        assert_eq!(d.tags("RUN").unwrap(), ["NN", "VB", "VBP"]);
        assert_eq!(d.tags("paris").unwrap(), ["NNP"]);
        assert_eq!(d.tags("London"), None);
        // The order of a `HashSet` of the tags: buckets 0 (DT, then NN), 1
        // (NNP), 3 (VBN), 5 (VBP), 9 (VBD) and 12 (VB) of a table of 16.
        assert_eq!(d.all_tags().unwrap(), ["DT", "NN", "NNP", "VBN", "VBP", "VBD", "VB"]);
        assert_eq!(d.tags_unknown_to(&tags(&["DT", "NN", "VB", "VBP", "VBD", "VBN"])).unwrap(), ["NNP"]);
        assert_eq!(d.tags_unknown_to(&tags(&["DT", "NNP"])).unwrap(), ["NN", "VBN", "VBP", "VBD", "VB"]);
        // What the JVM wrote into pos-fixture.bin from this file.
        let xml = String::from_utf8(d.serialize().unwrap()).unwrap();
        assert_eq!(
            xml,
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?><dictionary case_sensitive=\"false\">\n\
             \x20   <entry tags=\"DT\">\n        <token>the</token>\n    </entry>\n\
             \x20   <entry tags=\"VBD VBN\">\n        <token>fell</token>\n    </entry>\n\
             \x20   <entry tags=\"NNP\">\n        <token>paris</token>\n    </entry>\n\
             \x20   <entry tags=\"NN VB VBP\">\n        <token>run</token>\n    </entry>\n</dictionary>\n"
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
    fn case_variants_collapse_in_hash_order_when_read() {
        // "run" (hash 113291) and "Run" (hash 82539) lowercase to one word.
        // Both hashes fold to bucket 10 of a table of 16, so the first map
        // iterates them in insertion order and the second put, "Run"'s,
        // gives the tags. A later word in the file wins here, where the
        // sequence crate's reader (`TagDictionary::read`) agrees; the
        // order is the hash order, not the file's, and the fixture test
        // of the serialized bytes covers a file where the two differ.
        let xml = b"<dictionary case_sensitive=\"false\"><entry tags=\"A\"><token>run</token></entry>\
            <entry tags=\"B\"><token>Run</token></entry></dictionary>";
        let d = PosDictionary::read(xml).unwrap();
        assert_eq!(d.len(), 1);
        assert_eq!(d.tags("run").unwrap(), ["B"]);
        let lookup = turbo_sequence::postag::TagDictionary::read(xml).unwrap();
        assert_eq!(lookup.tags("run").unwrap(), ["B"]);
        // A word repeated as written keeps its first place and last tags.
        let xml = b"<dictionary case_sensitive=\"true\"><entry tags=\"A\"><token>x</token></entry>\
            <entry tags=\"B\"><token>y</token></entry><entry tags=\"C\"><token>x</token></entry></dictionary>";
        let d = PosDictionary::read(xml).unwrap();
        let words: Vec<(&str, &[String])> = d.words().collect();
        assert_eq!(words, [("x", &tags(&["C"])[..]), ("y", &tags(&["B"])[..])]);
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
}
