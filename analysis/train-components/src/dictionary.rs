//! The token n-gram dictionary as a training run or dictionary builder tool
//! holds and writes it: OpenNLP's `Dictionary`, a `HashSet<StringList>`,
//! `Dictionary.parseOneEntryPerLine` (`DictionaryBuilderTool`),
//! `CensusDictionaryCreatorTool.createDictionary`, and `Dictionary.serialize`,
//! which `DictionarySerializer` calls to write the `abbreviations.dictionary`
//! entry of a model file or a standalone dictionary XML file.
//!
//! `turbo_maxent::Dictionary` is the lookup side, and a case-insensitive
//! one keeps its entries lowercased; the JVM writes each entry as it was
//! read, in the order its hash set iterates. This type keeps both: the
//! entries with their casing, in insertion order, and the lookup
//! dictionary the context generators and `DictionaryNameFinder` are given.
//!
//! The XML is what the JDK's XSLT serializer (`com.sun.org.apache.xml.
//! internal.serializer.ToStream`) writes for the SAX events
//! `DictionaryEntryPersistor.serialize` sends with `INDENT` on: the
//! declaration with no line break after it, four spaces of indentation per
//! level, `&amp;`, `&lt;` and `&gt;` for the three characters it escapes
//! in text, `&#13;` for a carriage return, and a line feed after the root
//! element's end tag. An empty dictionary is one self-closing element. A
//! token holding a character XML 1.0 cannot carry makes the JDK serializer
//! throw, so the JVM cannot write such a dictionary; [`Error::InvalidXmlCharacter`]
//! stands for that.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use turbo_formats::Lines;
use turbo_formats::ner::CensusReader;
use turbo_maxent::{Dictionary, Entries};
use turbo_unicode::java::code_point::to_lower_case;

use crate::error::{Error, Result};
use crate::java_hash::{hash_map_order, string_list_hash_code};

/// A dictionary's entries with their casing and insertion order, and the
/// lookup dictionary made from them.
#[derive(Debug, Clone)]
pub struct AbbreviationDictionary {
    case_sensitive: bool,
    entries: Vec<Vec<String>>,
    lookup: Dictionary,
}

impl AbbreviationDictionary {
    /// `new Dictionary(InputStream)`: the entries of the XML in document
    /// order. The constructor puts every entry before it learns the
    /// `case_sensitive` attribute, so entries that differ only in case
    /// collapse to the first whatever the attribute says.
    pub fn read(xml: &[u8]) -> Result<AbbreviationDictionary> {
        let read = Entries::read(xml).map_err(|e| match e {
            turbo_maxent::Error::Dictionary { message: "an entry has no tokens", .. } => Error::EmptyEntry,
            other => Error::DictionaryXml(other),
        })?;
        let mut entries: Vec<Vec<String>> = Vec::new();
        let mut seen: HashSet<Vec<String>> = HashSet::new();
        for entry in read.iter() {
            if entry.tokens.is_empty() {
                return Err(Error::EmptyEntry);
            }
            let tokens: Vec<String> = entry.tokens.iter().map(|t| t.to_string()).collect();
            if seen.insert(fold_key(&tokens)) {
                entries.push(tokens);
            }
        }
        AbbreviationDictionary::assemble(read.is_case_sensitive(), entries)
    }

    /// `new Dictionary(caseSensitive)` followed by `put` of each entry:
    /// equal entries (by case when case-sensitive, ignoring case otherwise)
    /// collapse to the first. An entry with no token is refused as the
    /// JVM refuses it ([`Error::EmptyEntry`]); a case-sensitive dictionary
    /// with two entries that differ only in case is refused
    /// ([`Error::CaseVariants`]).
    pub fn new(case_sensitive: bool, entries: impl IntoIterator<Item = Vec<String>>) -> Result<AbbreviationDictionary> {
        let mut kept: Vec<Vec<String>> = Vec::new();
        let mut exact: HashSet<Vec<String>> = HashSet::new();
        // Each folded key and the index in `kept` of the entry that has it.
        let mut folded: HashMap<Vec<String>, usize> = HashMap::new();
        for tokens in entries {
            if tokens.is_empty() {
                return Err(Error::EmptyEntry);
            }
            let key = fold_key(&tokens);
            if case_sensitive {
                if exact.contains(&tokens) {
                    continue;
                }
                if let Some(&first) = folded.get(&key) {
                    return Err(Error::CaseVariants { first: kept[first].join(" "), second: tokens.join(" ") });
                }
                exact.insert(tokens.clone());
            } else if folded.contains_key(&key) {
                continue;
            }
            folded.insert(key, kept.len());
            kept.push(tokens);
        }
        AbbreviationDictionary::assemble(case_sensitive, kept)
    }

    /// `Dictionary.parseOneEntryPerLine(Reader)` (`DictionaryBuilderTool`):
    /// reads a case-insensitive dictionary with one entry per line (`\n`,
    /// `\r\n` or `\r`), tokens inside each line delimited by ASCII `' '`
    /// (`new StringTokenizer(line, " ")`), and lines with no tokens skipped.
    /// Non-space whitespace such as `\t` is kept inside tokens.
    pub fn parse_one_entry_per_line(source: &str) -> Result<AbbreviationDictionary> {
        let mut entries: Vec<Vec<String>> = Vec::new();
        for line in Lines::new(source)? {
            let tokens: Vec<String> = line.text().split(' ').filter(|s| !s.is_empty()).map(str::to_string).collect();
            if !tokens.is_empty() {
                entries.push(tokens);
            }
        }
        AbbreviationDictionary::new(false, entries)
    }

    /// `CensusDictionaryCreatorTool.createDictionary(new NameFinderCensus90NameStream(...))`:
    /// reads a 1990 US Census name list with [`CensusReader`] and builds a
    /// case-sensitive dictionary of single-token recased names, stopping at
    /// the first empty line or line without a space.
    ///
    /// Two names that recase to entries differing only in case (possible
    /// only when the list is not all upper case, such as `McDONALD` and
    /// `MCDONALD`) are refused with [`Error::CaseVariants`], as
    /// [`new`](Self::new) refuses them, where the JVM would keep both.
    pub fn from_census_names(source: &str) -> Result<AbbreviationDictionary> {
        let mut entries: Vec<Vec<String>> = Vec::new();
        for field in CensusReader::new(source)? {
            let field = field?;
            entries.push(vec![field.as_str().to_string()]);
        }
        AbbreviationDictionary::new(true, entries)
    }

    /// The lookup side of these entries. A token the JDK serializer could
    /// not write is refused here, so a dictionary that exists can always
    /// be written into a model file.
    fn assemble(case_sensitive: bool, entries: Vec<Vec<String>>) -> Result<AbbreviationDictionary> {
        for token in entries.iter().flatten() {
            if let Some(c) = token.chars().find(|&c| !xml_can_carry(c)) {
                return Err(Error::InvalidXmlCharacter(c as u32));
            }
        }
        let lookup = Dictionary::from_entries(case_sensitive, entries.iter().map(Vec::as_slice));
        Ok(AbbreviationDictionary { case_sensitive, entries, lookup })
    }

    pub fn is_case_sensitive(&self) -> bool {
        self.case_sensitive
    }

    /// The entries as they were read, in insertion order.
    pub fn entries(&self) -> &[Vec<String>] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The dictionary the context generators look abbreviations up in.
    pub fn lookup(&self) -> &Dictionary {
        &self.lookup
    }

    /// The entries in the order `Dictionary.iterator()` gives them: that
    /// of the `HashSet<StringList>`, by `StringList.hashCode`.
    pub fn java_order(&self) -> Result<Vec<&[String]>> {
        let hashes: Vec<i32> =
            self.entries.iter().map(|e| string_list_hash_code(e.iter().map(String::as_str))).collect();
        Ok(hash_map_order(&hashes)?.into_iter().map(|i| self.entries[i].as_slice()).collect())
    }

    /// `Dictionary.serialize`: the bytes of the `abbreviations.dictionary`
    /// entry.
    pub fn serialize(&self) -> Result<Vec<u8>> {
        write_xml(self.case_sensitive, self.java_order()?)
    }
}

/// `DictionaryBuilderTool`: reads one entry per line with
/// [`AbbreviationDictionary::parse_one_entry_per_line`] and serializes the
/// dictionary to XML bytes identical to `Dictionary.serialize`.
pub fn build_dictionary_xml(source: &str) -> Result<Vec<u8>> {
    AbbreviationDictionary::parse_one_entry_per_line(source)?.serialize()
}

/// `CensusDictionaryCreatorTool`: reads a 1990 US Census name list with
/// [`AbbreviationDictionary::from_census_names`] and serializes the
/// dictionary to XML bytes identical to `Dictionary.serialize`.
pub fn build_census_dictionary_xml(source: &str) -> Result<Vec<u8>> {
    AbbreviationDictionary::from_census_names(source)?.serialize()
}

/// The key two entries share when `StringList.compareToIgnoreCase` says
/// they are equal: the same number of tokens, each lowercased per code
/// point. (`String.compareToIgnoreCase` lowercases the upper case of each
/// `char`, which agrees with this for every character but a few with
/// irregular mappings, the same approximation `turbo_maxent::Dictionary`
/// makes.) The tokens stay separate, so an entry of one token holding a
/// tab and an entry of the two tokens around it are different, as
/// `StringList.equals` says.
fn fold_key(tokens: &[String]) -> Vec<String> {
    tokens.iter().map(|t| t.chars().map(|c| char::from_u32(to_lower_case(c as u32)).unwrap_or(c)).collect()).collect()
}

/// Whether XML 1.0 character data can hold `c`: everything but a C0
/// control other than tab, line feed and carriage return, U+FFFE and
/// U+FFFF (`ToStream` throws on the others).
fn xml_can_carry(c: char) -> bool {
    !(((c as u32) < 0x20 && !matches!(c, '\t' | '\n' | '\r')) || matches!(c, '\u{FFFE}' | '\u{FFFF}'))
}

/// The XML `DictionaryEntryPersistor.serialize` writes for these entries
/// in this order, none with attributes.
fn write_xml<'e>(case_sensitive: bool, entries: impl IntoIterator<Item = &'e [String]>) -> Result<Vec<u8>> {
    write_entries_xml(case_sensitive, entries.into_iter().map(|tokens| XmlEntry { tokens, tags: None }))
}

/// An entry as `DictionaryEntryPersistor.serialize` is given it: its
/// tokens, and for a tag dictionary its tags, which `POSDictionary` joins
/// with single spaces into the `tags` attribute of its `entry` element.
pub(crate) struct XmlEntry<'e> {
    pub tokens: &'e [String],
    pub tags: Option<&'e [String]>,
}

/// The XML `DictionaryEntryPersistor.serialize` writes for these entries
/// in this order: the JDK's `ToStream` with indentation, so a dictionary
/// with entries is `<dictionary case_sensitive="...">`, each entry on its
/// own lines indented by four and its tokens by eight spaces, and an empty
/// dictionary is one self-closing element.
pub(crate) fn write_entries_xml<'e>(
    case_sensitive: bool,
    entries: impl IntoIterator<Item = XmlEntry<'e>>,
) -> Result<Vec<u8>> {
    let mut out = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?><dictionary case_sensitive=\"");
    out.push_str(if case_sensitive { "true" } else { "false" });
    let mut any = false;
    for XmlEntry { tokens, tags } in entries {
        if tokens.is_empty() {
            return Err(Error::EmptyEntry);
        }
        if !any {
            out.push_str("\">\n");
            any = true;
        }
        out.push_str("    <entry");
        if let Some(tags) = tags {
            out.push_str(" tags=\"");
            for (i, tag) in tags.iter().enumerate() {
                if i > 0 {
                    out.push(' ');
                }
                push_attribute(&mut out, tag)?;
            }
            out.push('"');
        }
        out.push_str(">\n");
        for token in tokens {
            out.push_str("        <token>");
            push_text(&mut out, token)?;
            out.push_str("</token>\n");
        }
        out.push_str("    </entry>\n");
    }
    out.push_str(if any { "</dictionary>\n" } else { "\"/>\n" });
    Ok(out.into_bytes())
}

/// An attribute value as `ToStream.writeAttrString` writes it: like
/// character data, with `"` as `&quot;` and a tab, line feed or carriage
/// return as a character reference, since an XML parser would normalize
/// them to spaces otherwise.
fn push_attribute(out: &mut String, value: &str) -> Result<()> {
    for c in value.chars() {
        match c {
            '"' => out.push_str("&quot;"),
            '\t' => out.push_str("&#9;"),
            '\n' => out.push_str("&#10;"),
            c => push_text(out, c.encode_utf8(&mut [0; 4]))?,
        }
    }
    Ok(())
}

/// Character data as `ToStream.characters` writes it.
fn push_text(out: &mut String, text: &str) -> Result<()> {
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '\r' => out.push_str("&#13;"),
            '\t' | '\n' => out.push(c),
            c if !xml_can_carry(c) => return Err(Error::InvalidXmlCharacter(c as u32)),
            // The JDK serializer writes U+007F, the C1 controls U+0080 to
            // U+009F, and a character outside the Basic Multilingual Plane
            // as decimal character references.
            c if (0x7F..=0x9F).contains(&(c as u32)) || (c as u32) > 0xFFFF => {
                write!(out, "&#{};", c as u32).expect("writing to a String");
            }
            c => out.push(c),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
