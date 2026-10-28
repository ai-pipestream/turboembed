//! The WNUT-17 Emerging Entities reader.

use super::{NerError, Sentence, Tag, bio_names};
use crate::error::FormatError;
use crate::sample::NameSample;
use crate::source::{Lines, split_fields};

/// The name types a WNUT-17 reader keeps.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Wnut17Types {
    pub person: bool,
    pub location: bool,
    pub group: bool,
    pub creative_work: bool,
    pub corporation: bool,
    pub product: bool,
}

impl Wnut17Types {
    /// Every type.
    pub const ALL: Wnut17Types = Wnut17Types {
        person: true,
        location: true,
        group: true,
        creative_work: true,
        corporation: true,
        product: true,
    };

    /// The types a reader keeps from its `types` parameter: each type
    /// whose short name (`per`, `loc`, `grp`, `work`, `corp`, `prod`) the
    /// value contains anywhere.
    pub fn from_param(types: &str) -> Wnut17Types {
        Wnut17Types {
            person: types.contains("per"),
            location: types.contains("loc"),
            group: types.contains("grp"),
            creative_work: types.contains("work"),
            corporation: types.contains("corp"),
            product: types.contains("prod"),
        }
    }

    fn keeps(self, tag: &str) -> bool {
        let starts =
            |kind: &str| tag.strip_prefix("B-").or_else(|| tag.strip_prefix("I-")).is_some_and(|rest| rest == kind);
        if starts("person") {
            self.person
        } else if starts("location") {
            self.location
        } else if starts("group") {
            self.group
        } else if starts("creative-work") {
            self.creative_work
        } else if starts("corporation") {
            self.corporation
        } else if starts("product") {
            self.product
        } else {
            tag == "O"
        }
    }

    fn filter(self, tag: &str) -> &str {
        if tag != "O" && !self.keeps(tag) { "O" } else { tag }
    }
}

impl Default for Wnut17Types {
    fn default() -> Wnut17Types {
        Wnut17Types::ALL
    }
}

fn name_type(kind: &str) -> Option<&'static str> {
    Some(match kind {
        "person" => "person",
        "location" => "location",
        "group" => "group",
        "creative-work" => "creative-work",
        "corporation" => "corporation",
        "product" => "product",
        _ => return None,
    })
}

/// Reads name samples from WNUT-17: two tab-separated columns, the token
/// and its BIO name tag, with an empty line between sentences.
///
/// ```text
/// Empire    B-location
/// State     I-location
/// Building  I-location
/// ```
#[derive(Clone, Debug)]
pub struct Wnut17Reader<'a> {
    lines: Lines<'a>,
    types: Wnut17Types,
}

impl<'a> Wnut17Reader<'a> {
    /// A reader of `source` that keeps the names of `types`.
    ///
    /// # Errors
    ///
    /// [`FormatError::TooLong`] for a source longer than spans can index.
    pub fn new(source: &'a str, types: Wnut17Types) -> Result<Wnut17Reader<'a>, FormatError> {
        Ok(Wnut17Reader { lines: Lines::new(source)?, types })
    }
}

impl<'a> Iterator for Wnut17Reader<'a> {
    type Item = Result<NameSample<'a>, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let mut s = Sentence::default();
            let mut at_end = false;
            loop {
                let Some(line) = self.lines.next() else {
                    at_end = true;
                    break;
                };
                if line.text().is_empty() {
                    break;
                }
                s.cover(&line);
                let fields = split_fields(line.piece, '\t');
                if fields.len() != 2 {
                    let text = line.text().to_owned();
                    return Some(Err(NerError::Wnut17Fields { line: line.number, count: fields.len(), text }.into()));
                }
                s.tokens.push(fields[0].field());
                s.tags.push(Tag { text: fields[1].text, line: line.number });
            }
            if !s.tokens.is_empty() {
                let types = self.types;
                let names = bio_names(
                    &s.tags,
                    |t| types.filter(t),
                    |k, line| name_type(k).ok_or_else(|| FormatError::UnknownNameType { line, kind: k.to_owned() }),
                );
                return s.sample(names);
            }
            if at_end {
                return None;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(source: &str, types: Wnut17Types) -> Vec<Result<String, String>> {
        Wnut17Reader::new(source, types).unwrap().map(|r| r.map(|s| s.to_string()).map_err(|e| e.to_string())).collect()
    }

    const TEXT: &str =
        "Empire\tB-location\nState\tI-location\nBuilding\tI-location\n.\tO\n\nAHFA\tB-group\nextends\tO\n";

    #[test]
    fn reads_names_and_offsets() {
        let samples = read(TEXT, Wnut17Types::ALL);
        assert_eq!(samples.len(), 2);
        let first = samples[0].as_ref().unwrap();
        assert!(first.contains("location"));
        assert!(first.contains("Empire"));
        assert!(first.contains("State"));
        assert!(first.contains("Building"));
    }

    #[test]
    fn filters_types() {
        let types = Wnut17Types { location: false, ..Wnut17Types::ALL };
        let samples = read(TEXT, types);
        assert_eq!(samples.len(), 2);
        let first = samples[0].as_ref().unwrap();
        assert!(!first.contains("location"));
        let second = samples[1].as_ref().unwrap();
        assert!(second.contains("group"));
    }

    #[test]
    fn rejects_wrong_field_count() {
        let err = read("Empire\n", Wnut17Types::ALL);
        assert_eq!(err.len(), 1);
        let message = err[0].as_ref().unwrap_err();
        assert!(message.contains("Expected two tab-separated fields"), "{message}");
        assert!(message.contains("Empire"), "{message}");
    }
}
