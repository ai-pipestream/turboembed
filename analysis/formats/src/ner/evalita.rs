//! The Evalita reader.

use super::{NerError, Sentence, Tag, bio_names};
use crate::conll::DOCSTART;
use crate::error::FormatError;
use crate::sample::NameSample;
use crate::source::{Lines, split_fields};

/// The language of an Evalita file (`EvalitaNameSampleStream.LANGUAGE`):
/// Italian is the only one.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum EvalitaLanguage {
    /// Four columns: word, part of speech, chunk tag, name tag.
    #[default]
    Italian,
}

impl EvalitaLanguage {
    /// The language `EvalitaNameSampleStreamFactory` takes for its `lang`
    /// parameter: `it`.
    pub fn from_code(code: &str) -> Option<EvalitaLanguage> {
        (code == "it").then_some(EvalitaLanguage::Italian)
    }
}

/// The name types an Evalita reader keeps (the `GENERATE_*_ENTITIES`
/// flags of `EvalitaNameSampleStream`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct EvalitaTypes {
    pub person: bool,
    pub organization: bool,
    pub location: bool,
    pub gpe: bool,
}

impl EvalitaTypes {
    /// Every type.
    pub const ALL: EvalitaTypes = EvalitaTypes { person: true, organization: true, location: true, gpe: true };

    /// The types the factory takes from its `types` parameter: each type
    /// whose short name (`per`, `org`, `loc`, `gpe`) the value contains
    /// anywhere.
    pub fn from_param(types: &str) -> EvalitaTypes {
        EvalitaTypes {
            person: types.contains("per"),
            organization: types.contains("org"),
            location: types.contains("loc"),
            gpe: types.contains("gpe"),
        }
    }

    /// The tag the stream reads in place of `tag`: `O` when it ends with
    /// the name of a type left out.
    fn filter(self, tag: &str) -> &str {
        let left_out = (tag.ends_with("PER") && !self.person)
            || (tag.ends_with("ORG") && !self.organization)
            || (tag.ends_with("LOC") && !self.location)
            || (tag.ends_with("GPE") && !self.gpe);
        if left_out { "O" } else { tag }
    }
}

impl Default for EvalitaTypes {
    fn default() -> EvalitaTypes {
        EvalitaTypes::ALL
    }
}

/// OpenNLP's name for an Evalita type (`EvalitaNameSampleStream.extract`).
fn name_type(kind: &str) -> Option<&'static str> {
    Some(match kind {
        "PER" => "person",
        "LOC" => "location",
        "GPE" => "gpe",
        "ORG" => "organization",
        _ => return None,
    })
}

/// Reads name samples from Evalita 2007 (`EvalitaNameSampleStream`): four
/// space-separated columns, the word, its part of speech, its chunk tag
/// and its name tag. Every sample clears adaptive data. A `-DOCSTART-`
/// line, followed by an empty line, starts a document.
///
/// ```text
/// Il DET B-NP O
/// sindaco S I-NP O
/// di E B-PP O
/// Trento SP B-NP B-GPE
/// ```
#[derive(Clone, Debug)]
pub struct EvalitaReader<'a> {
    lines: Lines<'a>,
    types: EvalitaTypes,
}

impl<'a> EvalitaReader<'a> {
    /// A reader of `source` in `language` that keeps the names of `types`.
    ///
    /// # Errors
    ///
    /// [`FormatError::TooLong`] for a source longer than spans can index.
    pub fn new(
        source: &'a str,
        language: EvalitaLanguage,
        types: EvalitaTypes,
    ) -> Result<EvalitaReader<'a>, FormatError> {
        let EvalitaLanguage::Italian = language;
        Ok(EvalitaReader { lines: Lines::new(source)?, types })
    }
}

impl<'a> Iterator for EvalitaReader<'a> {
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
                if line.text().starts_with(DOCSTART) {
                    match self.lines.next() {
                        None => return Some(Err(FormatError::DocstartAtEnd { line: line.number })),
                        Some(after) if !after.text().is_empty() => {
                            let text = after.text().to_owned();
                            return Some(Err(FormatError::DocstartNotFollowedByEmpty { line: after.number, text }));
                        }
                        Some(_) => continue,
                    }
                }
                let fields = split_fields(line.piece, ' ');
                if fields.len() != 4 {
                    return Some(Err(
                        NerError::EvalitaFields { line: line.number, text: line.text().to_owned() }.into()
                    ));
                }
                s.tokens.push(fields[0].field());
                s.tags.push(Tag { text: fields[3].text, line: line.number });
            }
            // the stream clears adaptive data for every Italian sample
            s.clear_adaptive_data = true;
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

    fn read(source: &str, types: EvalitaTypes) -> Vec<Result<String, String>> {
        EvalitaReader::new(source, EvalitaLanguage::Italian, types)
            .unwrap()
            .map(|r| r.map(|s| s.to_string()).map_err(|e| e.to_string()))
            .collect()
    }

    #[test]
    fn reads_names_and_clears_every_sample() {
        let text = "-DOCSTART- -X- O O\n\nIl DET B-NP O\nsindaco S I-NP O\ndi E B-PP O\nTrento SP B-NP B-GPE\n\nRossi SP B-NP I-PER\nVerdi SP B-NP B-PER\n";
        assert_eq!(
            read(text, EvalitaTypes::ALL),
            [
                Ok("\nIl sindaco di <START:gpe> Trento <END>".to_owned()),
                Ok("\nRossi <START:person> Verdi <END>".to_owned()),
            ]
        );
        assert_eq!(
            read(text, EvalitaTypes::from_param("per")),
            [Ok("\nIl sindaco di Trento".to_owned()), Ok("\nRossi <START:person> Verdi <END>".to_owned())]
        );
    }

    #[test]
    fn languages_take_the_factory_code() {
        assert_eq!(EvalitaLanguage::from_code("it"), Some(EvalitaLanguage::Italian));
        assert_eq!(EvalitaLanguage::from_code("ita"), None);
    }

    #[test]
    fn document_starts() {
        assert_eq!(
            read("a B C O\n-DOCSTART- x\nb B C O\n\nc B C O\n", EvalitaTypes::ALL),
            [Err("Empty line after -DOCSTART- not empty: 'b B C O'!".to_owned()), Ok("\nc ".to_owned())]
        );
        assert_eq!(
            read("a B C O\n-DOCSTART-", EvalitaTypes::ALL),
            [Err("Cannot invoke \"java.lang.CharSequence.length()\" because \"theString\" is null".to_owned())]
        );
    }

    #[test]
    fn errors() {
        assert_eq!(
            read("a B C\nb B C O\n\nc B C B-MISC\n\nd B C X\n", EvalitaTypes::ALL),
            [
                Err("Incorrect number of fields per line for language: 'a B C'!".to_owned()),
                Ok("\nb ".to_owned()),
                Err("Unknown type: MISC".to_owned()),
                Err("Invalid tag: X".to_owned()),
            ]
        );
    }
}
