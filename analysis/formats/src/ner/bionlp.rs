//! The BioNLP 2004 reader.

use super::{NerError, Sentence, Tag, bio_names};
use crate::error::FormatError;
use crate::sample::NameSample;
use crate::source::{Lines, java_trim, split_fields};

/// The name types a BioNLP reader keeps (the `GENERATE_*_ENTITIES` flags
/// of `BioNLP2004NameSampleStream`).
///
/// `cell_line` is never read: the stream keeps `cell_line` names when
/// `cell_type` is set, as it tests the `cell_type` flag for both.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BioNlpTypes {
    pub dna: bool,
    pub protein: bool,
    pub cell_type: bool,
    pub cell_line: bool,
    pub rna: bool,
}

impl BioNlpTypes {
    /// Every type.
    pub const ALL: BioNlpTypes = BioNlpTypes { dna: true, protein: true, cell_type: true, cell_line: true, rna: true };

    /// The types the factory takes from its `types` parameter: each type
    /// whose name (`DNA`, `protein`, `cell_type`, `cell_line`, `RNA`) the
    /// value contains anywhere.
    pub fn from_param(types: &str) -> BioNlpTypes {
        BioNlpTypes {
            dna: types.contains("DNA"),
            protein: types.contains("protein"),
            cell_type: types.contains("cell_type"),
            cell_line: types.contains("cell_line"),
            rna: types.contains("RNA"),
        }
    }

    /// The tag the stream reads in place of `tag`: `O` when it ends with
    /// the name of a type left out.
    fn filter(self, tag: &str) -> &str {
        let left_out = (tag.ends_with("DNA") && !self.dna)
            || (tag.ends_with("protein") && !self.protein)
            || (tag.ends_with("cell_type") && !self.cell_type)
            // the stream's own test, which reads the cell_type flag
            || (tag.ends_with("cell_line") && !self.cell_type)
            || (tag.ends_with("RNA") && !self.rna);
        if left_out { "O" } else { tag }
    }
}

impl Default for BioNlpTypes {
    fn default() -> BioNlpTypes {
        BioNlpTypes::ALL
    }
}

/// Reads name samples from BioNLP 2004, the JNLPBA shared task
/// (`BioNLP2004NameSampleStream`): two tab-separated columns, the word and
/// its name tag. The type of a name is its tag after the prefix, as
/// written (`DNA`, `protein`, `cell_type`, `cell_line`, `RNA`). A
/// `###MEDLINE:` line starts an abstract and clears adaptive data for the
/// next sample; the line after it is skipped. With the tabs shown as
/// spaces:
///
/// ```text
/// ###MEDLINE:00000001
///
/// Binding    O
/// of         O
/// p50        B-protein
/// ```
#[derive(Clone, Debug)]
pub struct BioNlpReader<'a> {
    lines: Lines<'a>,
    types: BioNlpTypes,
}

impl<'a> BioNlpReader<'a> {
    /// A reader of `source` that keeps the names of `types`.
    ///
    /// # Errors
    ///
    /// [`FormatError::TooLong`] for a source longer than spans can index.
    pub fn new(source: &'a str, types: BioNlpTypes) -> Result<BioNlpReader<'a>, FormatError> {
        Ok(BioNlpReader { lines: Lines::new(source)?, types })
    }
}

impl<'a> Iterator for BioNlpReader<'a> {
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
                if java_trim(line.piece).text.is_empty() {
                    break;
                }
                s.cover(&line);
                if line.text().starts_with("###MEDLINE:") {
                    s.clear_adaptive_data = true;
                    if let Some(skipped) = self.lines.next() {
                        s.cover(&skipped);
                    }
                    continue;
                }
                if line.text().contains("ABSTRACT TRUNCATED") {
                    continue;
                }
                let fields = split_fields(line.piece, '\t');
                if fields.len() != 2 {
                    let text = line.text().to_owned();
                    return Some(Err(NerError::BioNlpFields { line: line.number, count: fields.len(), text }.into()));
                }
                s.tokens.push(fields[0].field());
                s.tags.push(Tag { text: fields[1].text, line: line.number });
            }
            if !s.tokens.is_empty() {
                let types = self.types;
                let names = bio_names(&s.tags, |t| types.filter(t), |k, _| Ok(k));
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

    fn read(source: &str, types: BioNlpTypes) -> Vec<Result<String, String>> {
        BioNlpReader::new(source, types).unwrap().map(|r| r.map(|s| s.to_string()).map_err(|e| e.to_string())).collect()
    }

    #[test]
    fn reads_abstracts_and_names() {
        let text = "###MEDLINE:1\n\nBinding\tO\nof\tO\np50\tB-protein\nand\tI-DNA\n.\tO\n \nIL-2\tB-DNA\n";
        assert_eq!(
            read(text, BioNlpTypes::ALL),
            [Ok("\nBinding of <START:protein> p50 and <END> .".to_owned()), Ok("<START:DNA> IL-2  <END>".to_owned())]
        );
    }

    #[test]
    fn the_line_after_an_abstract_start_is_skipped() {
        let text = "###MEDLINE:1\nlost\tO\nkept\tO\n";
        assert_eq!(read(text, BioNlpTypes::ALL), [Ok("\nkept ".to_owned())]);
    }

    #[test]
    fn truncated_abstract_lines_are_skipped() {
        let text = "a\tO\n[ABSTRACT TRUNCATED AT 250 WORDS]\nb\tO\n";
        assert_eq!(read(text, BioNlpTypes::ALL), [Ok("a b".to_owned())]);
    }

    #[test]
    fn cell_lines_follow_the_cell_type_flag() {
        let text = "a\tB-cell_line\nb\tB-cell_type\n";
        assert_eq!(read(text, BioNlpTypes::from_param("cell_line")), [Ok("a b".to_owned())]);
        assert_eq!(
            read(text, BioNlpTypes::from_param("cell_type")),
            [Ok("<START:cell_line> a <END> <START:cell_type> b <END>".to_owned())]
        );
    }

    #[test]
    fn errors() {
        assert_eq!(
            read("a\tO\tx\nb\tO\n", BioNlpTypes::ALL),
            [
                Err("Expected two fields per line in training data, got 3 for line 'a\tO\tx'!".to_owned()),
                Ok("b ".to_owned())
            ]
        );
        assert_eq!(
            read("a\tX\n\nb\tB-\n", BioNlpTypes::ALL),
            [Err("Invalid tag: X".to_owned()), Ok("<START:> b  <END>".to_owned())]
        );
    }
}
