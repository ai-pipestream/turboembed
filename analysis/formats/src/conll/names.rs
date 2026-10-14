//! The CoNLL-2002 and CoNLL-2003 name readers.

use turbo_analysis::Span;

use super::{Conll02Language, Conll03Language, DOCSTART, EntityTypes, name_type};
use crate::error::FormatError;
use crate::sample::{Name, NameSample};
use crate::source::{Field, Line, Lines, split_fields};

/// A name tag and the line it is on.
#[derive(Copy, Clone, Debug)]
struct Tag<'a> {
    text: &'a str,
    line: u32,
}

/// The lines of a sentence read so far.
#[derive(Default)]
struct Sentence<'a> {
    tokens: Vec<Field<'a>>,
    tags: Vec<Tag<'a>>,
    clear_adaptive_data: bool,
    /// From the start of the first line read to the end of the last.
    source: Option<Span>,
    first_line: u32,
}

impl<'a> Sentence<'a> {
    fn cover(&mut self, line: &Line<'a>) {
        let span = line.span();
        match self.source {
            None => {
                self.source = Some(span);
                self.first_line = line.number;
            }
            Some(s) => self.source = Some(Span::new(s.start, span.end)),
        }
    }

    fn sample(self, names: Result<Vec<Name<'static>>, FormatError>) -> Result<NameSample<'a>, FormatError> {
        let source = self.source.expect("a sentence with tokens has lines");
        NameSample::new(self.tokens, names?, self.clear_adaptive_data, source, self.first_line)
    }
}

/// The name of tokens `begin..end` whose first tag is `tag`
/// (`Conll02NameSampleStream.extract`): the type is the tag after its
/// two-character prefix.
fn extract(begin: usize, end: i64, tag: Tag<'_>) -> Result<Name<'static>, FormatError> {
    let kind = &tag.text[2..];
    let kind = name_type(kind).ok_or_else(|| FormatError::UnknownNameType { line: tag.line, kind: kind.to_owned() })?;
    let end = u32::try_from(end).expect("an open name ends after its first token");
    Name::new(begin as u32, end, Some(kind))
}

/// The names of a CoNLL-2002 sentence: a name starts at `B-` and grows
/// with each `I-`, whatever its type, until `O` or the next `B-`.
fn conll02_names(tags: &[Tag<'_>], types: EntityTypes) -> Result<Vec<Name<'static>>, FormatError> {
    let mut names = Vec::new();
    let mut begin: Option<usize> = None;
    // Java's endIndex, which an `I-` with no name open moves on from -1
    let mut end: i64 = -1;
    for (i, &t) in tags.iter().enumerate() {
        let tag = types.filter(t.text);
        if tag.starts_with("B-") {
            if let Some(b) = begin {
                names.push(extract(b, end, tags[b])?);
            }
            begin = Some(i);
            end = i as i64 + 1;
        } else if tag.starts_with("I-") {
            end += 1;
        } else if tag == "O" {
            if let Some(b) = begin.take() {
                names.push(extract(b, end, tags[b])?);
                end = -1;
            }
        } else {
            return Err(FormatError::InvalidNameTag { line: t.line, tag: tag.to_owned() });
        }
    }
    if let Some(b) = begin {
        names.push(extract(b, end, tags[b])?);
    }
    Ok(names)
}

/// The names of a CoNLL-2003 sentence: a name starts at `B-`, or at an
/// `I-` with no name open or whose type differs from the open name's.
fn conll03_names(tags: &[Tag<'_>], types: EntityTypes) -> Result<Vec<Name<'static>>, FormatError> {
    let mut names = Vec::new();
    let mut begin: Option<usize> = None;
    let mut end: i64 = -1;
    for (i, &t) in tags.iter().enumerate() {
        let tag = types.filter(t.text);
        if tag == "O" {
            if let Some(b) = begin.take() {
                names.push(extract(b, end, tags[b])?);
                end = -1;
            }
        } else if tag.starts_with("B-") {
            if let Some(b) = begin {
                names.push(extract(b, end, tags[b])?);
            }
            begin = Some(i);
            end = i as i64 + 1;
        } else if tag.starts_with("I-") {
            match begin {
                // the open name's tag less its first character: `-PER`
                Some(b) if tag.ends_with(&tags[b].text[1..]) => end += 1,
                Some(b) => {
                    names.push(extract(b, end, tags[b])?);
                    begin = Some(i);
                    end = i as i64 + 1;
                }
                None => {
                    begin = Some(i);
                    end = i as i64 + 1;
                }
            }
        } else {
            return Err(FormatError::InvalidNameTag { line: t.line, tag: tag.to_owned() });
        }
    }
    if let Some(b) = begin {
        names.push(extract(b, end, tags[b])?);
    }
    Ok(names)
}

/// Reads name samples from CoNLL-2002 (`Conll02NameSampleStream`): three
/// space-separated columns, the word, its part of speech and its name tag.
///
/// ```text
/// -DOCSTART- -DOCSTART- O
/// De Art O
/// Nederlandse Adj B-MISC
/// ```
#[derive(Clone, Debug)]
pub struct Conll02Reader<'a> {
    lines: Lines<'a>,
    language: Conll02Language,
    types: EntityTypes,
}

impl<'a> Conll02Reader<'a> {
    /// A reader of `source` in `language` that keeps the names of `types`.
    ///
    /// # Errors
    ///
    /// [`FormatError::TooLong`] for a source longer than spans can index.
    pub fn new(
        source: &'a str,
        language: Conll02Language,
        types: EntityTypes,
    ) -> Result<Conll02Reader<'a>, FormatError> {
        Ok(Conll02Reader { lines: Lines::new(source)?, language, types })
    }
}

impl<'a> Iterator for Conll02Reader<'a> {
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
                if self.language == Conll02Language::Dutch && line.text().starts_with(DOCSTART) {
                    s.clear_adaptive_data = true;
                    continue;
                }
                let fields = split_fields(line.piece, ' ');
                if fields.len() != 3 {
                    return Some(Err(FormatError::Conll02Fields {
                        line: line.number,
                        count: fields.len(),
                        text: line.text().to_owned(),
                    }));
                }
                s.tokens.push(fields[0].field());
                s.tags.push(Tag { text: fields[2].text, line: line.number });
            }
            if self.language == Conll02Language::Spanish {
                s.clear_adaptive_data = true;
            }
            if !s.tokens.is_empty() {
                let names = conll02_names(&s.tags, self.types);
                return Some(s.sample(names));
            }
            if at_end {
                return None;
            }
        }
    }
}

/// Reads name samples from CoNLL-2003 (`Conll03NameSampleStream`): four
/// space-separated columns in English (word, part of speech, chunk tag,
/// name tag) and five in German (word, lemma, part of speech, chunk tag,
/// name tag). A `-DOCSTART-` line, followed by an empty line, starts a
/// document and clears adaptive data for the next sample.
#[derive(Clone, Debug)]
pub struct Conll03Reader<'a> {
    lines: Lines<'a>,
    language: Conll03Language,
    types: EntityTypes,
}

impl<'a> Conll03Reader<'a> {
    /// A reader of `source` in `language` that keeps the names of `types`.
    ///
    /// # Errors
    ///
    /// [`FormatError::TooLong`] for a source longer than spans can index.
    pub fn new(
        source: &'a str,
        language: Conll03Language,
        types: EntityTypes,
    ) -> Result<Conll03Reader<'a>, FormatError> {
        Ok(Conll03Reader { lines: Lines::new(source)?, language, types })
    }
}

impl<'a> Iterator for Conll03Reader<'a> {
    type Item = Result<NameSample<'a>, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        let columns = self.language.columns();
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
                    s.clear_adaptive_data = true;
                    match self.lines.next() {
                        None => return Some(Err(FormatError::DocstartAtEnd { line: line.number })),
                        Some(after) if !after.text().is_empty() => {
                            return Some(Err(FormatError::DocstartNotFollowedByEmpty {
                                line: after.number,
                                text: after.text().to_owned(),
                            }));
                        }
                        Some(_) => continue,
                    }
                }
                let fields = split_fields(line.piece, ' ');
                if fields.len() != columns {
                    return Some(Err(FormatError::Conll03Fields { line: line.number, text: line.text().to_owned() }));
                }
                s.tokens.push(fields[0].field());
                s.tags.push(Tag { text: fields[columns - 1].text, line: line.number });
            }
            if !s.tokens.is_empty() {
                let names = conll03_names(&s.tags, self.types);
                return Some(s.sample(names));
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

    fn read02(source: &str, language: Conll02Language, types: EntityTypes) -> Vec<Result<String, String>> {
        Conll02Reader::new(source, language, types)
            .unwrap()
            .map(|r| r.map(|s| s.to_string()).map_err(|e| e.to_string()))
            .collect()
    }

    fn read03(source: &str, language: Conll03Language, types: EntityTypes) -> Vec<Result<String, String>> {
        Conll03Reader::new(source, language, types)
            .unwrap()
            .map(|r| r.map(|s| s.to_string()).map_err(|e| e.to_string()))
            .collect()
    }

    #[test]
    fn conll02_reads_names_and_document_starts() {
        let text =
            "-DOCSTART- -DOCSTART- O\nJan N B-PER\nJansen N I-PER\nwoont V O\nin Prep O\nGent N B-LOC\n\nHij Pron O\n";
        let dutch = read02(text, Conll02Language::Dutch, EntityTypes::ALL);
        assert_eq!(
            dutch,
            [
                Ok("\n<START:person> Jan Jansen <END> woont in <START:location> Gent <END>".to_owned()),
                Ok("Hij ".to_owned())
            ]
        );
        // Spanish reads the document start as a token and clears every sample
        let spanish = read02(text, Conll02Language::Spanish, EntityTypes::ALL);
        assert_eq!(
            spanish[0].as_ref().unwrap(),
            "\n-DOCSTART- <START:person> Jan Jansen <END> woont in <START:location> Gent <END>"
        );
        assert_eq!(spanish[1].as_ref().unwrap(), "\nHij ");
    }

    #[test]
    fn conll02_i_tags_never_start_or_split_a_name() {
        let text = "a X I-PER\nb X B-PER\nc X I-LOC\nd X O\ne X I-ORG\n";
        let read = read02(text, Conll02Language::Dutch, EntityTypes::ALL);
        assert_eq!(read, [Ok("a <START:person> b c <END> d e".to_owned())]);
    }

    #[test]
    fn conll03_i_tags_start_and_split_names() {
        let text = "a X X I-PER\nb X X I-PER\nc X X I-LOC\nd X X B-LOC\n";
        let read = read03(text, Conll03Language::English, EntityTypes::ALL);
        assert_eq!(read, [Ok("<START:person> a b <END> <START:location> c <END> <START:location> d <END>".to_owned())]);
    }

    #[test]
    fn left_out_types_read_as_o() {
        let text = "a X B-PER\nb X B-LOC\nc X I-LOC\n";
        let read = read02(text, Conll02Language::Dutch, EntityTypes::from_param("loc"));
        assert_eq!(read, [Ok("a <START:location> b c <END>".to_owned())]);
    }

    #[test]
    fn a_bad_line_ends_the_read_and_the_sentence_goes_on() {
        let text = "a X B-PER\nb X\nc X O\n\nd X O\n";
        let read = read02(text, Conll02Language::Dutch, EntityTypes::ALL);
        assert_eq!(
            read,
            [
                Err("Expected three fields per line in training data, got 2 for line 'b X'!".to_owned()),
                // a sample of one token is written with a space after it
                Ok("c ".to_owned()),
                Ok("d ".to_owned()),
            ]
        );
    }

    #[test]
    fn bad_tags_and_types_fail_after_the_sentence() {
        let text = "a X B-FOO\nb X Q\n\nc X O\n";
        let read = read02(text, Conll02Language::Dutch, EntityTypes::ALL);
        // the bad tag comes first: the name of a bad type is still open
        assert_eq!(read, [Err("Invalid tag: Q".to_owned()), Ok("c ".to_owned())]);
        let read = read02("a X B-FOO\nb X O\nc X Q\n", Conll02Language::Dutch, EntityTypes::ALL);
        assert_eq!(read, [Err("Unknown type: FOO".to_owned())]);
    }

    #[test]
    fn conll03_document_starts_need_an_empty_line() {
        let text = "-DOCSTART- -X- -X- O\n\nEU NNP B-NP B-ORG\n\n-DOCSTART- -X- -X- O\nx X X O\n\ny X X O\n";
        let read = read03(text, Conll03Language::English, EntityTypes::ALL);
        assert_eq!(
            read,
            [
                Ok("\n<START:organization> EU  <END>".to_owned()),
                Err("Empty line after -DOCSTART- not empty: 'x X X O'!".to_owned()),
                Ok("y ".to_owned()),
            ]
        );
        // at the end of the text, the sentence before is lost with it
        let read = read03("y X X O\n-DOCSTART- -X- O O\n", Conll03Language::English, EntityTypes::ALL);
        assert_eq!(read.len(), 1);
        assert!(read[0].as_ref().unwrap_err().contains("\"theString\" is null"));
    }

    #[test]
    fn conll03_german_has_five_columns() {
        let text = "Bonn Bonn NE I-NC I-LOC\nliegt liegen VVFIN I-VC O\n";
        assert_eq!(
            read03(text, Conll03Language::German, EntityTypes::ALL),
            [Ok("<START:location> Bonn <END> liegt".to_owned())]
        );
        assert_eq!(
            read03(text, Conll03Language::English, EntityTypes::ALL)[0],
            Err("Incorrect number of fields per line for language: 'Bonn Bonn NE I-NC I-LOC'!".to_owned())
        );
    }

    #[test]
    fn tokens_are_fields_of_the_source() {
        let text = "x X O\n\nJan N B-PER\n";
        let sample =
            Conll02Reader::new(text, Conll02Language::Dutch, EntityTypes::ALL).unwrap().nth(1).unwrap().unwrap();
        let token = &sample.tokens()[0];
        assert_eq!(&text[token.span.range()], "Jan");
        assert_eq!(&text[sample.source().range()], "Jan N B-PER");
    }
}
