//! The GermEval 2014 reader.

use super::{NerError, Sentence, Tag, bio_names};
use crate::error::FormatError;
use crate::sample::NameSample;
use crate::source::{Lines, java_trim, split_fields};

/// Which of GermEval's two tag columns a reader takes
/// (`GermEval2014NameSampleStream.NerLayer`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum GermEvalLayer {
    /// The third column: the names at the top level.
    #[default]
    Outer,
    /// The fourth column: the names nested inside them.
    Inner,
}

impl GermEvalLayer {
    /// The layer `GermEval2014NameSampleStreamFactory` takes for its
    /// `layer` parameter: `outer` when the parameter is not given.
    pub fn from_param(layer: Option<&str>) -> Option<GermEvalLayer> {
        match layer {
            None | Some("outer") => Some(GermEvalLayer::Outer),
            Some("inner") => Some(GermEvalLayer::Inner),
            Some(_) => None,
        }
    }
}

/// The name types a GermEval reader keeps (the `GENERATE_*_ENTITIES`
/// flags of `GermEval2014NameSampleStream`); `misc` is GermEval's `OTH`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct GermEvalTypes {
    pub person: bool,
    pub organization: bool,
    pub location: bool,
    pub misc: bool,
}

impl GermEvalTypes {
    /// Every type.
    pub const ALL: GermEvalTypes = GermEvalTypes { person: true, organization: true, location: true, misc: true };

    /// The types the factory takes from its `types` parameter: each type
    /// whose short name (`per`, `org`, `loc`, `misc`) the value contains
    /// anywhere.
    pub fn from_param(types: &str) -> GermEvalTypes {
        GermEvalTypes {
            person: types.contains("per"),
            organization: types.contains("org"),
            location: types.contains("loc"),
            misc: types.contains("misc"),
        }
    }

    /// `isTypeEnabled`: whether the stream keeps a tag, by the type its
    /// start names; `O` is kept and every other tag is not.
    fn keeps(self, tag: &str) -> bool {
        let starts = |kind: &str| {
            tag.strip_prefix("B-").or_else(|| tag.strip_prefix("I-")).is_some_and(|rest| rest.starts_with(kind))
        };
        if starts("PER") {
            self.person
        } else if starts("ORG") {
            self.organization
        } else if starts("LOC") {
            self.location
        } else if starts("OTH") {
            self.misc
        } else {
            tag == "O"
        }
    }

    /// The tag the stream reads in place of `tag`: `O` for every tag it
    /// does not keep.
    fn filter(self, tag: &str) -> &str {
        if tag != "O" && !self.keeps(tag) { "O" } else { tag }
    }
}

impl Default for GermEvalTypes {
    fn default() -> GermEvalTypes {
        GermEvalTypes::ALL
    }
}

/// OpenNLP's name for a GermEval type (`mapTagToType`).
fn name_type(kind: &str) -> Option<&'static str> {
    Some(match kind {
        "PER" => "person",
        "PERderiv" => "personderiv",
        "PERpart" => "personpart",
        "LOC" => "location",
        "LOCderiv" => "locationderiv",
        "LOCpart" => "locationpart",
        "ORG" => "organization",
        "ORGderiv" => "organizationderiv",
        "ORGpart" => "organizationpart",
        "OTH" => "misc",
        "OTHderiv" => "miscderiv",
        "OTHpart" => "miscpart",
        _ => return None,
    })
}

/// Reads name samples from GermEval 2014 (`GermEval2014NameSampleStream`):
/// four tab-separated columns, the word's number, the word, its outer
/// name tag and its inner name tag, with a comment line starting with `#`
/// before each sentence. A comment clears adaptive data for its sample.
/// With the tabs shown as spaces:
///
/// ```text
/// #  wiki.de vom 01.01.2014  [2014-01-01]
/// 1  Die      O      O
/// 2  Kölner   B-LOC  O
/// ```
#[derive(Clone, Debug)]
pub struct GermEvalReader<'a> {
    lines: Lines<'a>,
    types: GermEvalTypes,
    layer: GermEvalLayer,
}

impl<'a> GermEvalReader<'a> {
    /// A reader of `source` that keeps the names of `types` in `layer`.
    ///
    /// # Errors
    ///
    /// [`FormatError::TooLong`] for a source longer than spans can index.
    pub fn new(source: &'a str, types: GermEvalTypes, layer: GermEvalLayer) -> Result<GermEvalReader<'a>, FormatError> {
        Ok(GermEvalReader { lines: Lines::new(source)?, types, layer })
    }
}

impl<'a> Iterator for GermEvalReader<'a> {
    type Item = Result<NameSample<'a>, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let mut s = Sentence::default();
            let mut inner = Vec::new();
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
                if line.text().starts_with('#') {
                    s.clear_adaptive_data = true;
                    continue;
                }
                let fields = split_fields(line.piece, '\t');
                if fields.len() < 4 {
                    let text = line.text().to_owned();
                    return Some(Err(NerError::GermEvalFields { line: line.number, count: fields.len(), text }.into()));
                }
                s.tokens.push(fields[1].field());
                s.tags.push(Tag { text: fields[2].text, line: line.number });
                inner.push(Tag { text: java_trim(fields[3]).text, line: line.number });
            }
            if !s.tokens.is_empty() {
                let tags = match self.layer {
                    GermEvalLayer::Outer => &s.tags,
                    GermEvalLayer::Inner => &inner,
                };
                let types = self.types;
                let names = bio_names(
                    tags,
                    |t| types.filter(t),
                    |k, line| name_type(k).ok_or_else(|| NerError::UnknownType { line, kind: k.to_owned() }),
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

    fn read(source: &str, types: GermEvalTypes, layer: GermEvalLayer) -> Vec<Result<String, String>> {
        GermEvalReader::new(source, types, layer)
            .unwrap()
            .map(|r| r.map(|s| s.to_string()).map_err(|e| e.to_string()))
            .collect()
    }

    const TEXT: &str = "# a\n1\tDie\tO\tO\n2\tKölner\tB-ORG\tB-LOC\n3\tBank\tI-ORG\tO \n\n\n1\tWer\tO\tO\n";

    #[test]
    fn reads_either_layer() {
        let all = GermEvalTypes::ALL;
        assert_eq!(
            read(TEXT, all, GermEvalLayer::Outer),
            [Ok("\nDie <START:organization> Kölner Bank <END>".to_owned()), Ok("Wer ".to_owned())]
        );
        assert_eq!(
            read(TEXT, all, GermEvalLayer::Inner),
            [Ok("\nDie <START:location> Kölner <END> Bank".to_owned()), Ok("Wer ".to_owned())]
        );
    }

    #[test]
    fn layers_take_the_factory_values() {
        assert_eq!(GermEvalLayer::from_param(None), Some(GermEvalLayer::Outer));
        assert_eq!(GermEvalLayer::from_param(Some("inner")), Some(GermEvalLayer::Inner));
        assert_eq!(GermEvalLayer::from_param(Some("Inner")), None);
    }

    #[test]
    fn types_left_out_and_unknown_tags_are_o() {
        let t = GermEvalTypes::from_param("loc");
        assert_eq!(t.filter("B-PERderiv"), "O");
        assert_eq!(t.filter("B-LOCpart"), "B-LOCpart");
        assert_eq!(t.filter("X"), "O");
        assert_eq!(t.filter("B-"), "O");
        assert_eq!(t.filter("O"), "O");
        let text = "1\ta\tX\tO\n2\tb\tB-XYZ\tO\n3\tc\tI-LOC\tO\n";
        assert_eq!(read(text, t, GermEvalLayer::Outer), [Ok("a b c".to_owned())]);
    }

    #[test]
    fn a_kept_tag_of_an_unknown_type_is_refused() {
        let text = "1\ta\tB-PERfoo\tO\n\n1\tb\tB-PER \tB-PER \n";
        let all = GermEvalTypes::ALL;
        assert_eq!(
            read(text, all, GermEvalLayer::Outer),
            [Err("Unknown type: PERfoo".to_owned()), Err("Unknown type: PER ".to_owned())]
        );
        assert_eq!(
            read(text, all, GermEvalLayer::Inner),
            [Ok("a ".to_owned()), Ok("<START:person> b  <END>".to_owned())]
        );
    }

    #[test]
    fn a_short_line_ends_the_read_in_the_middle_of_a_sentence() {
        let text = "1\ta\tO\tO\n2\tb\tO\n3\tc\tB-OTH\tO\n";
        assert_eq!(
            read(text, GermEvalTypes::ALL, GermEvalLayer::Outer),
            [
                Err("Expected at least four tab-separated fields per line in GermEval 2014 data, got 3 for line '2\tb\tO'!"
                    .to_owned()),
                Ok("<START:misc> c  <END>".to_owned()),
            ]
        );
    }

    #[test]
    fn a_comment_before_an_empty_line_clears_nothing() {
        let text = "# a\n\n1\tb\tO\tO\n\n1\tc\tO\tO\n# d\n2\te\tO\tO\n";
        assert_eq!(read(text, GermEvalTypes::ALL, GermEvalLayer::Outer), [Ok("b ".to_owned()), Ok("\nc e".to_owned())]);
    }

    #[test]
    fn tokens_are_fields_of_the_source() {
        let sample =
            GermEvalReader::new(TEXT, GermEvalTypes::ALL, GermEvalLayer::Outer).unwrap().next().unwrap().unwrap();
        assert_eq!(&TEXT[sample.tokens()[1].span.range()], "Kölner");
        assert_eq!(&TEXT[sample.source().range()], "# a\n1\tDie\tO\tO\n2\tKölner\tB-ORG\tB-LOC\n3\tBank\tI-ORG\tO ");
    }
}
