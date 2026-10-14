//! The CoNLL shared task formats OpenNLP reads: CoNLL-2002 and CoNLL-2003
//! names (`Conll02NameSampleStream`, `Conll03NameSampleStream`) and
//! CoNLL-X part-of-speech tags (`ConllXPOSSampleStream`, and through the
//! conversion streams token and sentence samples).
//!
//! CoNLL-2000 chunks need no reader of their own: OpenNLP's native chunk
//! format is CoNLL-2000's three space-separated columns, read by
//! [`crate::native::ChunkReader`] (`ChunkSampleStream`, registered as the
//! chunker's default format).
//!
//! | Format | Reader | OpenNLP | Registry name |
//! |---|---|---|---|
//! | CoNLL-2002 names (Dutch, Spanish) | [`Conll02Reader`] | `Conll02NameSampleStream` | `conll02` |
//! | CoNLL-2003 names (English, German) | [`Conll03Reader`] | `Conll03NameSampleStream` | `conll03` |
//! | CoNLL-X part-of-speech tags | [`ConllXPosReader`] | `ConllXPOSSampleStream` | `conllx` |
//!
//! The readers take decoded text ([`crate::decode`]): OpenNLP reads
//! CoNLL-2002 as ISO-8859-1 ([`crate::Encoding::Latin1`]) and the others
//! as UTF-8. Tokens and tags are fields of the text with their byte spans.
//!
//! A sentence is a block of lines ended by a line of length zero; a line
//! of spaces is a line of fields. A name tag is `O`, or `B-` or `I-` and
//! one of `PER`, `ORG`, `LOC` and `MISC`, which become the name types
//! `person`, `organization`, `location` and `misc`.
//!
//! # What OpenNLP does here
//!
//! - A line with the wrong number of fields ends the read at that line,
//!   and the next read starts at the line after it, in the middle of the
//!   sentence: the rest of that sentence is read as a sample of its own.
//!   A bad tag or type is found after the whole sentence is read, so the
//!   next read starts at the next sentence.
//! - CoNLL-2002 starts a name only at `B-`. An `I-` tag with no name open
//!   is read as `O`, and an `I-` tag of another type continues the name
//!   open (`B-PER I-LOC` is one person). CoNLL-2003 starts a name at an
//!   `I-` tag with none open (the IOB1 tags of the original files), and an
//!   `I-` tag whose type differs from the open name's starts another.
//! - A tag is checked when the reader comes to it, and a type when its
//!   name ends, so of a bad tag and a bad type the reader reports the one
//!   it comes to first. A tag of a type left out of [`EntityTypes`] is read as `O`, but
//!   only when the tag ends with the type's name: the tag is matched by
//!   its end, so `B-XPER` is left out with `PER`.
//! - CoNLL-2002 Dutch takes a line that starts with `-DOCSTART-` as the
//!   start of a document, which clears adaptive data for the next sample;
//!   Spanish has no such lines and clears it for every sample, so a
//!   `-DOCSTART-` line in Spanish is a token. A document start followed by
//!   an empty line clears nothing: the empty sentence is skipped and its
//!   flag with it.
//! - CoNLL-2003 needs a line of length zero after a `-DOCSTART-` line
//!   ([`FormatError::DocstartNotFollowedByEmpty`]), and at the end of the
//!   text it throws `NullPointerException` ([`FormatError::DocstartAtEnd`]).
//! - CoNLL-X reads the second and fifth of at least five tab-separated
//!   columns. `ConllXPOSSampleStream` has a branch for a sentence without
//!   tokens that reads the next sentence and drops it; it cannot be
//!   reached, as `ParagraphStream` never returns an empty sentence.

#[cfg(doc)]
use crate::FormatError;

mod conllx;
mod names;

pub use conllx::ConllXPosReader;
pub use names::{Conll02Reader, Conll03Reader};

/// The name `-DOCSTART-` lines start with (`Conll02NameSampleStream.DOCSTART`,
/// and the same in CoNLL-2003 and Evalita).
pub const DOCSTART: &str = "-DOCSTART-";

/// The registry name of the CoNLL-2002 name format.
pub const CONLL02_FORMAT: &str = "conll02";

/// The registry name of the CoNLL-2003 name format.
pub const CONLL03_FORMAT: &str = "conll03";

/// The registry name of the CoNLL-X formats (`ConllXPOSSampleStreamFactory.CONLLX_FORMAT`).
pub const CONLLX_FORMAT: &str = "conllx";

/// The language of a CoNLL-2002 file (`Conll02NameSampleStream.LANGUAGE`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Conll02Language {
    /// Dutch: `-DOCSTART-` lines start documents.
    Dutch,
    /// Spanish: every sample clears adaptive data.
    Spanish,
}

impl Conll02Language {
    /// The language `Conll02NameSampleStreamFactory` takes for its `lang`
    /// parameter: `nl` or `nld`, `es` or `spa`.
    pub fn from_code(code: &str) -> Option<Conll02Language> {
        match code {
            "nl" | "nld" => Some(Conll02Language::Dutch),
            "es" | "spa" => Some(Conll02Language::Spanish),
            _ => None,
        }
    }
}

/// The language of a CoNLL-2003 file (`Conll03NameSampleStream.LANGUAGE`),
/// which fixes the number of columns and the one holding the name tag.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Conll03Language {
    /// Four columns: word, part of speech, chunk tag, name tag.
    English,
    /// Five columns: word, lemma, part of speech, chunk tag, name tag.
    German,
}

impl Conll03Language {
    /// The language `Conll03NameSampleStreamFactory` takes for its `lang`
    /// parameter: `eng` or `deu`.
    pub fn from_code(code: &str) -> Option<Conll03Language> {
        match code {
            "eng" => Some(Conll03Language::English),
            "deu" => Some(Conll03Language::German),
            _ => None,
        }
    }

    fn columns(self) -> usize {
        match self {
            Conll03Language::English => 4,
            Conll03Language::German => 5,
        }
    }
}

/// The name types a CoNLL name reader keeps (OpenNLP's `GENERATE_*_ENTITIES`
/// flags); the tags of the others are read as `O`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct EntityTypes {
    pub person: bool,
    pub organization: bool,
    pub location: bool,
    pub misc: bool,
}

impl EntityTypes {
    /// Every type.
    pub const ALL: EntityTypes = EntityTypes { person: true, organization: true, location: true, misc: true };

    /// The types the factories take from their `types` parameter: each
    /// type whose short name (`per`, `org`, `loc`, `misc`) the value
    /// contains anywhere, so `person` keeps persons and `xperx` does too.
    pub fn from_param(types: &str) -> EntityTypes {
        EntityTypes {
            person: types.contains("per"),
            organization: types.contains("org"),
            location: types.contains("loc"),
            misc: types.contains("misc"),
        }
    }

    /// `tag`, or `O` when it ends with the name of a type left out.
    fn filter(self, tag: &str) -> &str {
        let left_out = (tag.ends_with("PER") && !self.person)
            || (tag.ends_with("ORG") && !self.organization)
            || (tag.ends_with("LOC") && !self.location)
            || (tag.ends_with("MISC") && !self.misc);
        if left_out { "O" } else { tag }
    }
}

impl Default for EntityTypes {
    fn default() -> EntityTypes {
        EntityTypes::ALL
    }
}

/// The name type of a tag's type (`Conll02NameSampleStream.extract`), or
/// `None` for a type it does not know.
fn name_type(kind: &str) -> Option<&'static str> {
    match kind {
        "PER" => Some("person"),
        "LOC" => Some("location"),
        "MISC" => Some("misc"),
        "ORG" => Some("organization"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn languages_take_the_factory_codes() {
        assert_eq!(Conll02Language::from_code("nl"), Some(Conll02Language::Dutch));
        assert_eq!(Conll02Language::from_code("spa"), Some(Conll02Language::Spanish));
        assert_eq!(Conll02Language::from_code("en"), None);
        assert_eq!(Conll03Language::from_code("deu"), Some(Conll03Language::German));
        assert_eq!(Conll03Language::from_code("en"), None);
    }

    #[test]
    fn types_are_found_anywhere_in_the_parameter() {
        assert_eq!(EntityTypes::from_param("per,org,loc,misc"), EntityTypes::ALL);
        let t = EntityTypes::from_param("person");
        assert!(t.person && !t.organization && !t.location && !t.misc);
        assert_eq!(
            EntityTypes::from_param(""),
            EntityTypes { person: false, organization: false, location: false, misc: false }
        );
    }

    #[test]
    fn a_left_out_type_is_matched_by_the_tag_s_end() {
        let t = EntityTypes::from_param("loc");
        assert_eq!(t.filter("B-PER"), "O");
        assert_eq!(t.filter("I-XPER"), "O");
        assert_eq!(t.filter("B-LOC"), "B-LOC");
        assert_eq!(t.filter("B-PERS"), "B-PERS");
        assert_eq!(t.filter("O"), "O");
    }
}
