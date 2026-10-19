//! The brat standoff format: a corpus directory of documents, each a
//! text file (`.txt`) and an annotation file (`.ann`), read into name
//! samples as `BratNameSampleStreamFactory` reads it.
//!
//! [`read_brat_corpus`] lists and parses the documents
//! (`BratDocumentStream`, [`BratDocument`]), and a [`BratParser`] makes
//! the samples of each ([`BratSamples`] for a corpus). The factory takes
//! a sentence detector model or splits sentences at line ends
//! ([`NewlineSentences`](super::NewlineSentences)), and a tokenizer model
//! or a rule-based tokenizer by name, the whitespace tokenizer when given
//! neither ([`RuleTokenizer`](super::RuleTokenizer)). It needs an
//! annotation configuration ([`AnnotationConfiguration`]), which nothing
//! it reads depends on.
//!
//! An annotation file line is a whitespace-separated id and type, then
//! what the id's first letter says: `T` a text-bound annotation with
//! offsets (fragments separated by `;`) and text, `R` a relation, `A` an
//! attribute, `E` an event, `#` with the type `AnnotatorNotes` a note on
//! another annotation. With the tabs shown as spaces:
//!
//! ```text
//! T1  Person 0 9;10 16  Ada Byron King
//! T2  Location 30 36  London
//! #1  AnnotatorNotes T1  a note
//! ```
//!
//! # What OpenNLP does here
//!
//! - A line of two fields or fewer, such as an empty line, ends the
//!   reading of the annotation file: the lines after it are not read.
//!   Lines whose id starts with another letter, and `#` lines of another
//!   type, are skipped.
//! - An error in a line of an annotation file fails the whole document.
//!   The errors of the parsers are wrapped in an `IOException` naming
//!   the file ([`NerError::BratAnnotation`]); an offset that is negative
//!   or past the next is refused by `Span`'s constructor
//!   ([`NerError::BratSpan`]) and an array read past its end throws
//!   `ArrayIndexOutOfBoundsException` ([`NerError::BratIndex`]), neither
//!   of them wrapped: a fragment list ending with `;` with nothing after
//!   it, a text-bound line whose fields are all offsets, a relation with
//!   one argument and a note with no text.
//! - Offsets count UTF-16 code units, which brat does not.
//! - The text-bound annotations of the types kept are names; a sentence
//!   that starts inside one is joined to the one before (see
//!   [`BratParser`]).
//! - The configuration parser throws `NullPointerException` for a type
//!   before any section ([`NerError::BratConfigNoSection`]).
//!
//! # Where this differs from OpenNLP
//!
//! A relation argument is read from its sixth UTF-16 code unit on
//! (`RelationAnnotationParser.parseArg`). When that cut falls between the
//! two halves of a surrogate pair, Java keeps the pair's second half at the
//! start of the argument; a Rust string cannot hold half a pair, so here
//! the argument starts after the whole pair.

mod annotation;
mod document;
mod samples;

pub use annotation::{AnnotationClass, AnnotationConfiguration, AnnotationValue, BratAnnotation, BratCause, Fragment};
pub use document::{BratDocument, brat_documents, read_brat_corpus, write_brat_entities};
pub use samples::{BratParser, BratSamples};

#[cfg(doc)]
use super::NerError;

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_analysis::{Ix, Span, Spans, ViewId};
    use turbo_analysis_layers::Tokens;
    use turbo_unicode::WhitespaceMode;

    #[test]
    fn brat_standoff_round_trips_nested_and_discontinuous_entities() {
        let text = "Bank of New York took the fees off today";
        let mut tok_spans = Spans::new(ViewId::SOURCE);
        let mut pos = 0u32;
        for w in text.split(' ') {
            let end = pos + w.len() as u32;
            tok_spans.push(Span::new(pos, end));
            pos = end + 1;
        }
        let tokens = Tokens { spans: tok_spans, sentence: None };

        let ann = concat!(
            "T1\tOrg 0 16\tBank of New York\n",
            "T2\tLoc 8 16\tNew York\n",
            "T3\tPhrasalVerb 17 21;31 34\ttook off\n",
        );
        let doc = BratDocument::parse("doc1", text.as_bytes(), ann.as_bytes(), WhitespaceMode::Unicode).unwrap();
        let entities = doc.to_entities(&tokens).unwrap();
        assert_eq!(entities.spans.len(), 3);
        assert_eq!(entities.spans.get(Ix::new(0)), Span::new(0, 16));
        assert!(!entities.spans.is_discontinuous(Ix::new(0)));
        assert_eq!(entities.spans.get(Ix::new(1)), Span::new(8, 16));
        assert!(!entities.spans.is_discontinuous(Ix::new(1)));
        assert_eq!(entities.spans.get(Ix::new(2)), Span::new(17, 34));
        assert!(entities.spans.is_discontinuous(Ix::new(2)));
        assert_eq!(entities.spans.fragments(Ix::new(2)).collect::<Vec<_>>(), [Span::new(17, 21), Span::new(31, 34)],);

        let written = write_brat_entities(text, &entities);
        assert_eq!(written, ann);

        let reparsed =
            BratDocument::parse("doc1", text.as_bytes(), written.as_bytes(), WhitespaceMode::Unicode).unwrap();
        let round_tripped = reparsed.to_entities(&tokens).unwrap();
        assert_eq!(round_tripped.spans.len(), 3);
        assert_eq!(
            round_tripped.spans.fragments(Ix::new(2)).collect::<Vec<_>>(),
            [Span::new(17, 21), Span::new(31, 34)],
        );
    }

    #[test]
    fn brat_offsets_are_utf16_units_and_line_breaks_are_written_as_spaces() {
        // "é" is one unit of two bytes, the emoji two units of four bytes.
        let text = "café 😀 New\nYork took it off";
        let mut tok_spans = Spans::new(ViewId::SOURCE);
        let mut pos = 0u32;
        for w in text.split([' ', '\n']) {
            let end = pos + w.len() as u32;
            tok_spans.push(Span::new(pos, end));
            pos = end + 1;
        }
        let tokens = Tokens { spans: tok_spans, sentence: None };
        // "New\nYork" is units 8..16, bytes 11..19; "took" 17..21 and "off"
        // 25..28 in units, 20..24 and 28..31 in bytes; the touching
        // fragments "took" and " it" are one span once read.
        let ann = concat!(
            "T1\tLoc 8 16\tNew York\n",
            "T2\tVerb 17 21;25 28\ttook off\n",
            "T3\tVerb 17 21;21 24\ttook  it\n",
        );
        let doc = BratDocument::parse("doc1", text.as_bytes(), ann.as_bytes(), WhitespaceMode::Unicode).unwrap();
        let entities = doc.to_entities(&tokens).unwrap();
        assert_eq!(entities.spans.get(Ix::new(0)), Span::new(11, 19));
        assert_eq!(entities.spans.fragments(Ix::new(1)).collect::<Vec<_>>(), [Span::new(20, 24), Span::new(28, 31)]);
        assert_eq!(entities.spans.get(Ix::new(2)), Span::new(20, 27));
        assert!(!entities.spans.is_discontinuous(Ix::new(2)));

        let written = write_brat_entities(text, &entities);
        assert_eq!(
            written,
            concat!("T1\tLoc 8 16\tNew York\n", "T2\tVerb 17 21;25 28\ttook off\n", "T3\tVerb 17 24\ttook it\n",)
        );
        let reparsed =
            BratDocument::parse("doc1", text.as_bytes(), written.as_bytes(), WhitespaceMode::Unicode).unwrap();
        let again = reparsed.to_entities(&tokens).unwrap();
        assert_eq!(again.spans.get(Ix::new(0)), Span::new(11, 19));
        assert_eq!(again.spans.fragments(Ix::new(1)).collect::<Vec<_>>(), [Span::new(20, 24), Span::new(28, 31)]);
    }
}
