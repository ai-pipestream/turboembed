//! Dependency samples read directly from CoNLL-U:
//! `opennlp.tools.formats.conllu.ConlluDependencySampleStream`.
//!
//! Unlike [`super::ConlluReader`], which merges multiword token ranges
//! into their words for token, part-of-speech and lemma samples,
//! `ConlluDependencySampleStream` drops multiword range lines (`id`
//! containing `-`) and empty node lines (`id` containing `.`) while
//! keeping the syntactic word rows.
//!
//! Sentences are separated by any line that `StringUtil.isBlank` accepts
//! (empty or [`WhitespaceMode::Unicode`](turbo_unicode::WhitespaceMode::Unicode)
//! whitespace, including U+00A0). A leading UTF-8 byte order mark
//! (`\u{feff}`) on the first line is stripped. Word lines are split on
//! tabs retaining trailing empty fields; a word line without ten columns
//! fails with [`FormatError::ConlluDependencyColumns`]. Sentences whose
//! word ids are not `1..=n`, whose `HEAD` or `DEPREL` is `_` or not a
//! valid tree, or whose relation is blank are skipped and counted
//! ([`ConlluDependencyReader::skipped`] and
//! [`Warning::SkippedDependencySentence`]).

use turbo_analysis::Span;

use super::Tagset;
use crate::error::{FormatError, Warning};
use crate::masc::java::parse_int;
use crate::sample::dependency::is_blank;
use crate::sample::{DependencyGraph, DependencySample};
use crate::source::{Field, Lines, Piece};

const COLUMNS: usize = 10;
const ID: usize = 0;
const FORM: usize = 1;
const UPOS: usize = 3;
const XPOS: usize = 4;
const HEAD: usize = 6;
const DEPREL: usize = 7;
const PLACEHOLDER: &str = "_";
const BOM: char = '\u{feff}';

type SentenceWords<'a> = (Vec<[Piece<'a>; COLUMNS]>, u32, Span);

/// Splits `piece` on tabs, retaining empty fields at the end
/// (`ConlluDependencySampleStream.splitFields`).
pub(crate) fn split_tab_all(piece: Piece<'_>) -> Vec<Piece<'_>> {
    let mut fields = Vec::with_capacity(COLUMNS);
    let mut start = 0;
    for (i, c) in piece.text.char_indices() {
        if c == '\t' {
            fields.push(piece.slice(start, i));
            start = i + 1;
        }
    }
    fields.push(piece.slice(start, piece.text.len()));
    fields
}

/// Reads [`DependencySample`]s from CoNLL-U text
/// (`ConlluDependencySampleStream`).
#[derive(Clone, Debug)]
pub struct ConlluDependencyReader<'a> {
    source: &'a str,
    lines: Lines<'a>,
    tagset: Tagset,
    first_line: bool,
    skipped: usize,
    warnings: Vec<Warning>,
}

impl<'a> ConlluDependencyReader<'a> {
    /// Opens a CoNLL-U dependency sample reader over `source` using
    /// `tagset` (`Tagset::U` for `UPOS`, `Tagset::X` for `XPOS`).
    pub fn new(source: &'a str, tagset: Tagset) -> Result<ConlluDependencyReader<'a>, FormatError> {
        Ok(ConlluDependencyReader {
            source,
            lines: Lines::new(source)?,
            tagset,
            first_line: true,
            skipped: 0,
            warnings: Vec::new(),
        })
    }

    /// Rewinds the reader to the start of `source`
    /// (`ConlluDependencySampleStream.reset`).
    pub fn reset(&mut self) {
        self.lines = Lines::new(self.source).expect("source length already checked");
        self.first_line = true;
        self.skipped = 0;
        self.warnings.clear();
    }

    /// The number of sentences skipped so far because their basic
    /// dependency annotation was incomplete or not a valid tree.
    pub fn skipped(&self) -> usize {
        self.skipped
    }

    /// The warnings recorded for skipped sentences.
    pub fn warnings(&self) -> &[Warning] {
        &self.warnings
    }

    /// Reads the syntactic word lines of the next sentence.
    fn next_sentence(&mut self) -> Result<Option<SentenceWords<'a>>, FormatError> {
        let mut words: Vec<[Piece<'a>; COLUMNS]> = Vec::new();
        let mut first_line_num = 0u32;
        let mut first_start = 0u32;
        let mut last_end = 0u32;

        for line in self.lines.by_ref() {
            let mut piece = line.piece;
            if self.first_line {
                self.first_line = false;
                if piece.text.starts_with(BOM) {
                    piece = piece.slice(BOM.len_utf8(), piece.text.len());
                }
            }
            if is_blank(piece.text) {
                if !words.is_empty() {
                    return Ok(Some((words, first_line_num, Span::new(first_start, last_end))));
                }
                continue;
            }
            if piece.text.as_bytes()[0] == b'#' {
                continue;
            }
            let fields = split_tab_all(piece);
            let count = fields.len();
            let Ok(cols) = <[Piece<'a>; COLUMNS]>::try_from(fields) else {
                return Err(FormatError::ConlluDependencyColumns {
                    line: line.number,
                    count,
                    text: piece.text.to_owned(),
                });
            };
            let id = cols[ID].text;
            if !id.contains('-') && !id.contains('.') {
                if words.is_empty() {
                    first_line_num = line.number;
                    first_start = line.span().start;
                }
                last_end = line.span().end;
                words.push(cols);
            }
        }
        if words.is_empty() { Ok(None) } else { Ok(Some((words, first_line_num, Span::new(first_start, last_end)))) }
    }

    fn convert(&self, words: Vec<[Piece<'a>; COLUMNS]>, source: Span) -> Option<DependencySample<'a>> {
        let n = words.len();
        let tag_col = match self.tagset {
            Tagset::U => UPOS,
            Tagset::X => XPOS,
        };
        let mut tokens = Vec::with_capacity(n);
        let mut tags = Vec::with_capacity(n);
        let mut heads = Vec::with_capacity(n);
        let mut relations = Vec::with_capacity(n);

        for (i, word) in words.into_iter().enumerate() {
            if word[ID].text != (i + 1).to_string() {
                return None;
            }
            if word[DEPREL].text == PLACEHOLDER {
                return None;
            }
            let head = parse_int(word[HEAD].text)?.wrapping_sub(1);
            tokens.push(Field::from(word[FORM]));
            tags.push(Field::from(word[tag_col]));
            heads.push(head);
            relations.push(Field::from(word[DEPREL]));
        }
        let graph = DependencyGraph::of(heads, relations).ok()?;
        DependencySample::new(tokens, tags, graph, source).ok()
    }
}

impl<'a> Iterator for ConlluDependencyReader<'a> {
    type Item = Result<DependencySample<'a>, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (words, line_num, source) = match self.next_sentence() {
                Ok(Some(s)) => s,
                Ok(None) => return None,
                Err(e) => return Some(Err(e)),
            };
            if let Some(sample) = self.convert(words, source) {
                return Some(Ok(sample));
            }
            self.skipped += 1;
            self.warnings.push(Warning::SkippedDependencySentence { line: line_num });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample::DependencyArc;

    fn line(fields: &[&str]) -> String {
        fields.join("\t")
    }

    fn fixture() -> String {
        [
            "# sent_id = test-1",
            "# text = He bought the bonds",
            &line(&["1", "He", "he", "PRON", "PRP", "_", "2", "nsubj", "_", "_"]),
            &line(&["2", "bought", "buy", "VERB", "VBD", "_", "0", "root", "_", "_"]),
            &line(&["3", "the", "the", "DET", "DT", "_", "4", "det", "_", "_"]),
            &line(&["4", "bonds", "bond", "NOUN", "NNS", "_", "2", "obj", "_", "_"]),
            "",
            "# sent_id = test-2",
            "# text = Broken",
            &line(&["1", "Broken", "broken", "ADJ", "JJ", "_", "_", "_", "_", "_"]),
            "",
            "# sent_id = test-3",
            "# text = im Haus",
            &line(&["1-2", "im", "_", "_", "_", "_", "_", "_", "_", "_"]),
            &line(&["1", "in", "in", "ADP", "APPR", "_", "2", "case", "_", "_"]),
            &line(&["2", "Haus", "Haus", "NOUN", "NN", "_", "0", "root", "_", "_"]),
            "",
            "# sent_id = test-4",
            "# text = Dogs bark",
            &line(&["1", "Dogs", "dog", "NOUN", "NNS", "_", "2", "nsubj", "_", "_"]),
            &line(&["2", "bark", "bark", "VERB", "VBP", "_", "0", "root", "_", "_"]),
            "",
        ]
        .join("\n")
    }

    #[test]
    fn reads_samples_keeps_contractions_and_skips_unusable_sentences() {
        let src = fixture();
        let mut reader = ConlluDependencyReader::new(&src, Tagset::U).unwrap();

        let first = reader.next().unwrap().unwrap();
        let tok: Vec<&str> = first.tokens().iter().map(|f| f.as_str()).collect();
        let tags: Vec<&str> = first.tags().iter().map(|f| f.as_str()).collect();
        assert_eq!(tok, ["He", "bought", "the", "bonds"]);
        assert_eq!(tags, ["PRON", "VERB", "DET", "NOUN"]);
        assert_eq!(first.graph().head_of(0), Some(1));
        assert_eq!(first.graph().head_of(1), Some(DependencyArc::ROOT_HEAD));
        assert_eq!(first.graph().head_of(2), Some(3));
        assert_eq!(first.graph().relation_of(3).unwrap().as_str(), "obj");

        let second = reader.next().unwrap().unwrap();
        let tok2: Vec<&str> = second.tokens().iter().map(|f| f.as_str()).collect();
        assert_eq!(tok2, ["in", "Haus"]);
        assert_eq!(second.graph().head_of(0), Some(1));
        assert_eq!(second.graph().relation_of(0).unwrap().as_str(), "case");

        let third = reader.next().unwrap().unwrap();
        let tok3: Vec<&str> = third.tokens().iter().map(|f| f.as_str()).collect();
        assert_eq!(tok3, ["Dogs", "bark"]);

        assert!(reader.next().is_none());
        assert_eq!(reader.skipped(), 1);
        assert_eq!(reader.warnings(), &[Warning::SkippedDependencySentence { line: 10 }]);

        reader.reset();
        let after_reset = reader.next().unwrap().unwrap();
        assert_eq!(after_reset.tokens()[0].as_str(), "He");
    }

    #[test]
    fn xpos_tagset_and_bom_and_nbsp_separator() {
        let src = fixture();
        let mut reader = ConlluDependencyReader::new(&src, Tagset::X).unwrap();
        let first = reader.next().unwrap().unwrap();
        let tags: Vec<&str> = first.tags().iter().map(|f| f.as_str()).collect();
        assert_eq!(tags, ["PRP", "VBD", "DT", "NNS"]);

        let bom = format!("\u{feff}{}\n", line(&["1", "Word", "word", "NOUN", "NN", "_", "0", "root", "_", "_"]));
        let samples: Vec<_> =
            ConlluDependencyReader::new(&bom, Tagset::U).unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].tokens()[0].as_str(), "Word");

        let nbsp = [
            &line(&["1", "Dogs", "dog", "NOUN", "NNS", "_", "2", "nsubj", "_", "_"]),
            &line(&["2", "bark", "bark", "VERB", "VBP", "_", "0", "root", "_", "_"]),
            "\u{00a0}",
            &line(&["1", "Fine", "fine", "ADJ", "JJ", "_", "0", "root", "_", "_"]),
        ]
        .join("\n");
        let samples: Vec<_> =
            ConlluDependencyReader::new(&nbsp, Tagset::U).unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(samples.len(), 2);
    }

    #[test]
    fn rejects_wrong_column_counts_and_skips_invalid_trees() {
        let short = "1\ttoo\tfew\tcolumns\n";
        let err = ConlluDependencyReader::new(short, Tagset::U).unwrap().next().unwrap().unwrap_err();
        assert_eq!(
            err,
            FormatError::ConlluDependencyColumns { line: 1, count: 4, text: "1\ttoo\tfew\tcolumns".into() }
        );
        assert_eq!(err.to_string(), "CoNLL-U word line has 4 columns, expected 10: 1\ttoo\tfew\tcolumns");

        // Trailing empty column is retained by splitFields, not dropped.
        let trailing_empty = "1\tword\tword\tNOUN\tNN\t_\t0\troot\t_\t\n";
        let samples: Vec<_> =
            ConlluDependencyReader::new(trailing_empty, Tagset::U).unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(samples.len(), 1);

        let cycle = [
            &line(&["1", "far", "far", "ADV", "RB", "_", "5", "advmod", "_", "_"]),
            &line(&["2", "off", "off", "ADP", "RP", "_", "0", "root", "_", "_"]),
            "",
            &line(&["1", "loop", "loop", "NOUN", "NN", "_", "2", "dep", "_", "_"]),
            &line(&["2", "back", "back", "ADV", "RB", "_", "1", "dep", "_", "_"]),
            "",
            &line(&["1", "Fine", "fine", "ADJ", "JJ", "_", "0", "root", "_", "_"]),
        ]
        .join("\n");
        let mut reader = ConlluDependencyReader::new(&cycle, Tagset::U).unwrap();
        let only = reader.next().unwrap().unwrap();
        assert_eq!(only.tokens()[0].as_str(), "Fine");
        assert!(reader.next().is_none());
        assert_eq!(reader.skipped(), 2);
    }
}
