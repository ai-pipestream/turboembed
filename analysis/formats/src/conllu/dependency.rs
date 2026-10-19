//! Dependency samples from CoNLL-U: `ConlluDependencySampleStream` of the
//! OpenNLP fork's `ready/agents-md` branch, which reads the basic
//! dependencies (the `HEAD` and `DEPREL` columns) of each sentence.
//!
//! The stream reads lines itself, not through `ConlluStream`:
//!
//! - A sentence ends at a line that is empty or all whitespace in the
//!   whitespace mode (`StringUtil.isBlank`), unlike `ConlluStream`, which
//!   ends one only at a line of length zero. A byte order mark at the
//!   start of the first line is dropped.
//! - A line whose first character is `#` is a comment and is passed over;
//!   no comment is read.
//! - Every other line must have ten tab-separated columns, empty columns
//!   at its end included; a line without them is an error
//!   ([`FormatError::DependencyColumns`]). The words read so far into its
//!   sentence are dropped with it, and reading goes on with the next line,
//!   so the rest of that sentence is read as a sentence of its own.
//! - A line whose id holds a `-` (a multiword token) or a `.` (an empty
//!   node) is passed over, whatever else the id is; the words are the
//!   other lines. A sentence of comments, multiword tokens and empty nodes
//!   only has no words, so it does not end there: its lines join the next
//!   sentence's, and at the end of the text it ends the stream.
//!
//! A sentence becomes a sample when its words are numbered `1`, `2`, ...
//! in order (plain ASCII decimals without a leading zero that fit a Java
//! `int`), no selected tag or relation is `_`, every head is `0` or such
//! a decimal, and the heads and relations make a [`DependencyTree`]. Any
//! other sentence is skipped: Java counts it and logs the count at the end
//! of the text, and [`ConlluDependencyReader::skipped`] keeps each one with
//! its reason ([`SkipReason`]).
//!
//! The enhanced dependencies (the `DEPS` column) and the empty nodes they
//! use are not read, as Java does not read them.
//!
//! Beyond Java, the reader checks the ids of the multiword token and empty
//! node lines of each sentence it reads against the UD rules
//! ([`super::node_ids`]) and keeps each line that breaks them
//! ([`ConlluDependencyReader::node_id_issues`]); the samples and the
//! skipped sentences are Java's whatever the check finds.

use turbo_analysis::Span;
use turbo_unicode::WhitespaceMode;

use super::Tagset;
use super::node_ids::{NodeIdIssue, NodeLine, check_node_ids};
use crate::error::FormatError;
use crate::sample::dependency::{DependencySample, DependencyTree, Head, TreeError};
use crate::source::{Line, Lines, Piece};

/// The number of columns of a word line.
const COLUMNS: usize = 10;

/// The byte order mark, which the stream drops from the first line.
const BYTE_ORDER_MARK: char = '\u{feff}';

/// Why a sentence gave no sample.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SkipReason {
    /// A word whose id is not its place in the sentence counting from 1,
    /// written as a plain decimal. `expected` is that place.
    WordId { line: u32, expected: u32 },
    /// A word whose tag in the selected tag set is `_`.
    PlaceholderTag { line: u32 },
    /// A word whose relation is `_`.
    PlaceholderRelation { line: u32 },
    /// A word whose head is neither `0` nor a plain decimal that fits a
    /// Java `int`.
    Head { line: u32 },
    /// Heads and relations that are not a tree.
    Tree(TreeError),
}

/// A sentence the reader skipped, as Java skips and counts it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Skipped {
    /// The bytes of the sentence's word lines, from its first to its last.
    pub source: Span,
    pub reason: SkipReason,
}

/// One word line: its columns as pieces of the source, and its number.
struct WordColumns<'a> {
    columns: [Piece<'a>; COLUMNS],
    line: u32,
}

impl<'a> WordColumns<'a> {
    fn id(&self) -> Piece<'a> {
        self.columns[0]
    }
    fn form(&self) -> Piece<'a> {
        self.columns[1]
    }
    fn tag(&self, tagset: Tagset) -> Piece<'a> {
        match tagset {
            Tagset::U => self.columns[3],
            Tagset::X => self.columns[4],
        }
    }
    fn head(&self) -> Piece<'a> {
        self.columns[6]
    }
    fn relation(&self) -> Piece<'a> {
        self.columns[7]
    }
    /// From the start of the id to the end of the last column.
    fn span(&self) -> Span {
        Span::new(self.columns[0].span.start, self.columns[COLUMNS - 1].span.end)
    }
}

/// Reads dependency samples from CoNLL-U: `ConlluDependencySampleStream`.
#[derive(Clone, Debug)]
pub struct ConlluDependencyReader<'a> {
    lines: Lines<'a>,
    first_line: bool,
    tagset: Tagset,
    mode: WhitespaceMode,
    skipped: Vec<Skipped>,
    /// The multiword token and empty node lines of the sentence being read.
    nodes: Vec<NodeLine<'a>>,
    node_id_issues: Vec<NodeIdIssue>,
}

impl<'a> ConlluDependencyReader<'a> {
    /// A reader of `source` that takes its tags from `tagset` and tests
    /// lines and relations for whitespace in `mode`.
    pub fn new(source: &'a str, tagset: Tagset, mode: WhitespaceMode) -> Result<Self, FormatError> {
        Ok(ConlluDependencyReader {
            lines: Lines::new(source)?,
            first_line: true,
            tagset,
            mode,
            skipped: Vec::new(),
            nodes: Vec::new(),
            node_id_issues: Vec::new(),
        })
    }

    /// The sentences skipped so far, in the order read.
    pub fn skipped(&self) -> &[Skipped] {
        &self.skipped
    }

    /// The multiword token and empty node lines read so far whose ids
    /// break the UD rules, in the order read. The lines of a sentence are
    /// checked when the sentence ends, whether it gives a sample or is
    /// skipped; a sentence dropped for a line without ten columns is not.
    pub fn node_id_issues(&self) -> &[NodeIdIssue] {
        &self.node_id_issues
    }

    /// `nextSentence`: the word lines of the next sentence, none at the end
    /// of the text.
    fn next_sentence(&mut self) -> Result<Vec<WordColumns<'a>>, FormatError> {
        let mut words = Vec::new();
        self.nodes.clear();
        while let Some(mut line) = self.lines.next() {
            if std::mem::take(&mut self.first_line) && line.text().starts_with(BYTE_ORDER_MARK) {
                let len = line.text().len();
                line = Line { piece: line.piece.slice(BYTE_ORDER_MARK.len_utf8(), len), number: line.number };
            }
            if line.text().chars().all(|c| self.mode.is_whitespace(c as u32)) {
                if !words.is_empty() {
                    return Ok(words);
                }
                continue;
            }
            if line.text().starts_with('#') {
                continue;
            }
            let word = split_columns(line)?;
            let id = word.id().text;
            if !id.contains('-') && !id.contains('.') {
                words.push(word);
            } else {
                self.nodes.push(NodeLine { id: word.id(), line: word.line, words_before: words.len() as u32 });
            }
        }
        Ok(words)
    }

    /// `convert`: the sample of a sentence's words, or why there is none.
    fn convert(&self, words: &[WordColumns<'a>]) -> Result<DependencySample<'a>, SkipReason> {
        let mut tokens = Vec::with_capacity(words.len());
        let mut tags = Vec::with_capacity(words.len());
        let mut heads = Vec::with_capacity(words.len());
        let mut relations = Vec::with_capacity(words.len());
        for (place, word) in (1u32..).zip(words) {
            let line = word.line;
            if word_index(word.id().text) != Some(place) {
                return Err(SkipReason::WordId { line, expected: place });
            }
            let tag = word.tag(self.tagset);
            if tag.text == "_" {
                return Err(SkipReason::PlaceholderTag { line });
            }
            if word.relation().text == "_" {
                return Err(SkipReason::PlaceholderRelation { line });
            }
            let head = match word.head().text {
                "0" => Head::Root,
                h => Head::Word(word_index(h).ok_or(SkipReason::Head { line })? - 1),
            };
            tokens.push(word.form().field());
            tags.push(tag.field());
            heads.push(head);
            relations.push(word.relation().field());
        }
        let tree = DependencyTree::new(heads, relations, self.mode).map_err(SkipReason::Tree)?;
        let source = Span::new(words[0].span().start, words[words.len() - 1].span().end);
        Ok(DependencySample::new(tokens, tags, tree, source).expect("one tag and one head per token"))
    }
}

impl<'a> Iterator for ConlluDependencyReader<'a> {
    type Item = Result<DependencySample<'a>, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let words = match self.next_sentence() {
                Ok(words) => words,
                Err(e) => return Some(Err(e)),
            };
            if words.is_empty() {
                return None;
            }
            self.node_id_issues.extend(check_node_ids(&self.nodes, words.len() as u32));
            match self.convert(&words) {
                Ok(sample) => return Some(Ok(sample)),
                Err(reason) => {
                    let source = Span::new(words[0].span().start, words[words.len() - 1].span().end);
                    self.skipped.push(Skipped { source, reason });
                }
            }
        }
    }
}

/// `StringUtil.split(line, '\t', -1)`, which must give ten columns.
fn split_columns(line: Line<'_>) -> Result<WordColumns<'_>, FormatError> {
    let mut columns = [line.piece.slice(0, 0); COLUMNS];
    let mut count = 0;
    let mut start = 0;
    let text = line.text();
    for end in text.match_indices('\t').map(|(i, _)| i).chain([text.len()]) {
        if count < COLUMNS {
            columns[count] = line.piece.slice(start, end);
        }
        count += 1;
        start = end + 1;
    }
    if count != COLUMNS {
        return Err(FormatError::DependencyColumns { line: line.number, count, text: text.to_owned() });
    }
    Ok(WordColumns { columns, line: line.number })
}

/// `wordIndex`: a plain ASCII decimal without a leading zero that fits a
/// Java `int`, which is at least 1.
fn word_index(id: &str) -> Option<u32> {
    if id.is_empty() || id.starts_with('0') || !id.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    id.parse::<i32>().ok().map(|v| v as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(id: &str, form: &str, upos: &str, head: &str, rel: &str) -> String {
        format!("{id}\t{form}\t_\t{upos}\tx{upos}\t_\t{head}\t{rel}\t_\t_\n")
    }

    fn read(src: &str, mode: WhitespaceMode) -> (Vec<Result<String, String>>, Vec<Skipped>) {
        let mut r = ConlluDependencyReader::new(src, Tagset::U, mode).unwrap();
        let out = r.by_ref().map(|s| s.map(|s| s.to_string()).map_err(|e| e.to_string())).collect();
        (out, r.skipped().to_vec())
    }

    #[test]
    fn multiword_tokens_and_empty_nodes_are_left_out() {
        let src = format!(
            "\u{feff}# sent_id = 1\n{}{}{}{}{}{}",
            line("1", "Vamos", "VERB", "0", "root"),
            line("2-3", "al", "_", "_", "_"),
            line("2", "a", "ADP", "4", "case"),
            line("3", "el", "DET", "4", "det"),
            line("3.1", "e", "X", "_", "_"),
            line("4", "mar", "NOUN", "1", "obl"),
        );
        let mut r = ConlluDependencyReader::new(&src, Tagset::X, WhitespaceMode::Unicode).unwrap();
        let s = r.next().unwrap().unwrap();
        assert_eq!(
            s.to_string(),
            "1\tVamos\txVERB\t0\troot\n2\ta\txADP\t4\tcase\n3\tel\txDET\t4\tdet\n4\tmar\txNOUN\t1\tobl\n"
        );
        assert_eq!(&src[s.tokens()[3].span.range()], "mar");
        assert_eq!(&src[s.tags()[0].span.range()], "xVERB");
        assert_eq!(&src[s.tree().relations()[1].span.range()], "case");
        assert!(src[s.source().range()].starts_with("1\tVamos") && src[s.source().range()].ends_with("obl\t_\t_"));
        assert!(r.next().is_none());
    }

    #[test]
    fn node_ids_are_checked_without_changing_the_samples() {
        use crate::conllu::node_ids::NodeIdError;
        let src = format!(
            "{}{}{}{}{}\n{}",
            line("1", "a", "X", "0", "root"),
            line("1.2", "e", "X", "_", "_"),
            line("3-2", "bc", "_", "_", "_"),
            line("2", "b", "X", "1", "dep"),
            line("3", "c", "X", "1", "dep"),
            line("1", "d", "X", "0", "root"),
        );
        let mut r = ConlluDependencyReader::new(&src, Tagset::U, WhitespaceMode::Unicode).unwrap();
        assert_eq!(r.by_ref().filter(Result::is_ok).count(), 2);
        let issues: Vec<(u32, NodeIdError, &str)> =
            r.node_id_issues().iter().map(|i| (i.line, i.error, &src[i.id.range()])).collect();
        assert_eq!(
            issues,
            [(2, NodeIdError::EmptyNodeIndex { expected: 1 }, "1.2"), (3, NodeIdError::MultiwordSyntax, "3-2")]
        );
    }

    #[test]
    fn invalid_sentences_are_skipped_with_their_reason() {
        let ok = line("1", "a", "X", "0", "root");
        let src = [
            line("2", "a", "X", "0", "root"),
            line("1", "a", "_", "0", "root"),
            line("1", "a", "X", "0", "_"),
            line("1", "a", "X", "00", "root"),
            line("1", "a", "X", "+1", "root"),
            format!("{}{}", line("1", "a", "X", "2", "x"), line("2", "b", "X", "1", "y")),
            format!("{}{}", line("1", "a", "X", "0", "x"), line("2", "b", "X", "3", "y")),
            line("1", "a", "X", "0", " "),
            ok.clone(),
        ]
        .join("\n");
        let (samples, skipped) = read(&src, WhitespaceMode::Unicode);
        assert_eq!(samples, [Ok("1\ta\tX\t0\troot\n".to_owned())]);
        let reasons: Vec<SkipReason> = skipped.into_iter().map(|s| s.reason).collect();
        assert_eq!(
            reasons,
            [
                SkipReason::WordId { line: 1, expected: 1 },
                SkipReason::PlaceholderTag { line: 3 },
                SkipReason::PlaceholderRelation { line: 5 },
                SkipReason::Head { line: 7 },
                SkipReason::Head { line: 9 },
                SkipReason::Tree(TreeError::Roots { count: 0 }),
                SkipReason::Tree(TreeError::HeadOutOfRange { word: 1, head: 2 }),
                SkipReason::Tree(TreeError::BlankRelation { word: 0 }),
            ]
        );
    }

    #[test]
    fn a_bad_line_drops_its_sentence_so_far() {
        let src = format!(
            "{}1\tshort\n{}{}\n{}",
            line("1", "a", "X", "0", "root"),
            line("1", "b", "X", "0", "root"),
            line("2", "c", "X", "1", "dep"),
            line("1", "d", "X", "0", "root"),
        );
        let (samples, _) = read(&src, WhitespaceMode::Unicode);
        assert_eq!(
            samples,
            [
                Err("CoNLL-U word line has 2 columns, expected 10: 1\tshort".to_owned()),
                Ok("1\tb\tX\t0\troot\n2\tc\tX\t1\tdep\n".to_owned()),
                Ok("1\td\tX\t0\troot\n".to_owned()),
            ]
        );
        // An empty last column counts.
        let (samples, _) = read("1\ta\t_\tX\t_\t_\t0\troot\t_\t\n", WhitespaceMode::Unicode);
        assert_eq!(samples, [Ok("1\ta\tX\t0\troot\n".to_owned())]);
        let (samples, _) = read("1\ta\t_\tX\t_\t_\t0\troot\t_\t_\t\n", WhitespaceMode::Unicode);
        assert_eq!(
            samples[0].as_ref().unwrap_err(),
            "CoNLL-U word line has 11 columns, expected 10: 1\ta\t_\tX\t_\t_\t0\troot\t_\t_\t"
        );
    }

    #[test]
    fn blank_lines_follow_the_whitespace_mode() {
        let src = format!("{}\u{85}\n{}", line("1", "a", "X", "0", "root"), line("1", "b", "X", "0", "root"));
        // In the Unicode mode U+0085 ends the first sentence; in the legacy
        // mode it is a word line of one column.
        assert_eq!(read(&src, WhitespaceMode::Unicode).0.len(), 2);
        let (legacy, _) = read(&src, WhitespaceMode::Legacy);
        assert_eq!(legacy[0], Err("CoNLL-U word line has 1 columns, expected 10: \u{85}".to_owned()));
    }

    #[test]
    fn a_sentence_without_words_joins_the_next() {
        let src = format!(
            "# a\n{}\n{}{}\n{}",
            line("1-2", "ab", "_", "_", "_"),
            line("1", "a", "X", "0", "root"),
            line("2", "b", "X", "1", "dep"),
            line("1.1", "e", "X", "_", "_"),
        );
        let (samples, skipped) = read(&src, WhitespaceMode::Unicode);
        assert_eq!(samples.len(), 1);
        assert!(skipped.is_empty());
    }

    #[test]
    fn word_ids_are_java_ints() {
        assert_eq!(word_index("2147483647"), Some(2147483647));
        for bad in ["", "0", "01", "2147483648", "+1", "-1", "1a", "\u{661}"] {
            assert_eq!(word_index(bad), None, "{bad}");
        }
    }
}
