//! Mixed-language documents with gold spans, and how close a detector's
//! runs come to them.
//!
//! The file format (`tests/data/mixed-*.txt`): a line `## <id>` starts a
//! document, whose text is the lines after it up to the next `## ` line
//! or the end of the file, without the last line break. Inside the text,
//! `[xxx]` starts a span of language `xxx` at that point and is not part
//! of the text; every document starts with one. Lines starting with `# `
//! (or a lone `#`) are comments, and blank lines at the end of a document
//! are dropped.
//!
//! Where a boundary sits among the whitespace and punctuation between two
//! words is a matter of convention (does an opening quotation mark go
//! with the quote?), so boundaries are compared at the first letter at or
//! after them: two boundaries match when no letter lies between them.

use std::fmt;

use turbo_analysis::Span;
use turbo_unicode::category::general_category;
use turbo_unicode::script::script;

use crate::scripts::group;

/// One document: its text and its gold spans, which cover it from the
/// first marker to the end, in order.
#[derive(Clone, Debug, PartialEq)]
pub struct GoldDocument {
    pub id: String,
    pub text: String,
    pub spans: Vec<(Span, String)>,
}

/// Why a fixture file could not be read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseError {
    pub line: usize,
    pub message: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for ParseError {}

/// Reads every document of a fixture file.
pub fn parse(source: &str) -> Result<Vec<GoldDocument>, ParseError> {
    let mut docs = Vec::new();
    let mut current: Option<(usize, String, Vec<&str>)> = None;
    for (n, line) in source.lines().enumerate() {
        if let Some(id) = line.strip_prefix("## ") {
            if let Some(doc) = current.take() {
                docs.push(document(doc)?);
            }
            current = Some((n + 1, id.trim().to_string(), Vec::new()));
        } else if line == "#" || line.starts_with("# ") {
            continue;
        } else if let Some((_, _, lines)) = &mut current {
            lines.push(line);
        } else if !line.trim().is_empty() {
            return Err(ParseError { line: n + 1, message: "text before the first document".into() });
        }
    }
    if let Some(doc) = current.take() {
        docs.push(document(doc)?);
    }
    Ok(docs)
}

fn document((line, id, mut lines): (usize, String, Vec<&str>)) -> Result<GoldDocument, ParseError> {
    while lines.last().is_some_and(|l| l.trim().is_empty()) {
        lines.pop();
    }
    let marked = lines.join("\n");
    let err = |message: String| ParseError { line, message: format!("{id}: {message}") };
    let mut text = String::with_capacity(marked.len());
    let mut starts: Vec<(u32, String)> = Vec::new();
    let mut rest = marked.as_str();
    while let Some(open) = rest.find('[') {
        text.push_str(&rest[..open]);
        let close = rest[open..].find(']').ok_or_else(|| err("a marker without ]".into()))? + open;
        let code = &rest[open + 1..close];
        if code.len() != 3 || !code.bytes().all(|b| b.is_ascii_lowercase()) {
            return Err(err(format!("{code:?} is not a three-letter code")));
        }
        starts.push((text.len() as u32, code.to_string()));
        rest = &rest[close + 1..];
    }
    text.push_str(rest);
    if starts.first().is_none_or(|(s, _)| *s != 0) {
        return Err(err("the text does not start with a marker".into()));
    }
    let mut spans = Vec::with_capacity(starts.len());
    for (i, (start, code)) in starts.iter().enumerate() {
        let end = starts.get(i + 1).map_or(text.len() as u32, |(s, _)| *s);
        if end == *start {
            return Err(err(format!("an empty span of {code}")));
        }
        spans.push((Span::new(*start, end), code.clone()));
    }
    Ok(GoldDocument { id, text, spans })
}

/// A letter as the detector counts letters: a letter or mark of a
/// script group.
fn letter(c: char) -> bool {
    let cat = general_category(c);
    (cat.is_letter() || cat.is_mark()) && group(script(c)).is_some()
}

/// The first letter at or after `at`, or the end of the text.
fn next_letter(text: &str, at: u32) -> u32 {
    text[at as usize..].char_indices().find(|&(_, c)| letter(c)).map_or(text.len() as u32, |(i, _)| at + i as u32)
}

/// Totals over a set of documents.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Scores {
    pub documents: usize,
    /// Documents whose predicted spans match the gold spans exactly: as
    /// many, with the same languages and matching boundaries.
    pub exact_documents: usize,
    pub gold_spans: usize,
    /// Gold spans whose letters are mostly given their gold language.
    pub correct_spans: usize,
    pub gold_boundaries: usize,
    pub predicted_boundaries: usize,
    /// Predicted boundaries that match a gold boundary (and so gold
    /// boundaries matched by one).
    pub matched_boundaries: usize,
    pub letters: usize,
    pub correct_letters: usize,
    /// Documents whose most frequent predicted language (by letters) is
    /// the gold one.
    pub correct_document_languages: usize,
}

impl Scores {
    /// Scores one document: the predicted runs as (span, code), in order.
    pub fn add(&mut self, gold: &GoldDocument, predicted: &[(Span, &str)]) {
        let text = gold.text.as_str();
        self.documents += 1;
        self.gold_spans += gold.spans.len();

        // The predicted language of every letter, by byte offset.
        let label_at = |at: u32| predicted.iter().find(|(s, _)| s.start <= at && at < s.end).map(|(_, c)| *c);
        let mut letters_by_language: Vec<(&str, usize)> = Vec::new();
        let mut gold_by_language: Vec<(&str, usize)> = Vec::new();
        let mut all_spans_right = predicted.len() == gold.spans.len();
        for (span, code) in &gold.spans {
            let mut tally: Vec<(&str, usize)> = Vec::new();
            for (i, c) in text[span.range()].char_indices() {
                if !letter(c) {
                    continue;
                }
                let label = label_at(span.start + i as u32).unwrap_or("");
                bump(&mut tally, label);
                bump(&mut letters_by_language, label);
                bump(&mut gold_by_language, code);
                self.letters += 1;
                self.correct_letters += usize::from(label == code);
            }
            let majority = top(&tally);
            self.correct_spans += usize::from(majority == Some(code.as_str()));
        }

        let gold_bounds: Vec<u32> = gold.spans.iter().skip(1).map(|(s, _)| next_letter(text, s.start)).collect();
        let predicted_bounds: Vec<u32> = predicted.iter().skip(1).map(|(s, _)| next_letter(text, s.start)).collect();
        self.gold_boundaries += gold_bounds.len();
        self.predicted_boundaries += predicted_bounds.len();
        let matched = predicted_bounds.iter().filter(|b| gold_bounds.contains(b)).count();
        self.matched_boundaries += matched;
        if all_spans_right {
            all_spans_right = matched == gold_bounds.len()
                && gold.spans.iter().zip(predicted).all(|((_, g), (_, p))| g.as_str() == *p);
        }
        self.exact_documents += usize::from(all_spans_right);
        self.correct_document_languages += usize::from(top(&letters_by_language) == top(&gold_by_language));
    }

    /// The share of `part` in `whole`, 0 for an empty whole.
    fn share(part: usize, whole: usize) -> f64 {
        if whole == 0 { 0.0 } else { part as f64 / whole as f64 }
    }

    pub fn span_accuracy(&self) -> f64 {
        Scores::share(self.correct_spans, self.gold_spans)
    }

    pub fn boundary_precision(&self) -> f64 {
        Scores::share(self.matched_boundaries, self.predicted_boundaries)
    }

    pub fn boundary_recall(&self) -> f64 {
        Scores::share(self.matched_boundaries, self.gold_boundaries)
    }

    pub fn letter_accuracy(&self) -> f64 {
        Scores::share(self.correct_letters, self.letters)
    }
}

impl fmt::Display for Scores {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "documents                 {}", self.documents)?;
        writeln!(f, "documents exactly right   {}", self.exact_documents)?;
        writeln!(f, "document language right   {}", self.correct_document_languages)?;
        writeln!(
            f,
            "span accuracy             {}/{} = {:.3}",
            self.correct_spans,
            self.gold_spans,
            self.span_accuracy()
        )?;
        writeln!(
            f,
            "boundary precision        {}/{} = {:.3}",
            self.matched_boundaries,
            self.predicted_boundaries,
            self.boundary_precision()
        )?;
        writeln!(
            f,
            "boundary recall           {}/{} = {:.3}",
            self.matched_boundaries,
            self.gold_boundaries,
            self.boundary_recall()
        )?;
        write!(f, "letter accuracy           {}/{} = {:.3}", self.correct_letters, self.letters, self.letter_accuracy())
    }
}

fn bump<'a>(tally: &mut Vec<(&'a str, usize)>, key: &'a str) {
    match tally.iter_mut().find(|(k, _)| *k == key) {
        Some((_, n)) => *n += 1,
        None => tally.push((key, 1)),
    }
}

/// The key with the largest count, the first on a tie.
fn top<'a>(tally: &[(&'a str, usize)]) -> Option<&'a str> {
    let mut best: Option<(&str, usize)> = None;
    for &(k, n) in tally {
        if best.is_none_or(|(_, m)| n > m) {
            best = Some((k, n));
        }
    }
    best.map(|(k, _)| k)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FILE: &str =
        "# a comment\n\n## one\n[deu]Er sagte: [eng]\"Hello there.\"\n\n## two\n[fra]Bonjour\n[fra]encore\n";

    #[test]
    fn documents_parse_with_their_spans() {
        let docs = parse(FILE).unwrap();
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].id, "one");
        assert_eq!(docs[0].text, "Er sagte: \"Hello there.\"");
        assert_eq!(docs[0].spans, [(Span::new(0, 10), "deu".into()), (Span::new(10, 24), "eng".into())]);
        assert_eq!(docs[1].text, "Bonjour\nencore");
    }

    #[test]
    fn bad_files_say_where() {
        assert_eq!(parse("text\n## a\n[eng]x").unwrap_err().line, 1);
        assert!(parse("## a\nno marker").unwrap_err().message.contains("does not start"));
        assert!(parse("## a\n[english]x").unwrap_err().message.contains("three-letter"));
        assert!(parse("## a\n[eng").unwrap_err().message.contains("without ]"));
        assert!(parse("## a\n[eng][deu]x").unwrap_err().message.contains("empty span"));
    }

    #[test]
    fn boundaries_match_at_the_next_letter() {
        let doc = &parse(FILE).unwrap()[0];
        let mut s = Scores::default();
        // The predicted boundary is before the space, the gold one after
        // it, before the quotation mark: no letter between them.
        s.add(doc, &[(Span::new(0, 9), "deu"), (Span::new(9, 24), "eng")]);
        assert_eq!((s.matched_boundaries, s.exact_documents, s.correct_spans), (1, 1, 2));
        assert_eq!(s.letter_accuracy(), 1.0);

        let mut s = Scores::default();
        s.add(doc, &[(Span::new(0, 3), "deu"), (Span::new(3, 24), "eng")]);
        assert_eq!((s.matched_boundaries, s.predicted_boundaries, s.exact_documents), (0, 1, 0));
        // "sagte" is wrong: 5 of 17 letters.
        assert_eq!((s.correct_letters, s.letters), (12, 17));
        assert_eq!(s.correct_spans, 1);
        assert_eq!(s.correct_document_languages, 1);
    }
}
