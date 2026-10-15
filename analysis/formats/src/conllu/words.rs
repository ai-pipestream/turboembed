//! Word-preserving CoNLL-U reader and writer for Universal Dependencies
//! multi-word tokens (`1-2 del` -> `1 de` + `2 el`).
//!
//! OpenNLP's `ConlluStream` ([`super::ConlluReader`]) merges each multi-word
//! token range into a single surface line (`al` with `a+el` and `ADP+DET`) and
//! removes the constituent syntactic word lines. Universal Dependencies POS,
//! lemma, morphological-feature and dependency annotations, however, attach to
//! syntactic words (`turbo:words`), and round-tripping CoNLL-U requires
//! keeping both the multi-word token range lines (`1-2`) and the individual
//! syntactic word lines (`1`, `2`).

use std::fmt::Write as _;

use turbo_analysis::{IxRange, Span};
use turbo_analysis_layers::Words;

use super::{RangeError, TEXT_LANG_PREFIX, TextLang, WordLine, multiword_range, text_language};
use crate::error::FormatError;
use crate::source::{Field, Line, Lines, java_trim};

/// A multi-word token range line (`start-end`, such as `2-3\tal\t_\t...`) in a
/// CoNLL-U sentence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiwordToken<'a> {
    /// 1-based starting syntactic word ID (`2` in `2-3`).
    pub start_id: u32,
    /// 1-based ending syntactic word ID (`3` in `2-3`).
    pub end_id: u32,
    /// 0-based index range into [`SyntacticSentence::words`].
    pub words: IxRange<Words>,
    /// The 10 columns of the multi-word token line, carrying its surface
    /// `form` and `misc` (such as `SpaceAfter=No` or `SpacesInToken=...`).
    pub line_data: WordLine<'a>,
}

/// A CoNLL-U sentence with its syntactic words, multi-word token ranges, empty
/// nodes and comments preserved without lossy merging.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SyntacticSentence<'a> {
    /// Raw comment lines in the order they appeared in the sentence header.
    pub comments: Vec<Line<'a>>,
    /// Syntactic word lines (`1..=N`) in order, never merged or removed.
    pub words: Vec<WordLine<'a>>,
    /// Multi-word token range lines (`start-end`), in the order they appear.
    pub multiwords: Vec<MultiwordToken<'a>>,
    /// Enhanced-dependency empty node lines (`x.y`), paired with the index in
    /// `words` before which they appeared (`words.len()` when after the last
    /// word).
    pub empty_nodes: Vec<(usize, WordLine<'a>)>,
    /// The `sent_id` comment.
    pub sentence_id: Option<Field<'a>>,
    /// The `text` comment.
    pub text: Option<Field<'a>>,
    /// Whether a `newdoc` comment starts a document here.
    pub new_document: bool,
    pub document_id: Option<Field<'a>>,
    /// Whether a `newpar` comment starts a paragraph here.
    pub new_paragraph: bool,
    pub paragraph_id: Option<Field<'a>>,
    /// The `text_<language>` comments.
    pub text_lang: Vec<TextLang<'a>>,
    /// The `translit` comment.
    pub translit: Option<Field<'a>>,
    /// The 1-based line number of the sentence's first line.
    pub line: u32,
    /// The byte span of the sentence in the CoNLL-U source.
    pub source: Span,
}

/// Reads [`SyntacticSentence`]s from CoNLL-U text while preserving both
/// multi-word token ranges (`start-end`) and their constituent syntactic word
/// lines.
#[derive(Clone, Debug)]
pub struct ConlluWordReader<'a> {
    lines: Lines<'a>,
}

impl<'a> ConlluWordReader<'a> {
    pub fn new(source: &'a str) -> Result<ConlluWordReader<'a>, FormatError> {
        Ok(ConlluWordReader { lines: Lines::new(source)? })
    }

    fn block(&mut self) -> Vec<Line<'a>> {
        let mut block = Vec::new();
        for line in self.lines.by_ref() {
            if !line.text().is_empty() {
                block.push(line);
            } else if !block.is_empty() {
                break;
            }
        }
        block
    }
}

impl<'a> Iterator for ConlluWordReader<'a> {
    type Item = Result<SyntacticSentence<'a>, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        let block = self.block();
        let (first, last) = (block.first()?, block.last()?);
        let source = Span::new(first.span().start, last.span().end);
        Some(read_syntactic_sentence(&block, first.number, source))
    }
}

fn read_syntactic_sentence<'a>(
    block: &[Line<'a>],
    first: u32,
    source: Span,
) -> Result<SyntacticSentence<'a>, FormatError> {
    let mut sentence = SyntacticSentence {
        comments: Vec::new(),
        words: Vec::new(),
        multiwords: Vec::new(),
        empty_nodes: Vec::new(),
        sentence_id: None,
        text: None,
        new_document: false,
        document_id: None,
        new_paragraph: false,
        paragraph_id: None,
        text_lang: Vec::new(),
        translit: None,
        line: first,
        source,
    };

    let mut pending_mwt: Vec<(u32, u32, WordLine<'a>)> = Vec::new();
    for &line in block {
        let trimmed = java_trim(line.piece);
        if !trimmed.text.starts_with('#') {
            let parsed = WordLine::parse(line)?;
            let id = parsed.id.as_str();
            if id.contains('-') {
                let (start, end) = multiword_range(id).map_err(|e| match e {
                    RangeError::Invalid => FormatError::InvalidMultiwordId { line: parsed.line, id: id.to_owned() },
                    RangeError::Backwards => FormatError::BackwardsMultiwordId { line: parsed.line, id: id.to_owned() },
                })?;
                pending_mwt.push((start, end, parsed));
            } else if id.contains('.') {
                sentence.empty_nodes.push((sentence.words.len(), parsed));
            } else {
                sentence.words.push(parsed);
            }
            continue;
        }
        sentence.comments.push(line);
        let comment = trimmed.slice(1, trimmed.text.len());
        let Some(eq) = comment.text.find('=') else {
            match java_trim(comment).text {
                "newdoc" => sentence.new_document = true,
                "newpar" => sentence.new_paragraph = true,
                _ => {}
            }
            continue;
        };
        let key = java_trim(comment.slice(0, eq));
        let value = java_trim(comment.slice(eq + 1, comment.text.len()));
        if !value.text.is_empty() {
            match key.text {
                "newdoc id" => {
                    sentence.new_document = true;
                    sentence.document_id = Some(value.field());
                }
                "newpar id" => {
                    sentence.new_paragraph = true;
                    sentence.paragraph_id = Some(value.field());
                }
                "sent_id" => sentence.sentence_id = Some(value.field()),
                "text" => sentence.text = Some(value.field()),
                "translit" => sentence.translit = Some(value.field()),
                _ => {}
            }
        }
        if key.text.starts_with(TEXT_LANG_PREFIX) {
            let Some(language) = text_language(key) else {
                return Err(FormatError::InvalidLanguage { line: line.number, key: key.text.to_owned() });
            };
            let entry = TextLang { language: language.field(), text: value.field() };
            match sentence.text_lang.iter_mut().find(|t| t.language.as_str() == language.text) {
                Some(old) => *old = entry,
                None => sentence.text_lang.push(entry),
            }
        }
    }

    for (start_id, end_id, line_data) in pending_mwt {
        let mut first_idx = None;
        let mut last_idx = None;
        for j in u64::from(start_id)..=u64::from(end_id) {
            let target = j.to_string();
            let Some(pos) = sentence.words.iter().position(|w| w.id.as_str() == target) else {
                return Err(FormatError::MissingMultiwordLine {
                    line: line_data.line,
                    id: line_data.id.as_str().to_owned(),
                    missing: j,
                });
            };
            if first_idx.is_none() {
                first_idx = Some(pos);
            }
            last_idx = Some(pos + 1);
        }
        let words = IxRange::new(first_idx.unwrap_or(0) as u32, last_idx.unwrap_or(0) as u32);
        sentence.multiwords.push(MultiwordToken { start_id, end_id, words, line_data });
    }

    Ok(sentence)
}

/// Parses `SpaceAfter=No` and `SpacesAfter=...` from a CoNLL-U `MISC` field,
/// returning the trailing whitespace string after the token (default `" "`).
pub fn misc_trailing_space(misc: &str) -> String {
    if misc == "_" || misc.is_empty() {
        return " ".to_owned();
    }
    for part in misc.split('|') {
        if let Some(val) = part.strip_prefix("SpacesAfter=") {
            return unescape_conllu_spaces(val);
        }
        if part == "SpaceAfter=No" {
            return String::new();
        }
    }
    " ".to_owned()
}

/// Parses `SpacesInToken=...` from a CoNLL-U `MISC` field, if present.
pub fn misc_spaces_in_token(misc: &str) -> Option<String> {
    if misc == "_" || misc.is_empty() {
        return None;
    }
    for part in misc.split('|') {
        if let Some(val) = part.strip_prefix("SpacesInToken=") {
            return Some(unescape_conllu_spaces(val));
        }
    }
    None
}

/// The inverse of [`unescape_conllu_spaces`]: the whitespace between two
/// tokens as a `SpacesAfter` value.
pub(super) fn escape_conllu_spaces(gap: &str) -> String {
    let mut out = String::with_capacity(gap.len() * 2);
    for c in gap.chars() {
        match c {
            ' ' => out.push_str("\\s"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '|' => out.push_str("\\p"),
            '\\' => out.push_str("\\\\"),
            c => out.push(c),
        }
    }
    out
}

fn unescape_conllu_spaces(escaped: &str) -> String {
    let mut out = String::with_capacity(escaped.len());
    let mut chars = escaped.chars();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.next() {
                Some('s') => out.push(' '),
                Some('t') => out.push('\t'),
                Some('n') => out.push('\n'),
                Some('r') => out.push('\r'),
                Some('p') => out.push('|'),
                Some('\\') => out.push('\\'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(ch);
        }
    }
    out
}

impl<'a> SyntacticSentence<'a> {
    /// Reconstructs the surface text of the sentence from its surface tokens
    /// (multi-word tokens where present, otherwise individual syntactic words)
    /// and their `MISC` `SpaceAfter=No` / `SpacesAfter=...` attributes.
    pub fn reconstruct_surface_text(&self) -> String {
        let mut out = String::new();
        let mut w_idx = 0usize;
        let mut mwt_iter = self.multiwords.iter().peekable();
        while w_idx < self.words.len() {
            if let Some(mwt) = mwt_iter.peek()
                && mwt.words.start.get() == w_idx
            {
                let mwt = mwt_iter.next().unwrap();
                out.push_str(mwt.line_data.form.as_str());
                out.push_str(&misc_trailing_space(mwt.line_data.misc.as_str()));
                w_idx = mwt.words.end.get();
            } else {
                let w = &self.words[w_idx];
                out.push_str(w.form.as_str());
                out.push_str(&misc_trailing_space(w.misc.as_str()));
                w_idx += 1;
            }
        }
        if out.ends_with(' ') && self.text.as_ref().is_none_or(|t| !t.as_str().ends_with(' ')) {
            out.pop();
        }
        out
    }

    /// Appends this sentence in CoNLL-U format to `out`, keeping comments,
    /// multi-word token range lines (`start-end`), syntactic word lines, empty
    /// nodes and `MISC` annotations as read. A file in the format's own shape
    /// (LF line ends, one blank line after each sentence, comments before the
    /// word lines, no empty trailing columns) comes back as the same bytes.
    pub fn write_conllu(&self, out: &mut String) {
        if !self.comments.is_empty() {
            for c in &self.comments {
                out.push_str(c.text());
                out.push('\n');
            }
        } else {
            if self.new_document {
                match &self.document_id {
                    Some(id) => {
                        let _ = writeln!(out, "# newdoc id = {}", id.as_str());
                    }
                    None => out.push_str("# newdoc\n"),
                }
            }
            if self.new_paragraph {
                match &self.paragraph_id {
                    Some(id) => {
                        let _ = writeln!(out, "# newpar id = {}", id.as_str());
                    }
                    None => out.push_str("# newpar\n"),
                }
            }
            if let Some(id) = &self.sentence_id {
                let _ = writeln!(out, "# sent_id = {}", id.as_str());
            }
            if let Some(text) = &self.text {
                let _ = writeln!(out, "# text = {}", text.as_str());
            }
            for tl in &self.text_lang {
                if tl.text.as_str().is_empty() {
                    let _ = writeln!(out, "# text_{} =", tl.language.as_str());
                } else {
                    let _ = writeln!(out, "# text_{} = {}", tl.language.as_str(), tl.text.as_str());
                }
            }
            if let Some(tr) = &self.translit {
                let _ = writeln!(out, "# translit = {}", tr.as_str());
            }
        }

        let mut mwt_iter = self.multiwords.iter().peekable();
        let mut empty_iter = self.empty_nodes.iter().peekable();

        for (w_idx, word) in self.words.iter().enumerate() {
            while let Some((at, empty_node)) = empty_iter.peek()
                && *at == w_idx
            {
                write_word_line(out, empty_node);
                empty_iter.next();
            }
            while let Some(mwt) = mwt_iter.peek()
                && mwt.words.start.get() == w_idx
            {
                write_word_line(out, &mwt.line_data);
                mwt_iter.next();
            }
            write_word_line(out, word);
        }
        for (_, empty_node) in empty_iter {
            write_word_line(out, empty_node);
        }
        out.push('\n');
    }
}

fn write_word_line(out: &mut String, w: &WordLine<'_>) {
    let _ = writeln!(
        out,
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        w.id.as_str(),
        w.form.as_str(),
        w.lemma.as_str(),
        w.upos.as_str(),
        w.xpos.as_str(),
        w.feats.as_str(),
        w.head.as_str(),
        w.deprel.as_str(),
        w.deps.as_str(),
        w.misc.as_str(),
    );
}

/// Serializes a slice of [`SyntacticSentence`]s to CoNLL-U format.
pub fn write_conllu_sentences(sentences: &[SyntacticSentence<'_>]) -> String {
    let mut out = String::new();
    for s in sentences {
        s.write_conllu(&mut out);
    }
    out
}
