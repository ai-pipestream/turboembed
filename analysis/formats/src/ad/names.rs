//! Name samples from Arvores Deitadas: `ADNameSampleStream`.

use std::rc::Rc;

use turbo_analysis::Span;
use turbo_unicode::java::{is_digit, is_letter, lowercase_code_points, to_upper_char};

use super::contraction::contract;
use super::metadata;
use super::sentence::{AdSentence, SentenceRead, SentenceStream};
use super::tree::{Leaf, is_ws};
use super::{AdError, NullAt};
use crate::error::FormatError;
use crate::sample::{Name, NameSample};
use crate::source::{Field, Lines, Piece};

const LITERARY_PREFIX: &str = "LIT";
const SCIENTIFIC_PREFIX: &str = "CIE";

/// The name type of a named entity tag's content (`HAREM`), such as
/// `hum` for `<NER:hum>` or `<hum>`.
fn harem(tag: &str) -> Option<&'static str> {
    Some(match tag {
        "hum" | "official" | "member" => "person",
        "admin" | "org" | "inst" | "media" | "party" | "suborg" => "organization",
        "groupind" | "groupofficial" => "group",
        "top" | "civ" | "address" | "site" | "virtual" | "astro" => "place",
        "occ" | "event" | "history" => "event",
        "tit" | "pub" | "product" | "V" | "artwork" => "artprod",
        "brand" | "genre" | "school" | "idea" | "plan" | "author" | "absname" | "disease" => "abstract",
        "object" | "common" | "mat" | "class" | "plant" => "thing",
        "date" | "hour" | "period" | "cyclic" => "time",
        // `currency` is put as a thing and then again as numeric.
        "quantity" | "prednum" | "currency" => "numeric",
        _ => return None,
    })
}

/// The name type of a leaf's secondary tags (`getNER`): none when they
/// hold `<NER2>`, else the type of the first tag, split at whitespace, in
/// angle brackets with or without `NER:` that has one.
fn named_entity(tags: &str) -> Option<&'static str> {
    if tags.contains("<NER2>") {
        return None;
    }
    tags.split(is_ws).filter(|t| !t.is_empty()).find_map(|t| {
        let inner = t.strip_prefix('<')?.strip_suffix('>')?;
        harem(inner.strip_prefix("NER:").unwrap_or(inner))
    })
}

/// The parts of `s` between runs of `_` (`splitOnUnderscores`).
fn underscore_parts(s: Piece<'_>) -> impl Iterator<Item = Piece<'_>> {
    let mut start = 0;
    let text = s.text;
    std::iter::from_fn(move || {
        let rest = &text[start..];
        let skip = rest.len() - rest.trim_start_matches('_').len();
        let from = start + skip;
        if from == text.len() {
            return None;
        }
        let end = text[from..].find('_').map_or(text.len(), |i| from + i);
        start = end;
        Some(s.slice(from, end))
    })
}

/// Whether every character is a letter or a decimal digit, and there is
/// one (`isAlphaNumeric`).
fn is_alphanumeric(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| is_letter(c) || is_digit(c))
}

/// Whether `s` is longer than one UTF-16 unit.
fn longer_than_one_unit(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        None => false,
        Some(c) => c.len_utf16() > 1 || chars.next().is_some(),
    }
}

/// The end of the run of letters from `from`.
fn letters_end(s: &str, from: usize) -> usize {
    s[from..].char_indices().find(|&(_, c)| !is_letter(c)).map_or(s.len(), |(i, _)| from + i)
}

/// A hyphenated token's parts (`matchHyphenatedToken`): the letters
/// before the hyphen, the hyphen, the letters after it, and the rest.
struct Hyphenated<'a> {
    first: Option<Piece<'a>>,
    hyphen: Piece<'a>,
    second: Option<Piece<'a>>,
    rest: Option<Piece<'a>>,
}

fn split_hyphenated(tok: Piece<'_>) -> Option<Hyphenated<'_>> {
    let t = tok.text;
    let b = t.as_bytes();
    let len = t.len();
    if b[len - 1] == b'-' && letters_end(t, 0) == len - 1 {
        let hyphen = tok.slice(len - 1, len);
        return Some(Hyphenated { first: Some(tok.slice(0, len - 1)), hyphen, second: None, rest: None });
    }
    if b[0] == b'-' {
        let end = letters_end(t, 1);
        return (end > 1).then(|| Hyphenated {
            first: None,
            hyphen: tok.slice(0, 1),
            second: Some(tok.slice(1, end)),
            rest: Some(tok.slice(end, len)),
        });
    }
    let first = letters_end(t, 0);
    if first > 0 && first + 1 < len && b[first] == b'-' {
        let second = letters_end(t, first + 1);
        if second > first + 1 {
            return Some(Hyphenated {
                first: Some(tok.slice(0, first)),
                hyphen: tok.slice(first, first + 1),
                second: Some(tok.slice(first + 1, second)),
                rest: Some(tok.slice(second, len)),
            });
        }
    }
    None
}

/// Name samples read from an Arvores Deitadas text, one per sentence:
/// `ADNameSampleStream`.
///
/// The tokens are the leaves' words, split at underscores (`Rio_de_Janeiro`
/// is three tokens), with a leading `«` and a trailing `»`, `:`, `,` or
/// `!` cut off, and hyphenated words split at the hyphen when
/// `split_hyphenated_tokens` is set. A leaf tagged `<sam->` is the left
/// part of a contraction, joined with the next leaf where OpenNLP's table
/// has the pair (`de` and `o` make `do`). A leaf with a named entity tag
/// is a name of the type its tag maps to; a leaf tagged `<NER2>` extends
/// the name that ends right before its last token.
///
/// Every token is a field of the text, except a joined contraction, which
/// has a text of its own and the span from its left part to its right.
pub struct AdNameReader<'a> {
    sentences: SentenceStream<'a>,
    split_hyphenated_tokens: bool,
    /// The left part of a contraction, waiting for the next leaf; it
    /// carries over into the next sentence, as Java's field does.
    left: Option<Piece<'a>>,
    text_id: i32,
    /// The count of LIT and CIE texts seen, minus one, and the name of
    /// the last.
    literary_id: i32,
    literary_name: Rc<str>,
    done: bool,
}

impl<'a> AdNameReader<'a> {
    /// A reader of `source`, which must fit `u32` offsets.
    pub fn new(source: &'a str, split_hyphenated_tokens: bool) -> Result<AdNameReader<'a>, FormatError> {
        Ok(AdNameReader {
            sentences: SentenceStream::new(Lines::new(source)?),
            split_hyphenated_tokens,
            left: None,
            text_id: -1,
            literary_id: -1,
            literary_name: "".into(),
            done: false,
        })
    }

    /// `getTextID`: the id of the sentence's text, which clears adaptive
    /// data when it changes.
    fn text_id(&mut self, sentence: &AdSentence<'_>) -> Result<i32, AdError> {
        let meta = sentence.metadata.as_deref().ok_or(AdError::NullMetadata { at: NullAt::StartsWith })?;
        let literary = meta.starts_with(LITERARY_PREFIX);
        if literary || meta.starts_with(SCIENTIFIC_PREFIX) {
            let name = if literary { metadata::text_prefix(meta) } else { metadata::source(meta) };
            let name = name.ok_or_else(|| AdError::InvalidMetadata { metadata: meta.to_owned() })?;
            if name.is_empty() {
                return Ok(-1);
            }
            if name != &*self.literary_name {
                self.literary_id = self.literary_id.wrapping_add(1);
                self.literary_name = name.into();
            }
            return Ok(self.literary_id);
        }
        metadata::text_and_paragraph(meta)
            .map(|ids| ids.text)
            .ok_or_else(|| AdError::InvalidMetadata { metadata: meta.to_owned() })
    }

    fn sample(&mut self, sentence: AdSentence<'a>) -> Result<NameSample<'a>, FormatError> {
        let id = self.text_id(&sentence)?;
        let clear = id != self.text_id;
        self.text_id = id;
        let mut tokens = Vec::new();
        let mut names = Vec::new();
        for leaf in sentence.tree.leaves() {
            self.leaf(leaf, &mut tokens, &mut names)?;
        }
        // A left part carried over from the sentence before starts the
        // sample's source there.
        let start =
            tokens.first().map_or(sentence.source.start, |t: &Field<'_>| t.span.start.min(sentence.source.start));
        let source = Span::new(start, sentence.source.end);
        NameSample::new(tokens, names, clear, source, sentence.line)
    }

    /// `processLeaf`.
    fn leaf(&mut self, leaf: &Leaf<'a>, tokens: &mut Vec<Field<'a>>, names: &mut Vec<Name<'a>>) -> Result<(), AdError> {
        let mut joined = false;
        if let Some(left) = self.left {
            if let Some(parts) = contraction(left, leaf.lexeme)? {
                tokens.extend(parts);
                joined = true;
            } else {
                tokens.push(left.field());
            }
            self.left = None;
        }
        let mut kind = None;
        let mut extend = false;
        if let Some(tags) = leaf.secondary {
            if tags.text.contains("<sam->") && !joined {
                let mut parts = underscore_parts(leaf.lexeme).peekable();
                while let Some(part) = parts.next() {
                    if parts.peek().is_some() {
                        tokens.push(part.field());
                    } else {
                        self.left = Some(part);
                    }
                }
                return Ok(());
            }
            extend = tags.text.contains("<NER2>");
            kind = named_entity(tags.text);
        }
        let start = tokens.len() as u32;
        if !joined {
            self.lexeme(leaf.lexeme, tokens)?;
        }
        let end = tokens.len() as u32;
        if kind.is_some() {
            names.push(Name { start, end, kind, source: leaf.lexeme.span });
        }
        if extend
            && let Some(last) = names.last_mut()
            && i64::from(last.end) == i64::from(end) - 1
        {
            last.end = end;
            last.source = Span::new(last.source.start.min(leaf.lexeme.span.start), leaf.lexeme.span.end);
        }
        Ok(())
    }

    /// `processLexeme`: the parts between underscores, each longer than
    /// one unit and not alphanumeric cut further ([`Self::token`]).
    fn lexeme(&self, lexeme: Piece<'a>, tokens: &mut Vec<Field<'a>>) -> Result<(), AdError> {
        for part in underscore_parts(lexeme) {
            if longer_than_one_unit(part.text) && !is_alphanumeric(part.text) {
                self.token(part, tokens)?;
            } else {
                tokens.push(part.field());
            }
        }
        Ok(())
    }

    /// `processTok`, which Java calls again on what is left of the token
    /// after each cut; here a loop, with the cut suffixes kept to add in
    /// the order the calls return.
    fn token(&self, mut tok: Piece<'a>, tokens: &mut Vec<Field<'a>>) -> Result<(), AdError> {
        let mut suffixes = Vec::new();
        loop {
            let mut cut = false;
            if tok.text.starts_with('\u{AB}') {
                let open = '\u{AB}'.len_utf8();
                tokens.push(tok.slice(0, open).field());
                tok = tok.slice(open, tok.text.len());
                cut = true;
            }
            let last = tok.text.chars().next_back().ok_or(AdError::EmptyToken)?;
            if matches!(last, '\u{BB}' | ':' | ',' | '!') {
                let at = tok.text.len() - last.len_utf8();
                suffixes.push(tok.slice(at, tok.text.len()));
                tok = tok.slice(0, at);
                cut = true;
            }
            if self.split_hyphenated_tokens
                && tok.text.contains('-')
                && longer_than_one_unit(tok.text)
                && let Some(parts) = split_hyphenated(tok)
            {
                // The letter runs and the hyphen come out whole from
                // Java's processTok; the rest is cut again.
                tokens.extend(parts.first.filter(|p| !p.text.is_empty()).map(Piece::field));
                tokens.push(parts.hyphen.field());
                tokens.extend(parts.second.filter(|p| !p.text.is_empty()).map(Piece::field));
                match parts.rest.filter(|p| !p.text.is_empty()) {
                    Some(rest) => {
                        tok = rest;
                        continue;
                    }
                    None => break,
                }
            }
            if cut && longer_than_one_unit(tok.text) && !is_alphanumeric(tok.text) {
                continue;
            }
            tokens.push(tok.field());
            break;
        }
        tokens.extend(suffixes.into_iter().rev().map(Piece::field));
        Ok(())
    }
}

/// `toContraction` of a left part and the next leaf's word, as tokens:
/// `None` when the pair is not in the table. When the right word has
/// underscores, its first part is joined and the others follow, split at
/// whitespace; otherwise a left part that is in the table once lowercased
/// is joined and the result capitalized.
fn contraction<'a>(left: Piece<'a>, right: Piece<'a>) -> Result<Option<Vec<Field<'a>>>, AdError> {
    let span = Span::new(left.span.start.min(right.span.start), left.span.end.max(right.span.end));
    if let Some(c) = contract(left.text, right.text) {
        return Ok(Some(vec![Field::made(c.to_owned(), span)]));
    }
    // The left part has no underscore, so OpenNLP's split of it is the
    // part itself.
    if right.text.contains('_') {
        let mut parts: Vec<Piece<'a>> = right_parts(right);
        if parts.is_empty() {
            return Err(AdError::EmptyContraction);
        }
        let first = parts.remove(0);
        let Some(c) = contract(left.text, first.text) else {
            // Java tries the lowercased last part of the right word with
            // the whole right word, which has an underscore no key has.
            return Ok(None);
        };
        let joined = Span::new(span.start, left.span.end.max(first.span.end));
        let mut out = vec![Field::made(c.to_owned(), joined)];
        for part in parts {
            out.extend(whitespace_runs(part).map(Piece::field));
        }
        return Ok(Some(out));
    }
    let lower: String = lowercase_code_points(left.text);
    Ok(contract(&lower, right.text).map(|c| {
        let mut chars = c.chars();
        let first = chars.next().expect("no contraction is empty");
        let capitalized: String = std::iter::once(to_upper_char(first)).chain(chars).collect();
        vec![Field::made(capitalized, span)]
    }))
}

/// `StringUtil.split(right, '_')`: the pieces between underscores, with
/// the empty ones at the end dropped.
fn right_parts(right: Piece<'_>) -> Vec<Piece<'_>> {
    let mut out = Vec::new();
    let mut start = 0;
    for (i, c) in right.text.char_indices() {
        if c == '_' {
            out.push(right.slice(start, i));
            start = i + 1;
        }
    }
    out.push(right.slice(start, right.text.len()));
    while out.last().is_some_and(|p| p.text.is_empty()) {
        out.pop();
    }
    out
}

/// The runs of a piece that are not whitespace.
fn whitespace_runs(p: Piece<'_>) -> impl Iterator<Item = Piece<'_>> {
    let text = p.text;
    let mut at = 0;
    std::iter::from_fn(move || {
        let start = at + text[at..].char_indices().find(|&(_, c)| !is_ws(c))?.0;
        let end = text[start..].char_indices().find(|&(_, c)| is_ws(c)).map_or(text.len(), |(i, _)| start + i);
        at = end;
        Some(p.slice(start, end))
    })
}

impl<'a> Iterator for AdNameReader<'a> {
    type Item = Result<NameSample<'a>, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        match self.sentences.read() {
            Some(SentenceRead::Sentence(s)) => Some(self.sample(s)),
            Some(SentenceRead::Failed) | None => {
                self.done = true;
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(lexeme: &str, split: bool) -> Result<Vec<String>, AdError> {
        let reader = AdNameReader::new("", split).unwrap();
        let mut out = Vec::new();
        reader.lexeme(Piece { text: lexeme, span: Span::new(0, lexeme.len() as u32) }, &mut out)?;
        Ok(out.into_iter().map(|f| f.as_str().to_owned()).collect())
    }

    #[test]
    fn lexemes_are_cut_as_java_cuts_them() {
        assert_eq!(tokens("Rio__de_Janeiro_", true).unwrap(), ["Rio", "de", "Janeiro"]);
        // The rest after the hyphen, `»`, is cut again down to an empty token.
        assert_eq!(tokens("«carros-monstro»,", true).unwrap(), ["«", "carros", "-", "monstro", "", "»", ","]);
        assert_eq!(tokens("«carros-monstro»,", false).unwrap(), ["«", "carros-monstro", "»", ","]);
        assert_eq!(tokens("-ab-cd", true).unwrap(), ["-", "ab", "-", "cd"]);
        assert_eq!(tokens("ab-", true).unwrap(), ["ab", "-"]);
        assert_eq!(tokens("«»", true).unwrap(), ["«", "", "»"]);
        assert_eq!(tokens("ab-cd«", true), Err(AdError::EmptyToken));
    }

    #[test]
    fn contractions_join() {
        let p = |s: &'static str, at: u32| Piece { text: s, span: Span::new(at, at + s.len() as u32) };
        let texts = |r: Option<Vec<Field<'_>>>| r.map(|v| v.iter().map(|f| f.as_str().to_owned()).collect::<Vec<_>>());
        assert_eq!(texts(contraction(p("de", 0), p("o", 5)).unwrap()), Some(vec!["do".into()]));
        assert_eq!(texts(contraction(p("De", 0), p("o", 5)).unwrap()), Some(vec!["Do".into()]));
        assert_eq!(
            texts(contraction(p("em", 0), p("a_Rua x", 5)).unwrap()),
            Some(vec!["na".into(), "Rua".into(), "x".into()])
        );
        assert_eq!(texts(contraction(p("x", 0), p("o", 5)).unwrap()), None);
        assert_eq!(contraction(p("de", 0), p("__", 5)), Err(AdError::EmptyContraction));
        let joined = contraction(p("de", 0), p("o", 5)).unwrap().unwrap();
        assert_eq!(joined[0].span, Span::new(0, 6));
    }

    #[test]
    fn named_entity_tags() {
        assert_eq!(named_entity("<np-def> <NER:hum> <civ>"), Some("person"));
        assert_eq!(named_entity("<civ>"), Some("place"));
        assert_eq!(named_entity("<NER2> <civ>"), None);
        assert_eq!(named_entity("<currency>"), Some("numeric"));
        assert_eq!(named_entity("<x>"), None);
    }
}
