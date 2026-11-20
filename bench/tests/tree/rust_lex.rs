//! Rust source as tokens, for the tree checks that read code
//! (`duplicates.rs`, `orphans.rs`). Comments are dropped, nested block
//! comments included; string, character and lifetime tokens are kept whole
//! and unchanged, so nothing inside one is taken for code and two strings
//! that differ stay different. Every step is on a character boundary.
//!
//! This is not a parser: numbers come out as word and punctuation tokens,
//! and every punctuation character is a token of its own. The checks only
//! need words, brackets and literals.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// An identifier, keyword or the word part of a number. A raw
    /// identifier `r#name` is the word `name`.
    Word,
    /// One punctuation character.
    Punct,
    /// A string, byte string, raw string, character or lifetime, quotes and
    /// prefix included.
    Literal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Token<'a> {
    pub kind: Kind,
    pub text: &'a str,
    /// The line the token starts on, from 1.
    pub line: u32,
}

impl<'a> Token<'a> {
    pub fn is(&self, punct: char) -> bool {
        self.kind == Kind::Punct && self.text.starts_with(punct)
    }

    pub fn word(&self, word: &str) -> bool {
        self.kind == Kind::Word && self.text == word
    }

    /// The contents of a plain or raw string literal, escapes untouched.
    pub fn string(&self) -> Option<&'a str> {
        let open = self.text.find('"')?;
        let close = self.text.rfind('"')?;
        (self.kind == Kind::Literal && close > open).then(|| &self.text[open + 1..close])
    }
}

fn word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// The length of the string literal `rest` starts with, prefix and all:
/// `"..."`, `b"..."`, `c"..."`, `r"..."`, `r#"..."#`, `br#"..."#`.
fn string_len(rest: &str) -> Option<usize> {
    let prefix = ["br", "cr", "r", "b", "c", ""].into_iter().find(|p| {
        let after = rest.strip_prefix(p).unwrap_or("\0");
        if p.ends_with('r') { after.trim_start_matches('#').starts_with('"') } else { after.starts_with('"') }
    })?;
    let after = &rest[prefix.len()..];
    if prefix.ends_with('r') {
        let hashes = after.len() - after.trim_start_matches('#').len();
        let close = format!("\"{}", "#".repeat(hashes));
        let body = &after[hashes + 1..];
        return Some(prefix.len() + hashes + 1 + body.find(&close).map_or(body.len(), |end| end + close.len()));
    }
    let mut chars = after.char_indices().skip(1);
    while let Some((at, c)) = chars.next() {
        match c {
            '\\' => drop(chars.next()),
            '"' => return Some(prefix.len() + at + 1),
            _ => {}
        }
    }
    Some(rest.len())
}

/// The length of the character literal or lifetime `rest` (which starts
/// with `'`, or `b'`) starts with: `'x'`, `'\n'`, `'\u{1F600}'`, `'a`.
fn quote_len(rest: &str) -> usize {
    let lead = rest.find('\'').unwrap() + 1;
    let mut chars = rest[lead..].char_indices();
    match chars.next() {
        // the escaped character is skipped, so that '\'' ends at its last quote
        Some((_, '\\')) => {
            chars.next();
            chars.find(|&(_, c)| c == '\'').map_or(rest.len(), |(at, _)| lead + at + 1)
        }
        Some((_, first)) => match chars.next() {
            Some((at, '\'')) => lead + at + 1,
            // a lifetime: the quote and the word after it
            _ if word_char(first) => lead + rest[lead..].find(|c| !word_char(c)).unwrap_or(rest.len() - lead),
            _ => lead,
        },
        None => lead,
    }
}

pub fn tokens(text: &str) -> Vec<Token<'_>> {
    let mut out = Vec::new();
    let mut line = 1u32;
    let mut i = 0;
    while let Some(c) = text[i..].chars().next() {
        let rest = &text[i..];
        // how far this token or comment reaches, and its kind when it is a token
        let (len, kind) = if c.is_whitespace() {
            (c.len_utf8(), None)
        } else if rest.starts_with("//") {
            (rest.find('\n').unwrap_or(rest.len()), None)
        } else if rest.starts_with("/*") {
            let (mut depth, mut at) = (1usize, 2);
            while depth > 0 && at < rest.len() {
                let (open, close) = (rest[at..].find("/*"), rest[at..].find("*/"));
                match (open, close) {
                    (Some(o), Some(c)) if o < c => (depth, at) = (depth + 1, at + o + 2),
                    (_, Some(c)) => (depth, at) = (depth - 1, at + c + 2),
                    _ => at = rest.len(),
                }
            }
            (at, None)
        } else if let Some(len) = string_len(rest) {
            (len, Some(Kind::Literal))
        } else if c == '\'' || rest.starts_with("b'") {
            (quote_len(rest), Some(Kind::Literal))
        } else if word_char(c) {
            // r#name is the word name
            let raw = rest.strip_prefix("r#").filter(|r| r.starts_with(word_char)).map_or(0, |_| 2);
            if raw > 0 {
                i += raw;
                continue;
            }
            (rest.find(|c| !word_char(c)).unwrap_or(rest.len()), Some(Kind::Word))
        } else {
            (c.len_utf8(), Some(Kind::Punct))
        };
        let piece = &text[i..i + len];
        if let Some(kind) = kind {
            out.push(Token { kind, text: piece, line });
        }
        line += piece.matches('\n').count() as u32;
        i += len;
    }
    out
}

/// The index of the bracket that closes the one at `open`, or the last
/// token when it is never closed. Brackets of all three kinds nest.
pub fn closing(tokens: &[Token<'_>], open: usize) -> usize {
    let mut depth = 0usize;
    for (at, token) in tokens.iter().enumerate().skip(open) {
        if token.is('(') || token.is('[') || token.is('{') {
            depth += 1;
        } else if token.is(')') || token.is(']') || token.is('}') {
            depth = depth.saturating_sub(1);
            if depth == 0 {
                return at;
            }
        }
    }
    tokens.len().saturating_sub(1)
}

fn texts(source: &str) -> Vec<&str> {
    tokens(source).iter().map(|t| t.text).collect()
}

#[test]
fn comments_are_dropped_and_block_comments_nest() {
    assert_eq!(texts("a // b\nc /* d /* e */ f */ g /* h"), ["a", "c", "g"]);
    assert_eq!(texts("x /** doc */ y //! inner\n/// outer\nz"), ["x", "y", "z"]);
    let lines: Vec<u32> = tokens("a /* 1\n2 */ b\n// c\nd").iter().map(|t| t.line).collect();
    assert_eq!(lines, [1, 2, 4]);
}

#[test]
fn literals_are_one_token_whatever_they_hold() {
    assert_eq!(texts(r#"f("} // {", b"\"", c"x")"#), ["f", "(", r#""} // {""#, ",", r#"b"\"""#, ",", r#"c"x""#, ")"]);
    assert_eq!(texts(r###"r#"a "quoted" }"# br##"b"# c"## r"d" x"###).len(), 4);
    assert_eq!(
        texts(r"'{' '\'' '\\' '\u{1F600}' b'}' '\n'"),
        ["'{'", r"'\''", r"'\\'", r"'\u{1F600}'", "b'}'", r"'\n'"]
    );
    let raw = tokens("r#\"  two\n  lines\"# after");
    assert_eq!((raw[0].string(), raw[1].text, raw[1].line), (Some("  two\n  lines"), "after", 2));
}

#[test]
fn lifetimes_raw_identifiers_and_unicode_words_are_read() {
    assert_eq!(
        texts("fn f<'a>(x: &'a str) -> &'static str"),
        ["fn", "f", "<", "'a", ">", "(", "x", ":", "&", "'a", "str", ")", "-", ">", "&", "'static", "str"]
    );
    assert_eq!(
        texts("mod r#type; let café = 1; let 名前 = 'é';"),
        ["mod", "type", ";", "let", "café", "=", "1", ";", "let", "名前", "=", "'é'", ";"]
    );
    // text that ends inside a token is read to its end, not past it
    for cut in ["\"open", "'", "'\\", "r#\"open", "/* open", "b'", "é"] {
        let _ = tokens(cut);
    }
}

#[test]
fn closing_matches_nested_brackets() {
    let t = tokens("f([a, {b}], c) d");
    assert_eq!(t[closing(&t, 1)].text, ")");
    assert_eq!(closing(&t, 2), 8);
    assert_eq!(closing(&tokens("{ never"), 0), 1);
}
