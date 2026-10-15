//! Writing a glossary as a TBX termbase, which [`super::TbxReader`] reads
//! back to the same entries.

use std::fmt;

use turbo_glossary_layers::{EntryIx, Glossary, is_blank, trim_blank};

use super::tbx::TBX_NAMESPACE;

/// Which TBX to write.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TbxDialect {
    /// TBX 2 (`martif`, `termEntry`, `langSet`, `tig`), in no namespace.
    Martif,
    /// TBX 3 (`tbx`, `conceptEntry`, `langSec`, `termSec`) in the ISO
    /// 30042 namespace, TBX-Basic in the DCA style.
    Tbx,
}

/// Why a glossary cannot be written so that it reads back the same.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WriteError {
    /// The language tag is blank or holds a character XML does not allow.
    Language,
    /// The entry's term starts or ends with whitespace, which the reader
    /// trims.
    Untrimmed { entry: EntryIx },
    /// The entry's identifier or term holds a character XML 1.0 cannot
    /// carry, even as a reference (a control character, U+FFFE).
    Character { entry: EntryIx },
}

impl fmt::Display for WriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WriteError::Language => f.write_str("the language tag is blank or not XML text"),
            WriteError::Untrimmed { entry } => write!(f, "the term of entry {} is not trimmed", entry.0),
            WriteError::Character { entry } => write!(f, "entry {} holds a character XML cannot carry", entry.0),
        }
    }
}

impl std::error::Error for WriteError {}

fn xml_text_ok(s: &str) -> bool {
    s.chars().all(|c| matches!(c, '\t' | '\n' | '\r' | '\u{20}'..='\u{D7FF}' | '\u{E000}'..='\u{FFFD}' | '\u{10000}'..))
}

/// Appends `s` escaped for element content or an attribute value: the
/// markup characters as entities, and carriage returns, tabs and line
/// feeds as character references so the parser keeps them.
fn escape(s: &str, out: &mut String) {
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\r' => out.push_str("&#13;"),
            '\t' => out.push_str("&#9;"),
            '\n' => out.push_str("&#10;"),
            c => out.push(c),
        }
    }
}

/// Appends `glossary` to `out` as a TBX termbase whose terms are all in
/// language `language`. Consecutive entries with the same identifier are
/// one concept entry; an identifier that comes back later starts another.
pub fn write_tbx(glossary: &Glossary, language: &str, dialect: TbxDialect, out: &mut String) -> Result<(), WriteError> {
    if is_blank(language) || !xml_text_ok(language) {
        return Err(WriteError::Language);
    }
    for (ix, e) in glossary.iter() {
        if !xml_text_ok(e.id) || !xml_text_ok(e.term) {
            return Err(WriteError::Character { entry: ix });
        }
        if trim_blank(e.term).len() != e.term.len() {
            return Err(WriteError::Untrimmed { entry: ix });
        }
    }
    let (root, entry, lang, container) = match dialect {
        TbxDialect::Martif => ("martif", "termEntry", "langSet", "tig"),
        TbxDialect::Tbx => ("tbx", "conceptEntry", "langSec", "termSec"),
    };
    out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<");
    out.push_str(root);
    if dialect == TbxDialect::Tbx {
        out.push_str(" xmlns=\"");
        out.push_str(TBX_NAMESPACE);
        out.push_str("\" style=\"dca\"");
    }
    out.push_str(" type=\"TBX-Basic\" xml:lang=\"");
    escape(language, out);
    out.push_str("\">\n  <text>\n    <body>\n");
    let mut open: Option<&str> = None;
    for (_, e) in glossary.iter() {
        if open != Some(e.id) {
            if open.is_some() {
                close_entry(entry, lang, out);
            }
            for s in ["      <", entry, " id=\""] {
                out.push_str(s);
            }
            escape(e.id, out);
            for s in ["\">\n        <", lang, " xml:lang=\""] {
                out.push_str(s);
            }
            escape(language, out);
            out.push_str("\">\n");
            open = Some(e.id);
        }
        for s in ["          <", container, "><term>"] {
            out.push_str(s);
        }
        escape(e.term, out);
        for s in ["</term></", container, ">\n"] {
            out.push_str(s);
        }
    }
    if open.is_some() {
        close_entry(entry, lang, out);
    }
    for s in ["    </body>\n  </text>\n</", root, ">\n"] {
        out.push_str(s);
    }
    Ok(())
}

fn close_entry(entry: &str, lang: &str, out: &mut String) {
    for s in ["        </", lang, ">\n      </", entry, ">\n"] {
        out.push_str(s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::read::TbxReader;

    #[test]
    fn round_trips_in_both_dialects() {
        let mut g = Glossary::new();
        for (id, term) in [
            ("c1", "hot dog"),
            ("c1", "frankfurter"),
            ("a&b \"q\"", "R&D <x> \t tab"),
            ("c1", "line\r\nend"),
            ("東", "東京 🙂"),
        ] {
            g.push(id, term).unwrap();
        }
        for dialect in [TbxDialect::Martif, TbxDialect::Tbx] {
            let mut xml = String::new();
            write_tbx(&g, "en-US", dialect, &mut xml).unwrap();
            let back = TbxReader::new("en").unwrap().read(xml.as_bytes()).unwrap();
            assert_eq!(back.glossary, g, "{dialect:?}\n{xml}");
        }
    }

    #[test]
    fn refuses_what_would_not_read_back() {
        let mut g = Glossary::new();
        g.push("a", " padded").unwrap();
        let mut out = String::new();
        assert_eq!(write_tbx(&g, "en", TbxDialect::Martif, &mut out), Err(WriteError::Untrimmed { entry: EntryIx(0) }));
        let mut g = Glossary::new();
        g.push("a", "bell\u{7}").unwrap();
        assert_eq!(write_tbx(&g, "en", TbxDialect::Martif, &mut out), Err(WriteError::Character { entry: EntryIx(0) }));
        assert_eq!(write_tbx(&g, "", TbxDialect::Martif, &mut out), Err(WriteError::Language));
    }
}
