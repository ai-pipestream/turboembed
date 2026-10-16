//! TBX termbases (ISO 30042, TermBase eXchange), OpenNLP's
//! `TbxGlossaryReader`.

use turbo_analysis::Span;
use turbo_glossary_layers::{EntryError, Glossary, is_blank, trim_blank};
use turbo_unicode::java::to_lower_char;

use super::xml::{Event, Start, Xml};
use super::{GlossaryFile, decode};
use crate::error::ReadError;

/// The namespace of TBX 3 (ISO 30042:2019).
pub(super) const TBX_NAMESPACE: &str = "urn:iso:std:iso:30042:ed-2";

/// The structural levels on the way to a term; anything else is skipped
/// with its subtree.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Scope {
    Root,
    Text,
    Body,
    Entry,
    Language,
    NestedTerm,
    TermContainer,
    Term,
}

impl Scope {
    /// The level of a child element named `name` of this level, or `None`
    /// for a subtree to skip.
    fn child(self, name: &str) -> Option<Scope> {
        match (self, name) {
            (Scope::Root, "text") => Some(Scope::Text),
            (Scope::Text, "body") => Some(Scope::Body),
            (Scope::Body, "termEntry" | "conceptEntry") => Some(Scope::Entry),
            (Scope::Entry, "langSet" | "langSec") => Some(Scope::Language),
            (Scope::Language, "tig" | "termSec") => Some(Scope::TermContainer),
            (Scope::Language, "ntig") => Some(Scope::NestedTerm),
            (Scope::NestedTerm, "termGrp") => Some(Scope::TermContainer),
            (Scope::TermContainer, "term") => Some(Scope::Term),
            _ => None,
        }
    }
}

/// Reads the terms of one language from a TBX termbase: TBX 2
/// (`martif`, `termEntry`, `langSet`, `tig` or `ntig`/`termGrp`, `term`)
/// and TBX 3 (`tbx`, `conceptEntry`, `langSec`, `termSec`, `term`), the
/// elements in no namespace or in the TBX 3 namespace. Each term of a
/// language section that matches becomes an entry with its concept
/// entry's `id`.
///
/// The language tag matches an `xml:lang` equal to it or starting with it
/// and a hyphen, both lowercased per character (`Character.toLowerCase`):
/// `en` selects `en-US`, `en-GB` does not select `en-US`. Terms are read
/// only through `text`, `body` and the entry, language and term
/// containers; headers, back matter and other metadata are skipped with
/// whatever they hold. Text inside a term's inline markup is kept, and
/// the term is trimmed of Unicode whitespace. An entry without an
/// unqualified `id`, or a blank term in a selected language, is an error.
/// The full TBX schema is not checked.
///
/// The XML is read by a small parser of this crate: UTF-8 only, internal
/// entities expanded, external ones never loaded and refused when
/// declared, as OpenNLP refuses them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TbxReader {
    /// The tag, lowercased, and the same followed by a hyphen.
    language: String,
    prefix: String,
}

/// Lowercases `s` per character, as `StringUtil.toLowerCase` does.
fn java_lower(s: &str) -> String {
    s.chars().map(to_lower_char).collect()
}

impl TbxReader {
    /// A reader of the language `tag` (`en`, `en-US`), or `None` when the
    /// tag is blank.
    pub fn new(tag: &str) -> Option<TbxReader> {
        if is_blank(tag) {
            return None;
        }
        let language = java_lower(tag);
        let prefix = format!("{language}-");
        Some(TbxReader { language, prefix })
    }

    fn selects(&self, lang: Option<&str>) -> bool {
        lang.is_some_and(|l| {
            let l = java_lower(l);
            l == self.language || l.starts_with(&self.prefix)
        })
    }

    /// The entries of `bytes` in file order, each term with the span of
    /// its element's content in `bytes`.
    pub fn read(&self, bytes: &[u8]) -> Result<GlossaryFile, ReadError> {
        let (text, base) = decode(bytes)?;
        let mut xml = Xml::new(text, base as usize, TBX_NAMESPACE);
        let mut out = GlossaryFile { glossary: Glossary::new(), term_at: Vec::new() };
        loop {
            match xml.next()? {
                Event::Start(root) => {
                    if !root.known || !(root.local == "martif" || root.local == "tbx") {
                        return Err(ReadError::NotTbx { line: root.line, root: root.qname.to_string() });
                    }
                    self.children(&mut xml, Scope::Root, None, &mut out)?;
                }
                Event::Eof => return Ok(out),
                Event::End { .. } | Event::Text(_) => {}
            }
        }
    }

    /// Reads the children of an element of level `scope` up to its end tag.
    fn children(
        &self,
        xml: &mut Xml<'_>,
        scope: Scope,
        entry_id: Option<&str>,
        out: &mut GlossaryFile,
    ) -> Result<(), ReadError> {
        loop {
            let start = match xml.next()? {
                Event::End { .. } => return Ok(()),
                Event::Start(s) => s,
                Event::Text(_) => continue,
                Event::Eof => unreachable!("the parser ends an open element before the input"),
            };
            let child = if start.known { scope.child(start.local) } else { None };
            match child {
                None => skip(xml)?,
                Some(Scope::Language) if !self.selects(start.lang.as_deref()) => skip(xml)?,
                Some(Scope::Term) => self.term(xml, &start, entry_id, out)?,
                Some(Scope::Entry) => {
                    let id = start.id.as_deref().filter(|id| !is_blank(id));
                    let Some(id) = id else {
                        return Err(ReadError::Entry { line: start.line, error: EntryError::BlankId });
                    };
                    self.children(xml, Scope::Entry, Some(id), out)?;
                }
                Some(level) => self.children(xml, level, entry_id, out)?,
            }
        }
    }

    fn term(
        &self,
        xml: &mut Xml<'_>,
        start: &Start<'_>,
        entry_id: Option<&str>,
        out: &mut GlossaryFile,
    ) -> Result<(), ReadError> {
        let mut text = String::new();
        let mut depth = 1usize;
        let end = loop {
            match xml.next()? {
                Event::Start(_) => depth += 1,
                Event::End { at } => {
                    depth -= 1;
                    if depth == 0 {
                        break at;
                    }
                }
                Event::Text(t) => text.push_str(&t),
                Event::Eof => unreachable!("the parser ends an open element before the input"),
            }
        };
        let id = entry_id.expect("a term is read only inside an entry");
        out.glossary.push(id, trim_blank(&text)).map_err(|error| ReadError::Entry { line: start.line, error })?;
        out.term_at.push(Span::new(start.content_start as u32, end as u32));
        Ok(())
    }
}

/// Skips the rest of the element just started, whatever it holds.
fn skip(xml: &mut Xml<'_>) -> Result<(), ReadError> {
    let mut depth = 1usize;
    while depth > 0 {
        match xml.next()? {
            Event::Start(_) => depth += 1,
            Event::End { .. } => depth -= 1,
            Event::Text(_) => {}
            Event::Eof => unreachable!("the parser ends an open element before the input"),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(tag: &str, xml: &str) -> Result<Vec<(String, String)>, ReadError> {
        let f = TbxReader::new(tag).unwrap().read(xml.as_bytes())?;
        Ok(f.glossary.iter().map(|(_, e)| (e.id.to_string(), e.term.to_string())).collect())
    }

    fn pairs(list: &[(&str, &str)]) -> Vec<(String, String)> {
        list.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect()
    }

    const V2: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE martif SYSTEM "TBXBasiccoreStructV02.dtd">
<martif type="TBX-Basic" xml:lang="en">
  <martifHeader><fileDesc><sourceDesc><p>header</p></sourceDesc></fileDesc></martifHeader>
  <text>
    <body>
      <termEntry id="c1">
        <langSet xml:lang="en-US">
          <tig><term>hot dog</term></tig>
          <tig><term> frankfurter
          </term></tig>
        </langSet>
        <langSet xml:lang="de"><tig><term>Wiener W&#252;rstchen</term></tig></langSet>
      </termEntry>
      <termEntry id="c2">
        <langSet xml:lang="EN"><ntig><termGrp><term>New <hi>York</hi> City</term></termGrp></ntig></langSet>
      </termEntry>
    </body>
  </text>
</martif>
"#;

    #[test]
    fn tbx2_languages_and_inline_markup() {
        assert_eq!(
            read("en", V2).unwrap(),
            pairs(&[("c1", "hot dog"), ("c1", "frankfurter"), ("c2", "New York City")])
        );
        assert_eq!(read("de", V2).unwrap(), pairs(&[("c1", "Wiener Würstchen")]));
        assert_eq!(read("en-GB", V2).unwrap(), pairs(&[]));
        let f = TbxReader::new("en").unwrap().read(V2.as_bytes()).unwrap();
        assert_eq!(&V2[f.term_at[2].range()], "New <hi>York</hi> City");
        assert_eq!(TbxReader::new(" "), None);
    }

    #[test]
    fn tbx3_namespace_and_structure() {
        let xml = r#"<tbx xmlns="urn:iso:std:iso:30042:ed-2" xmlns:x="urn:x" type="TBX-Basic" style="dca" xml:lang="en">
<text><body>
<x:conceptEntry id="skip"><langSec xml:lang="en"><termSec><term>no</term></termSec></langSec></x:conceptEntry>
<conceptEntry id="ML" x:id="other"><descrip><term>no</term></descrip>
<langSec xml:lang="en"><termSec><term>machine learning</term></termSec><x:termSec><term>no</term></x:termSec></langSec>
</conceptEntry>
</body></text>
<back><conceptEntry id="B"><langSec xml:lang="en"><termSec><term>no</term></termSec></langSec></conceptEntry></back>
</tbx>"#;
        assert_eq!(read("en", xml).unwrap(), pairs(&[("ML", "machine learning")]));
    }

    #[test]
    fn errors() {
        assert!(matches!(read("en", "<root/>"), Err(ReadError::NotTbx { .. })));
        assert!(matches!(read("en", "<tbx xmlns='urn:other'/>"), Err(ReadError::NotTbx { .. })));
        let no_id = "<martif><text><body><termEntry x='1'/></body></text></martif>";
        assert_eq!(read("en", no_id), Err(ReadError::Entry { line: 1, error: EntryError::BlankId }));
        let blank = "<martif><text><body><termEntry id='a'><langSet xml:lang='en'><tig>\n<term> </term></tig></langSet></termEntry></body></text></martif>";
        assert_eq!(read("en", blank), Err(ReadError::Entry { line: 2, error: EntryError::BlankTerm }));
        // A blank term in a language not read is not looked at.
        assert_eq!(read("de", blank), Ok(Vec::new()));
        // Malformed XML in a skipped subtree is still an error.
        assert!(matches!(
            read("en", "<martif><martifHeader><a></b></martifHeader></martif>"),
            Err(ReadError::Xml { .. })
        ));
    }
}
