use super::super::payload::Decoder;
use super::{
    Chars, SVG_NAMESPACE, XML_NAMESPACE, XMLNS_NAMESPACE, declared_encoding, is_name_char, is_name_start,
    is_pubid_char, is_space, is_xml_char,
};

/// How far ahead the scan looks: the longest literal it compares,
/// `standalone`, fits.
const LOOKAHEAD: usize = 16;

pub(super) struct Xml<'t> {
    chars: Chars<'t>,
    look: [char; LOOKAHEAD],
    look_len: usize,
    xml11: bool,
}

/// One attribute of the root start tag: its qualified name and its
/// normalized value.
struct Attribute {
    name: String,
    value: String,
}

impl<'t> Xml<'t> {
    pub(super) fn new(bytes: Decoder<'t>) -> Xml<'t> {
        Xml { chars: Chars::new(bytes), look: ['\0'; LOOKAHEAD], look_len: 0, xml11: false }
    }

    fn peek_at(&mut self, k: usize) -> Option<char> {
        while self.look_len <= k {
            self.look[self.look_len] = self.chars.next()?;
            self.look_len += 1;
        }
        Some(self.look[k])
    }

    fn peek(&mut self) -> Option<char> {
        self.peek_at(0)
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.look.copy_within(1..self.look_len, 0);
        self.look_len -= 1;
        Some(c)
    }

    /// Whether the next characters are `literal`. Characters are decoded
    /// only as far as they match, so a mismatch never reads past the point
    /// where an encoding declaration takes effect.
    fn starts_with(&mut self, literal: &str) -> bool {
        literal.chars().enumerate().all(|(k, c)| self.peek_at(k) == Some(c))
    }

    fn expect(&mut self, literal: &str) -> Option<()> {
        if !self.starts_with(literal) {
            return None;
        }
        for _ in literal.chars() {
            self.bump();
        }
        Some(())
    }

    /// Skips white space and says whether there was any.
    fn skip_space(&mut self) -> bool {
        let mut any = false;
        while self.peek().is_some_and(is_space) {
            self.bump();
            any = true;
        }
        any
    }

    /// `true` when the root element is `svg` in the SVG namespace, `false`
    /// when it is another element, `None` when the prolog or the start tag
    /// is not well-formed or does not end within the decoded prefix.
    pub(super) fn svg_root(&mut self) -> Option<bool> {
        if self.starts_with("<?xml") && self.peek_at(5).is_some_and(is_space) {
            self.xml_declaration()?;
        }
        let mut doctype = false;
        loop {
            self.skip_space();
            if self.starts_with("<!--") {
                self.comment()?;
            } else if self.starts_with("<?") {
                self.processing_instruction()?;
            } else if self.starts_with("<!DOCTYPE") && !doctype {
                doctype = true;
                self.doctype()?;
            } else if self.peek() == Some('<') && self.peek_at(1).is_some_and(is_name_start) {
                return self.root_start_tag();
            } else {
                return None;
            }
        }
    }

    /// `<?xml version="1.x" encoding="..." standalone="..."?>`, and the
    /// switch to the declared encoding.
    fn xml_declaration(&mut self) -> Option<()> {
        self.expect("<?xml")?;
        self.skip_space();
        self.expect("version")?;
        let version = self.pseudo_attribute()?;
        if version != "1.0" && version != "1.1" {
            return None;
        }
        self.xml11 = version == "1.1";
        let mut space = self.skip_space();
        let mut declared = None;
        if space && self.starts_with("encoding") {
            self.expect("encoding")?;
            let name = self.pseudo_attribute()?;
            let mut chars = name.chars();
            if !chars.next().is_some_and(|c| c.is_ascii_alphabetic())
                || !chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
            {
                return None;
            }
            declared = Some(name);
            space = self.skip_space();
        }
        if space && self.starts_with("standalone") {
            self.expect("standalone")?;
            if !matches!(self.pseudo_attribute()?.as_str(), "yes" | "no") {
                return None;
            }
            self.skip_space();
        }
        self.expect("?>")?;
        if let Some(name) = declared {
            self.chars.encoding = declared_encoding(self.chars.encoding, &name)?;
        }
        Some(())
    }

    /// `S? = S? 'value'` of the XML declaration, with the value. The values
    /// it accepts are short ASCII names, so the length is capped.
    fn pseudo_attribute(&mut self) -> Option<String> {
        self.skip_space();
        self.expect("=")?;
        self.skip_space();
        let quote = self.bump().filter(|&q| q == '"' || q == '\'')?;
        let mut value = String::new();
        loop {
            let c = self.bump()?;
            if c == quote {
                return Some(value);
            }
            if value.len() >= 64 {
                return None;
            }
            value.push(c);
        }
    }

    /// `<!-- ... -->`, where the text may not hold `--`.
    fn comment(&mut self) -> Option<()> {
        self.expect("<!--")?;
        loop {
            if self.bump()? == '-' && self.peek()? == '-' {
                self.bump();
                return (self.bump()? == '>').then_some(());
            }
        }
    }

    /// `<?target ...?>`: the target is a name other than `xml` in any
    /// letter case. A colon in it is accepted, as the JDK parser accepts it.
    fn processing_instruction(&mut self) -> Option<()> {
        self.expect("<?")?;
        let mut target = String::new();
        self.name(&mut target)?;
        if target.eq_ignore_ascii_case("xml") {
            return None;
        }
        if self.starts_with("?>") {
            return self.expect("?>");
        }
        if !self.skip_space() {
            return None;
        }
        loop {
            if self.bump()? == '?' && self.peek()? == '>' {
                self.bump();
                return Some(());
            }
        }
    }

    /// `<!DOCTYPE name ExternalID? [internal subset]? >`, read as the JDK
    /// parser reads it with DTD support off: the external subset is never
    /// fetched, and the internal subset is skipped unread up to its first
    /// `]`, so its declarations are neither checked nor applied.
    fn doctype(&mut self) -> Option<()> {
        self.expect("<!DOCTYPE")?;
        if !self.skip_space() {
            return None;
        }
        self.name(&mut String::new())?;
        let space = self.skip_space();
        if space && self.starts_with("SYSTEM") {
            self.expect("SYSTEM")?;
            self.literal_after_space(false)?;
            self.skip_space();
        } else if space && self.starts_with("PUBLIC") {
            self.expect("PUBLIC")?;
            self.literal_after_space(true)?;
            self.literal_after_space(false)?;
            self.skip_space();
        }
        if self.peek()? == '[' {
            while self.bump()? != ']' {}
            self.skip_space();
        }
        (self.bump()? == '>').then_some(())
    }

    /// White space and then a quoted literal, of public identifier
    /// characters only when `public`.
    fn literal_after_space(&mut self, public: bool) -> Option<()> {
        if !self.skip_space() {
            return None;
        }
        let quote = self.bump().filter(|&q| q == '"' || q == '\'')?;
        loop {
            let c = self.bump()?;
            if c == quote {
                return Some(());
            }
            if public && !is_pubid_char(c) {
                return None;
            }
        }
    }

    /// Reads a `Name` into `out`.
    fn name(&mut self, out: &mut String) -> Option<()> {
        let first = self.bump().filter(|&c| is_name_start(c))?;
        out.push(first);
        while let Some(c) = self.peek().filter(|&c| is_name_char(c)) {
            self.bump();
            out.push(c);
        }
        Some(())
    }

    /// The root start tag, and whether it is the SVG root.
    fn root_start_tag(&mut self) -> Option<bool> {
        self.expect("<")?;
        let mut element = String::new();
        self.name(&mut element)?;
        let mut attributes: Vec<Attribute> = Vec::new();
        loop {
            let space = self.skip_space();
            if self.starts_with("/>") || self.starts_with(">") {
                break;
            }
            if !space {
                return None;
            }
            let mut name = String::new();
            self.name(&mut name)?;
            self.skip_space();
            self.expect("=")?;
            self.skip_space();
            let mut value = String::new();
            self.attribute_value(&mut value)?;
            if attributes.iter().any(|a| a.name == name) {
                return None;
            }
            attributes.push(Attribute { name, value });
        }
        let (prefix, local) = qualified(&element)?;
        let scope = Scope::declared(&attributes, self.xml11)?;
        let namespace = scope.resolve(prefix, true)?;
        scope.check_attributes(&attributes)?;
        Some(local == "svg" && namespace == Some(SVG_NAMESPACE))
    }

    /// A quoted attribute value with its references replaced and its
    /// white space characters normalized to spaces (XML 1.0 section 3.3.3,
    /// for an attribute no DTD declares).
    fn attribute_value(&mut self, out: &mut String) -> Option<()> {
        let quote = self.bump().filter(|&q| q == '"' || q == '\'')?;
        loop {
            match self.bump()? {
                c if c == quote => return Some(()),
                '<' => return None,
                '&' => out.push(self.reference()?),
                '\n' | '\t' => out.push(' '),
                c => out.push(c),
            }
        }
    }

    /// The character a reference after its `&` stands for: a character
    /// reference or one of the five predefined entities. Any other entity
    /// is undeclared, since no DTD is read.
    fn reference(&mut self) -> Option<char> {
        let c = if self.peek()? == '#' {
            self.bump();
            let radix = if self.peek()? == 'x' {
                self.bump();
                16
            } else {
                10
            };
            let mut value: u32 = 0;
            let mut digits = 0;
            while let Some(d) = self.peek().and_then(|c| c.to_digit(radix)) {
                self.bump();
                value = value.checked_mul(radix)?.checked_add(d)?;
                digits += 1;
            }
            if digits == 0 {
                return None;
            }
            char::from_u32(value).filter(|&c| is_xml_char(c))?
        } else {
            let mut name = String::new();
            self.name(&mut name)?;
            match name.as_str() {
                "lt" => '<',
                "gt" => '>',
                "amp" => '&',
                "apos" => '\'',
                "quot" => '"',
                _ => return None,
            }
        };
        (self.bump()? == ';').then_some(c)
    }
}

/// A qualified name split into prefix and local part, or `None` when it
/// is not a `QName`. As in the JDK parser, a colon in first place is part of
/// the name, the first colon after it splits the prefix from the local part,
/// and the local part must start with a name start character and may not
/// hold another colon.
fn qualified(name: &str) -> Option<(Option<&str>, &str)> {
    let first = name.chars().next().map_or(0, char::len_utf8);
    let Some(colon) = name[first..].find(':').map(|i| i + first) else {
        return Some((None, name));
    };
    let local = &name[colon + 1..];
    (local.starts_with(is_name_start) && !local.contains(':')).then_some((Some(&name[..colon]), local))
}

/// The namespace declarations of the root element (the only element in
/// scope).
struct Scope<'a> {
    default: Option<&'a str>,
    prefixes: Vec<(&'a str, &'a str)>,
}

impl<'a> Scope<'a> {
    /// Reads `xmlns` and `xmlns:p` attributes, enforcing the constraints
    /// of Namespaces in XML 1.0 (or 1.1 when `xml11` is set): `xmlns` is
    /// never declared, `xml` only to its own namespace, no other prefix to
    /// either reserved namespace, and no prefix to the empty name in XML 1.0.
    fn declared(attributes: &'a [Attribute], xml11: bool) -> Option<Scope<'a>> {
        let mut scope = Scope { default: None, prefixes: Vec::new() };
        for a in attributes {
            let value = a.value.as_str();
            if a.name == "xmlns" {
                if value == XML_NAMESPACE || value == XMLNS_NAMESPACE {
                    return None;
                }
                scope.default = (!value.is_empty()).then_some(value);
            } else if let Some(prefix) = a.name.strip_prefix("xmlns:") {
                let reserved = value == XML_NAMESPACE || value == XMLNS_NAMESPACE;
                let allowed = match prefix {
                    "xmlns" => false,
                    "xml" => value == XML_NAMESPACE,
                    _ => (!value.is_empty() || xml11) && !reserved,
                };
                if !allowed || qualified(&a.name).is_none() {
                    return None;
                }
                if value.is_empty() {
                    scope.prefixes.retain(|(q, _)| *q != prefix);
                } else {
                    scope.prefixes.push((prefix, value));
                }
            }
        }
        Some(scope)
    }

    /// The namespace of a prefix; an unprefixed element takes the default
    /// namespace, an unprefixed attribute none. `None` for an unbound
    /// prefix.
    fn resolve(&self, prefix: Option<&str>, element: bool) -> Option<Option<&'a str>> {
        match prefix {
            None => Some(if element { self.default } else { None }),
            Some("xml") => Some(Some(XML_NAMESPACE)),
            Some("xmlns") => None,
            Some(p) => self.prefixes.iter().find(|(q, _)| *q == p).map(|&(_, uri)| Some(uri)),
        }
    }

    /// Every attribute's prefix is bound, and no two attributes have the
    /// same namespace and local name.
    fn check_attributes(&self, attributes: &[Attribute]) -> Option<()> {
        let mut expanded: Vec<(Option<&str>, &str)> = Vec::new();
        for a in attributes {
            if a.name == "xmlns" || a.name.starts_with("xmlns:") {
                continue;
            }
            let (prefix, local) = qualified(&a.name)?;
            let name = (self.resolve(prefix, false)?, local);
            if expanded.contains(&name) {
                return None;
            }
            expanded.push(name);
        }
        Some(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xml11_allows_prefix_undeclaration_while_xml10_rejects_it() {
        let xml10 = [Attribute { name: "xmlns:p".into(), value: String::new() }];
        assert!(Scope::declared(&xml10, false).is_none());
        let scope = Scope::declared(&xml10, true).expect("XML 1.1 allows xmlns:p=\"\"");
        assert_eq!(scope.resolve(Some("p"), false), None);
    }
}
