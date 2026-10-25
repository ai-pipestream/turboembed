//! CommonMark 0.31.2 HTML block starts and ends. These are ASCII grammar
//! rules, not Unicode whitespace or case mappings.

#[derive(Copy, Clone, Debug)]
pub(crate) enum End {
    Script,
    Pre,
    Style,
    Textarea,
    Comment,
    Instruction,
    Declaration,
    Cdata,
    Blank,
}

impl End {
    pub(crate) fn closes(self, line: &str) -> bool {
        let raw = match self {
            Self::Script => "</script>",
            Self::Pre => "</pre>",
            Self::Style => "</style>",
            Self::Textarea => "</textarea>",
            Self::Comment => return line.contains("-->"),
            Self::Instruction => return line.contains("?>"),
            Self::Declaration => return line.contains('>'),
            Self::Cdata => return line.contains("]]>"),
            Self::Blank => return line.bytes().all(|c| matches!(c, b' ' | b'\t')),
        };
        line.as_bytes().windows(raw.len()).any(|s| s.eq_ignore_ascii_case(raw.as_bytes()))
    }
}

pub(crate) fn start(line: &str, paragraph: bool) -> Option<End> {
    let b = line.as_bytes();
    if b.first() != Some(&b'<') {
        return None;
    }
    for tag in ["script", "pre", "style", "textarea"] {
        let tail = b.get(1 + tag.len()..);
        if b.get(1..1 + tag.len()).is_some_and(|s| s.eq_ignore_ascii_case(tag.as_bytes()))
            && tail.is_some_and(|t| t.is_empty() || matches!(t[0], b' ' | b'\t' | b'>'))
        {
            return Some(match tag {
                "script" => End::Script,
                "pre" => End::Pre,
                "style" => End::Style,
                _ => End::Textarea,
            });
        }
    }
    if line.starts_with("<!--") {
        return Some(End::Comment);
    }
    if line.starts_with("<?") {
        return Some(End::Instruction);
    }
    if line.starts_with("<![CDATA[") {
        return Some(End::Cdata);
    }
    if line.starts_with("<!") && b.get(2).is_some_and(u8::is_ascii_alphabetic) {
        return Some(End::Declaration);
    }
    let begin = if b.get(1) == Some(&b'/') { 2 } else { 1 };
    let count = b[begin..].iter().take_while(|c| c.is_ascii_alphanumeric() || **c == b'-').count();
    let tag = &line[begin..begin + count];
    let tail = &line[begin + count..];
    if BLOCK_TAGS.iter().any(|t| tag.eq_ignore_ascii_case(t))
        && (tail.is_empty() || tail.starts_with([' ', '\t', '>']) || tail.starts_with("/>"))
    {
        return Some(End::Blank);
    }
    (!paragraph && complete_tag(b)).then_some(End::Blank)
}

const BLOCK_TAGS: &[&str] = &[
    "address",
    "article",
    "aside",
    "base",
    "basefont",
    "blockquote",
    "body",
    "caption",
    "center",
    "col",
    "colgroup",
    "dd",
    "details",
    "dialog",
    "dir",
    "div",
    "dl",
    "dt",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "form",
    "frame",
    "frameset",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "head",
    "header",
    "hr",
    "html",
    "iframe",
    "legend",
    "li",
    "link",
    "main",
    "menu",
    "menuitem",
    "nav",
    "noframes",
    "ol",
    "optgroup",
    "option",
    "p",
    "param",
    "search",
    "section",
    "summary",
    "table",
    "tbody",
    "td",
    "tfoot",
    "th",
    "thead",
    "title",
    "tr",
    "track",
    "ul",
];

/// Type 7: a whole opening or closing tag, with optional attributes.
fn complete_tag(b: &[u8]) -> bool {
    let mut i = 1;
    let closing = b.get(i) == Some(&b'/');
    if closing {
        i += 1;
    }
    if !b.get(i).is_some_and(u8::is_ascii_alphabetic) {
        return false;
    }
    i += 1;
    while b.get(i).is_some_and(|c| c.is_ascii_alphanumeric() || *c == b'-') {
        i += 1;
    }
    loop {
        let before = i;
        spaces(b, &mut i);
        if !closing && b.get(i) == Some(&b'/') {
            i += 1;
        }
        if b.get(i) == Some(&b'>') {
            i += 1;
            spaces(b, &mut i);
            return i == b.len();
        }
        if closing || i == before || !b.get(i).is_some_and(|c| c.is_ascii_alphabetic() || matches!(c, b'_' | b':')) {
            return false;
        }
        i += 1;
        while b.get(i).is_some_and(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b':' | b'.' | b'-')) {
            i += 1;
        }
        let after_name = i;
        spaces(b, &mut i);
        if b.get(i) != Some(&b'=') {
            i = after_name;
            continue;
        }
        i += 1;
        spaces(b, &mut i);
        if let Some(&q @ (b'\'' | b'"')) = b.get(i) {
            i += 1;
            while b.get(i).is_some_and(|c| *c != q) {
                i += 1;
            }
            if b.get(i) != Some(&q) {
                return false;
            }
            i += 1;
        } else {
            let from = i;
            while b.get(i).is_some_and(|c| !matches!(c, b' ' | b'\t' | b'"' | b'\'' | b'=' | b'<' | b'>' | b'`')) {
                i += 1;
            }
            if i == from {
                return false;
            }
        }
    }
}

fn spaces(b: &[u8], i: &mut usize) {
    while b.get(*i).is_some_and(|c| matches!(c, b' ' | b'\t')) {
        *i += 1;
    }
}
