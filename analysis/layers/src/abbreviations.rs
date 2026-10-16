//! Abbreviations and acronyms defined in a text view: the short and long
//! form of each definition ("World Health Organization (WHO)") and each
//! later use of a defined short form. The links between them are an
//! `Edges<Abbreviations, Abbreviations>` layer written beside this one:
//! every short form, defining or later, has one edge to the long form it
//! stands for.

use std::sync::Arc;

use turbo_analysis::export::{ExportCx, ExportError, ImportCx, LayerCodec, Reader, Writer, get_spans, put_spans};
use turbo_analysis::{Edges, Ix, Layer, Overlap, Span, Spans, TagSet, ViewId};

/// What one span of an abbreviation layer is.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum AbbreviationForm {
    /// The short form where it is defined: `WHO` in "World Health
    /// Organization (WHO)" and in "WHO (World Health Organization)".
    Short,
    /// The long form of a definition: `World Health Organization`.
    Long,
    /// A use of a defined short form after its definition.
    Mention,
}

/// The edge kind from a defining short form to its long form.
pub const DEFINITION: &str = "definition";
/// The edge kind from a later mention to the long form of the definition
/// it refers to.
pub const MENTION: &str = "mention";

/// The edge kinds of the links beside [`Abbreviations`], in tag order:
/// [`DEFINITION`] is tag 0 and [`MENTION`] tag 1.
pub fn abbreviation_links() -> Arc<TagSet> {
    TagSet::new("abbreviation-links", &[DEFINITION, MENTION])
}

/// Short forms, long forms and mentions, one span each, in text order (by
/// start, then by end). Spans may nest: a long form can contain a mention
/// of an abbreviation defined earlier ("the WHO Regional Office (WRO)").
pub struct Abbreviations {
    pub spans: Spans<Abbreviations>,
    pub form: Vec<AbbreviationForm>,
}

impl Abbreviations {
    pub fn new(view: ViewId) -> Abbreviations {
        Abbreviations { spans: Spans::new(view), form: Vec::new() }
    }

    pub fn push(&mut self, span: Span, form: AbbreviationForm) -> Ix<Abbreviations> {
        self.form.push(form);
        self.spans.push(span)
    }

    pub fn form(&self, i: Ix<Abbreviations>) -> AbbreviationForm {
        self.form[i.get()]
    }

    /// The long form that short form or mention `i` stands for, read from the
    /// links written beside the layer: its one outgoing edge. `None` for a
    /// long form.
    pub fn long_form(
        &self,
        links: &Edges<Abbreviations, Abbreviations>,
        i: Ix<Abbreviations>,
    ) -> Option<Ix<Abbreviations>> {
        links.outgoing(i).next().map(|(_, e)| e.to)
    }
}

impl Layer for Abbreviations {
    const ID: &'static str = "turbo:abbreviations";
    const OVERLAP: Overlap = Overlap::Nesting;
    positional!(Abbreviations);

    fn check(&self) -> Result<(), String> {
        if self.form.len() != self.spans.len() {
            return Err("a form for every span".into());
        }
        let mut last = Span::default();
        for (i, span) in self.spans.iter() {
            if span.is_empty() {
                return Err(format!("abbreviation span {} is empty at {}", i.get(), span.start));
            }
            if i.get() > 0 && (span.start, span.end) <= (last.start, last.end) {
                return Err(format!("abbreviation span {} is not after the one before it", i.get()));
            }
            last = span;
        }
        Ok(())
    }
}

impl LayerCodec for Abbreviations {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        for form in &self.form {
            w.u8(match form {
                AbbreviationForm::Short => 0,
                AbbreviationForm::Long => 1,
                AbbreviationForm::Mention => 2,
            });
        }
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let mut form = Vec::with_capacity(spans.len());
        for _ in 0..spans.len() {
            form.push(match r.u8()? {
                0 => AbbreviationForm::Short,
                1 => AbbreviationForm::Long,
                2 => AbbreviationForm::Mention,
                tag => {
                    return Err(ExportError::Malformed(format!("an abbreviation form tag of {tag}")));
                }
            });
        }
        if !r.is_done() {
            return Err(ExportError::Malformed("trailing bytes in an abbreviations payload".into()));
        }
        Ok(Abbreviations { spans, form })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_analysis::{Edge, Tag};

    #[test]
    fn links_lead_to_the_long_form() {
        // "World Health Organization (WHO) ... WHO"
        let mut a = Abbreviations::new(ViewId::SOURCE);
        let long = a.push(Span::new(0, 25), AbbreviationForm::Long);
        let short = a.push(Span::new(27, 30), AbbreviationForm::Short);
        let mention = a.push(Span::new(40, 43), AbbreviationForm::Mention);
        assert_eq!(a.check(), Ok(()));
        let kinds = abbreviation_links();
        assert_eq!(kinds.tag(DEFINITION), Some(Tag(0)));
        assert_eq!(kinds.tag(MENTION), Some(Tag(1)));
        let links = Edges::new(kinds, [Edge::new(short, long, Tag(0), 1.0), Edge::new(mention, long, Tag(1), 1.0)]);
        assert_eq!(a.long_form(&links, short), Some(long));
        assert_eq!(a.long_form(&links, mention), Some(long));
        assert_eq!(a.long_form(&links, long), None);
        assert_eq!(a.form(mention), AbbreviationForm::Mention);
        assert_eq!(Abbreviations::ID, "turbo:abbreviations");
    }

    #[test]
    fn check_wants_ordered_non_empty_spans() {
        let mut a = Abbreviations::new(ViewId::SOURCE);
        a.push(Span::new(0, 25), AbbreviationForm::Long);
        a.push(Span::new(4, 10), AbbreviationForm::Mention);
        assert_eq!(a.check(), Ok(()), "nested spans are allowed");
        a.push(Span::new(4, 10), AbbreviationForm::Mention);
        assert!(a.check().unwrap_err().contains("not after"));
        let mut a = Abbreviations::new(ViewId::SOURCE);
        a.push(Span::new(3, 3), AbbreviationForm::Short);
        assert!(a.check().unwrap_err().contains("empty"));
        let mut a = Abbreviations::new(ViewId::SOURCE);
        a.push(Span::new(0, 3), AbbreviationForm::Short);
        a.form.pop();
        assert_eq!(a.check().unwrap_err(), "a form for every span");
    }
}
