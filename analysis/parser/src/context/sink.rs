//! Feature sinks: the shared feature order written either to strings or to inference
//! buffers, and the lazily formatted constituent features they receive.

use std::fmt::{self, Display, Write};

use turbo_sequence::Features;

use crate::parse::Parse;

use super::*;

// Both public string contexts and inference buffers use the same feature order.
pub(crate) trait FeatureSink {
    fn emit(&mut self, value: fmt::Arguments<'_>);

    fn prepare<T: Display>(&self, value: T) -> Prepared<T> {
        Prepared::Direct(value)
    }
}

impl FeatureSink for Vec<String> {
    fn prepare<T: Display>(&self, value: T) -> Prepared<T> {
        let mut rendered = String::with_capacity(32);
        write!(rendered, "{value}").expect("string writes cannot fail");
        Prepared::Rendered(rendered)
    }

    fn emit(&mut self, value: fmt::Arguments<'_>) {
        // Reserve room for a typical combined feature instead of growing per piece.
        let mut rendered = String::with_capacity(64);
        rendered.write_fmt(value).expect("string writes cannot fail");
        self.push(rendered);
    }
}

impl FeatureSink for Features {
    fn prepare<T: Display>(&self, value: T) -> Prepared<T> {
        let mut rendered = InlineFeature::default();
        if write!(rendered, "{value}").is_ok() { Prepared::Inline(rendered) } else { Prepared::Direct(value) }
    }

    fn emit(&mut self, value: fmt::Arguments<'_>) {
        struct Writer<'a>(&'a mut Features);
        impl Write for Writer<'_> {
            fn write_str(&mut self, s: &str) -> fmt::Result {
                self.0.push_str(s);
                Ok(())
            }
        }
        Writer(self).write_fmt(value).expect("feature buffer writes cannot fail");
        self.finish();
    }
}

// Common inference pieces are rendered once on the stack. Oversized pieces
// retain direct formatting, so inference does not allocate for cache overflow.
#[derive(Clone)]
pub(crate) struct InlineFeature {
    bytes: [u8; 128],
    len: usize,
}

impl Default for InlineFeature {
    fn default() -> Self {
        Self { bytes: [0; 128], len: 0 }
    }
}

impl Write for InlineFeature {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        let Some(end) = self.len.checked_add(text.len()).filter(|&end| end <= self.bytes.len()) else {
            return Err(fmt::Error);
        };
        self.bytes[self.len..end].copy_from_slice(text.as_bytes());
        self.len = end;
        Ok(())
    }
}

impl Display for InlineFeature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(std::str::from_utf8(&self.bytes[..self.len]).expect("only complete strings are written"))
    }
}

pub(crate) enum Prepared<T> {
    Direct(T),
    Rendered(String),
    Inline(InlineFeature),
}

impl<T: Display> Display for Prepared<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Direct(value) => value.fmt(f),
            Self::Rendered(value) => f.write_str(value),
            Self::Inline(value) => value.fmt(f),
        }
    }
}

pub(crate) struct DisplayFn<F>(pub F);
impl<F: Fn(&mut fmt::Formatter<'_>) -> fmt::Result> Display for DisplayFn<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (self.0)(f)
    }
}

#[derive(Clone, Copy)]
pub(crate) enum ConsFeature<'a> {
    Text(&'a str),
    Node { node: Option<&'a Parse>, index: i32, labeled: bool, backoff: bool },
}

impl Display for ConsFeature<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Text(s) => f.write_str(s),
            Self::Node { node, index, labeled, backoff } => {
                write!(f, "{index}{}=", if backoff { "*" } else { "" })?;
                if let Some(p) = node {
                    if labeled && index < 0 {
                        write!(f, "{}|", p.label().unwrap_or("null"))?;
                    }
                    f.write_str(p.kind())?;
                    if !backoff {
                        write!(f, "|{}", p.head_text())?;
                    }
                    Ok(())
                } else {
                    f.write_str(EOS)
                }
            }
        }
    }
}

pub(crate) struct ConsRef<'a> {
    pub cons: Prepared<ConsFeature<'a>>,
    pub cons_bo: Prepared<ConsFeature<'a>>,
    pub index: i32,
    pub unigram: bool,
}

impl<'a> ConsRef<'a> {
    pub fn new(node: Option<&'a Parse>, index: i32, labeled: bool, unigram: bool, sink: &impl FeatureSink) -> Self {
        Self {
            cons: sink.prepare(ConsFeature::Node { node, index, labeled, backoff: false }),
            cons_bo: sink.prepare(ConsFeature::Node { node, index, labeled, backoff: true }),
            index,
            unigram,
        }
    }
}

impl<'a> From<&'a Cons> for ConsRef<'a> {
    fn from(c: &'a Cons) -> Self {
        Self {
            cons: Prepared::Direct(ConsFeature::Text(&c.cons)),
            cons_bo: Prepared::Direct(ConsFeature::Text(&c.cons_bo)),
            index: c.index,
            unigram: c.unigram,
        }
    }
}
