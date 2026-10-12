//! The scan both contraction normalizers run.

use turbo_analysis::ViewBuilder;

use super::rules::{Ambiguous, Contraction, Expansion, Reading, apostrophe_at, contraction_at};
use super::word_break::is_boundary;

/// Decides an ambiguous `'s` or `'d`: given the byte offset of the
/// apostrophe and which suffix it is, its reading, or `None` to leave the
/// word alone.
pub(crate) type Resolve<'r> = dyn FnMut(usize, Ambiguous) -> Option<Reading> + 'r;

/// Writes `text` with every contraction that is a complete word expanded.
/// `resolve` is consulted for `'s` and `'d` words; without one they are
/// not candidates at all, as in OpenNLP.
///
/// The search follows OpenNLP: each apostrophe in turn is tried as the
/// middle of a contraction; after a candidate, found complete or not, the
/// search goes on after its end.
pub(crate) fn expand(text: &str, out: &mut ViewBuilder<'_>, mut resolve: Option<&mut Resolve<'_>>) {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let Some(width) = apostrophe_at(bytes, i) else {
            i += 1;
            continue;
        };
        let Some(found) = contraction_at(bytes, i, width, resolve.is_some()) else {
            i += width;
            continue;
        };
        if is_boundary(text, found.start) && is_boundary(text, found.end) {
            write(out, &found, i, resolve.as_deref_mut());
        }
        i = found.end;
    }
}

/// Writes one contraction: the prefix kept or replaced, then the
/// expansion of the rest. An ambiguous suffix its resolver leaves alone is
/// not written, so the builder copies it unchanged later.
fn write(out: &mut ViewBuilder<'_>, c: &Contraction, apostrophe: usize, resolve: Option<&mut Resolve<'_>>) {
    let suffix = match c.suffix {
        Expansion::Fixed(s) => s,
        Expansion::Ambiguous { which, upper } => {
            let Some(reading) = resolve.and_then(|r| r(apostrophe, which)) else { return };
            reading.expansion(upper)
        }
    };
    out.keep(c.start);
    match c.prefix {
        Some(p) => out.replace(c.prefix_end, p),
        None => out.keep(c.prefix_end),
    }
    out.replace(c.suffix_end, suffix);
    out.keep(c.end);
}
