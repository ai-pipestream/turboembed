//! The masked view: a `View` of the text with the mentions masked, built
//! with `ViewBuilder`, so its alignment maps every masked byte back to
//! the character it stands for and every offset of the source forward.
//!
//! The text is the one [`super::mask`] writes for the same spans and
//! policies. Each masked character is its own edit, so a mask byte maps
//! back to exactly one source character and an offset between two masked
//! characters maps to the same offset in the source. A masked character
//! whose bytes already are the mask (a `*` masked with `*`) is left out of
//! the edits, since nothing changed there.

use turbo_analysis::{Annotator, Bound, Error, Key, Refs, Span, View, ViewBuilder};
use turbo_analysis_layers::{Pii, PiiKind};

use super::MaskPolicy;
use super::mask::{MaskError, check};
use crate::text::letter_or_digit;

/// The instance name of the masked view.
pub const MASKED: Key<View> = Key::named("masked");

/// One span being masked: where it ends, the first byte of the letters or
/// digits it keeps readable, and its place in the caller's order, which
/// decides the mask byte where two spans mask one character.
#[derive(Copy, Clone)]
struct Cover {
    start: u32,
    end: u32,
    keep_from: u32,
    order: usize,
    policy: MaskPolicy,
}

impl Cover {
    fn new(text: &str, span: Span, policy: MaskPolicy, order: usize) -> Cover {
        let piece = &text[span.start as usize..span.end as usize];
        // The keep_trailing-th letter or digit from the end starts the
        // readable tail; with fewer of them, all are readable.
        let mut keep_from = span.end;
        if policy.trailing() > 0 {
            keep_from = span.start;
            let mut kept = 0;
            for (at, c) in piece.char_indices().rev() {
                if letter_or_digit(c) {
                    kept += 1;
                    if kept == policy.trailing() {
                        keep_from = span.start + at as u32;
                        break;
                    }
                }
            }
        }
        Cover { start: span.start, end: span.end, keep_from, order, policy }
    }

    /// Whether this span masks the character `c` at byte `at`.
    fn masks(&self, c: char, at: u32) -> bool {
        if letter_or_digit(c) { at < self.keep_from } else { !self.policy.keeps_format() }
    }
}

/// Writes `text` into `out` with the characters the spans' policies select
/// replaced by mask bytes, one edit per changed character. Where spans
/// overlap, a character is masked when any of them masks it, with the
/// mask of the last of those in the caller's order, as [`super::mask`]
/// writes it.
pub fn mask_into(
    text: &str,
    spans: impl IntoIterator<Item = (Span, MaskPolicy)>,
    out: &mut ViewBuilder<'_>,
) -> Result<(), MaskError> {
    let mut covers = Vec::new();
    for (order, (span, policy)) in spans.into_iter().enumerate() {
        check(text, span)?;
        covers.push(Cover::new(text, span, policy, order));
    }
    covers.sort_by_key(|c| c.start);
    let bytes = text.as_bytes();
    let mut active: Vec<Cover> = Vec::new();
    let mut next = 0;
    let mut at = 0u32;
    loop {
        active.retain(|c| c.end > at);
        if active.is_empty() {
            let Some(c) = covers.get(next) else { break };
            at = at.max(c.start);
        }
        while let Some(&c) = covers.get(next).filter(|c| c.start <= at) {
            if c.end > at {
                active.push(c);
            }
            next += 1;
        }
        if active.is_empty() {
            continue;
        }
        // Every active span ends past `at`, which is a character boundary
        // of the text, so a character starts there.
        let ch = text[at as usize..].chars().next().expect("a character inside an active span");
        let len = ch.len_utf8();
        let by = active.iter().filter(|c| c.masks(ch, at)).max_by_key(|c| c.order);
        if let Some(c) = by {
            let mask = [c.policy.mask() as u8; 4];
            let from = at as usize;
            if bytes[from..from + len] != mask[..len] {
                out.keep(from);
                out.replace(from + len, std::str::from_utf8(&mask[..len]).expect("ASCII mask bytes"));
            }
        }
        at += len as u32;
    }
    Ok(())
}

/// `src` with the spans masked, as a view derived from it.
pub fn mask_view(src: &View, spans: impl IntoIterator<Item = (Span, MaskPolicy)>) -> Result<View, MaskError> {
    let text = src.text();
    let mut out = ViewBuilder::new(text);
    mask_into(text, spans, &mut out)?;
    let (masked, alignment) = out.finish();
    Ok(View::derived(src, masked, alignment))
}

/// Writes the `masked` view: the text it reads with every mention of the
/// pii layer over it masked with its kind's policy.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct MaskedView {
    policies: [MaskPolicy; PiiKind::ALL.len()],
}

impl MaskedView {
    /// Every mention masked with `policy`: `Masker.mask(document, layer,
    /// policy)`.
    pub fn all(policy: MaskPolicy) -> MaskedView {
        MaskedView { policies: [policy; PiiKind::ALL.len()] }
    }

    /// Each mention masked with its kind's default policy and `mask`:
    /// `Masker.mask` with `MaskPolicies.byType(mask)`.
    pub fn by_kind(mask: char) -> Result<MaskedView, MaskError> {
        let mut view = MaskedView::all(MaskPolicy::new(mask)?);
        for k in PiiKind::ALL {
            view.policies[k.rank() as usize] = MaskPolicy::for_kind(k, mask)?;
        }
        Ok(view)
    }

    /// This view with the mentions of `kind` masked with `policy`.
    pub fn with(mut self, kind: PiiKind, policy: MaskPolicy) -> MaskedView {
        self.policies[kind.rank() as usize] = policy;
        self
    }

    pub fn policy(&self, kind: PiiKind) -> MaskPolicy {
        self.policies[kind.rank() as usize]
    }

    /// The step for a pipeline, writing the view as [`MASKED`].
    pub fn bound(self) -> Bound<MaskedView> {
        Bound::new(self).write(0, MASKED.name)
    }

    /// `src` with the mentions of `pii`, a layer over it, masked.
    pub fn view(&self, src: &View, pii: &Pii) -> Result<View, MaskError> {
        mask_view(src, pii.mentions().map(|m| (m.span, self.policy(m.kind))))
    }
}

impl Annotator for MaskedView {
    const ID: &'static str = "turbo-pii:masked";
    type Reads = (View, Pii);
    type Writes = (View,);

    fn annotate(&self, (src, pii): Refs<'_, Self>) -> Result<(View,), Error> {
        if pii.spans.view() != src.id() {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!("the pii layer is over view {}, not view {}", pii.spans.view().0, src.id().0),
            });
        }
        let detail = |e: MaskError| Error::Annotator { annotator: Self::ID, detail: e.to_string() };
        Ok((self.view(src, pii).map_err(detail)?,))
    }

    /// FNV-1a over each kind's mask byte and options, fixed so that stored
    /// provenance stays comparable across builds.
    fn config_hash(&self) -> u64 {
        let words = self
            .policies
            .iter()
            .map(|p| u64::from(p.mask() as u8) | u64::from(p.keeps_format()) << 8 | (p.trailing() as u64) << 16);
        words.fold(0xcbf2_9ce4_8422_2325, |h, x| (h ^ x).wrapping_mul(0x0100_0000_01b3))
    }
}
