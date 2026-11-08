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
use turbo_analysis_layers::{CustomKinds, KindError, Pii, PiiKind};

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
    keep_until: u32,
    keep_from: u32,
    order: usize,
    policy: MaskPolicy,
}

impl Cover {
    fn new(text: &str, span: Span, policy: MaskPolicy, order: usize) -> Cover {
        let piece = &text[span.start as usize..span.end as usize];
        let mut keep_until = span.start;
        if policy.leading() > 0 {
            keep_until = span.end;
            let mut kept = 0;
            for (at, c) in piece.char_indices() {
                if letter_or_digit(c) {
                    kept += 1;
                    if kept == policy.leading() {
                        keep_until = span.start + (at + c.len_utf8()) as u32;
                        break;
                    }
                }
            }
        }
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
        Cover { start: span.start, end: span.end, keep_until, keep_from, order, policy }
    }

    /// Whether this span masks the character `c` at byte `at`.
    fn masks(&self, c: char, at: u32) -> bool {
        if letter_or_digit(c) { at >= self.keep_until && at < self.keep_from } else { !self.policy.keeps_format() }
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
///
/// Built-in policies are indexed by stable kind id. The view owns a
/// custom-kind table: a custom kind given to [`MaskedView::with`] indexes
/// it, and a layer's custom kind is matched to it by name.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MaskedView {
    policies: [MaskPolicy; BUILTIN_SLOTS],
    default_policy: MaskPolicy,
    origin: ViewOrigin,
    custom_kinds: CustomKinds,
    custom: Vec<Option<MaskPolicy>>,
}

/// The constructor a view came from, which gives every kind, including
/// one added to the registry later, its policy until a `with` overrides it.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum ViewOrigin {
    /// [`MaskedView::all`]: every kind masked with this policy.
    All(MaskPolicy),
    /// [`MaskedView::by_kind`]: every kind masked with its default for this mask.
    ByKind(char),
}

impl ViewOrigin {
    /// The policy `kind` has in a view of this origin before any override.
    fn default_for(self, kind: PiiKind) -> MaskPolicy {
        match self {
            ViewOrigin::All(policy) => policy,
            ViewOrigin::ByKind(mask) => MaskPolicy::for_kind(kind, mask).expect("a mask by_kind validated"),
        }
    }
}

/// One slot per built-in id; slot 0 (no kind) is unused.
const BUILTIN_SLOTS: usize = PiiKind::MAX_BUILTIN_ID as usize + 1;

impl MaskedView {
    /// Every mention masked with `policy`: `Masker.mask(document, layer,
    /// policy)`.
    pub fn all(policy: MaskPolicy) -> MaskedView {
        MaskedView {
            policies: [policy; BUILTIN_SLOTS],
            default_policy: policy,
            origin: ViewOrigin::All(policy),
            custom_kinds: CustomKinds::new(),
            custom: Vec::new(),
        }
    }

    /// Each mention masked with its kind's default policy and `mask`:
    /// `Masker.mask` with `MaskPolicies.byType(mask)`.
    pub fn by_kind(mask: char) -> Result<MaskedView, MaskError> {
        let mut view = MaskedView::all(MaskPolicy::new(mask)?);
        view.origin = ViewOrigin::ByKind(mask);
        for k in PiiKind::builtins() {
            view.policies[slot(k)] = MaskPolicy::for_kind(k, mask)?;
        }
        Ok(view)
    }

    /// This view with `kinds` as its custom-kind table, which every custom
    /// kind given to [`MaskedView::with`] afterwards indexes. Drops custom
    /// policies set under an earlier table.
    pub fn with_custom_kinds(mut self, kinds: CustomKinds) -> MaskedView {
        self.custom = vec![None; kinds.len()];
        self.custom_kinds = kinds;
        self
    }

    /// The table every custom kind of this view indexes.
    pub fn custom_kinds(&self) -> &CustomKinds {
        &self.custom_kinds
    }

    /// This view with the mentions of `kind` masked with `policy`.
    ///
    /// # Panics
    ///
    /// When `kind` is a custom kind outside the view's table; see
    /// [`MaskedView::try_with`].
    pub fn with(self, kind: PiiKind, policy: MaskPolicy) -> MaskedView {
        self.try_with(kind, policy).expect("a kind of the view's custom-kind table")
    }

    /// This view with the mentions of `kind` masked with `policy`, refused
    /// when `kind` is a custom kind outside [`MaskedView::custom_kinds`].
    pub fn try_with(mut self, kind: PiiKind, policy: MaskPolicy) -> Result<MaskedView, KindError> {
        self.custom_kinds.check(kind)?;
        match kind {
            PiiKind::Custom(c) => self.custom[usize::from(c.index())] = Some(policy),
            builtin => self.policies[slot(builtin)] = policy,
        }
        Ok(self)
    }

    /// The policy of `kind` of this view's table.
    pub fn policy(&self, kind: PiiKind) -> MaskPolicy {
        match kind {
            PiiKind::Custom(c) => {
                self.custom.get(usize::from(c.index())).copied().flatten().unwrap_or(self.default_policy)
            }
            builtin => self.policies[slot(builtin)],
        }
    }

    /// The policy of `kind` of a layer whose custom kinds index `kinds`: a
    /// custom kind is matched to this view's by name, and one the view does
    /// not list takes the default policy.
    pub fn policy_in(&self, kinds: &CustomKinds, kind: PiiKind) -> MaskPolicy {
        self.custom_kinds.translate(kinds, kind).map_or(self.default_policy, |k| self.policy(k))
    }

    /// The step for a pipeline, writing the view as [`MASKED`].
    pub fn bound(self) -> Bound<MaskedView> {
        Bound::new(self).write(0, MASKED.name)
    }

    /// `src` with the mentions of `pii`, a layer over it, masked. A layer
    /// over another view is refused: its spans are offsets of that view's
    /// text, so masking `src` with them would mask other bytes. A mention
    /// whose custom kind is outside the layer's table is refused.
    pub fn view(&self, src: &View, pii: &Pii) -> Result<View, MaskError> {
        let layer = pii.spans.view();
        if layer != src.id() {
            return Err(MaskError::ForeignView { layer, view: src.id() });
        }
        for m in pii.mentions() {
            pii.custom_kinds.check(m.kind).map_err(MaskError::Kind)?;
        }
        mask_view(src, pii.mentions().map(|m| (m.span, self.policy_in(&pii.custom_kinds, m.kind))))
    }
}

fn slot(kind: PiiKind) -> usize {
    usize::from(kind.builtin_id().unwrap_or(0))
}

impl Annotator for MaskedView {
    const ID: &'static str = "turbo-pii:masked";
    type Reads = (View, Pii);
    type Writes = (View,);

    fn annotate(&self, (src, pii): Refs<'_, Self>) -> Result<(View,), Error> {
        let detail = |e: MaskError| Error::Annotator { annotator: Self::ID, detail: e.to_string() };
        Ok((self.view(src, pii).map_err(detail)?,))
    }

    /// FNV-1a over the view's origin (`all` and its policy, or `by_kind`
    /// and its mask), then each built-in kind whose policy differs from
    /// that origin's default for it, by stable id in id order, then the
    /// custom table with every custom kind's effective policy. Two views of
    /// one origin hash alike exactly when they mask every kind alike, and a
    /// kind added to the registry later starts at its origin default and
    /// folds nothing, so every existing view keeps its hash.
    fn config_hash(&self) -> u64 {
        let fold_policy = |h: &mut turbo_analysis::Fnv1a, p: MaskPolicy| {
            h.u8(p.mask() as u8).u8(u8::from(p.keeps_format()));
            h.fold_u64(p.leading() as u64).fold_u64(p.trailing() as u64);
        };
        let mut h = turbo_analysis::Fnv1a::new();
        match self.origin {
            ViewOrigin::All(policy) => {
                h.u8(0);
                fold_policy(&mut h, policy);
            }
            ViewOrigin::ByKind(mask) => {
                h.u8(1).u8(mask as u8);
            }
        }
        for k in PiiKind::builtins() {
            let policy = self.policies[slot(k)];
            if policy != self.origin.default_for(k) {
                h.u16(k.builtin_id().expect("a built-in kind"));
                fold_policy(&mut h, policy);
            }
        }
        // Id 0 is no kind: it ends the overrides before the custom table.
        h.u16(0).fold_u64(self.custom_kinds.len() as u64);
        for (kind, name) in self.custom_kinds.kinds().zip(self.custom_kinds.names()) {
            h.str(name);
            fold_policy(&mut h, self.policy(kind));
        }
        h.finish()
    }
}
