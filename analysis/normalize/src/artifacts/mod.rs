//! Text artifact detection: damaged or hidden character sequences in a
//! view, reported as an `Artifacts` layer. The text is not changed.
//!
//! The detector is a port of OpenNLP's `CursorArtifactDetector` (the
//! text-artifacts add-on): replacement characters, controls,
//! noncharacters, private-use code points and bidirectional controls as
//! maximal runs of one class; zero-width characters and Unicode tags by
//! their context rules; mojibake by the Windows-1252 byte round trip.
//! Two kinds go further than OpenNLP and are off unless selected:
//! `MojibakeSequence` (`sequences.rs`) and `RepeatedLine`
//! (`repeated.rs`). docs/normalizers.md describes the rules and the
//! differences.
//!
//! Findings never overlap. Where they would, the more specific finding
//! keeps its span and the other is cut around it: round-trip mojibake
//! first, then mojibake sequences, then the character classes, then
//! repeated lines.

mod classes;
mod codepage;
pub(crate) mod lines;
mod mojibake;
mod repeated;
mod sequences;

use turbo_analysis::{Annotator, Error, Refs, Span, View};
use turbo_normalize_layers::{ArtifactKind, Artifacts};

pub use repeated::RepeatedLines;

/// One finding: where it is in the text and what it is.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Artifact {
    pub span: Span,
    pub kind: ArtifactKind,
}

impl Artifact {
    pub(crate) fn new(start: usize, end: usize, kind: ArtifactKind) -> Artifact {
        Artifact { span: Span::new(start as u32, end as u32), kind }
    }
}

/// A set of artifact kinds.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct KindSet(u16);

impl KindSet {
    /// The kinds OpenNLP's detector reports by default.
    pub const OPENNLP: KindSet = KindSet::of(&ArtifactKind::OPENNLP);

    /// Every kind.
    pub const ALL: KindSet = KindSet::of(&ArtifactKind::ALL);

    pub const fn of(kinds: &[ArtifactKind]) -> KindSet {
        let mut bits = 0;
        let mut i = 0;
        while i < kinds.len() {
            bits |= 1 << kinds[i] as u16;
            i += 1;
        }
        KindSet(bits)
    }

    pub const fn contains(self, kind: ArtifactKind) -> bool {
        self.0 & (1 << kind as u16) != 0
    }

    pub const fn with(self, kind: ArtifactKind) -> KindSet {
        KindSet(self.0 | 1 << kind as u16)
    }

    pub const fn without(self, kind: ArtifactKind) -> KindSet {
        KindSet(self.0 & !(1 << kind as u16))
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Whether any of the character classes the class scan reports is in
    /// the set.
    fn has_class(self) -> bool {
        !KindSet(
            self.0
                & !KindSet::of(&[ArtifactKind::Mojibake, ArtifactKind::MojibakeSequence, ArtifactKind::RepeatedLine]).0,
        )
        .is_empty()
    }
}

/// Finds artifacts in a text. `ArtifactDetector::default()` reports the
/// nine kinds OpenNLP reports, as OpenNLP does; `builder` selects others.
/// It holds only its configuration and is safe to share between threads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactDetector {
    kinds: KindSet,
    repeated: RepeatedLines,
}

impl Default for ArtifactDetector {
    fn default() -> ArtifactDetector {
        ArtifactDetector { kinds: KindSet::OPENNLP, repeated: RepeatedLines::default() }
    }
}

/// Configures an `ArtifactDetector`. It starts from the default: OpenNLP's
/// nine kinds.
#[derive(Clone, Debug)]
pub struct ArtifactDetectorBuilder {
    detector: ArtifactDetector,
}

impl ArtifactDetectorBuilder {
    /// Reports exactly `kinds`, as OpenNLP's `CursorArtifactDetector(Set)`
    /// does. An empty set reports nothing.
    pub fn only(mut self, kinds: &[ArtifactKind]) -> ArtifactDetectorBuilder {
        self.detector.kinds = KindSet::of(kinds);
        self
    }

    /// Adds one kind.
    pub fn with(mut self, kind: ArtifactKind) -> ArtifactDetectorBuilder {
        self.detector.kinds = self.detector.kinds.with(kind);
        self
    }

    /// Removes one kind.
    pub fn without(mut self, kind: ArtifactKind) -> ArtifactDetectorBuilder {
        self.detector.kinds = self.detector.kinds.without(kind);
        self
    }

    /// Reports repeated lines with these options (and adds the kind).
    pub fn repeated_lines(mut self, options: RepeatedLines) -> ArtifactDetectorBuilder {
        self.detector.repeated = options;
        self.with(ArtifactKind::RepeatedLine)
    }

    pub fn build(self) -> ArtifactDetector {
        self.detector
    }
}

impl ArtifactDetector {
    pub fn builder() -> ArtifactDetectorBuilder {
        ArtifactDetectorBuilder { detector: ArtifactDetector::default() }
    }

    /// The kinds this detector reports.
    pub fn kinds(&self) -> KindSet {
        self.kinds
    }

    /// The findings in `text`, in text order, not overlapping, with UTF-8
    /// byte offsets.
    pub fn detect(&self, text: &str) -> Vec<Artifact> {
        let mut classes = Vec::new();
        if self.kinds.has_class() {
            classes::scan(text, self.kinds, &mut classes);
        }
        let mut damage = Vec::new();
        if self.kinds.contains(ArtifactKind::Mojibake) {
            mojibake::scan(text, &mut damage);
        }
        if self.kinds.contains(ArtifactKind::MojibakeSequence) {
            let skip: Vec<Span> = damage.iter().map(|a| a.span).collect();
            let mut found = Vec::new();
            sequences::scan(text, &skip, &mut found);
            let sequences = found.into_iter().map(|span| Artifact { span, kind: ArtifactKind::MojibakeSequence });
            damage = merge_sorted(damage, sequences.collect());
        }
        let mut found = overlay(classes, damage);
        if self.kinds.contains(ArtifactKind::RepeatedLine) {
            let mut lines = Vec::new();
            repeated::scan(text, self.repeated, &mut lines);
            found = overlay(lines, found);
        }
        found
    }

    /// The findings in `text` as a layer over `view`.
    pub fn layer(&self, text: &str, view: turbo_analysis::ViewId) -> Artifacts {
        let found = self.detect(text);
        let mut layer = Artifacts::new(view);
        layer.kind.reserve(found.len());
        for a in found {
            layer.spans.push(a.span);
            layer.kind.push(a.kind);
        }
        layer
    }
}

impl Annotator for ArtifactDetector {
    const ID: &'static str = "turbo:artifacts";
    type Reads = (View,);
    type Writes = (Artifacts,);

    fn annotate(&self, (view,): Refs<'_, Self>) -> Result<(Artifacts,), Error> {
        Ok((self.layer(view.text(), view.id()),))
    }

    fn config_hash(&self) -> u64 {
        crate::hash_config(&(self.kinds, self.repeated))
    }
}

/// Two lists of findings that do not overlap each other, each in text
/// order, as one list in text order.
fn merge_sorted(a: Vec<Artifact>, b: Vec<Artifact>) -> Vec<Artifact> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let mut b = b.into_iter().peekable();
    for x in a {
        while let Some(y) = b.next_if(|y| y.span.start < x.span.start) {
            out.push(y);
        }
        out.push(x);
    }
    out.extend(b);
    out
}

/// Lays `top` over `base`: every finding of `top` is kept, and each
/// finding of `base` keeps only the parts no finding of `top` covers (a
/// part before, between or after them). Both lists are in text order and
/// free of overlaps; so is the result. With the character classes as
/// `base` and the round-trip mojibake as `top`, this is the merge of
/// OpenNLP's `CursorArtifactDetector`.
fn overlay(base: Vec<Artifact>, top: Vec<Artifact>) -> Vec<Artifact> {
    if top.is_empty() {
        return base;
    }
    let mut out = Vec::with_capacity(base.len() + top.len());
    let mut top = top.into_iter().peekable();
    for b in base {
        let mut rest = b.span;
        loop {
            while let Some(t) = top.next_if(|t| t.span.end <= rest.start) {
                out.push(t);
            }
            match top.peek() {
                Some(t) if t.span.start < rest.end => {
                    if rest.start < t.span.start {
                        out.push(Artifact { span: Span::new(rest.start, t.span.start), kind: b.kind });
                    }
                    if t.span.end >= rest.end {
                        // The rest of `b` is covered. `t` may reach into the
                        // next finding of `base`, so it stays for that one.
                        break;
                    }
                    rest = Span::new(t.span.end, rest.end);
                    out.extend(top.next());
                }
                _ => {
                    out.push(Artifact { span: rest, kind: b.kind });
                    break;
                }
            }
        }
    }
    out.extend(top);
    out
}

/// The character starting at byte `at`, which is on a character boundary
/// before the end of `text`.
#[inline]
fn char_at(text: &str, at: usize) -> char {
    text[at..].chars().next().expect("a character at a boundary before the end")
}

/// The character ending at byte `at`, if any.
#[inline]
fn char_before(text: &str, at: usize) -> Option<char> {
    text[..at].chars().next_back()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a(start: u32, end: u32, kind: ArtifactKind) -> Artifact {
        Artifact { span: Span::new(start, end), kind }
    }

    #[test]
    fn overlay_cuts_base_findings_around_top_findings() {
        use ArtifactKind::{Control, Mojibake};
        let base = vec![a(0, 10, Control), a(12, 14, Control)];
        let top = vec![a(2, 4, Mojibake), a(6, 8, Mojibake), a(9, 13, Mojibake), a(20, 21, Mojibake)];
        assert_eq!(
            overlay(base, top),
            vec![
                a(0, 2, Control),
                a(2, 4, Mojibake),
                a(4, 6, Control),
                a(6, 8, Mojibake),
                a(8, 9, Control),
                a(9, 13, Mojibake),
                a(13, 14, Control),
                a(20, 21, Mojibake),
            ]
        );
    }

    #[test]
    fn overlay_drops_base_findings_a_top_finding_covers() {
        use ArtifactKind::{Control, Mojibake};
        assert_eq!(overlay(vec![a(1, 2, Control), a(2, 3, Control)], vec![a(0, 4, Mojibake)]), vec![a(0, 4, Mojibake)]);
        assert_eq!(overlay(vec![], vec![a(0, 4, Mojibake)]), vec![a(0, 4, Mojibake)]);
        assert_eq!(overlay(vec![a(0, 4, Control)], vec![]), vec![a(0, 4, Control)]);
    }

    #[test]
    fn kind_sets() {
        assert!(KindSet::OPENNLP.contains(ArtifactKind::UnpairedSurrogate));
        assert!(!KindSet::OPENNLP.contains(ArtifactKind::RepeatedLine));
        assert!(KindSet::ALL.contains(ArtifactKind::RepeatedLine));
        assert!(!KindSet::of(&[ArtifactKind::Mojibake]).has_class());
        assert!(KindSet::of(&[ArtifactKind::UnpairedSurrogate]).has_class());
        assert!(KindSet::default().is_empty());
    }
}
