//! Label policy configuration for the PII detector.

use turbo_analysis::Span;
use turbo_analysis_layers::PiiKind;

use super::PiiDetector;
use crate::window::LabelPolicy;

impl PiiDetector {
    /// This detector with `kind`'s context-window [`LabelPolicy`] set to
    /// `policy`.
    pub fn with_label_policy(mut self, kind: PiiKind, policy: LabelPolicy) -> PiiDetector {
        self.window.set_policy(kind, policy);
        self
    }

    /// This detector requiring a context label within `kind`'s word window
    /// to admit a `kind` candidate ([`LabelPolicy::Required`]), raising its
    /// confidence (`Format -> Labelled`, `Checksum -> LabelledChecksum`).
    pub fn label_required(self, kind: PiiKind) -> PiiDetector {
        self.with_label_policy(kind, LabelPolicy::Required)
    }

    /// This detector boosting `kind`'s confidence (`Format -> Labelled`,
    /// `Checksum -> LabelledChecksum`) when a label appears within `kind`'s
    /// word window ([`LabelPolicy::Boosts`]) while still admitting
    /// unlabelled candidates at their base confidence.
    pub fn label_boosts(self, kind: PiiKind) -> PiiDetector {
        self.with_label_policy(kind, LabelPolicy::Boosts)
    }

    /// This detector with `kind`'s context-word window width set to
    /// `window_words` words on either side of a candidate span.
    pub fn with_label_window(mut self, kind: PiiKind, window_words: usize) -> PiiDetector {
        self.window.set_window(kind, window_words);
        self
    }

    /// This detector with `label` (a single word or multi-word phrase,
    /// matched case-insensitively on ASCII) added to `kind`'s context-label
    /// list.
    pub fn with_extra_label(mut self, kind: PiiKind, label: impl Into<String>) -> PiiDetector {
        self.window.add_extra_label(kind, label.into());
        self
    }

    /// This detector with every term in `labels` added to `kind`'s
    /// context-label list.
    pub fn with_extra_labels<I, S>(mut self, kind: PiiKind, labels: I) -> PiiDetector
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        for label in labels {
            self.window.add_extra_label(kind, label.into());
        }
        self
    }

    /// The effective [`LabelPolicy`] for `kind` on this detector.
    pub fn label_policy(&self, kind: PiiKind) -> LabelPolicy {
        self.window.policy(kind)
    }

    /// The effective context-word window width (`N` words on either side)
    /// for `kind` on this detector.
    pub fn label_window(&self, kind: PiiKind) -> usize {
        self.window.window_words(kind)
    }

    /// Caller-supplied extra label terms configured for `kind`.
    pub fn extra_labels(&self, kind: PiiKind) -> &[String] {
        self.window.extra_labels(kind)
    }

    /// Whether a label for `kind` (default or caller-added) appears within
    /// `kind`'s configured word window around `span` in `text`.
    pub fn has_label(&self, text: &str, span: Span, kind: PiiKind) -> bool {
        self.window.has_label(text, span, kind)
    }
}
