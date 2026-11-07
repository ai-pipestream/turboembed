//! The redacted view: a pipeline step writing the text with every mention
//! replaced by its policy's strategy as a derived `View`, together with
//! the `turbo:pii-edits` layer over it that says what each replacement
//! covered and which rule wrote it.
//!
//! The step writes the view first and its edits second, so the edits,
//! made over the pending view, are bound to the id the document gives the
//! view as the two are stored.

use turbo_analysis::{Annotator, Bound, Error, Fnv1a, Key, Refs, View};
use turbo_analysis_layers::{Pii, PiiEdits};

use super::RewritePolicy;

/// The instance name of the redacted view and of its edits layer.
pub const REDACTED: Key<View> = Key::named("redacted");

/// Writes the `redacted` view and its `turbo:pii-edits` layer from the pii
/// layer over the view it reads, with one [`RewritePolicy`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RedactedView {
    policy: RewritePolicy,
}

impl RedactedView {
    pub fn new(policy: RewritePolicy) -> RedactedView {
        RedactedView { policy }
    }

    pub fn policy(&self) -> &RewritePolicy {
        &self.policy
    }

    /// The step for a pipeline, writing the view and its edits as
    /// [`REDACTED`].
    pub fn bound(self) -> Bound<RedactedView> {
        Bound::new(self).write(0, REDACTED.name).write(1, REDACTED.name)
    }
}

impl Annotator for RedactedView {
    const ID: &'static str = "turbo-pii:redacted";
    type Reads = (View, Pii);
    type Writes = (View, PiiEdits);

    fn annotate(&self, (src, pii): Refs<'_, Self>) -> Result<(View, PiiEdits), Error> {
        let rewrite = self
            .policy
            .rewrite(src, pii)
            .map_err(|e| Error::Annotator { annotator: Self::ID, detail: e.to_string() })?;
        Ok(rewrite.into_parts())
    }

    /// FNV-1a over the whole policy: its revision, whether it refuses
    /// overlaps, and every rule with each parameter that changes what its
    /// strategy writes, so two policies that write differently never share
    /// a hash. An HMAC key and a lookup table enter only through a SHA-256
    /// digest, never as themselves.
    fn config_hash(&self) -> u64 {
        let mut h = Fnv1a::new();
        self.policy.fold_config(&mut h);
        h.finish()
    }
}
