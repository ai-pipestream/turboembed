//! Guard row joins between the default layers selected by ProjectChunks.

use turbo_analysis::{Document, Key, Layer};

use super::bad;
use crate::api::Failure;

/// A column selected by its default key may be validly anchored to another
/// instance. Refuse that join rather than assigning its rows to default output
/// spans. `made_on` includes bindings resolved from both `on` and `reads`.
pub(super) fn require_default_binding<L: Layer, A: Layer>(doc: &Document) -> Result<(), Failure> {
    let key = Key::<L>::DEFAULT.id();
    let target = Key::<A>::DEFAULT.id();
    for actual in doc.made_on(key).iter().filter(|actual| actual.ty == target.ty) {
        if *actual != target {
            return Err(bad(format!("{key} is bound to instance {actual}, but projection selects {target}")));
        }
    }
    Ok(())
}
