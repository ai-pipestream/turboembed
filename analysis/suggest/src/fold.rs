//! Unicode case and diacritic folding for surface-prefix lookup keys.

use turbo_normalize::{AccentFold, CaseFold, Chain, UnicodeForm, apply};

/// Folds `text` into a canonical prefix-lookup key by applying Unicode NFC,
/// lowercase case-folding, and diacritic/accent folding (`"Café"` -> `"cafe"`,
/// `"COURTESY"` -> `"courtesy"`, `"naïve"` -> `"naive"`).
pub fn fold_surface_key(text: &str) -> String {
    if text.is_ascii() {
        return text.to_ascii_lowercase();
    }
    let chain = Chain::new().then(UnicodeForm::nfc()).then(CaseFold::lowercase()).then(AccentFold::new());
    apply(&chain, text).into_owned()
}

/// Lowercases `text` in NFC form while keeping diacritics intact so canonical
/// surface completions retain human-readable orthography (`"Café"` -> `"café"`).
pub fn canonical_surface_lowercase(text: &str) -> String {
    if text.is_ascii() {
        return text.to_ascii_lowercase();
    }
    let chain = Chain::new().then(UnicodeForm::nfc()).then(CaseFold::lowercase());
    apply(&chain, text).into_owned()
}
