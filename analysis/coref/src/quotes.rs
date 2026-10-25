//! Quotation regions over token forms, shared by the speaker sieve and
//! [`QuotationAnnotator`](crate::QuotationAnnotator).

use crate::input::lowercase;
use crate::lexicon;

/// The quotations of a token sequence. Quote `q` (0-based) opens at token
/// `starts[q]` and closes at `ends[q]` (the last token when it never
/// closes); `quote_of[t]` is `q + 1` for a token strictly inside quote
/// `q`, 0 for every other token, the marks included.
pub(crate) struct QuoteRegions {
    pub quote_of: Vec<usize>,
    pub starts: Vec<usize>,
    pub ends: Vec<usize>,
}

impl QuoteRegions {
    pub(crate) fn scan(forms: &[&str]) -> QuoteRegions {
        let n = forms.len();
        let mut out = QuoteRegions { quote_of: vec![0; n], starts: Vec::new(), ends: Vec::new() };
        let mut open = 0;
        for (t, form) in forms.iter().enumerate() {
            if open == 0 && lexicon::opens_quote(form) {
                out.starts.push(t);
                out.ends.push(n - 1);
                open = out.starts.len();
            } else if open > 0 && lexicon::closes_quote(form) {
                out.ends[open - 1] = t;
                open = 0;
            } else if open > 0 {
                out.quote_of[t] = open;
            }
        }
        out
    }

    /// Whether token `t` is a quotation mark that opens or closes a quote.
    pub(crate) fn is_mark(&self, t: usize) -> bool {
        self.starts.contains(&t) || self.ends.iter().zip(&self.starts).any(|(&e, &s)| e == t && e != s)
    }
}

/// Whether `form` is a verb of speech (`said`, `told`, `warned`).
pub(crate) fn speech_verb(form: &str) -> bool {
    lexicon::speech_verb(&lowercase(form))
}
