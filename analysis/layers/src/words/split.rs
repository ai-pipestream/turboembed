//! A [`Words`] layer from each token's decoded words, as a token split
//! head gives them: the token itself, pieces that tile its bytes, or
//! explicit forms.

use turbo_analysis::{Ix, IxRange, Span};

use super::Words;
use crate::Tokens;

/// Byte sub-spans when `ws` is two or more non-empty pieces that tile
/// `surface`, which starts at `span.start`, from start to end.
fn subspans_of(surface: &str, span: Span, ws: &[String]) -> Option<Vec<Span>> {
    if ws.len() < 2 {
        return None;
    }
    let mut out = Vec::with_capacity(ws.len());
    let mut off = 0usize;
    for w in ws {
        if w.is_empty() || !surface[off..].starts_with(w.as_str()) {
            return None;
        }
        let next = off + w.len();
        out.push(Span::new(span.start + off as u32, span.start + next as u32));
        off = next;
    }
    (off == surface.len()).then_some(out)
}

impl Words {
    /// The words of `tokens` over `text`, `split[i]` token `i`'s words in
    /// order: a 1-to-1 word when it is the token's own text, byte
    /// sub-spans when the words tile the token's bytes, and explicit forms
    /// otherwise (a fusional `"del"` as `"de"` and `"el"`). Checked against
    /// the tokens before it is returned.
    pub fn from_splits(text: &str, tokens: &Tokens, split: &[Vec<String>]) -> Result<Words, String> {
        if split.len() != tokens.spans.len() {
            return Err(format!("{} word lists for {} tokens", split.len(), tokens.spans.len()));
        }
        let mut out = Words::new();
        for (i, ws) in split.iter().enumerate() {
            let t = Ix::<Tokens>::new(i as u32);
            let span = tokens.spans.get(t);
            let surface = text.get(span.range()).ok_or_else(|| format!("token {i} is outside the text"))?;
            let range = IxRange::new(i as u32, i as u32 + 1);
            if ws.len() == 1 && ws[0] == surface {
                out.push_token(t);
            } else if let Some(subs) = subspans_of(surface, span, ws) {
                for sub in subs {
                    out.push_word(range, None, Some(sub));
                }
            } else {
                for w in ws {
                    out.push_word(range, Some(w), None);
                }
            }
        }
        out.check_against(text, tokens)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use turbo_analysis::{Spans, ViewId};

    use super::*;

    fn tokens(text: &str) -> Tokens {
        let mut spans = Spans::new(ViewId::SOURCE);
        let mut start = 0;
        for word in text.split(' ') {
            spans.push(Span::new(start as u32, (start + word.len()) as u32));
            start += word.len() + 1;
        }
        Tokens { spans, sentence: None }
    }

    fn owned(ws: &[&str]) -> Vec<String> {
        ws.iter().map(|w| w.to_string()).collect()
    }

    #[test]
    fn kept_tiled_and_fusional_words() {
        let text = "can't del día";
        let toks = tokens(text);
        let split = [owned(&["ca", "n't"]), owned(&["de", "el"]), owned(&["día"])];
        let words = Words::from_splits(text, &toks, &split).unwrap();
        let got: Vec<(&str, Option<Span>, usize)> =
            words.iter(text, &toks).map(|w| (w.form, w.subspan, w.tokens.start.get())).collect();
        assert_eq!(
            got,
            [
                ("ca", Some(Span::new(0, 2)), 0),
                ("n't", Some(Span::new(2, 5)), 0),
                ("de", None, 1),
                ("el", None, 1),
                ("día", None, 2),
            ]
        );
        // A word that is neither the token nor a tiling is an explicit form.
        let words = Words::from_splits(text, &toks, &[owned(&["can"]), owned(&["del"]), owned(&["día"])]).unwrap();
        assert_eq!(words.form.as_ref().unwrap()[0].as_deref(), Some("can"));
        assert!(Words::from_splits(text, &toks, &[owned(&["can't"])]).is_err());
    }
}
