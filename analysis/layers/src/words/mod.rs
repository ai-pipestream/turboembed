//! Syntactic words beside surface tokens for Universal Dependencies
//! multi-word tokens (`opennlp:words`), word-anchored morphological layers,
//! and the rule-based multi-word token expander for Romance, German and
//! English contractions.
//!
//! In Universal Dependencies, a single surface token can correspond to two or
//! more syntactic words whose forms are not substrings of the surface token:
//! Spanish `"del"` is `"de"` + `"el"`, French `"au"` is `"à"` + `"le"`,
//! Portuguese `"pelos"` is `"por"` + `"os"`, Italian `"nella"` is `"in"` +
//! `"la"`, and German `"zum"` is `"zu"` + `"dem"`. Conversely, an English
//! clitic contraction such as `"can't"` splits into `"ca"` + `"n't"`, whose
//! pieces are contiguous byte substrings of the surface token.
//!
//! [`Words`] (`opennlp:words`) sits beside [`Tokens`]:
//! - Every syntactic word points at a non-empty [`IxRange<Tokens>`] (normally a
//!   single surface token `t..t+1`, or a multi-token range when a syntactic
//!   word spans several tokens).
//! - When a language or document has no multi-word tokens, `form` and `subspan`
//!   are `None`, so [`Words`] costs only the single `Vec<IxRange<Tokens>>`
//!   column (`8` bytes per word) and reads each word's surface slice directly
//!   from the document view.
//! - When a word's form is not equal to its token's slice, `form` carries its
//!   syntactic form (`"de"`, `"el"`), and `subspan` carries `Some(Span)` only
//!   when that form is a contiguous byte substring of the token (`"ca"` and
//!   `"n't"` inside `"can't"`).
//! - Every word maps to its token's span (`Words::token_span`), so UTF-16 and
//!   code-point conversions at the edge (`Words::utf16_span`,
//!   `Words::code_point_span`) never split a surface token between words.

mod mwt;

use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

use turbo_analysis::{
    Anchor, CodePointOffset, Ix, IxRange, Key, KeyId, Layer, LayerName, OffsetTable, Span, Tag, TagSet, Utf16Offset,
};

pub use mwt::{MwtLanguage, RuleMwtExpander, TokenExpansion};

use crate::{Sentences, Strs, Tokens};

/// One syntactic word read from [`Words`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Word<'a> {
    /// The surface token range (normally one token `t..t + 1`) this syntactic
    /// word belongs to.
    pub tokens: IxRange<Tokens>,
    /// The UTF-8 byte span of the word's surface token(s) in the view.
    pub token_span: Span,
    /// Present only when the word is a proper contiguous byte substring of its
    /// surface token (for example `"ca"` and `"n't"` in `"can't"`). `None` for
    /// 1-to-1 words and for fusional contractions (`"del"` -> `"de"` + `"el"`).
    pub subspan: Option<Span>,
    /// The syntactic word's surface or expanded form.
    pub form: &'a str,
}

/// Universal Dependencies syntactic words (`opennlp:words`).
///
/// Anchored to the document and indexing [`Tokens`] (default or named via
/// `tokens_key`), because multi-word token expansion produces `Words.len() >=
/// Tokens.len()`.
#[derive(Clone, Debug)]
pub struct Words {
    /// The surface token range (`r.start < r.end`) each syntactic word belongs
    /// to, in non-decreasing token order.
    pub tokens: Vec<IxRange<Tokens>>,
    /// Explicit word forms when one or more words have a form that is not a
    /// substring of the view (`"del"` -> `"de"`, `"el"`). `None` when every
    /// word's form is its `subspan` or `token_span` slice in the view.
    pub form: Option<Vec<Option<Box<str>>>>,
    /// Optional byte sub-span inside the surface token when a split word's form
    /// is a contiguous substring of the token (`"can't"` -> `"ca"` `0..2`,
    /// `"n't"` `2..5`). `None` when no word carries a sub-span.
    pub subspan: Option<Vec<Option<Span>>>,
    /// Which [`Tokens`] instance `tokens` indexes (`Key::DEFAULT` unless set
    /// via [`Words::on`]).
    pub tokens_key: Key<Tokens>,
}

impl Default for Words {
    fn default() -> Self {
        Self::new()
    }
}

impl Words {
    /// Creates an empty `Words` layer indexing the default [`Tokens`] instance.
    pub fn new() -> Words {
        Words { tokens: Vec::new(), form: None, subspan: None, tokens_key: Key::DEFAULT }
    }

    /// Creates an empty `Words` layer indexing the named `tokens_key` instance.
    pub fn on(tokens_key: Key<Tokens>) -> Words {
        Words { tokens: Vec::new(), form: None, subspan: None, tokens_key }
    }

    /// Builds a 1-to-1 `Words` layer from `tokens` when a language or document
    /// has no multi-word tokens. Allocates only the single `Vec<IxRange<Tokens>>`
    /// index column (`form` and `subspan` remain `None`).
    pub fn from_tokens(tokens: &Tokens) -> Words {
        let n = tokens.spans.len() as u32;
        Words {
            tokens: (0..n).map(|i| IxRange::new(i, i + 1)).collect(),
            form: None,
            subspan: None,
            tokens_key: Key::DEFAULT,
        }
    }

    /// Whether every word in this layer is 1-to-1 with a single token and
    /// carries neither an explicit form nor a sub-span.
    pub fn is_identity(&self) -> bool {
        self.form.is_none()
            && self.subspan.is_none()
            && self.tokens.iter().enumerate().all(|(i, r)| r.start.get() == i && r.end.get() == i + 1)
    }

    /// Appends a 1-to-1 word for surface token `t` whose form is the token's
    /// own text slice.
    pub fn push_token(&mut self, t: Ix<Tokens>) -> Ix<Words> {
        let idx = t.get() as u32;
        self.push_word(IxRange::new(idx, idx + 1), None, None)
    }

    /// Appends a syntactic word covering `tokens`, with an optional explicit
    /// `form` (when not a substring of the view) and an optional `subspan`
    /// (when the word is a proper substring of its token).
    pub fn push_word(&mut self, tokens: IxRange<Tokens>, form: Option<&str>, subspan: Option<Span>) -> Ix<Words> {
        let n_before = self.tokens.len();
        let ix = Ix::new(n_before as u32);
        self.tokens.push(tokens);
        match (&mut self.form, form) {
            (Some(forms), f) => forms.push(f.map(Box::from)),
            (None, Some(f)) => {
                let mut forms = vec![None; n_before];
                forms.push(Some(Box::from(f)));
                self.form = Some(forms);
            }
            (None, None) => {}
        }
        match (&mut self.subspan, subspan) {
            (Some(subs), s) => subs.push(s),
            (None, Some(s)) => {
                let mut subs = vec![None; n_before];
                subs.push(Some(s));
                self.subspan = Some(subs);
            }
            (None, None) => {}
        }
        ix
    }

    /// The contiguous range of syntactic words belonging to surface token `t`,
    /// found by binary search (`O(log N)`).
    pub fn of_token(&self, t: Ix<Tokens>) -> IxRange<Words> {
        let a = self.tokens.partition_point(|r| r.end <= t);
        let b = a + self.tokens[a..].partition_point(|r| r.start <= t);
        IxRange::new(a as u32, b as u32)
    }

    /// The contiguous range of syntactic words in sentence `s`, found by
    /// binary search over `tokens.of(s)`.
    pub fn of_sentence(&self, tokens: &Tokens, s: Ix<Sentences>) -> IxRange<Words> {
        let tok_range = tokens.of(s);
        if tok_range.is_empty() {
            let at = self.tokens.partition_point(|r| r.start < tok_range.start) as u32;
            return IxRange::new(at, at);
        }
        let a = self.tokens.partition_point(|r| r.end <= tok_range.start);
        let b = a + self.tokens[a..].partition_point(|r| r.start < tok_range.end);
        IxRange::new(a as u32, b as u32)
    }

    /// The byte span of word `w`'s surface token(s) in `tokens`'s view. Every
    /// word maps to its token's span, even when multiple words share one
    /// fusional token (`"del"` -> `"de"` + `"el"`).
    pub fn token_span(&self, w: Ix<Words>, tokens: &Tokens) -> Span {
        let r = self.tokens[w.get()];
        let first = tokens.spans.get(r.start);
        let last = tokens.spans.get(Ix::new(r.end.get() as u32 - 1));
        Span::new(first.start, last.end)
    }

    /// The optional substring span of word `w`, present only when `w`'s form is
    /// a contiguous substring of its surface token (`"can't"` -> `"ca"` +
    /// `"n't"`).
    pub fn subspan_of(&self, w: Ix<Words>) -> Option<Span> {
        self.subspan.as_ref().and_then(|s| s[w.get()])
    }

    /// Converts word `w` to a UTF-16 `(start, end)` span at the edge using its
    /// surface token's span, so UTF-16 conversion never splits a surface token
    /// between words.
    pub fn utf16_span(
        &self,
        w: Ix<Words>,
        text: &str,
        tokens: &Tokens,
        offsets: &OffsetTable,
    ) -> (Utf16Offset, Utf16Offset) {
        offsets.span_to_utf16(text, self.token_span(w, tokens))
    }

    /// Converts word `w` to a Unicode code-point `(start, end)` span at the
    /// edge using its surface token's span.
    pub fn code_point_span(
        &self,
        w: Ix<Words>,
        text: &str,
        tokens: &Tokens,
        offsets: &OffsetTable,
    ) -> (CodePointOffset, CodePointOffset) {
        offsets.span_to_code_point(text, self.token_span(w, tokens))
    }

    /// Returns the surface or expanded string of word `w`, borrowing from
    /// `self.form` when present, or slicing `view_text` by `subspan` or
    /// `token_span` with zero allocation.
    pub fn form<'a>(&'a self, w: Ix<Words>, view_text: &'a str, tokens: &Tokens) -> &'a str {
        let idx = w.get();
        if let Some(forms) = &self.form
            && let Some(f) = &forms[idx]
        {
            return f.as_ref();
        }
        let span = self.subspan_of(w).unwrap_or_else(|| self.token_span(w, tokens));
        &view_text[span.range()]
    }

    /// Reads word `w` with its token range, token span, optional sub-span, and
    /// form.
    pub fn get<'a>(&'a self, w: Ix<Words>, view_text: &'a str, tokens: &Tokens) -> Word<'a> {
        Word {
            tokens: self.tokens[w.get()],
            token_span: self.token_span(w, tokens),
            subspan: self.subspan_of(w),
            form: self.form(w, view_text, tokens),
        }
    }

    /// Iterates over all syntactic words in order.
    pub fn iter<'a>(&'a self, view_text: &'a str, tokens: &'a Tokens) -> impl ExactSizeIterator<Item = Word<'a>> + 'a {
        (0..self.tokens.len() as u32).map(move |i| self.get(Ix::new(i), view_text, tokens))
    }
}

impl Layer for Words {
    const ID: &'static str = "opennlp:words";

    fn anchor() -> Anchor {
        Anchor::Document
    }

    fn len(&self) -> usize {
        self.tokens.len()
    }

    fn indexes(&self) -> Vec<(KeyId, usize)> {
        let max_tok = self.tokens.iter().map(|r| r.end.get()).max().unwrap_or(0);
        vec![(self.tokens_key.id(), max_tok)]
    }

    fn check(&self) -> Result<(), String> {
        let n = self.tokens.len();
        if let Some(forms) = &self.form {
            if forms.len() != n {
                return Err(format!("Words.form length ({}) does not match tokens length ({n})", forms.len()));
            }
            if let Some(i) = forms.iter().position(|f| f.as_deref().is_some_and(str::is_empty)) {
                return Err(format!("word {i} has an empty explicit form"));
            }
        }
        if let Some(subs) = &self.subspan
            && subs.len() != n
        {
            return Err(format!("Words.subspan length ({}) does not match tokens length ({n})", subs.len()));
        }
        for (i, r) in self.tokens.iter().enumerate() {
            if r.start >= r.end {
                return Err(format!(
                    "word {i} has an empty or backwards token range {}..{}",
                    r.start.get(),
                    r.end.get()
                ));
            }
            if i > 0 {
                let prev = self.tokens[i - 1];
                if r.start < prev.start || r.end < prev.end {
                    return Err(format!(
                        "word {i} token range {}..{} is out of token order after {}..{}",
                        r.start.get(),
                        r.end.get(),
                        prev.start.get(),
                        prev.end.get()
                    ));
                }
                if r != &prev && r.start < prev.end {
                    return Err(format!(
                        "word {i} token range {}..{} partially overlaps {}..{}",
                        r.start.get(),
                        r.end.get(),
                        prev.start.get(),
                        prev.end.get()
                    ));
                }
            }
            if let Some(subs) = &self.subspan
                && let Some(sub) = subs[i]
            {
                if sub.start >= sub.end {
                    return Err(format!("word {i} has an empty or backwards subspan {}..{}", sub.start, sub.end));
                }
                if i > 0
                    && self.tokens[i - 1] == *r
                    && let Some(prev_sub) = subs[i - 1]
                    && sub.start < prev_sub.end
                {
                    return Err(format!(
                        "word {i} subspan {}..{} overlaps previous subspan {}..{} in the same token",
                        sub.start, sub.end, prev_sub.start, prev_sub.end
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Sealed marker trait for the two granularities morphosyntactic layers can
/// anchor on: surface [`Tokens`] (`OF = None`, keeping the existing layer names
/// `opennlp:pos`, `opennlp:lemmas`, `opennlp:dependencies`) and syntactic
/// [`Words`] (`OF = Some(Words::NAME)`, giving `opennlp:pos<opennlp:words>`,
/// `opennlp:lemmas<opennlp:words>`, `opennlp:dependencies<opennlp:words>`,
/// `opennlp:feats<opennlp:words>` so token-based and word-based layers can
/// coexist on the same [`turbo_analysis::Document`]).
pub trait WordOrTokenAnchor: Layer {
    const TARGET_OF: Option<&'static LayerName>;
}

impl WordOrTokenAnchor for Tokens {
    const TARGET_OF: Option<&'static LayerName> = None;
}

impl WordOrTokenAnchor for Words {
    const TARGET_OF: Option<&'static LayerName> = Some(Words::NAME);
}

/// One part-of-speech tag per syntactic [`Words`] item
/// (`opennlp:pos<opennlp:words>`). Coexists with token-anchored [`crate::PosTags`]
/// (`opennlp:pos`) on the same document.
#[derive(Clone, Debug)]
pub struct WordPosTags {
    pub tagset: Arc<TagSet>,
    pub tags: Vec<Tag>,
    pub prob: Option<Vec<f64>>,
}

impl Layer for WordPosTags {
    const ID: &'static str = "opennlp:pos";
    const OF: Option<&'static LayerName> = Some(Words::NAME);

    fn anchor() -> Anchor {
        Anchor::aligned::<Words>()
    }

    fn len(&self) -> usize {
        self.tags.len()
    }

    fn check(&self) -> Result<(), String> {
        if self.tags.iter().any(|t| t.0 as usize >= self.tagset.len()) {
            return Err(format!("a tag outside {}", self.tagset.name));
        }
        if self.prob.as_ref().is_some_and(|p| p.len() != self.tags.len()) {
            return Err("a probability for every word tag".into());
        }
        Ok(())
    }
}

/// One lemma per syntactic [`Words`] item (`opennlp:lemmas<opennlp:words>`).
/// Coexists with token-anchored [`crate::Lemmas`] (`opennlp:lemmas`) on the same
/// document.
#[derive(Clone, Debug, Default)]
pub struct WordLemmas {
    pub lemma: Strs,
}

impl Layer for WordLemmas {
    const ID: &'static str = "opennlp:lemmas";
    const OF: Option<&'static LayerName> = Some(Words::NAME);

    fn anchor() -> Anchor {
        Anchor::aligned::<Words>()
    }

    fn len(&self) -> usize {
        self.lemma.len()
    }
}

/// Universal Dependencies morphological features (`FEATS` column, such as
/// `Gender=Masc|Number=Sing` or `_`), aligned to `A` (by default [`Words`]:
/// `opennlp:feats<opennlp:words>`).
pub struct MorphFeatures<A: WordOrTokenAnchor = Words> {
    pub feats: Strs,
    pub _of: PhantomData<fn() -> A>,
}

impl<A: WordOrTokenAnchor> Clone for MorphFeatures<A> {
    fn clone(&self) -> Self {
        MorphFeatures { feats: self.feats.clone(), _of: PhantomData }
    }
}

impl<A: WordOrTokenAnchor> Default for MorphFeatures<A> {
    fn default() -> Self {
        Self::new(Strs::default())
    }
}

impl<A: WordOrTokenAnchor> fmt::Debug for MorphFeatures<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MorphFeatures").field("feats", &self.feats).finish()
    }
}

impl<A: WordOrTokenAnchor> MorphFeatures<A> {
    pub fn new(feats: Strs) -> MorphFeatures<A> {
        MorphFeatures { feats, _of: PhantomData }
    }
}

impl<A: WordOrTokenAnchor> Layer for MorphFeatures<A> {
    const ID: &'static str = "opennlp:feats";
    const OF: Option<&'static LayerName> = A::TARGET_OF;

    fn anchor() -> Anchor {
        Anchor::aligned::<A>()
    }

    fn len(&self) -> usize {
        self.feats.len()
    }
}
