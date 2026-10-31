use super::{Word, WordLemmas, WordPosTags, Words};
use crate::{Lemmas, PosTags, Tokens};
use turbo_analysis::{Ix, IxRange, Layer, LayerName, Tag, TagSet, View};

/// Typed POS access shared by token and word layers.
pub trait MorphPosTags: Layer {
    fn label(&self, index: usize) -> Option<&str>;
    /// The tag of item `index`, when it is inside the tag set.
    fn tag(&self, index: usize) -> Option<Tag>;
    /// The tag set the tags index.
    fn tagset(&self) -> &TagSet;
}
impl MorphPosTags for PosTags {
    fn label(&self, index: usize) -> Option<&str> {
        self.tag(index).map(|t| self.tagset.label(t))
    }
    fn tag(&self, index: usize) -> Option<Tag> {
        self.tags.get(index).copied().filter(|t| (t.0 as usize) < self.tagset.len())
    }
    fn tagset(&self) -> &TagSet {
        &self.tagset
    }
}
impl MorphPosTags for WordPosTags {
    fn label(&self, index: usize) -> Option<&str> {
        self.tag(index).map(|t| self.tagset.label(t))
    }
    fn tag(&self, index: usize) -> Option<Tag> {
        self.tags.get(index).copied().filter(|t| (t.0 as usize) < self.tagset.len())
    }
    fn tagset(&self) -> &TagSet {
        &self.tagset
    }
}
/// Typed lemma access shared by token and word layers.
pub trait MorphLemmas: Layer {
    /// The lemma of item `index`.
    fn lemma(&self, index: usize) -> Option<&str>;
}
impl MorphLemmas for Lemmas {
    fn lemma(&self, index: usize) -> Option<&str> {
        (index < self.lemma.len()).then(|| self.lemma.get(index))
    }
}
impl MorphLemmas for WordLemmas {
    fn lemma(&self, index: usize) -> Option<&str> {
        (index < self.lemma.len()).then(|| self.lemma.get(index))
    }
}
/// The two granularities of morphosyntactic layers. Their POS types differ at
/// compile time, so word dependencies cannot accidentally consume token tags.
/// `TARGET_OF = None` keeps token layer names unqualified; `Some(Words::NAME)`
/// qualifies word layers with `<turbo:words>`.
pub trait WordOrTokenAnchor: Layer {
    const TARGET_OF: Option<&'static LayerName>;
    type Pos: MorphPosTags;
    /// The lemma layer aligned to this anchor (`Lemmas` for tokens,
    /// `WordLemmas` for words).
    type Lemmas: MorphLemmas;
    /// Resolves forms and their surface ranges, checking UTF-8 and alignment.
    fn resolved<'a>(&'a self, view: &'a View, tokens: &'a Tokens) -> Result<Vec<Word<'a>>, String>;
}
fn check_tokens(view: &View, tokens: &Tokens) -> Result<(), String> {
    tokens.check()?;
    let (starts, ends) = tokens.spans.columns();
    if tokens.len() >= u32::MAX as usize
        || !starts.windows(2).all(|w| w[0] <= w[1])
        || !ends.windows(2).all(|w| w[0] <= w[1])
    {
        return Err("tokens must fit typed ranges and follow text order".into());
    }
    if tokens.spans.view() != view.id() {
        return Err("token view mismatch".into());
    }
    for (i, span) in tokens.spans.iter() {
        if tokens.spans.is_discontinuous(i) || span.start > span.end || view.text().get(span.range()).is_none() {
            return Err("invalid token span".into());
        }
    }
    Ok(())
}
impl WordOrTokenAnchor for Tokens {
    const TARGET_OF: Option<&'static LayerName> = None;
    type Pos = PosTags;
    type Lemmas = Lemmas;
    fn resolved<'a>(&'a self, view: &'a View, tokens: &'a Tokens) -> Result<Vec<Word<'a>>, String> {
        check_tokens(view, self)?;
        check_tokens(view, tokens)?;
        if self.spans.columns() != tokens.spans.columns() || self.sentence != tokens.sentence {
            return Err("token anchor mismatch".into());
        }
        Ok(self
            .spans
            .iter()
            .map(|(i, span)| Word {
                tokens: IxRange::new(i.get() as u32, i.get() as u32 + 1),
                token_span: span,
                subspan: None,
                form: &view.text()[span.range()],
            })
            .collect())
    }
}
impl WordOrTokenAnchor for Words {
    const TARGET_OF: Option<&'static LayerName> = Some(Words::NAME);
    type Pos = WordPosTags;
    type Lemmas = WordLemmas;
    fn resolved<'a>(&'a self, view: &'a View, tokens: &'a Tokens) -> Result<Vec<Word<'a>>, String> {
        check_tokens(view, tokens)?;
        if self.len() >= u32::MAX as usize {
            return Err("too many words".into());
        }
        self.check_against(view.text(), tokens)?;
        Ok((0..self.len()).map(|i| self.get(Ix::new(i as u32), view.text(), tokens)).collect())
    }
}
