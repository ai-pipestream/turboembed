//! How the shared layers are carried across an edit
//! (`turbo_analysis::Splice`): spans move with their partition, indexes
//! move with the layers they point into, payload columns are copied,
//! and a tag set must be the same on both sides.

use std::ops::Range;
use std::sync::Arc;

use turbo_analysis::{Splice, SpliceCx, TagSet, ViewId};

use crate::{Dependencies, Entities, Lemmas, PosTags, Sentences, Strs, Structure, Tokens, WordOrTokenAnchor};

fn same(a: &Arc<TagSet>, b: &Arc<TagSet>, what: &str) -> Result<(), String> {
    if Arc::ptr_eq(a, b) || a.content_hash() == b.content_hash() {
        Ok(())
    } else {
        Err(format!("the {what} tag sets differ: {} and {}", a.name, b.name))
    }
}

impl Splice for Sentences {
    fn empty_like(_like: &Sentences, view: ViewId) -> Sentences {
        Sentences { spans: turbo_analysis::Spans::new(view) }
    }
    fn append(&mut self, from: &Sentences, range: Range<usize>, cx: &mut SpliceCx<'_>) -> Result<(), String> {
        cx.spans(&mut self.spans, &from.spans, range)
    }
}

impl Splice for Structure {
    fn empty_like(_like: &Structure, view: ViewId) -> Structure {
        Structure { spans: turbo_analysis::Spans::new(view), kind: Vec::new(), parent: Vec::new(), depth: Vec::new() }
    }
    fn append(&mut self, from: &Structure, range: Range<usize>, cx: &mut SpliceCx<'_>) -> Result<(), String> {
        cx.spans(&mut self.spans, &from.spans, range.clone())?;
        self.kind.extend_from_slice(&from.kind[range.clone()]);
        self.depth.extend_from_slice(&from.depth[range.clone()]);
        for &p in &from.parent[range] {
            self.parent.push(p.map(|p| cx.ix(p)).transpose()?);
        }
        Ok(())
    }
}

impl Splice for Tokens {
    fn empty_like(like: &Tokens, view: ViewId) -> Tokens {
        Tokens { spans: turbo_analysis::Spans::new(view), sentence: like.sentence.as_ref().map(|_| Vec::new()) }
    }
    fn append(&mut self, from: &Tokens, range: Range<usize>, cx: &mut SpliceCx<'_>) -> Result<(), String> {
        match (&mut self.sentence, &from.sentence) {
            (None, None) => {}
            (Some(out), Some(col)) => {
                for &s in &col[range.clone()] {
                    out.push(cx.ix(s)?);
                }
            }
            _ => return Err("tokens with a sentence column and tokens without".into()),
        }
        cx.spans(&mut self.spans, &from.spans, range)
    }
}

impl Splice for PosTags {
    fn empty_like(like: &PosTags, _view: ViewId) -> PosTags {
        PosTags { tagset: like.tagset.clone(), tags: Vec::new(), prob: like.prob.as_ref().map(|_| Vec::new()) }
    }
    fn append(&mut self, from: &PosTags, range: Range<usize>, _cx: &mut SpliceCx<'_>) -> Result<(), String> {
        same(&self.tagset, &from.tagset, "POS")?;
        match (&mut self.prob, &from.prob) {
            (None, None) => {}
            (Some(out), Some(p)) => out.extend_from_slice(&p[range.clone()]),
            _ => return Err("tags with probabilities and tags without".into()),
        }
        self.tags.extend_from_slice(&from.tags[range]);
        Ok(())
    }
}

impl Splice for Lemmas {
    fn empty_like(_like: &Lemmas, _view: ViewId) -> Lemmas {
        Lemmas { lemma: Strs::default() }
    }
    fn append(&mut self, from: &Lemmas, range: Range<usize>, _cx: &mut SpliceCx<'_>) -> Result<(), String> {
        for i in range {
            self.lemma.push(from.lemma.get(i));
        }
        Ok(())
    }
}

impl<A: WordOrTokenAnchor> Splice for Dependencies<A> {
    fn empty_like(like: &Dependencies<A>, _view: ViewId) -> Dependencies<A> {
        Dependencies { head: Vec::new(), relation: Vec::new(), relations: like.relations.clone() }
    }
    fn append(&mut self, from: &Dependencies<A>, range: Range<usize>, cx: &mut SpliceCx<'_>) -> Result<(), String> {
        same(&self.relations, &from.relations, "relation")?;
        for &h in &from.head[range.clone()] {
            self.head.push(h.map(|h| cx.ix(h)).transpose()?);
        }
        self.relation.extend_from_slice(&from.relation[range]);
        Ok(())
    }
}

impl Splice for Entities {
    fn empty_like(like: &Entities, view: ViewId) -> Entities {
        Entities {
            spans: turbo_analysis::Spans::new(view),
            tokens: Vec::new(),
            kind: Vec::new(),
            kinds: like.kinds.clone(),
            score: Vec::new(),
        }
    }
    fn append(&mut self, from: &Entities, range: Range<usize>, cx: &mut SpliceCx<'_>) -> Result<(), String> {
        same(&self.kinds, &from.kinds, "entity kind")?;
        for &r in &from.tokens[range.clone()] {
            self.tokens.push(cx.ix_range(r)?);
        }
        self.kind.extend_from_slice(&from.kind[range.clone()]);
        self.score.extend_from_slice(&from.score[range.clone()]);
        cx.spans(&mut self.spans, &from.spans, range)
    }
}
