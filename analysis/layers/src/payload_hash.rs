//! Deterministic payload fingerprints (`Layer::content_hash`) for the shared
//! segmentation and syntax layers.

use turbo_analysis::Fnv1a;

use crate::words::WordOrTokenAnchor;
use crate::{
    Block, Boundaries, ChunkTags, Chunks, Cut, Dependencies, Entities, PhraseChunks, Phrases, PosTags, Structure,
    Subwords, Terms, Tokens,
};

impl Block {
    pub(crate) const fn tag(self) -> u8 {
        self as u8
    }
}

impl Cut {
    pub(crate) fn mix_into(self, h: &mut Fnv1a) {
        match self {
            Cut::DocumentStart => h.u8(0),
            Cut::Sentence => h.u8(1),
            Cut::Section => h.u8(2),
            Cut::Heading => h.u8(3),
            Cut::Paragraph => h.u8(4),
            Cut::TopicShift => h.u8(5),
            Cut::Clause => h.u8(6),
            Cut::Item => h.u8(7),
            Cut::Inside(b) => h.u8(8).u8(b.tag()),
            Cut::Overlap(b) => h.u8(9).u8(b.tag()),
        };
    }
}

impl Structure {
    pub(crate) fn payload_hash(&self) -> u64 {
        let mut h = Fnv1a::new();
        h.usize(self.kind.len());
        for i in 0..self.kind.len() {
            h.u8(self.kind[i].tag()).u8(self.depth[i]);
            match self.parent[i] {
                None => h.u8(0),
                Some(p) => h.u8(1).u32(p.get() as u32),
            };
        }
        h.finish()
    }
}

impl Tokens {
    pub(crate) fn payload_hash(&self) -> u64 {
        let mut h = Fnv1a::new();
        match &self.sentence {
            None => {
                h.u8(0);
            }
            Some(s) => {
                h.u8(1).usize(s.len());
                for &ix in s {
                    h.u32(ix.get() as u32);
                }
            }
        }
        h.finish()
    }
}

impl Subwords {
    pub(crate) fn payload_hash(&self) -> u64 {
        let mut h = Fnv1a::new();
        h.str(&self.model.id).str(&self.model.revision).str(&self.model.sha256).usize(self.ids.len());
        for &id in &self.ids {
            h.i32(id);
        }
        match &self.token {
            None => {
                h.u8(0);
            }
            Some(toks) => {
                h.u8(1).usize(toks.len());
                for &t in toks {
                    h.u32(t.get() as u32);
                }
            }
        }
        h.finish()
    }
}

impl Boundaries {
    pub(crate) fn payload_hash(&self) -> u64 {
        let mut h = Fnv1a::new();
        h.usize(self.cost.len());
        for (&c, &r) in self.cost.iter().zip(&self.reason) {
            h.i32(c);
            r.mix_into(&mut h);
        }
        h.finish()
    }
}

impl Chunks {
    pub(crate) fn payload_hash(&self) -> u64 {
        let mut h = Fnv1a::new();
        h.u32(self.prompt_overhead).usize(self.level.len());
        for i in 0..self.level.len() {
            h.u8(self.level[i])
                .u32(self.sentences[i].start.get() as u32)
                .u32(self.sentences[i].end.get() as u32)
                .i64(self.cost[i])
                .u32(self.model_tokens[i])
                .bool(self.over_budget[i]);
            self.cut[i].mix_into(&mut h);
            match self.parent[i] {
                None => h.u8(0),
                Some(p) => h.u8(1).u32(p.get() as u32),
            };
        }
        h.finish()
    }
}

impl PosTags {
    pub(crate) fn payload_hash(&self) -> u64 {
        let mut h = Fnv1a::new();
        h.u64(self.tagset.content_hash()).usize(self.tags.len());
        for &t in &self.tags {
            h.u16(t.0);
        }
        match &self.prob {
            None => {
                h.u8(0);
            }
            Some(prob) => {
                h.u8(1);
                for &p in prob {
                    h.f64(p);
                }
            }
        }
        h.finish()
    }
}

impl Terms {
    pub(crate) fn payload_hash(&self) -> u64 {
        let mut h = Fnv1a::new();
        h.u64(self.surface.content_hash()).u64(self.folded.content_hash());
        for opt in [&self.stem, &self.lemma] {
            match opt {
                None => h.u8(0),
                Some(col) => h.u8(1).u64(col.content_hash()),
            };
        }
        h.finish()
    }
}

impl<A: WordOrTokenAnchor> Dependencies<A> {
    pub(crate) fn payload_hash(&self) -> u64 {
        let mut h = Fnv1a::new();
        h.u64(self.relations.content_hash()).usize(self.head.len());
        for (&hd, &rel) in self.head.iter().zip(&self.relation) {
            match hd {
                None => h.u8(0),
                Some(ix) => h.u8(1).u32(ix.get() as u32),
            };
            h.u16(rel.0);
        }
        h.finish()
    }
}

impl Entities {
    pub(crate) fn payload_hash(&self) -> u64 {
        let mut h = Fnv1a::new();
        h.u64(self.kinds.content_hash()).usize(self.tokens.len());
        for i in 0..self.tokens.len() {
            h.u32(self.tokens[i].start.get() as u32)
                .u32(self.tokens[i].end.get() as u32)
                .u16(self.kind[i].0)
                .f64(self.score[i]);
        }
        h.finish()
    }
}

impl ChunkTags {
    pub(crate) fn payload_hash(&self) -> u64 {
        let mut h = Fnv1a::new();
        h.u64(self.tagset.content_hash()).usize(self.tags.len());
        for &t in &self.tags {
            h.u16(t.0);
        }
        match &self.prob {
            None => {
                h.u8(0);
            }
            Some(prob) => {
                h.u8(1);
                for &p in prob {
                    h.f64(p);
                }
            }
        }
        h.finish()
    }
}

impl PhraseChunks {
    pub(crate) fn payload_hash(&self) -> u64 {
        let mut h = Fnv1a::new();
        h.u64(self.kinds.content_hash()).usize(self.tokens.len());
        for (r, &k) in self.tokens.iter().zip(&self.kind) {
            h.u32(r.start.get() as u32).u32(r.end.get() as u32).u16(k.0);
        }
        h.finish()
    }
}

impl Phrases {
    pub(crate) fn payload_hash(&self) -> u64 {
        let mut h = Fnv1a::new();
        h.u64(self.kinds.content_hash()).usize(self.tokens.len());
        for i in 0..self.tokens.len() {
            h.u32(self.tokens[i].start.get() as u32)
                .u32(self.tokens[i].end.get() as u32)
                .u16(self.kind[i].0)
                .u32(self.head[i].get() as u32);
            match self.parent[i] {
                None => h.u8(0),
                Some(p) => h.u8(1).u32(p.get() as u32),
            };
        }
        h.finish()
    }
}
