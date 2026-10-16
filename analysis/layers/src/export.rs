//! [`LayerCodec`]s for the shared layers, so a populated pipeline
//! document crosses the graph export boundary in core.
//!
//! [`register_codecs`] registers every codec in this module (plus the
//! abbreviation relations) on a [`Codecs`] set. Each codec writes its
//! layer's columns with the boundary's [`Writer`] and reads them back
//! with [`Reader`], refusing anything that does not hold together:
//! parallel columns must match the span or item count, every tag must
//! index the decoded tag set, every stored index into another layer
//! must resolve against the layers restored before it, and a parent
//! column must point strictly backwards, so no cycle or out-of-bounds
//! reference can be smuggled in through bytes. What a layer's own
//! `check` already proves is then re-proved by `Document::add` on
//! import; the codec checks are the ones that need the graph around
//! the layer.

use turbo_analysis::export::{
    Codecs, ExportCx, ExportError, ImportCx, LayerCodec, LayerRef, NodeRef, Reader, Writer, get_spans, get_tagset,
    put_spans, put_tagset,
};
use turbo_analysis::{Edges, Groups, Ix, IxRange, Key, Layer, ModelRef, Tag, TagSet, View};
use turbo_unicode::Script;

use crate::categories::{Categories, CategoryScope};
use crate::date::Date;
use crate::language_spans::LanguageSpans;
use crate::temporal::{DocumentDate, Granularity, Origin, Temporals};
use crate::words::{WordLemmas, WordOrTokenAnchor, WordPosTags, Words};
use crate::{
    Abbreviations, Block, Boundaries, ChunkTags, Chunks, Cut, Dependencies, Entities, Language, Lemmas, PhraseChunks,
    PosTags, Sentences, Stems, Strs, Structure, Subwords, Terms, Tokens,
};

fn malformed(detail: impl Into<String>) -> ExportError {
    ExportError::Malformed(detail.into())
}

fn want_len(got: usize, want: usize, what: &str) -> Result<(), ExportError> {
    if got != want {
        return Err(malformed(format!("{got} {what} for {want} items")));
    }
    Ok(())
}

fn done(r: &Reader<'_>, what: &str) -> Result<(), ExportError> {
    if r.is_done() { Ok(()) } else { Err(malformed(format!("trailing bytes in a {what} payload"))) }
}

fn put_strs(w: &mut Writer, strs: &Strs) -> Result<(), ExportError> {
    w.usize(strs.len())?;
    for s in strs.iter() {
        w.str(s)?;
    }
    Ok(())
}

fn get_strs(r: &mut Reader<'_>) -> Result<Strs, ExportError> {
    let mut strs = Strs::default();
    for _ in 0..r.usize()? {
        strs.push(&r.string()?);
    }
    Ok(strs)
}

fn put_opt_strs(w: &mut Writer, strs: &Option<Strs>) -> Result<(), ExportError> {
    match strs {
        None => w.u8(0),
        Some(s) => {
            w.u8(1);
            put_strs(w, s)?;
        }
    }
    Ok(())
}

fn get_opt_strs(r: &mut Reader<'_>) -> Result<Option<Strs>, ExportError> {
    Ok(match r.u8()? {
        0 => None,
        1 => Some(get_strs(r)?),
        tag => return Err(malformed(format!("an optional column tag of {tag}"))),
    })
}

fn put_tags(w: &mut Writer, tags: &[Tag]) -> Result<(), ExportError> {
    w.usize(tags.len())?;
    for t in tags {
        w.u16(t.0);
    }
    Ok(())
}

fn get_tags(r: &mut Reader<'_>, tagset: &TagSet) -> Result<Vec<Tag>, ExportError> {
    let mut tags = Vec::new();
    for _ in 0..r.usize()? {
        let tag = Tag(r.u16()?);
        if tag.0 as usize >= tagset.len() {
            return Err(malformed(format!("a tag outside {}", tagset.name)));
        }
        tags.push(tag);
    }
    Ok(tags)
}

fn put_f64s(w: &mut Writer, values: &[f64]) -> Result<(), ExportError> {
    w.usize(values.len())?;
    for v in values {
        w.f64(*v);
    }
    Ok(())
}

fn get_f64s(r: &mut Reader<'_>) -> Result<Vec<f64>, ExportError> {
    let mut values = Vec::new();
    for _ in 0..r.usize()? {
        values.push(r.f64()?);
    }
    Ok(values)
}

fn put_opt_f64s(w: &mut Writer, values: &Option<Vec<f64>>) -> Result<(), ExportError> {
    match values {
        None => w.u8(0),
        Some(v) => {
            w.u8(1);
            put_f64s(w, v)?;
        }
    }
    Ok(())
}

fn get_opt_f64s(r: &mut Reader<'_>) -> Result<Option<Vec<f64>>, ExportError> {
    Ok(match r.u8()? {
        0 => None,
        1 => Some(get_f64s(r)?),
        tag => return Err(malformed(format!("an optional column tag of {tag}"))),
    })
}

fn put_ref(w: &mut Writer, layer: &LayerRef) -> Result<(), ExportError> {
    w.str(&layer.layer)?;
    w.str(&layer.instance)
}

fn get_ref(r: &mut Reader<'_>) -> Result<LayerRef, ExportError> {
    Ok(LayerRef { layer: r.string()?, instance: r.string()? })
}

fn want_target<Target: Layer>(named: &LayerRef, cx: &ImportCx) -> Result<(), ExportError> {
    if named.layer != Target::NAME.to_string() {
        return Err(ExportError::InvalidLayer {
            layer: named.layer.clone(),
            detail: format!("an index column into {}, not {}", named.layer, Target::NAME),
        });
    }
    let _ = cx;
    Ok(())
}

/// An index column into another layer: the resolved target instance,
/// then the indexes. Import validates every index against the target
/// instance restored before this layer.
fn put_ix_col<Target: Layer>(
    w: &mut Writer,
    cx: &ExportCx,
    key: &Key<Target>,
    values: &[Ix<Target>],
) -> Result<(), ExportError> {
    put_ref(w, &cx.resolve(key))?;
    w.usize(values.len())?;
    for v in values {
        w.u32(v.get() as u32);
    }
    Ok(())
}

fn get_ix_col<Target: Layer>(r: &mut Reader<'_>, cx: &mut ImportCx) -> Result<Vec<Ix<Target>>, ExportError> {
    let target = get_ref(r)?;
    want_target::<Target>(&target, cx)?;
    let mut values = Vec::new();
    for _ in 0..r.usize()? {
        let index = r.u32()?;
        cx.require_node(&NodeRef { layer: target.clone(), index })?;
        values.push(Ix::new(index));
    }
    Ok(values)
}

fn put_opt_ix_col<Target: Layer>(
    w: &mut Writer,
    cx: &ExportCx,
    key: &Key<Target>,
    values: &Option<Vec<Ix<Target>>>,
) -> Result<(), ExportError> {
    match values {
        None => w.u8(0),
        Some(v) => {
            w.u8(1);
            put_ix_col(w, cx, key, v)?;
        }
    }
    Ok(())
}

fn get_opt_ix_col<Target: Layer>(
    r: &mut Reader<'_>,
    cx: &mut ImportCx,
) -> Result<Option<Vec<Ix<Target>>>, ExportError> {
    Ok(match r.u8()? {
        0 => None,
        1 => Some(get_ix_col(r, cx)?),
        tag => return Err(malformed(format!("an optional column tag of {tag}"))),
    })
}

/// A column of ranges into another layer: the resolved target
/// instance, then start/end pairs. Import requires each range to end
/// at or before the target's item count.
fn put_ranges<Target: Layer>(
    w: &mut Writer,
    cx: &ExportCx,
    key: &Key<Target>,
    values: &[IxRange<Target>],
) -> Result<(), ExportError> {
    put_ref(w, &cx.resolve(key))?;
    w.usize(values.len())?;
    for v in values {
        w.u32(v.start.get() as u32);
        w.u32(v.end.get() as u32);
    }
    Ok(())
}

fn get_ranges<Target: Layer>(r: &mut Reader<'_>, cx: &mut ImportCx) -> Result<Vec<IxRange<Target>>, ExportError> {
    let target = get_ref(r)?;
    want_target::<Target>(&target, cx)?;
    let len = cx.layer_len(&target)?;
    let mut values = Vec::new();
    for _ in 0..r.usize()? {
        let (start, end) = (r.u32()?, r.u32()?);
        if start > end || end as usize > len {
            return Err(ExportError::DanglingEndpoint {
                layer: target.layer.clone(),
                endpoint: NodeRef { layer: target.clone(), index: end },
            });
        }
        values.push(IxRange::new(start, end));
    }
    Ok(values)
}

/// A parent column into the layer itself. Import requires every
/// parent to sit strictly before its item, which makes cycles
/// unrepresentable.
fn put_parents<L>(w: &mut Writer, values: &[Option<Ix<L>>]) -> Result<(), ExportError> {
    w.usize(values.len())?;
    for v in values {
        match v {
            None => w.u8(0),
            Some(p) => {
                w.u8(1);
                w.u32(p.get() as u32);
            }
        }
    }
    Ok(())
}

fn get_parents<L>(r: &mut Reader<'_>) -> Result<Vec<Option<Ix<L>>>, ExportError> {
    let mut values = Vec::new();
    for i in 0..r.usize()? {
        values.push(match r.u8()? {
            0 => None,
            1 => {
                let p = r.u32()?;
                if p as usize >= i {
                    return Err(malformed(format!("item {i} has a parent at or after it")));
                }
                Some(Ix::new(p))
            }
            tag => return Err(malformed(format!("an optional index tag of {tag}"))),
        });
    }
    Ok(values)
}

macro_rules! spans_only {
    ($t:ty, $what:literal) => {
        impl LayerCodec for $t {
            fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
                let mut w = Writer::new();
                put_spans(&mut w, &self.spans, cx)?;
                Ok(w.finish())
            }
            fn import(data: &[u8], cx: &mut ImportCx) -> Result<Self, ExportError> {
                let mut r = Reader::new(data);
                let spans = get_spans(&mut r, cx)?;
                done(&r, $what)?;
                Ok(Self { spans })
            }
        }
    };
}

spans_only!(Sentences, "sentences");

impl LayerCodec for Tokens {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        put_opt_ix_col(&mut w, cx, &Key::<Sentences>::DEFAULT, &self.sentence)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let sentence = get_opt_ix_col(&mut r, cx)?;
        if let Some(s) = &sentence {
            want_len(s.len(), spans.len(), "sentence indexes")?;
        }
        done(&r, "tokens")?;
        Ok(Tokens { spans, sentence })
    }
}

fn put_model(w: &mut Writer, model: &ModelRef) -> Result<(), ExportError> {
    w.str(&model.id)?;
    w.str(&model.revision)?;
    w.str(&model.sha256)
}

fn get_model(r: &mut Reader<'_>) -> Result<ModelRef, ExportError> {
    Ok(ModelRef { id: r.string()?, revision: r.string()?, sha256: r.string()? })
}

impl LayerCodec for Subwords {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        w.usize(self.ids.len())?;
        for id in &self.ids {
            w.u32(*id as u32);
        }
        put_model(&mut w, &self.model)?;
        put_opt_ix_col(&mut w, cx, &Key::<Tokens>::DEFAULT, &self.token)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let mut ids = Vec::new();
        for _ in 0..r.usize()? {
            ids.push(r.u32()? as i32);
        }
        want_len(ids.len(), spans.len(), "subword ids")?;
        let model = get_model(&mut r)?;
        let token = get_opt_ix_col(&mut r, cx)?;
        if let Some(t) = &token {
            want_len(t.len(), spans.len(), "token indexes")?;
        }
        done(&r, "subwords")?;
        Ok(Subwords { spans, ids, model, token })
    }
}

macro_rules! tag_column {
    ($t:ty, $what:literal) => {
        impl LayerCodec for $t {
            fn export(&self, _cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
                let mut w = Writer::new();
                put_tagset(&mut w, &self.tagset)?;
                put_tags(&mut w, &self.tags)?;
                put_opt_f64s(&mut w, &self.prob)?;
                Ok(w.finish())
            }
            fn import(data: &[u8], _cx: &mut ImportCx) -> Result<Self, ExportError> {
                let mut r = Reader::new(data);
                let tagset = get_tagset(&mut r)?;
                let tags = get_tags(&mut r, &tagset)?;
                let prob = get_opt_f64s(&mut r)?;
                if let Some(p) = &prob {
                    want_len(p.len(), tags.len(), "probabilities")?;
                }
                done(&r, $what)?;
                Ok(Self { tagset, tags, prob })
            }
        }
    };
}

tag_column!(PosTags, "pos tags");
tag_column!(ChunkTags, "chunk tags");
tag_column!(WordPosTags, "word pos tags");

macro_rules! strs_column {
    ($t:ty, $field:ident, $what:literal) => {
        impl LayerCodec for $t {
            fn export(&self, _cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
                let mut w = Writer::new();
                put_strs(&mut w, &self.$field)?;
                Ok(w.finish())
            }
            fn import(data: &[u8], _cx: &mut ImportCx) -> Result<Self, ExportError> {
                let mut r = Reader::new(data);
                let $field = get_strs(&mut r)?;
                done(&r, $what)?;
                Ok(Self { $field })
            }
        }
    };
}

strs_column!(Lemmas, lemma, "lemmas");
strs_column!(Stems, stem, "stems");
strs_column!(WordLemmas, lemma, "word lemmas");

impl LayerCodec for Terms {
    fn export(&self, _cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_strs(&mut w, &self.surface)?;
        put_strs(&mut w, &self.folded)?;
        put_opt_strs(&mut w, &self.stem)?;
        put_opt_strs(&mut w, &self.lemma)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], _cx: &mut ImportCx) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let surface = get_strs(&mut r)?;
        let folded = get_strs(&mut r)?;
        let stem = get_opt_strs(&mut r)?;
        let lemma = get_opt_strs(&mut r)?;
        want_len(folded.len(), surface.len(), "folded forms")?;
        if let Some(s) = &stem {
            want_len(s.len(), surface.len(), "stems")?;
        }
        if let Some(l) = &lemma {
            want_len(l.len(), surface.len(), "lemmas")?;
        }
        done(&r, "terms")?;
        Ok(Terms { surface, folded, stem, lemma })
    }
}

impl<A: WordOrTokenAnchor> LayerCodec for Dependencies<A> {
    fn export(&self, _cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_tagset(&mut w, &self.relations)?;
        w.usize(self.head.len())?;
        for h in &self.head {
            match h {
                None => w.u8(0),
                Some(h) => {
                    w.u8(1);
                    w.u32(h.get() as u32);
                }
            }
        }
        put_tags(&mut w, &self.relation)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], _cx: &mut ImportCx) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let relations = get_tagset(&mut r)?;
        let n = r.usize()?;
        let mut head = Vec::with_capacity(n);
        for i in 0..n {
            head.push(match r.u8()? {
                0 => None,
                1 => {
                    let h = r.u32()?;
                    if h as usize >= n {
                        return Err(malformed(format!("dependency head {h} at {i} is out of bounds ({n})")));
                    }
                    if h as usize == i {
                        return Err(malformed(format!("dependency item {i} is its own head")));
                    }
                    Some(Ix::new(h))
                }
                tag => return Err(malformed(format!("an optional head tag of {tag}"))),
            });
        }
        // Heads must reach a root: walking each chain with a visited
        // mark refuses a cycle the layer's own check would let through.
        let mut state = vec![0u8; n];
        for start in 0..n {
            let mut at = start;
            let mut chain = Vec::new();
            while state[at] == 0 {
                state[at] = 1;
                chain.push(at);
                let Some(h) = head[at] else { break };
                at = h.get();
                if state[at] == 1 && chain.contains(&at) {
                    return Err(malformed(format!("dependency heads cycle at item {at}")));
                }
            }
            for c in chain {
                state[c] = 2;
            }
        }
        let relation = get_tags(&mut r, &relations)?;
        want_len(relation.len(), n, "relations")?;
        done(&r, "dependencies")?;
        Ok(Dependencies { head, relation, relations })
    }
}

impl LayerCodec for Entities {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        put_ranges(&mut w, cx, &Key::<Tokens>::DEFAULT, &self.tokens)?;
        put_tagset(&mut w, &self.kinds)?;
        put_tags(&mut w, &self.kind)?;
        put_f64s(&mut w, &self.score)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let tokens = get_ranges(&mut r, cx)?;
        let kinds = get_tagset(&mut r)?;
        let kind = get_tags(&mut r, &kinds)?;
        let score = get_f64s(&mut r)?;
        want_len(tokens.len(), spans.len(), "token ranges")?;
        want_len(kind.len(), spans.len(), "kinds")?;
        want_len(score.len(), spans.len(), "scores")?;
        done(&r, "entities")?;
        Ok(Entities { spans, tokens, kind, kinds, score })
    }
}

impl LayerCodec for PhraseChunks {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        put_ranges(&mut w, cx, &Key::<Tokens>::DEFAULT, &self.tokens)?;
        put_tagset(&mut w, &self.kinds)?;
        put_tags(&mut w, &self.kind)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let tokens = get_ranges(&mut r, cx)?;
        let kinds = get_tagset(&mut r)?;
        let kind = get_tags(&mut r, &kinds)?;
        want_len(tokens.len(), spans.len(), "token ranges")?;
        want_len(kind.len(), spans.len(), "kinds")?;
        done(&r, "phrase chunks")?;
        Ok(PhraseChunks { spans, tokens, kind, kinds })
    }
}

const BLOCKS: [Block; 14] = [
    Block::Section,
    Block::Heading,
    Block::Paragraph,
    Block::List,
    Block::ListItem,
    Block::Table,
    Block::TableRow,
    Block::TableCell,
    Block::Code,
    Block::Quote,
    Block::Caption,
    Block::Footnote,
    Block::PageHeader,
    Block::PageFooter,
];

const CUTS: [(&str, Cut); 7] = [
    ("document", Cut::DocumentStart),
    ("sentence", Cut::Sentence),
    ("section", Cut::Section),
    ("heading", Cut::Heading),
    ("paragraph", Cut::Paragraph),
    ("topic", Cut::TopicShift),
    ("clause", Cut::Clause),
];

fn put_cut(w: &mut Writer, cut: Cut) -> Result<(), ExportError> {
    match cut {
        Cut::Inside(block) => {
            w.u8(8);
            w.u8(BLOCKS.iter().position(|b| *b == block).expect("a block in the table") as u8);
        }
        Cut::Overlap(block) => {
            w.u8(9);
            w.u8(BLOCKS.iter().position(|b| *b == block).expect("a block in the table") as u8);
        }
        Cut::Item => w.u8(7),
        other => {
            let at = CUTS.iter().position(|(_, c)| *c == other).expect("a cut in the table");
            w.u8(at as u8);
        }
    }
    Ok(())
}

fn get_cut(r: &mut Reader<'_>) -> Result<Cut, ExportError> {
    let block = |r: &mut Reader<'_>| -> Result<Block, ExportError> {
        BLOCKS.get(r.u8()? as usize).copied().ok_or_else(|| malformed("a block outside the block table"))
    };
    Ok(match r.u8()? {
        7 => Cut::Item,
        8 => Cut::Inside(block(r)?),
        9 => Cut::Overlap(block(r)?),
        tag => CUTS.get(tag as usize).map(|(_, c)| *c).ok_or_else(|| malformed(format!("a cut tag of {tag}")))?,
    })
}

impl LayerCodec for Boundaries {
    fn export(&self, _cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        w.usize(self.cost.len())?;
        for c in &self.cost {
            w.u32(*c as u32);
        }
        for reason in &self.reason {
            put_cut(&mut w, *reason)?;
        }
        Ok(w.finish())
    }
    fn import(data: &[u8], _cx: &mut ImportCx) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let n = r.usize()?;
        let mut cost = Vec::with_capacity(n);
        for _ in 0..n {
            cost.push(r.u32()? as i32);
        }
        let mut reason = Vec::with_capacity(n);
        for _ in 0..n {
            reason.push(get_cut(&mut r)?);
        }
        done(&r, "boundaries")?;
        Ok(Boundaries { cost, reason })
    }
}

impl LayerCodec for Structure {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        w.usize(self.kind.len())?;
        for k in &self.kind {
            w.u8(BLOCKS.iter().position(|b| b == k).expect("a block in the table") as u8);
        }
        put_parents(&mut w, &self.parent)?;
        w.usize(self.depth.len())?;
        for d in &self.depth {
            w.u8(*d);
        }
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let mut kind = Vec::new();
        for _ in 0..r.usize()? {
            kind.push(*BLOCKS.get(r.u8()? as usize).ok_or_else(|| malformed("a block outside the block table"))?);
        }
        let parent = get_parents(&mut r)?;
        let mut depth = Vec::new();
        for _ in 0..r.usize()? {
            depth.push(r.u8()?);
        }
        want_len(kind.len(), spans.len(), "kinds")?;
        want_len(parent.len(), spans.len(), "parents")?;
        want_len(depth.len(), spans.len(), "depths")?;
        done(&r, "structure")?;
        Ok(Structure { spans, kind, parent, depth })
    }
}

impl LayerCodec for Chunks {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        w.usize(self.level.len())?;
        for l in &self.level {
            w.u8(*l);
        }
        put_parents(&mut w, &self.parent)?;
        put_ranges(&mut w, cx, &Key::<Sentences>::DEFAULT, &self.sentences)?;
        for cut in &self.cut {
            put_cut(&mut w, *cut)?;
        }
        w.usize(self.cost.len())?;
        for c in &self.cost {
            w.u64(*c as u64);
        }
        w.usize(self.model_tokens.len())?;
        for t in &self.model_tokens {
            w.u32(*t);
        }
        w.usize(self.over_budget.len())?;
        for b in &self.over_budget {
            w.u8(u8::from(*b));
        }
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let n = spans.len();
        let mut level = Vec::new();
        for _ in 0..r.usize()? {
            level.push(r.u8()?);
        }
        let parent = get_parents(&mut r)?;
        let sentences = get_ranges(&mut r, cx)?;
        let mut cut = Vec::new();
        for _ in 0..n {
            cut.push(get_cut(&mut r)?);
        }
        let mut cost = Vec::new();
        for _ in 0..r.usize()? {
            cost.push(r.u64()? as i64);
        }
        let mut model_tokens = Vec::new();
        for _ in 0..r.usize()? {
            model_tokens.push(r.u32()?);
        }
        let mut over_budget = Vec::new();
        for _ in 0..r.usize()? {
            over_budget.push(match r.u8()? {
                0 => false,
                1 => true,
                tag => return Err(malformed(format!("a boolean tag of {tag}"))),
            });
        }
        want_len(level.len(), n, "levels")?;
        want_len(parent.len(), n, "parents")?;
        want_len(sentences.len(), n, "sentence ranges")?;
        want_len(cost.len(), n, "costs")?;
        want_len(model_tokens.len(), n, "model token counts")?;
        want_len(over_budget.len(), n, "over-budget flags")?;
        done(&r, "chunks")?;
        Ok(Chunks { spans, level, parent, sentences, cut, cost, model_tokens, over_budget })
    }
}

impl LayerCodec for Language {
    fn export(&self, _cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_strs(&mut w, &self.code)?;
        put_f64s(&mut w, &self.score)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], _cx: &mut ImportCx) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let code = get_strs(&mut r)?;
        let score = get_f64s(&mut r)?;
        want_len(score.len(), code.len(), "scores")?;
        done(&r, "language")?;
        Ok(Language { code, score })
    }
}

impl LayerCodec for Words {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_ranges(&mut w, cx, &self.tokens_key, &self.tokens)?;
        match &self.form {
            None => w.u8(0),
            Some(forms) => {
                w.u8(1);
                w.usize(forms.len())?;
                for f in forms {
                    match f {
                        None => w.u8(0),
                        Some(f) => {
                            w.u8(1);
                            w.str(f)?;
                        }
                    }
                }
            }
        }
        match &self.subspan {
            None => w.u8(0),
            Some(spans) => {
                w.u8(1);
                w.usize(spans.len())?;
                for s in spans {
                    match s {
                        None => w.u8(0),
                        Some(s) => {
                            w.u8(1);
                            w.span(*s);
                        }
                    }
                }
            }
        }
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        // The token ranges carry their target ref; read it through a
        // reader that also yields the key to restore.
        let target = get_ref(&mut r)?;
        want_target::<Tokens>(&target, cx)?;
        let len = cx.layer_len(&target)?;
        let mut tokens = Vec::new();
        for _ in 0..r.usize()? {
            let (start, end) = (r.u32()?, r.u32()?);
            if start > end || end as usize > len {
                return Err(ExportError::DanglingEndpoint {
                    layer: target.layer.clone(),
                    endpoint: NodeRef { layer: target.clone(), index: end },
                });
            }
            tokens.push(IxRange::new(start, end));
        }
        let tokens_key = cx.key::<Tokens>(&target.instance.clone());
        let n = tokens.len();
        let form = match r.u8()? {
            0 => None,
            1 => {
                let mut forms = Vec::new();
                for _ in 0..r.usize()? {
                    forms.push(match r.u8()? {
                        0 => None,
                        1 => Some(r.string()?.into_boxed_str()),
                        tag => return Err(malformed(format!("an optional form tag of {tag}"))),
                    });
                }
                want_len(forms.len(), n, "word forms")?;
                Some(forms)
            }
            tag => return Err(malformed(format!("an optional column tag of {tag}"))),
        };
        let subspan = match r.u8()? {
            0 => None,
            1 => {
                let mut spans = Vec::new();
                for _ in 0..r.usize()? {
                    spans.push(match r.u8()? {
                        0 => None,
                        1 => Some(r.span()?),
                        tag => return Err(malformed(format!("an optional span tag of {tag}"))),
                    });
                }
                want_len(spans.len(), n, "subspans")?;
                Some(spans)
            }
            tag => return Err(malformed(format!("an optional column tag of {tag}"))),
        };
        done(&r, "words")?;
        Ok(Words { tokens, form, subspan, tokens_key })
    }
}

impl<L: CategoryScope> LayerCodec for Categories<L> {
    fn export(&self, _cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_tagset(&mut w, &self.categories)?;
        put_tags(&mut w, &self.category)?;
        put_f64s(&mut w, &self.scores)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], _cx: &mut ImportCx) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let categories = get_tagset(&mut r)?;
        let category = get_tags(&mut r, &categories)?;
        let scores = get_f64s(&mut r)?;
        let want =
            category.len().checked_mul(categories.len()).ok_or_else(|| malformed("a score count that overflows"))?;
        want_len(scores.len(), want, "scores")?;
        done(&r, "categories")?;
        let mut out = Categories::with_capacity(categories, category.len());
        let k = out.categories.len();
        for (i, best) in category.into_iter().enumerate() {
            out.push(best, &scores[i * k..(i + 1) * k]);
        }
        Ok(out)
    }
}

const GRANULARITIES: [Granularity; 5] =
    [Granularity::Day, Granularity::Week, Granularity::Month, Granularity::Quarter, Granularity::Year];

impl LayerCodec for Temporals {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        w.usize(self.granularity.len())?;
        for g in &self.granularity {
            w.u8(GRANULARITIES.iter().position(|x| x == g).expect("a granularity in the table") as u8);
        }
        w.usize(self.origin.len())?;
        for o in &self.origin {
            w.u8(match o {
                Origin::Absolute => 0,
                Origin::Relative => 1,
            });
        }
        w.usize(self.year.len())?;
        for y in &self.year {
            w.u32(*y as u32);
        }
        w.usize(self.period.len())?;
        for p in &self.period {
            w.u8(*p);
        }
        w.usize(self.day.len())?;
        for d in &self.day {
            w.u8(*d);
        }
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let n = spans.len();
        let mut granularity = Vec::new();
        for _ in 0..r.usize()? {
            granularity.push(
                *GRANULARITIES.get(r.u8()? as usize).ok_or_else(|| malformed("a granularity outside the table"))?,
            );
        }
        let mut origin = Vec::new();
        for _ in 0..r.usize()? {
            origin.push(match r.u8()? {
                0 => Origin::Absolute,
                1 => Origin::Relative,
                tag => return Err(malformed(format!("an origin tag of {tag}"))),
            });
        }
        let mut year = Vec::new();
        for _ in 0..r.usize()? {
            year.push(r.u32()? as i32);
        }
        let mut period = Vec::new();
        for _ in 0..r.usize()? {
            period.push(r.u8()?);
        }
        let mut day = Vec::new();
        for _ in 0..r.usize()? {
            day.push(r.u8()?);
        }
        want_len(granularity.len(), n, "granularities")?;
        want_len(origin.len(), n, "origins")?;
        want_len(year.len(), n, "years")?;
        want_len(period.len(), n, "periods")?;
        want_len(day.len(), n, "days")?;
        done(&r, "temporals")?;
        Ok(Temporals { spans, granularity, origin, year, period, day })
    }
}

fn put_date(w: &mut Writer, date: Date) {
    w.u32(date.year() as u32);
    w.u8(date.month());
    w.u8(date.day());
}

fn get_date(r: &mut Reader<'_>) -> Result<Date, ExportError> {
    let (year, month, day) = (r.u32()? as i32, r.u8()?, r.u8()?);
    Date::new(year, month, day).ok_or_else(|| malformed(format!("a date that does not exist: {year}-{month}-{day}")))
}

impl LayerCodec for DocumentDate {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        match self.date {
            None => w.u8(0),
            Some(date) => {
                w.u8(1);
                put_date(&mut w, date);
            }
        }
        match self.mention {
            None => w.u8(0),
            Some(m) => {
                w.u8(1);
                put_ref(&mut w, &cx.resolve(&Key::<Temporals>::DEFAULT))?;
                w.u32(m.get() as u32);
            }
        }
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let date = match r.u8()? {
            0 => None,
            1 => Some(get_date(&mut r)?),
            tag => return Err(malformed(format!("an optional date tag of {tag}"))),
        };
        let mention = match r.u8()? {
            0 => None,
            1 => {
                let target = get_ref(&mut r)?;
                want_target::<Temporals>(&target, cx)?;
                let index = r.u32()?;
                cx.require_node(&NodeRef { layer: target, index })?;
                Some(Ix::new(index))
            }
            tag => return Err(malformed(format!("an optional mention tag of {tag}"))),
        };
        done(&r, "document date")?;
        Ok(DocumentDate { date, mention })
    }
}

impl LayerCodec for LanguageSpans {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        put_tagset(&mut w, &self.languages)?;
        put_tags(&mut w, &self.language)?;
        w.usize(self.script.len())?;
        for s in &self.script {
            w.str(s.name())?;
        }
        put_f64s(&mut w, &self.confidence)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let languages = get_tagset(&mut r)?;
        let language = get_tags(&mut r, &languages)?;
        let mut script = Vec::new();
        for _ in 0..r.usize()? {
            let name = r.string()?;
            script.push(Script::from_name(&name).ok_or_else(|| malformed(format!("an unknown script {name:?}")))?);
        }
        let confidence = get_f64s(&mut r)?;
        want_len(language.len(), spans.len(), "languages")?;
        want_len(script.len(), spans.len(), "scripts")?;
        want_len(confidence.len(), spans.len(), "confidences")?;
        done(&r, "language spans")?;
        Ok(LanguageSpans { spans, language, languages, script, confidence })
    }
}

/// Registers the codecs for the shared layers on `codecs`: the core
/// pipeline layers of this crate and the abbreviation relations over
/// them. Layers without a codec here (the sentiment and numeric
/// families, vectors, and each add-on's own layers) still refuse to
/// cross until their crates register one.
pub fn register_codecs(codecs: &mut Codecs) {
    codecs
        .register::<Sentences>()
        .register::<Tokens>()
        .register::<Words>()
        .register::<Subwords>()
        .register::<PosTags>()
        .register::<WordPosTags>()
        .register::<Lemmas>()
        .register::<WordLemmas>()
        .register::<Stems>()
        .register::<Terms>()
        .register::<Dependencies<Tokens>>()
        .register::<Dependencies<Words>>()
        .register::<Entities>()
        .register::<ChunkTags>()
        .register::<PhraseChunks>()
        .register::<Boundaries>()
        .register::<Structure>()
        .register::<Chunks>()
        .register::<Language>()
        .register::<LanguageSpans>()
        .register::<Temporals>()
        .register::<DocumentDate>()
        .register::<Categories<View>>()
        .register::<Categories<Sentences>>()
        .register::<Categories<Structure>>()
        .register::<Categories<Chunks>>()
        .register::<Abbreviations>()
        .register::<Edges<Abbreviations, Abbreviations>>()
        .register::<Groups<Abbreviations>>();
}

/// A [`Codecs`] set with the shared layers registered: see
/// [`register_codecs`].
pub fn codecs() -> Codecs {
    let mut codecs = Codecs::new();
    register_codecs(&mut codecs);
    codecs
}
