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
    Abbreviations, Aspects, Block, Boundaries, ChunkTags, Chunks, Confidence, Containment, Currency, Cut, Dependencies,
    Entities, Incivility, IncivilityLabel, IncivilityTarget, Incongruity, Language, Lemmas, Levels, Matches, Modifier,
    Money, Multiplier, NumberNotation, Numbers, Opinions, OrdinalBase, Ordinals, PhraseChunks, Phrases, Pii,
    PiiConfidence, PiiKind, Portion, PosTags, Pragmatics, Quantities, RegionCode, Regions, Relation, ScaleMarker,
    Sentences, Stems, Strs, Structure, Subwords, Terms, TimeGrain, TimeKind, Times, Tokens, Unit, Utf16Spans, Vectors,
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

fn put_bools(w: &mut Writer, values: &[bool]) -> Result<(), ExportError> {
    w.usize(values.len())?;
    for v in values {
        w.u8(u8::from(*v));
    }
    Ok(())
}

fn get_bools(r: &mut Reader<'_>) -> Result<Vec<bool>, ExportError> {
    let mut values = Vec::new();
    for _ in 0..r.usize()? {
        values.push(match r.u8()? {
            0 => false,
            1 => true,
            tag => return Err(malformed(format!("a boolean tag of {tag}"))),
        });
    }
    Ok(values)
}

fn put_u32s(w: &mut Writer, values: &[u32]) -> Result<(), ExportError> {
    w.usize(values.len())?;
    for v in values {
        w.u32(*v);
    }
    Ok(())
}

fn get_u32s(r: &mut Reader<'_>) -> Result<Vec<u32>, ExportError> {
    let mut values = Vec::new();
    for _ in 0..r.usize()? {
        values.push(r.u32()?);
    }
    Ok(values)
}

fn put_i32s(w: &mut Writer, values: &[i32]) -> Result<(), ExportError> {
    w.usize(values.len())?;
    for v in values {
        w.u32(*v as u32);
    }
    Ok(())
}

fn get_i32s(r: &mut Reader<'_>) -> Result<Vec<i32>, ExportError> {
    let mut values = Vec::new();
    for _ in 0..r.usize()? {
        values.push(r.u32()? as i32);
    }
    Ok(values)
}

fn put_i64s(w: &mut Writer, values: &[i64]) -> Result<(), ExportError> {
    w.usize(values.len())?;
    for v in values {
        w.u64(*v as u64);
    }
    Ok(())
}

fn get_i64s(r: &mut Reader<'_>) -> Result<Vec<i64>, ExportError> {
    let mut values = Vec::new();
    for _ in 0..r.usize()? {
        values.push(r.u64()? as i64);
    }
    Ok(values)
}

fn put_opt_i64s(w: &mut Writer, values: &[Option<i64>]) -> Result<(), ExportError> {
    w.usize(values.len())?;
    for v in values {
        match v {
            None => w.u8(0),
            Some(v) => {
                w.u8(1);
                w.u64(*v as u64);
            }
        }
    }
    Ok(())
}

fn get_opt_i64s(r: &mut Reader<'_>) -> Result<Vec<Option<i64>>, ExportError> {
    let mut values = Vec::new();
    for _ in 0..r.usize()? {
        values.push(match r.u8()? {
            0 => None,
            1 => Some(r.u64()? as i64),
            tag => return Err(malformed(format!("an optional number tag of {tag}"))),
        });
    }
    Ok(values)
}

/// A fieldless enum as its position in `table`, the declared order.
fn put_variant<E: Copy + PartialEq>(w: &mut Writer, table: &[E], value: E) {
    w.u8(table.iter().position(|v| *v == value).expect("a variant in the table") as u8);
}

fn get_variant<E: Copy>(r: &mut Reader<'_>, table: &[E], what: &str) -> Result<E, ExportError> {
    table.get(r.u8()? as usize).copied().ok_or_else(|| malformed(format!("a {what} outside the table")))
}

fn put_variants<E: Copy + PartialEq>(w: &mut Writer, table: &[E], values: &[E]) -> Result<(), ExportError> {
    w.usize(values.len())?;
    for v in values {
        put_variant(w, table, *v);
    }
    Ok(())
}

fn get_variants<E: Copy>(r: &mut Reader<'_>, table: &[E], what: &str) -> Result<Vec<E>, ExportError> {
    let mut values = Vec::new();
    for _ in 0..r.usize()? {
        values.push(get_variant(r, table, what)?);
    }
    Ok(values)
}

/// Ranges into another layer plus the target instance to restore as a
/// key, for layers that carry their target key in the layer itself.
fn put_keyed_ranges<Target: Layer>(
    w: &mut Writer,
    cx: &ExportCx,
    key: &Key<Target>,
    values: &[IxRange<Target>],
) -> Result<(), ExportError> {
    put_ranges(w, cx, key, values)
}

fn get_keyed_ranges<Target: Layer>(
    r: &mut Reader<'_>,
    cx: &mut ImportCx,
) -> Result<(Key<Target>, Vec<IxRange<Target>>), ExportError> {
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
    Ok((cx.key::<Target>(&target.instance.clone()), values))
}

fn validate_range<Target: Layer>(cx: &ImportCx, target: &LayerRef, range: IxRange<Target>) -> Result<(), ExportError> {
    let len = cx.layer_len(target)?;
    if range.start > range.end || range.end.get() > len {
        return Err(ExportError::DanglingEndpoint {
            layer: target.layer.clone(),
            endpoint: NodeRef { layer: target.clone(), index: range.end.get() as u32 },
        });
    }
    Ok(())
}

impl<L: Layer> LayerCodec for Vectors<L> {
    fn export(&self, _cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        w.u32(self.dim);
        w.usize(self.len())?;
        w.usize(self.data.len())?;
        for v in &self.data {
            w.u32(v.to_bits());
        }
        w.usize(self.empty.len())?;
        for e in &self.empty {
            w.u32(e.get() as u32);
        }
        put_model(&mut w, &self.model)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], _cx: &mut ImportCx) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let dim = r.u32()?;
        if dim == 0 {
            return Err(malformed("a vector dimension of zero"));
        }
        let rows = r.usize()?;
        let want = rows.checked_mul(dim as usize).ok_or_else(|| malformed("a vector block that overflows"))?;
        let count = r.usize()?;
        if count != want {
            return Err(malformed(format!("{count} vector values for {rows} rows of dimension {dim}")));
        }
        let mut data = Vec::with_capacity(count);
        for _ in 0..count {
            data.push(f32::from_bits(r.u32()?));
        }
        let mut empty = Vec::new();
        for _ in 0..r.usize()? {
            let e = r.u32()?;
            if e as usize >= rows {
                return Err(malformed(format!("an empty row {e} outside {rows} rows")));
            }
            if empty.last().is_some_and(|&last: &Ix<L>| last.get() as u32 >= e) {
                return Err(malformed("empty rows out of order or repeated"));
            }
            empty.push(Ix::new(e));
        }
        let model = get_model(&mut r)?;
        done(&r, "vectors")?;
        Ok(Vectors::new(dim, data, empty, model))
    }
}

impl<L: Layer> LayerCodec for Confidence<L> {
    fn export(&self, _cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_f64s(&mut w, &self.value)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], _cx: &mut ImportCx) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let value = get_f64s(&mut r)?;
        done(&r, "confidence")?;
        Ok(Confidence::new(value))
    }
}

impl<L: Layer> LayerCodec for Utf16Spans<L> {
    fn export(&self, _cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_u32s(&mut w, &self.start16)?;
        put_u32s(&mut w, &self.end16)?;
        put_bools(&mut w, &self.cut)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], _cx: &mut ImportCx) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let start16 = get_u32s(&mut r)?;
        let end16 = get_u32s(&mut r)?;
        let cut = get_bools(&mut r)?;
        want_len(end16.len(), start16.len(), "UTF-16 ends")?;
        want_len(cut.len(), start16.len(), "cut flags")?;
        done(&r, "utf16 spans")?;
        Ok(Utf16Spans::new(start16, end16, cut))
    }
}

impl LayerCodec for Opinions {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_keyed_ranges(&mut w, cx, &self.tokens_key, &self.tokens)?;
        put_f64s(&mut w, &self.polarity)?;
        put_f64s(&mut w, &self.intensity)?;
        put_bools(&mut w, &self.negated)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let (tokens_key, tokens) = get_keyed_ranges(&mut r, cx)?;
        let polarity = get_f64s(&mut r)?;
        let intensity = get_f64s(&mut r)?;
        let negated = get_bools(&mut r)?;
        want_len(polarity.len(), tokens.len(), "polarities")?;
        want_len(intensity.len(), tokens.len(), "intensities")?;
        want_len(negated.len(), tokens.len(), "negation flags")?;
        done(&r, "opinions")?;
        Ok(Opinions { tokens, polarity, intensity, negated, tokens_key })
    }
}

impl LayerCodec for Aspects {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_ref(&mut w, &cx.resolve(&self.tokens_key))?;
        put_ref(&mut w, &cx.resolve(&self.sentences_key))?;
        put_ref(&mut w, &cx.resolve(&self.opinions_key))?;
        w.usize(self.tokens.len())?;
        for t in &self.tokens {
            match t {
                None => w.u8(0),
                Some(range) => {
                    w.u8(1);
                    w.u32(range.start.get() as u32);
                    w.u32(range.end.get() as u32);
                }
            }
        }
        w.usize(self.sentence.len())?;
        for s in &self.sentence {
            match s {
                None => w.u8(0),
                Some(s) => {
                    w.u8(1);
                    w.u32(s.get() as u32);
                }
            }
        }
        put_tagset(&mut w, &self.categories)?;
        w.usize(self.category.len())?;
        for c in &self.category {
            match c {
                None => w.u8(0),
                Some(t) => {
                    w.u8(1);
                    w.u16(t.0);
                }
            }
        }
        put_f64s(&mut w, &self.positive)?;
        put_f64s(&mut w, &self.negative)?;
        put_f64s(&mut w, &self.neutral)?;
        w.usize(self.opinions.len())?;
        for o in &self.opinions {
            w.u32(o.get() as u32);
        }
        put_u32s(&mut w, &self.opinion_end)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let tokens_ref = get_ref(&mut r)?;
        want_target::<Tokens>(&tokens_ref, cx)?;
        let sentences_ref = get_ref(&mut r)?;
        want_target::<Sentences>(&sentences_ref, cx)?;
        let opinions_ref = get_ref(&mut r)?;
        want_target::<Opinions>(&opinions_ref, cx)?;
        let tokens_key = cx.key::<Tokens>(&tokens_ref.instance.clone());
        let sentences_key = cx.key::<Sentences>(&sentences_ref.instance.clone());
        let opinions_key = cx.key::<Opinions>(&opinions_ref.instance.clone());
        let mut tokens = Vec::new();
        for _ in 0..r.usize()? {
            tokens.push(match r.u8()? {
                0 => None,
                1 => {
                    let range = IxRange::new(r.u32()?, r.u32()?);
                    validate_range(cx, &tokens_ref, range)?;
                    Some(range)
                }
                tag => return Err(malformed(format!("an optional range tag of {tag}"))),
            });
        }
        let n = tokens.len();
        let mut sentence = Vec::new();
        for _ in 0..r.usize()? {
            sentence.push(match r.u8()? {
                0 => None,
                1 => {
                    let index = r.u32()?;
                    cx.require_node(&NodeRef { layer: sentences_ref.clone(), index })?;
                    Some(Ix::new(index))
                }
                tag => return Err(malformed(format!("an optional index tag of {tag}"))),
            });
        }
        let categories = get_tagset(&mut r)?;
        let mut category = Vec::new();
        for _ in 0..r.usize()? {
            category.push(match r.u8()? {
                0 => None,
                1 => {
                    let tag = Tag(r.u16()?);
                    if tag.0 as usize >= categories.len() {
                        return Err(malformed(format!("a category outside {}", categories.name)));
                    }
                    Some(tag)
                }
                tag => return Err(malformed(format!("an optional category tag of {tag}"))),
            });
        }
        let positive = get_f64s(&mut r)?;
        let negative = get_f64s(&mut r)?;
        let neutral = get_f64s(&mut r)?;
        let mut opinions = Vec::new();
        for _ in 0..r.usize()? {
            let index = r.u32()?;
            cx.require_node(&NodeRef { layer: opinions_ref.clone(), index })?;
            opinions.push(Ix::new(index));
        }
        let opinion_end = get_u32s(&mut r)?;
        want_len(sentence.len(), n, "sentences")?;
        want_len(category.len(), n, "categories")?;
        want_len(positive.len(), n, "positive scores")?;
        want_len(negative.len(), n, "negative scores")?;
        want_len(neutral.len(), n, "neutral scores")?;
        want_len(opinion_end.len(), n, "opinion ends")?;
        if opinion_end.windows(2).any(|w| w[0] > w[1])
            || opinion_end.last().copied().unwrap_or(0) as usize != opinions.len()
        {
            return Err(malformed("opinion ends do not partition the opinions"));
        }
        done(&r, "aspects")?;
        Ok(Aspects {
            tokens,
            sentence,
            category,
            categories,
            positive,
            negative,
            neutral,
            opinions,
            opinion_end,
            tokens_key,
            sentences_key,
            opinions_key,
        })
    }
}

impl LayerCodec for Incivility {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_ref(&mut w, &cx.resolve(&self.tokens_key))?;
        put_ref(&mut w, &cx.resolve(&self.sentences_key))?;
        put_ref(&mut w, &cx.resolve(&self.chunks_key))?;
        put_ref(&mut w, &cx.resolve(&self.structure_key))?;
        w.usize(self.target.len())?;
        for t in &self.target {
            match t {
                IncivilityTarget::Sentence(i) => {
                    w.u8(0);
                    w.u32(i.get() as u32);
                }
                IncivilityTarget::Chunk(i) => {
                    w.u8(1);
                    w.u32(i.get() as u32);
                }
                IncivilityTarget::Block(i) => {
                    w.u8(2);
                    w.u32(i.get() as u32);
                }
            }
        }
        for scores in [
            &self.insult,
            &self.profanity,
            &self.threat,
            &self.identity_attack,
            &self.intolerance,
            &self.severe_toxicity,
        ] {
            put_f64s(&mut w, scores)?;
        }
        w.usize(self.evidence_label.len())?;
        for l in &self.evidence_label {
            w.u8(*l as u8);
        }
        w.usize(self.evidence_tokens.len())?;
        for t in &self.evidence_tokens {
            w.u32(t.start.get() as u32);
            w.u32(t.end.get() as u32);
        }
        put_u32s(&mut w, &self.evidence_end)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let tokens_ref = get_ref(&mut r)?;
        want_target::<Tokens>(&tokens_ref, cx)?;
        let sentences_ref = get_ref(&mut r)?;
        want_target::<Sentences>(&sentences_ref, cx)?;
        let chunks_ref = get_ref(&mut r)?;
        want_target::<Chunks>(&chunks_ref, cx)?;
        let structure_ref = get_ref(&mut r)?;
        want_target::<Structure>(&structure_ref, cx)?;
        let tokens_key = cx.key::<Tokens>(&tokens_ref.instance.clone());
        let sentences_key = cx.key::<Sentences>(&sentences_ref.instance.clone());
        let chunks_key = cx.key::<Chunks>(&chunks_ref.instance.clone());
        let structure_key = cx.key::<Structure>(&structure_ref.instance.clone());
        let mut target = Vec::new();
        for _ in 0..r.usize()? {
            target.push(match r.u8()? {
                0 => {
                    let index = r.u32()?;
                    cx.require_node(&NodeRef { layer: sentences_ref.clone(), index })?;
                    IncivilityTarget::Sentence(Ix::new(index))
                }
                1 => {
                    let index = r.u32()?;
                    cx.require_node(&NodeRef { layer: chunks_ref.clone(), index })?;
                    IncivilityTarget::Chunk(Ix::new(index))
                }
                2 => {
                    let index = r.u32()?;
                    cx.require_node(&NodeRef { layer: structure_ref.clone(), index })?;
                    IncivilityTarget::Block(Ix::new(index))
                }
                tag => return Err(malformed(format!("an incivility target tag of {tag}"))),
            });
        }
        let n = target.len();
        let [insult, profanity, threat, identity_attack, intolerance, severe_toxicity] = [
            get_f64s(&mut r)?,
            get_f64s(&mut r)?,
            get_f64s(&mut r)?,
            get_f64s(&mut r)?,
            get_f64s(&mut r)?,
            get_f64s(&mut r)?,
        ];
        let mut evidence_label = Vec::new();
        for _ in 0..r.usize()? {
            evidence_label.push(
                *IncivilityLabel::ALL
                    .get(r.u8()? as usize)
                    .ok_or_else(|| malformed("an incivility label outside the table"))?,
            );
        }
        let mut evidence_tokens = Vec::new();
        for _ in 0..r.usize()? {
            let range = IxRange::new(r.u32()?, r.u32()?);
            validate_range(cx, &tokens_ref, range)?;
            evidence_tokens.push(range);
        }
        let evidence_end = get_u32s(&mut r)?;
        for (scores, what) in [
            (&insult, "insult scores"),
            (&profanity, "profanity scores"),
            (&threat, "threat scores"),
            (&identity_attack, "identity attack scores"),
            (&intolerance, "intolerance scores"),
            (&severe_toxicity, "severe toxicity scores"),
        ] {
            want_len(scores.len(), n, what)?;
        }
        want_len(evidence_tokens.len(), evidence_label.len(), "evidence spans")?;
        want_len(evidence_end.len(), n, "evidence ends")?;
        if evidence_end.windows(2).any(|w| w[0] > w[1])
            || evidence_end.last().copied().unwrap_or(0) as usize != evidence_label.len()
        {
            return Err(malformed("evidence ends do not partition the evidence"));
        }
        done(&r, "incivility")?;
        Ok(Incivility {
            target,
            insult,
            profanity,
            threat,
            identity_attack,
            intolerance,
            severe_toxicity,
            evidence_label,
            evidence_tokens,
            evidence_end,
            tokens_key,
            sentences_key,
            chunks_key,
            structure_key,
        })
    }
}

impl LayerCodec for Pragmatics {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_ref(&mut w, &cx.resolve(&self.tokens_key))?;
        put_f64s(&mut w, &self.score)?;
        w.usize(self.evidence.len())?;
        for e in &self.evidence {
            match e {
                None => w.u8(0),
                Some(inc) => {
                    w.u8(1);
                    for range in [inc.positive, inc.negative] {
                        w.u32(range.start.get() as u32);
                        w.u32(range.end.get() as u32);
                    }
                }
            }
        }
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let tokens_ref = get_ref(&mut r)?;
        want_target::<Tokens>(&tokens_ref, cx)?;
        let tokens_key = cx.key::<Tokens>(&tokens_ref.instance.clone());
        let score = get_f64s(&mut r)?;
        let mut evidence = Vec::new();
        for _ in 0..r.usize()? {
            evidence.push(match r.u8()? {
                0 => None,
                1 => {
                    let positive = IxRange::new(r.u32()?, r.u32()?);
                    let negative = IxRange::new(r.u32()?, r.u32()?);
                    validate_range(cx, &tokens_ref, positive)?;
                    validate_range(cx, &tokens_ref, negative)?;
                    Some(Incongruity { positive, negative })
                }
                tag => return Err(malformed(format!("an optional evidence tag of {tag}"))),
            });
        }
        want_len(evidence.len(), score.len(), "evidence slots")?;
        done(&r, "pragmatics")?;
        Ok(Pragmatics { score, evidence, tokens_key })
    }
}

const NOTATIONS: [NumberNotation; 2] = [NumberNotation::LatinUs, NumberNotation::LatinEu];
const MULTIPLIERS: [Multiplier; 5] =
    [Multiplier::One, Multiplier::Thousand, Multiplier::Million, Multiplier::Billion, Multiplier::Trillion];
const MARKERS: [ScaleMarker; 3] = [ScaleMarker::None, ScaleMarker::Suffix, ScaleMarker::Word];
const BASES: [OrdinalBase; 3] = [OrdinalBase::Start, OrdinalBase::End, OrdinalBase::Current];
const TIME_KINDS: [TimeKind; 8] = [
    TimeKind::Date,
    TimeKind::Time,
    TimeKind::DateTime,
    TimeKind::DateRange,
    TimeKind::TimeRange,
    TimeKind::DateTimeRange,
    TimeKind::Duration,
    TimeKind::Set,
];
const TIME_GRAINS: [TimeGrain; 14] = [
    TimeGrain::Second,
    TimeGrain::Minute,
    TimeGrain::Hour,
    TimeGrain::PartOfDay,
    TimeGrain::Day,
    TimeGrain::Week,
    TimeGrain::Weekend,
    TimeGrain::Month,
    TimeGrain::Quarter,
    TimeGrain::Half,
    TimeGrain::Season,
    TimeGrain::Year,
    TimeGrain::Decade,
    TimeGrain::Century,
];
const RELATIONS: [Relation; 8] = [
    Relation::None,
    Relation::Before,
    Relation::After,
    Relation::Since,
    Relation::Until,
    Relation::More,
    Relation::Less,
    Relation::Approx,
];
const PORTIONS: [Portion; 6] =
    [Portion::Whole, Portion::Start, Portion::Mid, Portion::End, Portion::Approx, Portion::Undefined];
const PII_CONFIDENCES: [PiiConfidence; 3] =
    [PiiConfidence::Format, PiiConfidence::Checksum, PiiConfidence::LabelledChecksum];

impl LayerCodec for Numbers {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        put_u32s(&mut w, &self.numeral_end)?;
        put_bools(&mut w, &self.negative)?;
        put_strs(&mut w, &self.coefficient)?;
        put_u32s(&mut w, &self.scale)?;
        put_variants(&mut w, &MULTIPLIERS, &self.multiplier)?;
        put_variants(&mut w, &MARKERS, &self.marker)?;
        put_variants(&mut w, &NOTATIONS, &self.notation)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let numeral_end = get_u32s(&mut r)?;
        let negative = get_bools(&mut r)?;
        let coefficient = get_strs(&mut r)?;
        let scale = get_u32s(&mut r)?;
        let multiplier = get_variants(&mut r, &MULTIPLIERS, "multiplier")?;
        let marker = get_variants(&mut r, &MARKERS, "scale marker")?;
        let notation = get_variants(&mut r, &NOTATIONS, "number notation")?;
        let n = spans.len();
        want_len(numeral_end.len(), n, "numeral ends")?;
        want_len(negative.len(), n, "negatives")?;
        want_len(coefficient.len(), n, "coefficients")?;
        want_len(scale.len(), n, "scales")?;
        want_len(multiplier.len(), n, "multipliers")?;
        want_len(marker.len(), n, "markers")?;
        want_len(notation.len(), n, "notations")?;
        done(&r, "numbers")?;
        Ok(Numbers { spans, numeral_end, negative, coefficient, scale, multiplier, marker, notation })
    }
}

impl LayerCodec for Money {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        put_bools(&mut w, &self.negative)?;
        put_strs(&mut w, &self.coefficient)?;
        put_i32s(&mut w, &self.scale)?;
        w.usize(self.currency.len())?;
        for c in &self.currency {
            w.str(c.as_str())?;
        }
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let negative = get_bools(&mut r)?;
        let coefficient = get_strs(&mut r)?;
        let scale = get_i32s(&mut r)?;
        let mut currency = Vec::new();
        for _ in 0..r.usize()? {
            let code = r.string()?;
            currency
                .push(Currency::parse(&code).ok_or_else(|| malformed(format!("an invalid currency code {code:?}")))?);
        }
        let n = spans.len();
        want_len(negative.len(), n, "negatives")?;
        want_len(coefficient.len(), n, "coefficients")?;
        want_len(scale.len(), n, "scales")?;
        want_len(currency.len(), n, "currencies")?;
        done(&r, "money")?;
        Ok(Money { spans, negative, coefficient, scale, currency })
    }
}

impl LayerCodec for Ordinals {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        put_variants(&mut w, &BASES, &self.base)?;
        put_i64s(&mut w, &self.offset)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let base = get_variants(&mut r, &BASES, "ordinal base")?;
        let offset = get_i64s(&mut r)?;
        want_len(base.len(), spans.len(), "bases")?;
        want_len(offset.len(), spans.len(), "offsets")?;
        done(&r, "ordinals")?;
        Ok(Ordinals { spans, base, offset })
    }
}

impl LayerCodec for Quantities {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        put_bools(&mut w, &self.negative)?;
        put_strs(&mut w, &self.coefficient)?;
        put_i32s(&mut w, &self.scale)?;
        put_strs(&mut w, &self.unit)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let negative = get_bools(&mut r)?;
        let coefficient = get_strs(&mut r)?;
        let scale = get_i32s(&mut r)?;
        let unit = get_strs(&mut r)?;
        let n = spans.len();
        want_len(negative.len(), n, "negatives")?;
        want_len(coefficient.len(), n, "coefficients")?;
        want_len(scale.len(), n, "scales")?;
        want_len(unit.len(), n, "units")?;
        done(&r, "quantities")?;
        Ok(Quantities { spans, negative, coefficient, scale, unit })
    }
}

impl LayerCodec for Times {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        put_variants(&mut w, &TIME_KINDS, &self.kind)?;
        put_variants(&mut w, &TIME_GRAINS, &self.grain)?;
        w.usize(self.modifier.len())?;
        for m in &self.modifier {
            put_variant(&mut w, &RELATIONS, m.relation);
            put_variant(&mut w, &PORTIONS, m.portion);
        }
        put_opt_i64s(&mut w, &self.start)?;
        put_opt_i64s(&mut w, &self.end)?;
        put_opt_i64s(&mut w, &self.duration_ms)?;
        put_bools(&mut w, &self.ambiguous)?;
        put_strs(&mut w, &self.timex)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let kind = get_variants(&mut r, &TIME_KINDS, "time kind")?;
        let grain = get_variants(&mut r, &TIME_GRAINS, "time grain")?;
        let mut modifier = Vec::new();
        for _ in 0..r.usize()? {
            modifier.push(Modifier {
                relation: get_variant(&mut r, &RELATIONS, "relation")?,
                portion: get_variant(&mut r, &PORTIONS, "portion")?,
            });
        }
        let start = get_opt_i64s(&mut r)?;
        let end = get_opt_i64s(&mut r)?;
        let duration_ms = get_opt_i64s(&mut r)?;
        let ambiguous = get_bools(&mut r)?;
        let timex = get_strs(&mut r)?;
        let n = spans.len();
        want_len(kind.len(), n, "kinds")?;
        want_len(grain.len(), n, "grains")?;
        want_len(modifier.len(), n, "modifiers")?;
        want_len(start.len(), n, "starts")?;
        want_len(end.len(), n, "ends")?;
        want_len(duration_ms.len(), n, "durations")?;
        want_len(ambiguous.len(), n, "ambiguous flags")?;
        want_len(timex.len(), n, "timex values")?;
        done(&r, "times")?;
        Ok(Times { spans, kind, grain, modifier, start, end, duration_ms, ambiguous, timex })
    }
}

impl LayerCodec for Pii {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        w.usize(self.kind.len())?;
        for k in &self.kind {
            w.str(k.name())?;
        }
        put_variants(&mut w, &PII_CONFIDENCES, &self.confidence)?;
        put_strs(&mut w, &self.normalized)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let mut kind = Vec::new();
        for _ in 0..r.usize()? {
            let name = r.string()?;
            kind.push(
                PiiKind::ALL
                    .iter()
                    .copied()
                    .find(|k| k.name() == name)
                    .ok_or_else(|| malformed(format!("an unknown pii kind {name:?}")))?,
            );
        }
        let confidence = get_variants(&mut r, &PII_CONFIDENCES, "pii confidence")?;
        let normalized = get_strs(&mut r)?;
        want_len(kind.len(), spans.len(), "kinds")?;
        want_len(confidence.len(), spans.len(), "confidences")?;
        want_len(normalized.len(), spans.len(), "normalized values")?;
        done(&r, "pii")?;
        Ok(Pii { spans, kind, confidence, normalized })
    }
}

impl LayerCodec for Regions {
    fn export(&self, _cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        w.usize(self.country.len())?;
        for c in &self.country {
            w.str(c.as_str())?;
        }
        put_f64s(&mut w, &self.share)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], _cx: &mut ImportCx) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let mut country = Vec::new();
        for _ in 0..r.usize()? {
            let code = r.string()?;
            country.push(RegionCode::parse(&code).map_err(|e| malformed(e.to_string()))?);
        }
        let share = get_f64s(&mut r)?;
        want_len(share.len(), country.len(), "shares")?;
        done(&r, "regions")?;
        Ok(Regions { country, share })
    }
}

impl LayerCodec for Matches {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        put_ranges(&mut w, cx, &Key::<Tokens>::DEFAULT, &self.tokens)?;
        put_tagset(&mut w, &self.labels)?;
        put_tags(&mut w, &self.label)?;
        put_u32s(&mut w, &self.rule)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let tokens = get_ranges(&mut r, cx)?;
        let labels = get_tagset(&mut r)?;
        let label = get_tags(&mut r, &labels)?;
        let rule = get_u32s(&mut r)?;
        want_len(tokens.len(), spans.len(), "token ranges")?;
        want_len(label.len(), spans.len(), "labels")?;
        want_len(rule.len(), spans.len(), "rules")?;
        done(&r, "matches")?;
        Ok(Matches { spans, tokens, label, labels, rule })
    }
}

impl LayerCodec for Levels {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        put_ref(&mut w, &cx.resolve(&Key::<Chunks>::DEFAULT))?;
        put_ref(&mut w, &cx.resolve(&Key::<Structure>::DEFAULT))?;
        w.usize(self.unit.len())?;
        for u in &self.unit {
            match u {
                Unit::Chunk(i) => {
                    w.u8(0);
                    w.u32(i.get() as u32);
                }
                Unit::Section(i) => {
                    w.u8(1);
                    w.u32(i.get() as u32);
                }
                Unit::Document => w.u8(2),
            }
        }
        put_parents(&mut w, &self.parent)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let chunks_ref = get_ref(&mut r)?;
        want_target::<Chunks>(&chunks_ref, cx)?;
        let structure_ref = get_ref(&mut r)?;
        want_target::<Structure>(&structure_ref, cx)?;
        let mut unit = Vec::new();
        for _ in 0..r.usize()? {
            unit.push(match r.u8()? {
                0 => {
                    let index = r.u32()?;
                    cx.require_node(&NodeRef { layer: chunks_ref.clone(), index })?;
                    Unit::Chunk(Ix::new(index))
                }
                1 => {
                    let index = r.u32()?;
                    cx.require_node(&NodeRef { layer: structure_ref.clone(), index })?;
                    Unit::Section(Ix::new(index))
                }
                2 => Unit::Document,
                tag => return Err(malformed(format!("a unit tag of {tag}"))),
            });
        }
        let parent = get_parents(&mut r)?;
        want_len(unit.len(), spans.len(), "units")?;
        want_len(parent.len(), spans.len(), "parents")?;
        done(&r, "levels")?;
        Ok(Levels { spans, unit, parent })
    }
}

impl LayerCodec for Containment {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        put_strs(&mut w, &self.ancestor_id)?;
        put_strs(&mut w, &self.ancestor_name)?;
        put_strs(&mut w, &self.ancestor_kind)?;
        put_u32s(&mut w, &self.chain_end)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let ancestor_id = get_strs(&mut r)?;
        let ancestor_name = get_strs(&mut r)?;
        let ancestor_kind = get_strs(&mut r)?;
        let chain_end = get_u32s(&mut r)?;
        want_len(ancestor_name.len(), ancestor_id.len(), "ancestor names")?;
        want_len(ancestor_kind.len(), ancestor_id.len(), "ancestor kinds")?;
        want_len(chain_end.len(), spans.len(), "chain ends")?;
        if chain_end.windows(2).any(|w| w[0] > w[1])
            || chain_end.last().copied().unwrap_or(0) as usize != ancestor_id.len()
        {
            return Err(malformed("chain ends do not partition the ancestors"));
        }
        done(&r, "containment")?;
        Ok(Containment { spans, ancestor_id, ancestor_name, ancestor_kind, chain_end })
    }
}

impl LayerCodec for Phrases {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        put_ranges(&mut w, cx, &Key::<Tokens>::DEFAULT, &self.tokens)?;
        put_tagset(&mut w, &self.kinds)?;
        put_tags(&mut w, &self.kind)?;
        w.usize(self.head.len())?;
        for h in &self.head {
            w.u32(h.get() as u32);
        }
        put_parents(&mut w, &self.parent)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let tokens = get_ranges(&mut r, cx)?;
        let kinds = get_tagset(&mut r)?;
        let kind = get_tags(&mut r, &kinds)?;
        let mut head = Vec::new();
        for (i, _) in (0..r.usize()?).enumerate() {
            let h = r.u32()?;
            let range = tokens.get(i).ok_or_else(|| malformed("a head without a token range"))?;
            if (h as usize) < range.start.get() || (h as usize) >= range.end.get() {
                return Err(malformed(format!("phrase {i} head {h} outside its token range")));
            }
            head.push(Ix::new(h));
        }
        let parent = get_parents(&mut r)?;
        want_len(tokens.len(), spans.len(), "token ranges")?;
        want_len(kind.len(), spans.len(), "kinds")?;
        want_len(head.len(), spans.len(), "heads")?;
        want_len(parent.len(), spans.len(), "parents")?;
        done(&r, "phrases")?;
        Ok(Phrases { spans, tokens, kind, kinds, head, parent })
    }
}

/// Registers the codecs for the shared layers on `codecs`: the core
/// pipeline layers, the sentiment family, the numeric family, the
/// vector and confidence layers at their common scopes, and the
/// abbreviation relations. An add-on's own layer types still refuse to
/// cross until their crates register a codec.
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
        .register::<Phrases>()
        .register::<Opinions>()
        .register::<Aspects>()
        .register::<Incivility>()
        .register::<Pragmatics>()
        .register::<Numbers>()
        .register::<Money>()
        .register::<Ordinals>()
        .register::<Quantities>()
        .register::<Times>()
        .register::<Pii>()
        .register::<Regions>()
        .register::<Matches>()
        .register::<Levels>()
        .register::<Containment>()
        .register::<Vectors<View>>()
        .register::<Vectors<Sentences>>()
        .register::<Vectors<Tokens>>()
        .register::<Vectors<Chunks>>()
        .register::<Vectors<Structure>>()
        .register::<Confidence<Tokens>>()
        .register::<Confidence<Sentences>>()
        .register::<Confidence<Entities>>()
        .register::<Utf16Spans<Tokens>>()
        .register::<Utf16Spans<Sentences>>()
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
