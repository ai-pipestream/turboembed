//! The document graph codecs for this crate's layers, so a
//! chunked document's vocabulary, indexed terms, term statistics,
//! term weights, chunk facets and chunk entity headers cross the export
//! boundary with the rest of the graph.

use std::marker::PhantomData;

use turbo_analysis::export::{Codecs, ExportCx, ExportError, ImportCx, LayerCodec, LayerRef, NodeRef, Reader, Writer};
use turbo_analysis::{Document, Ix, Layer, Span, View};
use turbo_analysis_layers::{
    Aspects, Chunks, Entities, IncivilityLabel, IncivilityScores, Levels, Polarity, Sentences, Strs, Structure, Tokens,
};

use crate::entity::ChunkEntities;

use crate::clauses::{ClauseBreak, ClauseCuts, SentenceClauseCuts};
use crate::coref::{AntecedentLink, ChunkAntecedents};
use crate::hashes::{ChunkId, ContentHashes};
use crate::sentiment::ChunkFacets;
use crate::terms::{IndexedTerms, TermStats, Vocabulary, WeightedTerms};

mod density;

/// The polarities in wire order: a variant crosses as its position
/// here, so a position outside the table is refused rather than
/// read as some other polarity.
const POLARITIES: [Polarity; 3] = [Polarity::Positive, Polarity::Negative, Polarity::Neutral];

fn malformed(detail: impl Into<String>) -> ExportError {
    ExportError::Malformed(detail.into())
}

fn checked<L: Layer>(layer: L, cx: &ImportCx) -> Result<L, ExportError> {
    layer.check().map_err(|detail| ExportError::InvalidLayer { layer: cx.layer().to_owned(), detail })?;
    Ok(layer)
}

fn done(r: &Reader<'_>, what: &str) -> Result<(), ExportError> {
    if r.is_done() { Ok(()) } else { Err(malformed(format!("trailing bytes in a {what} payload"))) }
}

/// The restored instance of `Target` this record's references bind
/// to: the instance its provenance resolves `Target` to, else the
/// default. That is the resolution `Document::add` applies to the
/// default keys these layers index through.
fn resolved<Target: Layer>(cx: &ImportCx) -> LayerRef {
    let name = Target::NAME.to_string();
    let instance = cx.resolved_instance(&name).to_owned();
    LayerRef { layer: name, instance }
}

/// The item count of the already restored `Target` instance an
/// aligned layer on this record is aligned to.
fn aligned_len<Target: Layer>(cx: &ImportCx) -> Result<usize, ExportError> {
    cx.layer_len(&resolved::<Target>(cx))
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

fn put_u32s(w: &mut Writer, values: &[u32]) -> Result<(), ExportError> {
    w.usize(values.len())?;
    for &v in values {
        w.u32(v);
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

fn put_f64s(w: &mut Writer, values: &[f64]) -> Result<(), ExportError> {
    w.usize(values.len())?;
    for &v in values {
        w.f64(v);
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

fn put_variants<E: Copy + PartialEq>(w: &mut Writer, table: &[E], values: &[E], what: &str) -> Result<(), ExportError> {
    w.usize(values.len())?;
    for &v in values {
        let at = table.iter().position(|t| *t == v).ok_or_else(|| malformed(format!("a {what} outside its table")))?;
        w.u8(u8::try_from(at).map_err(|_| malformed(format!("a {what} table longer than a u8 can index")))?);
    }
    Ok(())
}

fn get_variants<E: Copy>(r: &mut Reader<'_>, table: &[E], what: &str) -> Result<Vec<E>, ExportError> {
    let mut values = Vec::new();
    for _ in 0..r.usize()? {
        let at = r.u8()? as usize;
        values.push(table.get(at).copied().ok_or_else(|| malformed(format!("a {what} discriminant of {at}")))?);
    }
    Ok(values)
}

/// A wire index into `Target`, refused before the typed index is
/// built when it is not an item of the restored target instance:
/// the typed constructor panics on the reserved `u32::MAX`.
fn get_ix<Target: Layer>(r: &mut Reader<'_>, cx: &ImportCx, target: &LayerRef) -> Result<Ix<Target>, ExportError> {
    let index = r.u32()?;
    cx.require_node(&NodeRef { layer: target.clone(), index })?;
    Ok(Ix::new(index))
}

impl LayerCodec for Vocabulary {
    fn export(&self, _cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_strs(&mut w, &self.text)?;
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let text = get_strs(&mut r)?;
        done(&r, "vocabulary")?;
        // The layer's own check holds the terms to strictly
        // ascending byte order, each once, which is what makes a
        // term's place its id.
        checked(Vocabulary { text }, cx)
    }
}

impl LayerCodec for IndexedTerms {
    fn export(&self, _cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        w.usize(self.term.len())?;
        for t in &self.term {
            match t {
                None => w.u8(0),
                Some(ix) => {
                    w.u8(1);
                    w.u32(ix.get() as u32);
                }
            }
        }
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let vocab = resolved::<Vocabulary>(cx);
        let mut term = Vec::new();
        for _ in 0..r.usize()? {
            term.push(match r.u8()? {
                0 => None,
                1 => Some(get_ix::<Vocabulary>(&mut r, cx, &vocab)?),
                tag => return Err(malformed(format!("an optional term tag of {tag}"))),
            });
        }
        done(&r, "indexed terms")?;
        let want = aligned_len::<Tokens>(cx)?;
        if term.len() != want {
            return Err(malformed(format!("{} indexed terms for {want} tokens", term.len())));
        }
        checked(IndexedTerms { term }, cx)
    }
}

impl<L: Layer> LayerCodec for TermStats<L> {
    fn export(&self, _cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_u32s(&mut w, &self.row)?;
        w.usize(self.term.len())?;
        for t in &self.term {
            w.u32(t.get() as u32);
        }
        put_u32s(&mut w, &self.tf)?;
        put_u32s(&mut w, &self.length)?;
        put_u32s(&mut w, &self.df)?;
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let row = get_u32s(&mut r)?;
        let vocab = resolved::<Vocabulary>(cx);
        let mut term = Vec::new();
        for _ in 0..r.usize()? {
            term.push(get_ix::<Vocabulary>(&mut r, cx, &vocab)?);
        }
        let tf = get_u32s(&mut r)?;
        let length = get_u32s(&mut r)?;
        let df = get_u32s(&mut r)?;
        done(&r, "term stats")?;
        let units = aligned_len::<L>(cx)?;
        if length.len() != units {
            return Err(malformed(format!("{} unit lengths for {units} aligned units", length.len())));
        }
        let vocab_len = cx.layer_len(&vocab)?;
        if df.len() != vocab_len {
            return Err(malformed(format!("{} df counts for a vocabulary of {vocab_len}", df.len())));
        }
        // The layer's own check settles the sparse rows: one row
        // start per unit ending past the last entry, terms ascending
        // per unit, no zero counts, each length the sum of its
        // counts, and df the number of units holding each term.
        checked(TermStats { row, term, tf, length, df, _of: PhantomData }, cx)
    }
}

impl<L: Layer> LayerCodec for WeightedTerms<L> {
    fn export(&self, _cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_u32s(&mut w, &self.row)?;
        put_f64s(&mut w, &self.weight)?;
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let row = get_u32s(&mut r)?;
        let weight = get_f64s(&mut r)?;
        done(&r, "weighted terms")?;
        let units = aligned_len::<L>(cx)?;
        if row.len() != units + 1 {
            return Err(malformed(format!("{} row starts for {units} aligned units", row.len())));
        }
        // The layer's own check settles the rest: rows from zero to
        // the last weight, in unit order, and every weight finite.
        checked(WeightedTerms { row, weight, _of: PhantomData }, cx)
    }
}

impl LayerCodec for ChunkFacets {
    fn export(&self, _cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_variants(&mut w, &POLARITIES, &self.polarity, "polarity")?;
        put_f64s(&mut w, &self.polarity_score)?;
        put_u32s(&mut w, &self.aspect_row)?;
        w.usize(self.aspect.len())?;
        for a in &self.aspect {
            w.u32(a.get() as u32);
        }
        put_strs(&mut w, &self.aspect_label)?;
        put_variants(&mut w, &POLARITIES, &self.aspect_polarity, "aspect polarity")?;
        put_u32s(&mut w, &self.incivility_row)?;
        put_variants(&mut w, &IncivilityLabel::ALL, &self.incivility_label, "incivility label")?;
        w.usize(self.incivility_max.len())?;
        for s in &self.incivility_max {
            for label in IncivilityLabel::ALL {
                w.f64(s.get(label));
            }
        }
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let polarity = get_variants(&mut r, &POLARITIES, "polarity")?;
        let polarity_score = get_f64s(&mut r)?;
        let aspect_row = get_u32s(&mut r)?;
        let aspects = resolved::<Aspects>(cx);
        let mut aspect = Vec::new();
        for _ in 0..r.usize()? {
            aspect.push(get_ix::<Aspects>(&mut r, cx, &aspects)?);
        }
        let aspect_label = get_strs(&mut r)?;
        let aspect_polarity = get_variants(&mut r, &POLARITIES, "aspect polarity")?;
        let incivility_row = get_u32s(&mut r)?;
        let incivility_label = get_variants(&mut r, &IncivilityLabel::ALL, "incivility label")?;
        let mut incivility_max = Vec::new();
        for _ in 0..r.usize()? {
            incivility_max.push(IncivilityScores {
                insult: r.f64()?,
                profanity: r.f64()?,
                threat: r.f64()?,
                identity_attack: r.f64()?,
                intolerance: r.f64()?,
                severe_toxicity: r.f64()?,
            });
        }
        done(&r, "chunk facets")?;
        let chunks = aligned_len::<Chunks>(cx)?;
        if polarity.len() != chunks {
            return Err(malformed(format!("{} polarities for {chunks} chunks", polarity.len())));
        }
        // The layer's own check settles the rest: the row offsets
        // against their flat columns, strictly ascending incivility
        // labels per chunk, finite polarity scores in [-1, 1] and
        // incivility maxima in [0, 1].
        checked(
            ChunkFacets {
                polarity,
                polarity_score,
                aspect_row,
                aspect,
                aspect_label,
                aspect_polarity,
                incivility_row,
                incivility_label,
                incivility_max,
            },
            cx,
        )
    }
}

impl LayerCodec for ChunkEntities {
    fn export(&self, _cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        w.usize(self.first.len())?;
        for e in &self.first {
            w.u32(e.get() as u32);
        }
        put_u32s(&mut w, &self.row)?;
        put_u32s(&mut w, &self.key)?;
        put_u32s(&mut w, &self.mentions)?;
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let entities = resolved::<Entities>(cx);
        let mut first = Vec::new();
        for _ in 0..r.usize()? {
            first.push(get_ix::<Entities>(&mut r, cx, &entities)?);
        }
        let row = get_u32s(&mut r)?;
        let key = get_u32s(&mut r)?;
        let mentions = get_u32s(&mut r)?;
        done(&r, "chunk entities")?;
        let chunks = aligned_len::<Chunks>(cx)?;
        if row.len() != chunks + 1 {
            return Err(malformed(format!("{} row starts for {chunks} chunks", row.len())));
        }
        // The layer's own check settles the rest: rows from zero to the
        // last entry, keys inside `first`, one count per entry, each key
        // once per chunk and mentioned entries before carried ones.
        checked(ChunkEntities { first, row, key, mentions }, cx)
    }
}

impl<L: Layer> LayerCodec for ContentHashes<L> {
    fn export(&self, _cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        w.usize(self.id.len())?;
        for id in &self.id {
            w.bytes(id.as_bytes())?;
        }
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let mut id = Vec::new();
        for _ in 0..r.usize()? {
            let bytes: [u8; 16] =
                r.bytes()?.try_into().map_err(|_| malformed("a content hash that is not 16 bytes"))?;
            id.push(ChunkId::from_bytes(bytes));
        }
        done(&r, "content hashes")?;
        let units = aligned_len::<L>(cx)?;
        if id.len() != units {
            return Err(malformed(format!("{} content hashes for {units} aligned units", id.len())));
        }
        checked(ContentHashes { id, _of: PhantomData }, cx)
    }
}

mod extra;

/// Registers this crate's layer codecs on `codecs`: the vocabulary,
/// the indexed terms, term statistics at the chunk, sentence,
/// structure and whole-view levels, term weights at the chunk, level
/// and whole-view levels, the chunk facets, the chunk entity headers, and content
/// hashes over the chunks and over the level units. The
/// token-vector windows late pooling checks against are a
/// shared layer, registered by `turbo-analysis-layers`.
pub fn register_codecs(codecs: &mut Codecs) {
    density::register(codecs);
    codecs
        .register::<Vocabulary>()
        .register::<IndexedTerms>()
        .register::<TermStats<Chunks>>()
        .register::<TermStats<Sentences>>()
        .register::<TermStats<Structure>>()
        .register::<TermStats<View>>()
        .register::<TermStats<Levels>>()
        .register::<WeightedTerms<Chunks>>()
        .register::<WeightedTerms<Levels>>()
        .register::<WeightedTerms<View>>()
        .register::<ChunkFacets>()
        .register::<ChunkEntities>()
        .register::<ContentHashes<Chunks>>()
        .register::<ContentHashes<Levels>>()
        .register::<ClauseCuts>()
        .register::<ChunkAntecedents>();
}

/// A registry holding this crate's codecs and nothing else.
pub fn codecs() -> Codecs {
    let mut codecs = Codecs::new();
    register_codecs(&mut codecs);
    codecs
}
