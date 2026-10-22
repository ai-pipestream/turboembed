//! turbo-chunker's units at one level of `Levels` as what is retrieved:
//! the chunks, the sections or the whole document, each with the vector
//! the pipeline embeds for it, or with a vector `LevelVectors` pools.
//!
//! - Chunk: the chunks the optimizer cut, the leaves.
//! - Section: every section of the structure, nested ones included, and
//!   each chunk outside every section (text before the first heading or
//!   after the last section), so every byte a chunk covers is still
//!   reachable. A document with no structure has no sections, and its
//!   units are its chunks.
//! - Document: the document unit, the whole view.
//!
//! With `vectors` set, each unit's vector is `LevelVectors`' local (its
//! own text, in windows of its own pieces) or contextual one (pooled in
//! its parent's windows). Without it the pipeline embeds each unit's text
//! as one row, cut where the bundle cuts it, as it embeds any chunk.

use std::collections::BTreeSet;
use std::sync::Arc;

use turbo_analysis::{Analyzer, Document, Ix, Key};
use turbo_analysis_embed::{LevelVectors, VectorAnnotator};
use turbo_analysis_layers::{Chunks, Level, Levels, Sentences, Structure, Subwords, Unit, VectorKind, Vectors};
use turbo_chunker::levels::LevelBuilder;

use super::{Chunker, TurboChunker, chunk_span, need, setting};
use crate::Result;
use crate::embedder::Embedder;
use crate::record::Setting;
use crate::spans::Span;

/// The level whose units are retrieved.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Retrieve {
    Chunk,
    Section,
    Document,
}

impl Retrieve {
    pub fn name(self) -> &'static str {
        match self {
            Retrieve::Chunk => "chunk",
            Retrieve::Section => "section",
            Retrieve::Document => "document",
        }
    }
}

/// See the module comment.
#[derive(Clone)]
pub struct LevelChunker {
    pub chunker: TurboChunker,
    pub level: Retrieve,
    /// The pooled vectors and which of the two to retrieve by; None to
    /// embed each unit's text.
    pub vectors: Option<(Arc<VectorAnnotator>, VectorKind)>,
}

impl LevelChunker {
    /// The document's levels and, when asked for, its vectors of the kind
    /// asked for.
    fn levels(&self, doc: &Document, embedder: &mut dyn Embedder) -> Result<Document> {
        let fail = |e: turbo_analysis::Error| format!("turbo-chunker levels: {e}");
        let doc = self.chunker.chunked(doc, embedder)?;
        let mut b = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .given(Key::<Chunks>::DEFAULT)
            .given(Key::<Subwords>::DEFAULT);
        if doc.get::<Structure>().is_some() {
            b = b.given(Key::<Structure>::DEFAULT);
        }
        b = b.add(LevelBuilder);
        if let Some((annotator, _)) = &self.vectors {
            b = b.step(LevelVectors(annotator.clone()).bound());
        }
        let analyzer = b.build().map_err(fail)?;
        let mut docs = vec![doc];
        analyzer.run(&mut docs).remove(0).map_err(fail)?;
        Ok(docs.remove(0))
    }
}

/// The rows of `levels` retrieved at `level`, in row order.
pub fn units(levels: &Levels, level: Retrieve) -> Vec<Ix<Levels>> {
    (0..levels.unit.len() as u32)
        .map(Ix::new)
        .filter(|&i| match (level, levels.unit[i.get()]) {
            (Retrieve::Chunk, Unit::Chunk(_)) | (Retrieve::Document, Unit::Document) => true,
            (Retrieve::Section, Unit::Section(_)) => true,
            (Retrieve::Section, Unit::Chunk(_)) => {
                levels.parent[i.get()].is_some_and(|p| levels.level(p) == Level::Document)
            }
            _ => false,
        })
        .collect()
}

impl Chunker for LevelChunker {
    fn name(&self) -> &'static str {
        "turbo-chunker"
    }

    fn config(&self) -> Vec<Setting> {
        let mut out = self.chunker.config();
        out.push(setting("level", self.level.name()));
        out.push(setting(
            "level_vectors",
            match &self.vectors {
                None => "each unit's text embedded as one row",
                Some((_, VectorKind::Local)) => "local, pooled by LevelVectors",
                Some((_, VectorKind::Contextual)) => "contextual, pooled by LevelVectors",
            },
        ));
        out
    }

    fn chunk(&self, doc: &Document, embedder: &mut dyn Embedder) -> Result<Vec<Span>> {
        Ok(self.chunk_vectors(doc, embedder)?.0)
    }

    fn chunk_vectors(&self, doc: &Document, embedder: &mut dyn Embedder) -> Result<(Vec<Span>, Option<Vec<f32>>)> {
        let doc = self.levels(doc, embedder)?;
        let levels = need::<Levels>(&doc, "turbo-chunker levels")?;
        let pooled = match &self.vectors {
            Some((_, kind)) => Some(
                doc.get_key::<Vectors<Levels>>(&kind.key())
                    .ok_or_else(|| format!("turbo-chunker levels: no {} vectors", kind.name()))?,
            ),
            None => None,
        };
        let (mut spans, mut vectors, mut seen) = (Vec::new(), Vec::new(), BTreeSet::new());
        for i in units(levels, self.level) {
            // An empty unit, or one with the bytes of a unit before it,
            // is not retrieved twice.
            let Some(span) = chunk_span(levels.spans.get(i)) else { continue };
            if !seen.insert(span) {
                continue;
            }
            spans.push(span);
            if let Some(v) = pooled {
                let dim = v.dim as usize;
                vectors.extend_from_slice(&v.data[i.get() * dim..(i.get() + 1) * dim]);
            }
        }
        Ok((spans, pooled.map(|_| vectors)))
    }
}

#[cfg(test)]
#[path = "levels_tests.rs"]
mod tests;
