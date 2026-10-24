//! Evaluation traces without changing the persisted Chunks layer.
use crate::{Chunker, ClauseCuts, DecisionMargin, no_term};
use turbo_analysis::{Error, Layer, Spans};
use turbo_analysis_layers::{Boundaries, Chunks, Sentences, Structure, Subwords};

/// All chunk rows and their closing DP decisions. Parent rows and clause
/// splits have no decision: those cuts were not chosen by the sentence DP.
/// A run's last decision does not price the forced next section boundary.
pub struct ChunkTrace {
    chunks: Chunks,
    decisions: Vec<Option<DecisionMargin>>,
}

impl ChunkTrace {
    pub fn chunks(&self) -> &Chunks {
        &self.chunks
    }
    pub fn decisions(&self) -> &[Option<DecisionMargin>] {
        &self.decisions
    }
    pub fn into_chunks(self) -> Chunks {
        self.chunks
    }

    /// Flat leaf spans and their closing margins for `compare_chunk_cuts`.
    /// A forced section boundary has no margin, even when its preceding
    /// row has a DP decision about the end of the preceding run.
    pub fn leaf_cuts(&self) -> (Spans<Chunks>, Vec<Option<i128>>) {
        let mut spans = Spans::new(self.chunks.spans.view());
        let mut margins = Vec::new();
        let mut previous_parent = None;
        for (ix, span) in self.chunks.spans.iter() {
            let i = ix.get();
            if self.chunks.level[i] != 0 {
                continue;
            }
            if !margins.is_empty() && self.chunks.parent[i] != previous_parent {
                *margins.last_mut().unwrap() = None;
            }
            spans.push(span);
            margins.push(self.decisions[i].and_then(DecisionMargin::margin));
            previous_parent = self.chunks.parent[i];
        }
        (spans, margins)
    }
}

impl Chunker {
    /// Run the normal sentence chunker with its optional DP trace. Terms
    /// are zero, as for the `Chunker` annotator. The normal path does not
    /// allocate runner-up tables. Input units must be nonempty and ordered.
    pub fn chunk_with_trace(
        &self,
        sentences: &Sentences,
        subwords: &Subwords,
        bounds: &Boundaries,
        structure: Option<&Structure>,
        clauses: Option<&ClauseCuts>,
    ) -> Result<ChunkTrace, Error> {
        let invalid = |detail| Error::Annotator { annotator: "turbo:chunker-trace", detail };
        sentences.check().map_err(invalid)?;
        subwords.check().map_err(invalid)?;
        if let Some(s) = structure {
            s.check().map_err(invalid)?;
        }
        if let Some(c) = clauses {
            c.check().map_err(invalid)?;
        }
        if sentences.spans.iter().any(|(_, s)| s.start >= s.end) {
            return Err(Error::Annotator { annotator: "turbo:chunker-trace", detail: "empty sentence unit".into() });
        }
        let mut decisions = Vec::new();
        let chunks = self.chunk_record(
            "turbo:chunker-trace",
            sentences,
            subwords,
            bounds,
            structure,
            clauses,
            |_| no_term,
            Some(&mut decisions),
        )?;
        Ok(ChunkTrace { chunks, decisions })
    }
}
