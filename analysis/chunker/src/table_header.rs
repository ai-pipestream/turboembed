//! Table boundary costs and source-preserving header context.
use turbo_analysis::{Annotator, Error, Ix, Refs, View};
use turbo_analysis_layers::{Block, Boundaries, ChunkTableHeaders, Chunks, Cut, Sentences, Tables, Tokens};

mod rows;
pub(crate) use rows::table_rows;

/// The optimizer minimizes costs: an inside-row cut costs more than a row edge.
/// `Chunker` additionally reads `Tables` to keep logical rows atomic, including
/// rows larger than the budget. Costs remain usable beside other signals.
#[derive(Debug, Clone, Copy)]
pub struct TableBoundaryAnnotator {
    pub inside_cost: i32,
    pub row_boundary_cost: i32,
}
impl Default for TableBoundaryAnnotator {
    fn default() -> Self {
        Self { inside_cost: 100_000, row_boundary_cost: 0 }
    }
}
impl Annotator for TableBoundaryAnnotator {
    const ID: &'static str = "turbo:table-boundaries";
    type Reads = (View, Sentences, Tokens, Tables);
    type Writes = (Boundaries,);
    fn annotate(&self, (view, sentences, tokens, tables): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let fail = |detail: &str| Error::Annotator { annotator: Self::ID, detail: detail.into() };
        if self.inside_cost <= self.row_boundary_cost {
            return Err(fail("inside-row cost must exceed row-edge cost"));
        }
        if [sentences.spans.view(), tokens.spans.view(), tables.spans.view()].iter().any(|&v| v != view.id()) {
            return Err(fail("table boundary inputs are in different views"));
        }
        sentences.spans.check(Sentences::OVERLAP).map_err(|e| fail(&e))?;
        tokens.spans.check(Tokens::OVERLAP).map_err(|e| fail(&e))?;
        tables.check().map_err(|e| fail(&e))?;
        let (token_start, token_end) = tokens.spans.columns();
        if !token_start.windows(2).all(|w| w[0] <= w[1]) || !token_end.windows(2).all(|w| w[0] <= w[1]) {
            return Err(fail("table boundary tokens must have ordered source endpoints"));
        }
        let row = |offset| rows::row_at(tables, offset);
        let anchors: Vec<_> = sentences
            .spans
            .iter()
            .map(|(_, s)| {
                let first = token_end.partition_point(|&e| e <= s.start);
                let end = token_start.partition_point(|&a| a < s.end);
                if first < end {
                    (row(token_start[first]), row(token_start[end - 1]))
                } else {
                    (row(s.start), row(s.end.saturating_sub(1)))
                }
            })
            .collect();
        let mut out = Boundaries { cost: vec![0; anchors.len()], reason: vec![Cut::Sentence; anchors.len()] };
        for i in 0..anchors.len().saturating_sub(1) {
            let (left, right) = (anchors[i].1, anchors[i + 1].0);
            if left.is_some() && left == right {
                out.cost[i] = self.inside_cost;
                out.reason[i] = Cut::Inside(Block::TableRow);
            } else if left.is_some() || right.is_some() {
                out.cost[i] = self.row_boundary_cost;
                out.reason[i] = Cut::Item;
            }
        }
        Ok((out,))
    }
    fn config_hash(&self) -> u64 {
        [self.inside_cost, self.row_boundary_cost]
            .iter()
            .fold(0xcbf2_9ce4_8422_2325, |h, &x| (h ^ u64::from(x as u32)).wrapping_mul(0x0100_0000_01b3))
    }
}
use turbo_analysis::Layer;

#[derive(Debug, Default, Clone, Copy)]
pub struct TableHeaderAnnotator;
impl Annotator for TableHeaderAnnotator {
    const ID: &'static str = "turbo:table-headers";
    type Reads = (View, Chunks, Tables);
    type Writes = (ChunkTableHeaders,);
    fn annotate(&self, (view, chunks, tables): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let fail = |detail: String| Error::Annotator { annotator: Self::ID, detail };
        if view.id() != chunks.spans.view() {
            return Err(fail("header source is in a different view".into()));
        }
        Ok((ChunkTableHeaders::build(chunks, tables, view.text()).map_err(fail)?,))
    }
    fn config_hash(&self) -> u64 {
        1
    }
}
