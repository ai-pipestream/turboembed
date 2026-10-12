//! The structural term of `turbo:boundaries`: what a cut after each
//! sentence costs, from the strongest break the document's structure
//! makes there.

use turbo_analysis::{Annotator, Error, Refs};
use turbo_analysis_layers::{Boundaries, Cut, Sentences, Structure};

use crate::gaps::gaps;

/// What a cut costs at each kind of break, in `Boundaries::UNIT`s. A plain
/// sentence boundary costs one unit; a stronger break costs less, and a
/// cut that would part a heading from what it heads costs extra.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Costs {
    pub section: i32,
    pub heading: i32,
    /// The end of a paragraph, list, table, code block, quote, caption,
    /// footnote or page header or footer.
    pub paragraph: i32,
    /// Between two list items or two table rows.
    pub item: i32,
    pub sentence: i32,
    /// Added where a heading ends.
    pub after_heading: i32,
}

impl Default for Costs {
    fn default() -> Costs {
        let u = Boundaries::UNIT;
        Costs {
            section: 0,
            heading: u / 10,
            paragraph: u * 4 / 10,
            item: u * 6 / 10,
            sentence: u,
            after_heading: u * 4,
        }
    }
}

impl Costs {
    fn of(&self, reason: Cut) -> i32 {
        match reason {
            Cut::Section => self.section,
            Cut::Heading => self.heading,
            Cut::Paragraph => self.paragraph,
            Cut::Item => self.item,
            _ => self.sentence,
        }
    }
}

/// Writes the structural boundary costs: a cost and a reason after each
/// sentence. Without structure every gap is a sentence boundary. It writes
/// the default instance while it is the only boundary signal; once there
/// are others, it is bound to write its own named instance and a summing
/// step writes the default one.
///
/// The hard limits (no cut inside a table row, code block or quote that
/// fits the budget) are not costs: they depend on the budget, so the
/// chunker applies them from the same structure.
pub struct StructureBoundaries {
    pub costs: Costs,
}

impl Annotator for StructureBoundaries {
    const ID: &'static str = "turbo:structure-boundaries";
    type Reads = (Sentences, Option<Structure>);
    type Writes = (Boundaries,);

    fn annotate(&self, (sentences, structure): Refs<'_, Self>) -> Result<(Boundaries,), Error> {
        if structure.is_some_and(|s| s.spans.view() != sentences.spans.view()) {
            return Err(Error::Annotator { annotator: Self::ID, detail: "its inputs are in different views".into() });
        }
        let gaps = gaps(&sentences.spans, structure).gap;
        let cost = gaps
            .iter()
            .map(|g| self.costs.of(g.reason).saturating_add(if g.after_heading { self.costs.after_heading } else { 0 }))
            .collect();
        let reason = gaps.iter().map(|g| g.reason).collect();
        Ok((Boundaries { cost, reason },))
    }

    fn config_hash(&self) -> u64 {
        let c = self.costs;
        [c.section, c.heading, c.paragraph, c.item, c.sentence, c.after_heading]
            .iter()
            .fold(0xcbf2_9ce4_8422_2325, |h, &x| (h ^ x as u32 as u64).wrapping_mul(0x0100_0000_01b3))
    }
}
