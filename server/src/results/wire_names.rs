//! The wire form of the analysis values more than one answer sends
//! (#2458): the name each enum variant goes out under, and the quality
//! metrics message. `InspectDocumentGraph` (`typed`), `ProjectSearchDocument`
//! (`search_projection`) and `AnalyzeQuery` (`analyze_query`) all read them
//! here, so one value cannot have two names on two surfaces.
//!
//! These strings are part of the wire contract: clients match on them. A
//! rename is a breaking change, and the tests below make it a visible one.

use turbo_analysis_layers::{Block, BoilerplateKind, Cut, EpistemicStatus, QualityMetrics, TimeGrain};

use crate::proto::result::QualityMetricsAnnotation;

pub(crate) fn time_grain(g: TimeGrain) -> &'static str {
    match g {
        TimeGrain::Second => "second",
        TimeGrain::Minute => "minute",
        TimeGrain::Hour => "hour",
        TimeGrain::PartOfDay => "part_of_day",
        TimeGrain::Day => "day",
        TimeGrain::Week => "week",
        TimeGrain::Weekend => "weekend",
        TimeGrain::Month => "month",
        TimeGrain::Quarter => "quarter",
        TimeGrain::Half => "half",
        TimeGrain::Season => "season",
        TimeGrain::Year => "year",
        TimeGrain::Decade => "decade",
        TimeGrain::Century => "century",
    }
}

pub(crate) fn block(b: Block) -> &'static str {
    match b {
        Block::Section => "section",
        Block::Heading => "heading",
        Block::Paragraph => "paragraph",
        Block::List => "list",
        Block::ListItem => "list_item",
        Block::Table => "table",
        Block::TableRow => "table_row",
        Block::TableCell => "table_cell",
        Block::Code => "code",
        Block::Quote => "quote",
        Block::Caption => "caption",
        Block::Footnote => "footnote",
        Block::PageHeader => "page_header",
        Block::PageFooter => "page_footer",
        Block::LooseList => "loose_list",
        Block::Html => "html",
    }
}

/// Why a chunk or a boundary was cut. A cut made inside or across a block
/// names the block after a colon: `inside:table`, `overlap:code`.
pub(crate) fn cut(c: Cut) -> String {
    match c {
        Cut::DocumentStart => "document_start".to_owned(),
        Cut::Sentence => "sentence".to_owned(),
        Cut::Section => "section".to_owned(),
        Cut::Heading => "heading".to_owned(),
        Cut::Paragraph => "paragraph".to_owned(),
        Cut::TopicShift => "topic_shift".to_owned(),
        Cut::Clause => "clause".to_owned(),
        Cut::Item => "item".to_owned(),
        Cut::Inside(b) => format!("inside:{}", block(b)),
        Cut::Overlap(b) => format!("overlap:{}", block(b)),
    }
}

pub(crate) fn boilerplate_kind(k: BoilerplateKind) -> &'static str {
    match k {
        BoilerplateKind::Header => "header",
        BoilerplateKind::Footer => "footer",
        BoilerplateKind::Navigation => "navigation",
        BoilerplateKind::Disclaimer => "disclaimer",
        BoilerplateKind::Template => "template",
    }
}

pub(crate) fn epistemic(e: EpistemicStatus) -> &'static str {
    match e {
        EpistemicStatus::Asserted => "asserted",
        EpistemicStatus::Hedged => "hedged",
        EpistemicStatus::Reported => "reported",
        EpistemicStatus::Conditional => "conditional",
        EpistemicStatus::Presupposed => "presupposed",
    }
}

pub(crate) fn quality_metrics(m: QualityMetrics) -> QualityMetricsAnnotation {
    QualityMetricsAnnotation {
        composite_score: Some(m.composite_score),
        mtld_lexical_diversity: Some(m.mtld_lexical_diversity),
        flesch_kincaid_grade: Some(m.flesch_kincaid_grade),
        automated_readability_index: Some(m.automated_readability_index),
        coleman_liau_index: Some(m.coleman_liau_index),
        information_density: Some(m.information_density),
        alpha_ratio: Some(m.alpha_ratio),
        digit_ratio: Some(m.digit_ratio),
        symbol_ratio: Some(m.symbol_ratio),
        whitespace_ratio: Some(m.whitespace_ratio),
        replacement_ratio: Some(m.replacement_ratio),
        flags: u32::from(m.flags.0),
    }
}

#[cfg(test)]
#[path = "wire_names_tests.rs"]
mod tests;
