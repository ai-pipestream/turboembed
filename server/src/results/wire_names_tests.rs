//! Every variant's wire name, pinned. Each enum is walked by a `next`
//! function whose match is exhaustive, so a new variant does not compile
//! here until it is given a place, and then its name has to be listed.

use turbo_analysis_layers::{Block, BoilerplateKind, Cut, EpistemicStatus, QualityFlags, QualityMetrics, TimeGrain};

use super::*;

/// The variants from `first` on, in the order `next` gives.
fn walk<T: Copy>(first: T, next: fn(T) -> Option<T>) -> Vec<T> {
    std::iter::successors(Some(first), |&v| next(v)).collect()
}

fn next_grain(g: TimeGrain) -> Option<TimeGrain> {
    use TimeGrain::*;
    Some(match g {
        Second => Minute,
        Minute => Hour,
        Hour => PartOfDay,
        PartOfDay => Day,
        Day => Week,
        Week => Weekend,
        Weekend => Month,
        Month => Quarter,
        Quarter => Half,
        Half => Season,
        Season => Year,
        Year => Decade,
        Decade => Century,
        Century => return None,
    })
}

fn next_block(b: Block) -> Option<Block> {
    use Block::*;
    Some(match b {
        Section => Heading,
        Heading => Paragraph,
        Paragraph => List,
        List => ListItem,
        ListItem => Table,
        Table => TableRow,
        TableRow => TableCell,
        TableCell => Code,
        Code => Quote,
        Quote => Caption,
        Caption => Footnote,
        Footnote => PageHeader,
        PageHeader => PageFooter,
        PageFooter => LooseList,
        LooseList => Html,
        Html => return None,
    })
}

fn next_cut(c: Cut) -> Option<Cut> {
    use Cut::*;
    Some(match c {
        DocumentStart => Sentence,
        Sentence => Section,
        Section => Heading,
        Heading => Paragraph,
        Paragraph => TopicShift,
        TopicShift => Clause,
        Clause => Item,
        Item => Inside(Block::Table),
        Inside(_) => Overlap(Block::Code),
        Overlap(_) => return None,
    })
}

fn next_boilerplate(k: BoilerplateKind) -> Option<BoilerplateKind> {
    use BoilerplateKind::*;
    Some(match k {
        Header => Footer,
        Footer => Navigation,
        Navigation => Disclaimer,
        Disclaimer => Template,
        Template => return None,
    })
}

fn next_epistemic(e: EpistemicStatus) -> Option<EpistemicStatus> {
    use EpistemicStatus::*;
    Some(match e {
        Asserted => Hedged,
        Hedged => Reported,
        Reported => Conditional,
        Conditional => Presupposed,
        Presupposed => return None,
    })
}

#[test]
fn time_grains_keep_their_names() {
    let names: Vec<&str> = walk(TimeGrain::Second, next_grain).into_iter().map(time_grain).collect();
    assert_eq!(
        names,
        [
            "second",
            "minute",
            "hour",
            "part_of_day",
            "day",
            "week",
            "weekend",
            "month",
            "quarter",
            "half",
            "season",
            "year",
            "decade",
            "century"
        ]
    );
}

#[test]
fn blocks_keep_their_names() {
    let names: Vec<&str> = walk(Block::Section, next_block).into_iter().map(block).collect();
    assert_eq!(
        names,
        [
            "section",
            "heading",
            "paragraph",
            "list",
            "list_item",
            "table",
            "table_row",
            "table_cell",
            "code",
            "quote",
            "caption",
            "footnote",
            "page_header",
            "page_footer",
            "loose_list",
            "html"
        ]
    );
}

#[test]
fn cuts_keep_their_names_and_name_the_block_they_cut() {
    let names: Vec<String> = walk(Cut::DocumentStart, next_cut).into_iter().map(cut).collect();
    assert_eq!(
        names,
        [
            "document_start",
            "sentence",
            "section",
            "heading",
            "paragraph",
            "topic_shift",
            "clause",
            "item",
            "inside:table",
            "overlap:code"
        ]
    );
    // the part after the colon is the block's own name, for every block
    for b in walk(Block::Section, next_block) {
        assert_eq!(cut(Cut::Inside(b)), format!("inside:{}", block(b)));
        assert_eq!(cut(Cut::Overlap(b)), format!("overlap:{}", block(b)));
    }
}

#[test]
fn boilerplate_kinds_and_epistemic_statuses_keep_their_names() {
    let kinds = walk(BoilerplateKind::Header, next_boilerplate);
    assert_eq!(kinds, BoilerplateKind::ALL, "the walk reaches every kind the layer lists");
    let names: Vec<&str> = kinds.into_iter().map(boilerplate_kind).collect();
    assert_eq!(names, ["header", "footer", "navigation", "disclaimer", "template"]);
    let statuses = walk(EpistemicStatus::Asserted, next_epistemic);
    for (byte, status) in statuses.iter().enumerate() {
        assert_eq!(EpistemicStatus::from_byte(byte as u8), Some(*status));
    }
    assert_eq!(EpistemicStatus::from_byte(statuses.len() as u8), None);
    let names: Vec<&str> = statuses.into_iter().map(epistemic).collect();
    assert_eq!(names, ["asserted", "hedged", "reported", "conditional", "presupposed"]);
}

#[test]
fn names_are_distinct_within_an_enum() {
    fn distinct(mut names: Vec<String>) {
        let n = names.len();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), n, "{names:?}");
    }
    distinct(walk(TimeGrain::Second, next_grain).into_iter().map(|g| time_grain(g).to_owned()).collect());
    distinct(walk(Block::Section, next_block).into_iter().map(|b| block(b).to_owned()).collect());
    distinct(walk(Block::Section, next_block).into_iter().map(|b| cut(Cut::Inside(b))).collect());
}

#[test]
fn each_quality_metric_lands_in_its_own_field() {
    // every field has a different value, so a swapped pair shows
    let m = QualityMetrics {
        composite_score: 0.5,
        mtld_lexical_diversity: 1.5,
        flesch_kincaid_grade: 2.5,
        automated_readability_index: 3.5,
        coleman_liau_index: 4.5,
        information_density: 5.5,
        alpha_ratio: 6.5,
        digit_ratio: 7.5,
        symbol_ratio: 8.5,
        whitespace_ratio: 9.5,
        replacement_ratio: 10.5,
        flags: QualityFlags(0x8003),
    };
    let wire = quality_metrics(m);
    assert_eq!(wire.composite_score, Some(0.5));
    assert_eq!(wire.mtld_lexical_diversity, Some(1.5));
    assert_eq!(wire.flesch_kincaid_grade, Some(2.5));
    assert_eq!(wire.automated_readability_index, Some(3.5));
    assert_eq!(wire.coleman_liau_index, Some(4.5));
    assert_eq!(wire.information_density, Some(5.5));
    assert_eq!(wire.alpha_ratio, Some(6.5));
    assert_eq!(wire.digit_ratio, Some(7.5));
    assert_eq!(wire.symbol_ratio, Some(8.5));
    assert_eq!(wire.whitespace_ratio, Some(9.5));
    assert_eq!(wire.replacement_ratio, Some(10.5));
    assert_eq!(wire.flags, 0x8003);
    // a clean measurement still says each number was measured
    let clean = quality_metrics(QualityMetrics::default());
    assert_eq!((clean.composite_score, clean.replacement_ratio, clean.flags), (Some(0.0), Some(0.0), 0));
}
