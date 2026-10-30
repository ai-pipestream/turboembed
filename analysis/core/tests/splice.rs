//! `Analyzer::run_spliced` against a from-scratch run over toy annotators:
//! the spliced document is the same document, layer for layer, through
//! the graph export; the partition-local steps run only on the windows
//! an edit touched; a declaration the pipeline cannot bear is downgraded.

#[path = "splice/toys.rs"]
mod toys;

use turbo_analysis::export::export;
use turbo_analysis::*;
use turbo_analysis_layers::export::codecs;
use turbo_analysis_layers::{Sentences, Structure};

use toys::*;

const BASE: &str = "Alice meets Bob in Paris. The weather is warm. They walk along the river.\n\n\
Carol writes a long letter. Dave reads it twice. The letter mentions London.\n\n\
Eve plays chess with Frank. Frank loses again. Everyone laughs loudly.";

fn analyzed(text: &str) -> Document {
    analyzed_by(&analyzer(), text)
}

fn analyzed_by(analyzer: &Analyzer, text: &str) -> Document {
    let mut docs = vec![Document::new(text)];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    docs.pop().unwrap()
}

/// `base` edited: every `(from, to)` replaces the first occurrence of
/// `from` after the one before it, in one alignment.
fn edited(base: &str, edits: &[(&str, &str)]) -> (String, Alignment) {
    let mut b = ViewBuilder::new(base);
    for (from, to) in edits {
        let at = b.position() + base[b.position()..].find(from).expect("the edit's text");
        b.keep(at);
        b.replace(at + from.len(), to);
    }
    b.finish()
}

const LOCAL: [&str; 6] = [Tokenizer::ID, SectionTokens::ID, Tagger::ID, Lemmatizer::ID, Parser::ID, NameFinder::ID];

/// Splices `edits` onto the analyzed base and checks the result against
/// a from-scratch run: the same graph, the local steps spliced, nothing
/// left to do for `analyze_incremental`. Returns the report.
fn check(edits: &[(&str, &str)]) -> SpliceReport {
    let base = analyzed(BASE);
    let (text, alignment) = edited(BASE, edits);
    let analyzer = analyzer();
    let (mut spliced, report) = analyzer.run_spliced(&base, text.as_str(), &alignment).unwrap();
    let scratch = analyzed(&text);
    let codecs = codecs();
    assert_eq!(export(&spliced, &codecs).unwrap(), export(&scratch, &codecs).unwrap(), "edits {edits:?}");
    for id in LOCAL {
        assert!(matches!(report.step(id), Some(StepRun::Spliced { .. })), "{id}: {:?}", report.step(id));
    }
    assert_eq!(report.step(Sentencer::ID), Some(&StepRun::Document));
    assert_eq!(report.step(Counter::ID), Some(&StepRun::Document));
    assert!(matches!(report.step(Scaled::ID), Some(StepRun::Rerun(r)) if r.contains("document-wide")));
    let again = analyzer.analyze_incremental(&mut spliced).unwrap();
    assert_eq!(again.stages_run, 0, "{again:?}");
    report
}

/// The windows the sentence-local steps ran on, as text.
fn sentence_windows<'a>(report: &SpliceReport, text: &'a str) -> Vec<&'a str> {
    let key = Key::<Sentences>::DEFAULT.id();
    let (_, windows) = report.windows.iter().find(|(k, _)| *k == key).expect("sentence windows");
    windows.iter().map(|s| &text[s.range()]).collect()
}

#[test]
fn replacement_inside_a_sentence_reruns_its_neighbourhood_only() {
    let report = check(&[("warm", "cold")]);
    let (text, _) = edited(BASE, &[("warm", "cold")]);
    assert_eq!(
        sentence_windows(&report, &text),
        ["Alice meets Bob in Paris. The weather is cold. They walk along the river."]
    );
    let structure = Key::<Structure>::DEFAULT.id();
    let (_, sections) = report.windows.iter().find(|(k, _)| *k == structure).unwrap();
    assert_eq!(sections.len(), 1, "one section dirty, with its neighbour");
}

#[test]
fn insertion_at_a_sentence_boundary() {
    check(&[("They walk", "It rains. They walk")]);
}

#[test]
fn deletion_across_a_sentence_boundary() {
    let report = check(&[("twice. The letter", "")]);
    let (text, _) = edited(BASE, &[("twice. The letter", "")]);
    let windows = sentence_windows(&report, &text);
    assert_eq!(windows.len(), 1);
    assert!(windows[0].starts_with("Carol") && windows[0].ends_with("with Frank."), "{windows:?}");
}

#[test]
fn merging_two_sentences() {
    check(&[("Frank. Frank", "Frank and Frank")]);
}

#[test]
fn splitting_a_sentence() {
    check(&[(" laughs loudly", ". Laughs loudly")]);
}

#[test]
fn several_edits_at_once_make_one_window_per_dirty_run() {
    let edits = [("warm", "cold"), ("Frank. Frank", "Frank and Frank"), (" laughs loudly", ". Laughs loudly")];
    let report = check(&edits);
    let (text, _) = edited(BASE, &edits);
    let windows = sentence_windows(&report, &text);
    assert_eq!(windows.len(), 2, "{windows:?}");
    assert!(windows[0].starts_with("Alice") && windows[0].ends_with("river."), "{windows:?}");
    assert!(windows[1].starts_with("The letter") && windows[1].ends_with("loudly."), "{windows:?}");
}

#[test]
fn an_edit_inside_one_section_leaves_the_other_sections_alone() {
    let report = check(&[("a long", "a short")]);
    let (text, _) = edited(BASE, &[("a long", "a short")]);
    let structure = Key::<Structure>::DEFAULT.id();
    let (_, sections) = report.windows.iter().find(|(k, _)| *k == structure).unwrap();
    let sections: Vec<&str> = sections.iter().map(|s| &text[s.range()]).collect();
    assert_eq!(sections.len(), 1, "{sections:?}");
    assert!(sections[0].starts_with("Alice") && sections[0].ends_with("loudly."), "{sections:?}");
}

#[test]
fn edits_made_one_after_another_compose() {
    let base = analyzed(BASE);
    let (first, a1) = edited(BASE, &[("warm", "cold")]);
    let (second, a2) = edited(&first, &[("loudly", "softly")]);
    let alignment = a1.compose(&a2);
    let analyzer = analyzer();
    let (spliced, report) = analyzer.run_spliced(&base, second.as_str(), &alignment).unwrap();
    let codecs = codecs();
    assert_eq!(export(&spliced, &codecs).unwrap(), export(&analyzed(&second), &codecs).unwrap());
    assert_eq!(sentence_windows(&report, &second).len(), 2);
}

#[test]
fn an_alignment_that_does_not_match_the_texts_is_refused() {
    let base = analyzed(BASE);
    let (text, alignment) = edited(BASE, &[("warm", "cold")]);
    let other = text.replace("Carol", "Carla");
    match analyzer().run_spliced(&base, other.as_str(), &alignment) {
        Err(Error::Source(_)) => {}
        Err(e) => panic!("{e}"),
        Ok(_) => panic!("accepted"),
    }
}

#[test]
fn a_declaration_the_pipeline_cannot_bear_is_downgraded_at_build_time() {
    let sentences = Key::<Sentences>::DEFAULT.id();
    let localities = analyzer().localities();
    let of = |id: &str| localities.iter().find(|(s, _, _)| *s == id).map(|&(_, d, e)| (d, e)).unwrap();
    assert_eq!(of(Scaled::ID), (Locality::Partition(sentences), Locality::Document));
    assert_eq!(of(Tagger::ID), (Locality::Partition(sentences), Locality::Partition(sentences)));
    assert_eq!(of(Counter::ID), (Locality::Document, Locality::Document));
    assert_eq!(of(SectionTokens::ID).1, Locality::Partition(Key::<Structure>::DEFAULT.id()));
}

/// Splices `edits` onto `base` with `analyzer`, whose tagger differs
/// from the base's: the tags and what is made from them are computed
/// again, the rest is spliced, and the result is the analyzer's own.
fn check_stale_tags(base: &Document, analyzer: &Analyzer, edits: &[(&str, &str)], reason: &str) -> Document {
    let (text, alignment) = edited(BASE, edits);
    let (spliced, report) = analyzer.run_spliced(base, text.as_str(), &alignment).unwrap();
    let scratch = analyzed_by(analyzer, &text);
    let codecs = codecs();
    assert_eq!(export(&spliced, &codecs).unwrap(), export(&scratch, &codecs).unwrap(), "edits {edits:?}");
    assert!(
        matches!(report.step(Tagger::ID), Some(StepRun::Rerun(r)) if r.contains(reason)),
        "{:?}",
        report.step(Tagger::ID)
    );
    assert!(
        matches!(report.step(Lemmatizer::ID), Some(StepRun::Rerun(r)) if r.contains("not spliced")),
        "{:?}",
        report.step(Lemmatizer::ID)
    );
    for id in [Tokenizer::ID, SectionTokens::ID, Parser::ID, NameFinder::ID] {
        assert!(matches!(report.step(id), Some(StepRun::Spliced { .. })), "{id}: {:?}", report.step(id));
    }
    spliced
}

#[test]
fn a_base_made_by_another_configuration_is_not_copied() {
    let base = analyzed(BASE);
    let retuned = analyzer_with(200);
    let mut doc = check_stale_tags(&base, &retuned, &[("warm", "cold")], "another configuration");
    assert_eq!(retuned.analyze_incremental(&mut doc).unwrap().stages_run, 0);
    // The same text, aligned to itself: nothing to splice, yet the stale
    // tags are still computed again.
    check_stale_tags(&base, &retuned, &[], "another configuration");
}

#[test]
fn a_step_that_may_not_be_cached_is_not_copied() {
    let uncached = analyzer_uncached();
    let base = analyzed_by(&uncached, BASE);
    check_stale_tags(&base, &uncached, &[("warm", "cold")], "not cacheable");
}

/// A hundred sentences in ten sections.
fn long_text() -> String {
    let mut text = String::new();
    for p in 0..10 {
        if p > 0 {
            text.push_str("\n\n");
        }
        for s in 0..10 {
            if s > 0 {
                text.push(' ');
            }
            text.push_str(&format!("Writer{p} drafts section {p} sentence {s} about Topic{s} with care."));
        }
    }
    text
}

/// Timing, for the acceptance figures: `cargo test --release -p
/// turbo-analysis --test splice -- --ignored --nocapture`. The toy steps
/// cost next to nothing, so the splice's own bookkeeping shows at
/// `work` 0; a tagger that costs what a model costs is where the splice
/// pays.
#[test]
#[ignore = "timing, run in release by hand"]
fn splicing_five_edits_beats_a_run_from_scratch() {
    use std::time::Instant;
    let base_text = long_text();
    let edits = [
        ("sentence 3 about Topic3", "sentence 3 about Topic9"),
        ("Writer2 drafts section 2 sentence 5", "Writer2 redrafts section 2 sentence 5"),
        ("Writer5 drafts section 5 sentence 0", "Writer5 drafts section 5 sentence 0 again"),
        ("sentence 8 about Topic8 with care. Writer7", "sentence 8 about Topic8 with care and Writer7"),
        ("Writer9 drafts section 9 sentence 9", "Writer9 drafts. Section 9 sentence 9"),
    ];
    let (text, alignment) = edited(&base_text, &edits);
    let (identity_text, identity) = edited(&base_text, &[]);
    let best = |f: &dyn Fn()| {
        (0..7)
            .map(|_| {
                let t = Instant::now();
                f();
                t.elapsed()
            })
            .min()
            .unwrap()
    };
    for work in [0u32, 200, 2000] {
        let analyzer = analyzer_with(work);
        let mut docs = vec![Document::new(base_text.as_str())];
        analyzer.run(&mut docs);
        let base = &docs[0];
        let scratch = best(&|| {
            let mut docs = vec![Document::new(text.as_str())];
            analyzer.run(&mut docs);
        });
        let spliced = best(&|| {
            analyzer.run_spliced(base, text.as_str(), &alignment).unwrap();
        });
        let unchanged = best(&|| {
            analyzer.run_spliced(base, identity_text.as_str(), &identity).unwrap();
        });
        let (doc, report) = analyzer.run_spliced(base, text.as_str(), &alignment).unwrap();
        let mut docs = vec![Document::new(text.as_str())];
        analyzer.run(&mut docs);
        let codecs = codecs();
        assert_eq!(export(&doc, &codecs).unwrap(), export(&docs[0], &codecs).unwrap());
        let windows = sentence_windows(&report, &text);
        println!(
            "tagger work {work}: from scratch {scratch:?}, spliced {spliced:?} ({} windows), no edits {unchanged:?}",
            windows.len()
        );
        if work > 0 {
            assert!(spliced < scratch, "spliced {spliced:?} is not faster than {scratch:?}");
        }
    }
}
