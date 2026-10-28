//! The bottom-up constituency parsers trained here: the chunking and tree-insertion event
//! streams, `chunking.Parser.train` and `treeinsert.Parser.train`, the model file read back,
//! `ParserEvaluator`, `ParserCrossValidator`, `ParserAnnotator` and `ParserTool` on a trained
//! model. Each is checked against the JVM in `tests/parser_goldens.rs`; these tests pin
//! behaviour on a small project-written corpus.

use std::str::FromStr;

use turbo_formats::evaluate::LabeledSpan;
use turbo_parser::treeinsert::{ATTACH_DAUGHTER, ATTACH_SISTER, DONE};
use turbo_parser::{EnglishHeadRules, INC_NODE, Parse, ParseSampleStream, ParserEventType, build_dictionary};
use turbo_train_components::parser::{ParserEvaluator, chunking, treeinsert};

fn english_head_rules() -> EnglishHeadRules {
    let path = turbo_goldens::data_dir().join("fixtures/en-head-rules.txt");
    EnglishHeadRules::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// Project-authored Penn Treebank training corpus for constituency parser tests.
const SAMPLE_TREEBANK: &str = "\
(TOP (S (NP-SBJ (DT The) (JJ quick) (NN fox)) (VP (VBD jumped) (PP (IN over) (NP (DT the) (JJ lazy) (NN dog)))) (. .)))
(TOP (S (NP-SBJ (DT The) (JJ lazy) (NN dog)) (VP (VBD slept) (PP (IN under) (NP (DT the) (JJ old) (NN tree)))) (. .)))
(TOP (S (NP-SBJ (DT A) (JJ small) (NN bird)) (VP (VBD sang) (PP (IN in) (NP (DT the) (NN garden)))) (. .)))
(TOP (S (NP-SBJ (NNP Alice)) (VP (VBD wrote) (NP (DT a) (JJ new) (NN program)) (PP (IN in) (NP (NNP Rust)))) (. .)))
(TOP (S (NP-SBJ (NNP Bob)) (VP (VBD tested) (NP (DT the) (NN parser)) (PP (IN with) (NP (JJ several) (NNS sentences)))) (. .)))
(TOP (S (NP-SBJ (DT The) (NN parser)) (VP (VBZ builds) (NP (JJ syntactic) (NNS trees)) (PP (IN from) (NP (NNS tokens)))) (. .)))
(TOP (S (NP-SBJ (DT Every) (NN model)) (VP (VBZ reads) (NP (NN input)) (CC and) (VBZ writes) (NP (NNS phrases))) (. .)))
(TOP (S (NP-SBJ (NNP Martin)) (VP (VBZ is) (VP (VBG testing) (NP (DT the) (NN system)))) (. .)))
(TOP (S (NP-SBJ (NNP Eric)) (VP (VBZ is) (VP (VBG testing) (NP (DT the) (NN code)))) (. .)))
(TOP (S (NP-SBJ (PRP She)) (VP (VBD saw) (NP (DT the) (NN ship)) (, ,) (CC and) (VP (VBD waved) (PP (IN from) (NP (DT the) (NN dock))))) (. .)))
(TOP (S (PP (IN On) (NP (NNP Monday))) (, ,) (NP-SBJ (DT the) (NN team)) (VP (VBD shipped) (NP (DT the) (NN release))) (. .)))
(TOP (S (NP-SBJ (DT The) (NN cat)) (VP (VBD sat) (PP (IN on) (NP (DT the) (NN mat)))) (. .)))
";

fn sample_parses() -> Vec<Parse> {
    ParseSampleStream::new(SAMPLE_TREEBANK).collect::<Result<Vec<_>, _>>().expect("valid sample treebank")
}

#[test]
fn chunking_and_treeinsert_event_streams_generate_events() {
    let rules = english_head_rules();
    let samples = sample_parses();
    let dict = build_dictionary(&samples, &rules, 1);

    let build_events =
        chunking::ParserEventStream::generate_events(&samples, &rules, ParserEventType::Build, Some(&dict)).unwrap();
    assert!(!build_events.is_empty());
    assert!(build_events.iter().any(|e| e.outcome() == "S-S" || e.outcome() == "S-VP"));

    let check_events =
        chunking::ParserEventStream::generate_events(&samples, &rules, ParserEventType::Check, None).unwrap();
    assert!(!check_events.is_empty());
    assert!(check_events.iter().any(|e| e.outcome() == turbo_parser::COMPLETE));
    assert!(check_events.iter().any(|e| e.outcome() == turbo_parser::INCOMPLETE));

    let ti_build =
        treeinsert::ParserEventStream::generate_events(&samples, &rules, ParserEventType::Build, Some(&dict)).unwrap();
    assert!(!ti_build.is_empty());
    assert!(ti_build.iter().any(|e| e.outcome() == DONE));

    let ti_attach =
        treeinsert::ParserEventStream::generate_events(&samples, &rules, ParserEventType::Attach, None).unwrap();
    assert!(!ti_attach.is_empty());
    assert!(ti_attach.iter().any(|e| e.outcome() == ATTACH_SISTER || e.outcome() == ATTACH_DAUGHTER));

    let ti_check =
        treeinsert::ParserEventStream::generate_events(&samples, &rules, ParserEventType::Check, None).unwrap();
    assert!(!ti_check.is_empty());
}

/// Both streams make their `TAG` events as `AbstractParserEventStream.addTagEvents` does: one per
/// token, the outcome its tag and the context `DefaultPOSContextGenerator(null)`'s, which reads the
/// tags of the two tokens before it.
#[test]
fn tag_events_are_default_pos_contexts_in_both_streams() {
    let rules = english_head_rules();
    let samples = [Parse::parse("(TOP (S (NP (DT The) (NN cat-o)) (VP (VBD sat) (NP (CD 2))) (. .)))").unwrap()];
    let events = |etype| {
        let chunking = chunking::ParserEventStream::generate_events(&samples, &rules, etype, None).unwrap();
        let tree_insertion = treeinsert::ParserEventStream::generate_events(&samples, &rules, etype, None).unwrap();
        let show = |events: Vec<turbo_train::Event>| -> Vec<String> {
            let context = |e: &turbo_train::Event| e.context().iter().map(|c| c.to_string_lossy()).collect::<Vec<_>>();
            events.iter().map(|e| format!("{} | {}", e.outcome().to_string_lossy(), context(e).join(" "))).collect()
        };
        (show(chunking), show(tree_insertion))
    };
    let (chunking, tree_insertion) = events(ParserEventType::Tag);
    assert_eq!(
        chunking,
        [
            "DT | default w=The suf=e suf=he suf=The suf=The pre=T pre=Th pre=The pre=The c p=*SB* n=cat-o nn=sat",
            "NN | default w=cat-o suf=o suf=-o suf=t-o suf=at-o pre=c pre=ca pre=cat pre=cat- h p=The t=DT \
             pp=*SB* n=sat nn=2",
            "VBD | default w=sat suf=t suf=at suf=sat suf=sat pre=s pre=sa pre=sat pre=sat p=cat-o t=NN pp=The \
             t2=DT,NN n=2 nn=.",
            "CD | default w=2 suf=2 suf=2 suf=2 suf=2 pre=2 pre=2 pre=2 pre=2 d p=sat t=VBD pp=cat-o t2=NN,VBD \
             n=. nn=*SE*",
            ". | default w=. suf=. suf=. suf=. suf=. pre=. pre=. pre=. pre=. p=2 t=CD pp=sat t2=VBD,CD n=*SE*",
        ]
    );
    assert_eq!(tree_insertion, chunking);
}

#[test]
fn parser_evaluator_constituent_spans() {
    // Ported from OpenNLP's ParserEvaluatorTest.testGetConstituencySpans
    let parse_str = "(TOP (S (NP-SBJ (NNP Mr.) (NNP Vinken) )(VP (VBZ is)(NP-PRD (NP (NN chairman) )(PP (IN of)(NP (NP (NNP Elsevier) (NNP N.V.) )(, ,)(NP (DT the) (NNP Dutch) (VBG publishing) (NN group) )))))(. .) ))";
    let p = Parse::parse(parse_str).unwrap();
    let spans = ParserEvaluator::get_constituent_spans(&p);
    // OpenNLP pops a stack of each node's children, so a node comes before its children and the
    // last child of a node first.
    let expected = [
        LabeledSpan::new(0, 70, Some("S")),
        LabeledSpan::new(11, 68, Some("VP")),
        LabeledSpan::new(14, 68, Some("NP")),
        LabeledSpan::new(23, 68, Some("PP")),
        LabeledSpan::new(26, 68, Some("NP")),
        LabeledSpan::new(42, 68, Some("NP")),
        LabeledSpan::new(26, 39, Some("NP")),
        LabeledSpan::new(14, 22, Some("NP")),
        LabeledSpan::new(0, 10, Some("NP")),
    ];
    assert_eq!(spans, expected);

    // The root is left out whatever its type, so an `INC` guess gives only the spans below it.
    let mut guess = Parse::parse("(NP (DT The) (NN cat)) (VP (VBD sat))").unwrap();
    guess.set_kind(INC_NODE);
    assert_eq!(
        ParserEvaluator::get_constituent_spans(&guess),
        [LabeledSpan::new(8, 11, Some("VP")), LabeledSpan::new(0, 7, Some("NP"))]
    );
}

#[test]
fn chunking_event_stream_revisits_constituent_reduced_at_index_zero() {
    // After `NP VP` reduce into `S` at index 0, OpenNLP's loop sets `ci = reduceStart - 1` and
    // increments, so it visits `S` next and emits its `S-TOP` build event and `TOP` check event.
    let rules = english_head_rules();
    let samples = vec![Parse::parse("(TOP (S (NP (DT The) (NN dog)) (VP (VBD ran))))").unwrap()];

    let build = chunking::ParserEventStream::generate_events(&samples, &rules, ParserEventType::Build, None).unwrap();
    let outcomes: Vec<String> = build.iter().map(|e| e.outcome().to_string_lossy()).collect();
    assert_eq!(outcomes, ["S-S", "C-S", "S-TOP"]);

    let check = chunking::ParserEventStream::generate_events(&samples, &rules, ParserEventType::Check, None).unwrap();
    let outcomes: Vec<String> = check.iter().map(|e| e.outcome().to_string_lossy()).collect();
    assert_eq!(outcomes, [turbo_parser::INCOMPLETE, turbo_parser::COMPLETE, turbo_parser::COMPLETE]);
}
