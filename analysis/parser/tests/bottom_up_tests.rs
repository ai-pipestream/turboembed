//! Unit and integration tests for `turbo-parser`'s `CHUNKING` and `TREEINSERT`
//! constituency parsers, trainers, model packaging, context generators,
//! `ParserEvaluator`, `ParserCrossValidator`, `ParserAnnotator`, and
//! `ParserTool`.

use std::str::FromStr;
use std::sync::Arc;

use turbo_analysis::{Analyzer, Document, Ix, Key, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::{Phrases, Sentences, Tokens};
use turbo_formats::evaluate::LabeledSpan;
use turbo_parser::{
    ChunkContextGenerator, EnglishHeadRules, HeadRules, HeadRulesArtifact, INC_NODE, Parse, ParseSampleStream, Parser,
    ParserAnnotator, ParserChunkerSequenceValidator, ParserCrossValidator, ParserEvaluator, ParserEventType,
    ParserFactory, ParserModel, ParserTool, ParserTrainingParameters, ParserType, build_dictionary, chunking,
    collapse_punctuation, dictionary_key, treeinsert,
};

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
fn chunk_context_generator_and_sequence_validator() {
    // The 19 features of `ChunkContextGenerator.getContext`, in its order.
    let mut cg = ChunkContextGenerator::with_cache(4);
    let words = ["The", "quick", "fox", "jumped"];
    let tags = ["DT", "JJ", "NN", "VBD"];
    let preds = ["S-NP", "C-NP", "C-NP", "S-VP"];
    let ctx1 = cg.get_context(1, &words, &tags, &preds);
    assert_eq!(
        ctx1,
        [
            "default",
            "-2=eos|eos|eos",
            "-2*=eos|eos",
            "-1=The|DT|S-NP",
            "-1*=DT|S-NP",
            "0=quick|JJ",
            "0*=JJ",
            "1=fox|NN",
            "1*=NN",
            "2=jumped|VBD",
            "2*=VBD",
            "-1=The|DT|S-NP,0=quick|JJ",
            "-1*=DT|S-NP,0=quick|JJ",
            "-1=The|DT|S-NP,0*=JJ",
            "-1*=DT|S-NP,0*=JJ",
            "0=quick|JJ,1=fox|NN",
            "0*=JJ,1=fox|NN",
            "0=quick|JJ,1*=NN",
            "0*=JJ,1*=NN",
        ]
    );
    // Hit cache on repeated call
    assert_eq!(cg.get_context(1, &words, &tags, &preds), ctx1);

    let ctx3 = cg.get_context(3, &words, &tags, &preds);
    assert_eq!(ctx3.len(), 19);
    assert_eq!(&ctx3[1..3], ["-2=quick|JJ|C-NP", "-2*=JJ|C-NP"]);
    assert_eq!(&ctx3[7..11], ["1=eos|eos", "1*=eos", "2=eos|eos", "2*=eos"]);

    let validator = ParserChunkerSequenceValidator::new(&["S-NP", "C-NP", "S-VP", "C-VP", "O"]);
    assert!(validator.valid_outcome(&[], "S-NP"));
    assert!(validator.valid_outcome(&[], "O"));
    assert!(!validator.valid_outcome(&[], "C-NP"));
    assert!(validator.valid_outcome(&["S-NP"], "C-NP"));
    assert!(!validator.valid_outcome(&["S-NP"], "C-VP"));
    assert!(!validator.valid_outcome(&["O"], "C-NP"));
}

/// The chunks of `tree` after `collapsePunctuation`, with heads, the first labeled `S-S`.
fn labeled_chunks(tree: &str) -> Vec<Parse> {
    let rules = english_head_rules();
    let mut p = Parse::parse(tree).unwrap();
    p.update_heads(&rules);
    let chunks: Vec<Parse> = p.initial_chunks().into_iter().cloned().collect();
    let mut collapsed = collapse_punctuation(&chunks, rules.punctuation_tags());
    collapsed[0].set_label(Some("S-S".to_owned()));
    collapsed
}

#[test]
fn chunking_build_context_matches_opennlp() {
    // `chunking.BuildContextGenerator.getContext(chunks, 1)` without a dictionary: every lexical
    // feature, labels on the nodes before the current one, and the comma before `VP` as `-1=,`.
    let chunks = labeled_chunks("(TOP (S (NP (PRP She)) (, ,) (VP (VBD left))))");
    let ctx = chunking::BuildContextGenerator::new().get_context(&chunks, 1);
    let expected = [
        "default",
        "0=VP|left",
        "0*=VP",
        "-2=eos",
        "-2*=eos",
        "-1=S-S|NP|She",
        "-1*=S-S|NP",
        "1=eos",
        "1*=eos",
        "2=eos",
        "2*=eos",
        // cons2(c0, c1)
        "0=VP|left,1=eos",
        "0*=VP,1=eos",
        "0=VP|left,1*=eos",
        "0*=VP,1*=eos",
        // cons2(c_1, c0) across the comma
        "-1=,",
        "-1=,,0=VP|left",
        "-1=,,0*=VP",
        "-1=S-S|NP|She,-1=,,0=VP|left",
        "-1*=S-S|NP,-1=,,0=VP|left",
        "-1=S-S|NP|She,-1=,,0*=VP",
        "-1*=S-S|NP,-1=,,0*=VP",
        // cons3(c0, c1, c2)
        "0=VP|left,1=eos,2=eos",
        "0*=VP,1=eos,2=eos",
        "0=VP|left,1*=eos,2=eos",
        "0=VP|left,1=eos,2*=eos",
        "0*=VP,1*=eos,2=eos",
        "0*=VP,1=eos,2*=eos",
        "0=VP|left,1*=eos,2*=eos",
        "0*=VP,1*=eos,2*=eos",
        // cons3(c_2, c_1, c0), the comma between c_1 and c0
        "-2=eos,-1=S-S|NP|She,-1=,,0=VP|left",
        "-2*=eos,-1=S-S|NP|She,-1=,,0=VP|left",
        "-2=eos,-1*=S-S|NP,-1=,,0=VP|left",
        "-2=eos,-1=S-S|NP|She,-1=,,0*=VP",
        "-2*=eos,-1*=S-S|NP,-1=,,0=VP|left",
        "-2*=eos,-1=S-S|NP|She,-1=,,0*=VP",
        "-2=eos,-1*=S-S|NP,-1=,,0*=VP",
        "-2*=eos,-1*=S-S|NP,-1=,,0*=VP",
        // cons3(c_1, c0, c1), the comma between c_1 and c0
        "-1=S-S|NP|She,-1=,,0=VP|left,1=eos",
        "-1*=S-S|NP,-1=,,0=VP|left,1=eos",
        "-1=S-S|NP|She,-1=,,0*=VP,1=eos",
        "-1=S-S|NP|She,-1=,,0=VP|left,1*=eos",
        "-1*=S-S|NP,-1=,,0*=VP,1=eos",
        "-1*=S-S|NP,-1=,,0=VP|left,1*=eos",
        "-1=S-S|NP|She,-1=,,0*=VP,1*=eos",
        "-1*=S-S|NP,-1=,,0*=VP,1*=eos",
    ];
    assert_eq!(ctx, expected);
}

#[test]
fn chunking_check_context_matches_opennlp() {
    // `chunking.CheckContextGenerator.getContext(chunks, "S", 0, 1)`.
    let chunks = labeled_chunks("(TOP (S (NP (DT The) (NN dog)) (VP (VBD ran))))");
    let ctx = chunking::CheckContextGenerator::new().get_context(&chunks, "S", 0, 1);
    let expected = [
        "default",
        "fl=S-S",
        "cbegin=NP|dog|S",
        "cbegin*=NP|S",
        "clast=VP|ran|S",
        "clast*=VP|S",
        "cil=S,NP|dog,VP|ran",
        "ci*l=S,NP,VP|ran",
        "cil*=S,NP|dog,VP",
        "ci*l*=S,NP,VP",
        "p=S->NP,VP",
        "pp=S->NP,VP",
        "s-1=S|eos",
        "s-1*=S|eos",
        "s-2=S|eos",
        "s-2*=S|eos",
        "s1=S|eos",
        "s1*=S|eos",
        "s2=S|eos",
        "s2*=S|eos",
    ];
    assert_eq!(ctx, expected);
}

#[test]
fn collapse_punctuation_and_build_dictionary() {
    let rules = english_head_rules();
    let p = Parse::parse("(TOP (S (NP (PRP She)) (, ,) (VP (VBD left)) (. .)))").unwrap();
    let chunks: Vec<Parse> = p.initial_chunks().into_iter().cloned().collect();
    let collapsed = collapse_punctuation(&chunks, rules.punctuation_tags());
    assert_eq!(collapsed.len(), 2);
    assert_eq!(collapsed[0].kind(), "NP");
    assert_eq!(collapsed[1].kind(), "VP");
    assert_eq!(collapsed[0].next_punctuation().len(), 1);
    assert_eq!(collapsed[0].next_punctuation()[0].kind(), ",");
    assert_eq!(collapsed[1].previous_punctuation().len(), 1);
    assert_eq!(collapsed[1].previous_punctuation()[0].kind(), ",");

    let samples = sample_parses();
    let dict = build_dictionary(&samples, &rules, 1);
    assert!(dict.contains(&dictionary_key(&["fox"])));
    assert!(dict.contains(&dictionary_key(&["dog"])));
    // A token with a comma is not the bigram of its halves.
    assert_ne!(dictionary_key(&["1,000"]), dictionary_key(&["1", "000"]));
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
    assert!(ti_build.iter().any(|e| e.outcome() == treeinsert::DONE));

    let ti_attach =
        treeinsert::ParserEventStream::generate_events(&samples, &rules, ParserEventType::Attach, None).unwrap();
    assert!(!ti_attach.is_empty());
    assert!(
        ti_attach
            .iter()
            .any(|e| e.outcome() == treeinsert::ATTACH_SISTER || e.outcome() == treeinsert::ATTACH_DAUGHTER)
    );

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
fn chunking_parser_trains_serializes_and_parses() {
    let rules = HeadRulesArtifact::English(english_head_rules());
    let samples = sample_parses();
    let params = ParserTrainingParameters::with_iterations_and_cutoff(50, 1);

    let model = chunking::Parser::train("en", &samples, rules, &params).expect("train chunking parser");
    assert_eq!(model.parser_type(), ParserType::Chunking);
    assert_eq!(model.language(), "en");
    assert!(model.attach_model().is_none());

    // Round-trip model through bytes
    let bytes = model.as_bytes().to_vec();
    let reloaded = ParserModel::from_bytes(bytes).expect("reload chunking model");
    assert_eq!(reloaded.parser_type(), ParserType::Chunking);
    assert_eq!(reloaded.sha256(), model.sha256());

    let parser = ParserFactory::create(Arc::new(reloaded));
    let flat = Parse::from_tokens(&["The", "quick", "fox", "jumped", "."]).unwrap();
    let parsed = parser.parse(&flat).unwrap().expect("produce parse");
    assert_eq!(parsed.kind(), "TOP");
    assert!(parsed.complete());
    assert_eq!(parsed.text(), "The quick fox jumped .");

    // Top-k parses
    let top2 = parser.parse_n(&flat, 2).unwrap();
    assert!(!top2.is_empty());
    if top2.len() == 2 {
        assert!(top2[0].prob() >= top2[1].prob());
    }
}

#[test]
fn treeinsert_parser_trains_serializes_and_parses() {
    let rules = HeadRulesArtifact::English(english_head_rules());
    // OpenNLP's parser reads the attach model's sister outcome, so the corpus needs a constituent
    // with a child of its own type for sister adjunction to be trained.
    let mut samples = sample_parses();
    samples.push(
        Parse::parse(
            "(TOP (S (NP-SBJ (NP (DT The) (NN dog)) (PP (IN in) (NP (DT the) (NN yard)))) (VP (VBD barked)) (. .)))",
        )
        .unwrap(),
    );
    let params = ParserTrainingParameters::with_iterations_and_cutoff(50, 1);

    let model = treeinsert::Parser::train("en", &samples, rules, &params).expect("train treeinsert parser");
    assert_eq!(model.parser_type(), ParserType::TreeInsert);
    assert_eq!(model.language(), "en");
    assert!(model.attach_model().is_some());

    let bytes = model.as_bytes().to_vec();
    let reloaded = ParserModel::from_bytes(bytes).expect("reload treeinsert model");
    assert_eq!(reloaded.parser_type(), ParserType::TreeInsert);

    let parser = ParserFactory::create(Arc::new(reloaded));
    let flat = Parse::from_tokens(&["The", "lazy", "dog", "slept", "."]).unwrap();
    let parsed = parser.parse(&flat).unwrap().expect("produce treeinsert parse");
    assert_eq!(parsed.kind(), "TOP");
    assert!(parsed.complete());
    assert_eq!(parsed.text(), "The lazy dog slept .");
}

#[test]
fn parser_evaluator_constituent_spans_and_evaluation() {
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

    // Evaluate trained model on training corpus
    let rules = HeadRulesArtifact::English(english_head_rules());
    let samples = sample_parses();
    let params = ParserTrainingParameters::with_iterations_and_cutoff(50, 1);
    let model = Arc::new(chunking::Parser::train("en", &samples, rules.clone(), &params).unwrap());
    let engine = ParserFactory::create(model);
    let mut evaluator = ParserEvaluator::new(&engine);
    evaluator.evaluate(&samples[..4]).unwrap();
    let fm = evaluator.fmeasure();
    assert!(fm.precision() > 0.0);
    assert!(fm.recall() > 0.0);
    assert!(fm.f_measure() > 0.0);

    // Cross-validator
    let mut cv = ParserCrossValidator::new("en", params, ParserType::Chunking);
    cv.evaluate(&samples[..6], 2, &rules).expect("cross-validate");
    assert!(cv.fmeasure().f_measure() >= 0.0);
}

#[test]
fn parser_annotator_and_parser_tool() {
    let rules = HeadRulesArtifact::English(english_head_rules());
    let samples = sample_parses();
    let params = ParserTrainingParameters::with_iterations_and_cutoff(50, 1);
    let model = Arc::new(chunking::Parser::train("en", &samples, rules, &params).unwrap());

    // Test ParserTool::parse_line with untokenized parens
    let engine = ParserFactory::create(Arc::clone(&model));
    let line = "The quick (fox) jumped .";
    let parsed_lines = ParserTool::parse_line(line, &engine, 1).unwrap();
    assert_eq!(parsed_lines.len(), 1);
    assert_eq!(parsed_lines[0].text(), line);

    // Test ParserAnnotator on Document
    let doc_text = "The quick fox jumped.";
    let mut doc = Document::new(doc_text);
    let view_id = ViewId(0);
    let mut sent_spans = Spans::<Sentences>::new(view_id);
    sent_spans.push(Span::new(0, 21));
    doc.add(&Key::<Sentences>::DEFAULT, Sentences { spans: sent_spans }, Provenance::by("test")).unwrap();

    let mut tok_spans = Spans::<Tokens>::new(view_id);
    tok_spans.push(Span::new(0, 3)); // The
    tok_spans.push(Span::new(4, 9)); // quick
    tok_spans.push(Span::new(10, 13)); // fox
    tok_spans.push(Span::new(14, 20)); // jumped
    tok_spans.push(Span::new(20, 21)); // .
    doc.add(
        &Key::<Tokens>::DEFAULT,
        Tokens { spans: tok_spans, sentence: Some(vec![Ix::new(0); 5]) },
        Provenance::by("test"),
    )
    .unwrap();

    let annotator = ParserAnnotator::new(model);
    let analyzer = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .add(annotator)
        .build()
        .unwrap();
    assert_eq!(analyzer.run(std::slice::from_mut(&mut doc)), [Ok(())]);

    let phrases = doc.get::<Phrases>().unwrap();
    assert!(!phrases.spans.is_empty());
    let tokens = doc.get::<Tokens>().unwrap();
    for i in 0..phrases.spans.len() {
        let ix = Ix::<Phrases>::new(i as u32);
        let head_sp = phrases.head_span(ix, tokens);
        let phrase_sp = phrases.spans.get(ix);
        assert!(phrase_sp.contains(head_sp));
    }
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

#[test]
fn chunk_context_cache_does_not_leak_into_another_sentence() {
    // The second vector can be allocated where the first one was; its words must still be used.
    let mut cg = ChunkContextGenerator::with_cache(10);
    let tags = ["DT", "NN"];
    let preds = ["S-NP", "C-NP"];
    let first = {
        let words: Vec<&str> = vec!["The", "dog"];
        cg.get_context(0, &words, &tags, &preds)
    };
    assert!(first.iter().any(|f| f == "0=The|DT"));
    let words: Vec<&str> = vec!["A", "cat"];
    let second = cg.get_context(0, &words, &tags, &preds);
    assert!(second.iter().any(|f| f == "0=A|DT"), "{second:?}");
    assert!(second.iter().any(|f| f == "1=cat|NN"), "{second:?}");
}
