//! Tree-insertion event stream and context generators against OpenNLP's
//! `opennlp.tools.parser.treeinsert` feature strings. Each expected list was worked out by hand
//! from the OpenNLP sources (`treeinsert.ParserEventStream.addParseEvents`, the three
//! `treeinsert` context generators and `AbstractContextGenerator`), for the gold parse
//! `(TOP (S (NP (DT The) (NN dog)) (VP (VBD barked)) (. .)))` with the English head rules:
//! the `NP` is headed by `dog` (token 1), the `VP` and the full `S` by `barked` (token 2), and
//! the final `.` is collapsed into the next-punctuation set of the `VP` chunk.

use std::str::FromStr;

use turbo_parser::{EnglishHeadRules, Parse, ParserEventType, treeinsert};
use turbo_train::Event;

fn english_head_rules() -> EnglishHeadRules {
    let path = turbo_goldens::data_dir().join("fixtures/en-head-rules.txt");
    EnglishHeadRules::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

const GOLD: &str = "(TOP (S (NP (DT The) (NN dog)) (VP (VBD barked)) (. .)))";

fn events(etype: ParserEventType) -> Vec<(String, Vec<String>)> {
    let rules = english_head_rules();
    let gold = Parse::parse(GOLD).unwrap();
    let events = treeinsert::ParserEventStream::generate_events(&[gold], &rules, etype, None).unwrap();
    events.iter().map(owned).collect()
}

fn owned(e: &Event) -> (String, Vec<String>) {
    let s = |j: &turbo_train::JavaString| j.to_str().unwrap().into_owned();
    (s(e.outcome()), e.context().iter().map(s).collect())
}

fn strs(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| (*s).to_owned()).collect()
}

/// `BuildContextGenerator` features for the chunk at index 0 when the next constituent is the
/// `VP` chunk followed by `.`, with `kind` the type of the constituent at index 0 (the `NP` chunk,
/// then the `S` built over it, both headed by `dog`). Index 0 has an empty right frontier, so the
/// left context is `eos` and the `eos,` features close the list.
fn build_at_zero(kind: &str) -> Vec<String> {
    let c0 = format!("0={kind}|dog");
    let c0b = format!("0*={kind}");
    let mut f = strs(&["default", "-2=eos", "-2*=eos", "-1=eos", "-1*=eos"]);
    f.extend([c0.clone(), c0b.clone()]);
    f.extend(strs(&["1=VP|barked", "1*=VP", "2=eos", "2*=eos"]));
    // cons2(c0, c1, null)
    f.extend([
        format!("{c0},1=VP|barked"),
        format!("{c0b},1=VP|barked"),
        format!("{c0},1*=VP"),
        format!("{c0b},1*=VP"),
    ]);
    // cons2(c_1, c0, null)
    f.extend([format!("-1=eos,{c0}"), format!("-1*=eos,{c0}"), format!("-1=eos,{c0b}"), format!("-1*=eos,{c0b}")]);
    // cons3(c0, c1, c2, null, {.}): punctbo(2) first, then the eight features with ",2=.,"
    f.push("2=.".to_owned());
    f.extend([
        format!("{c0},1=VP|barked,2=.,2=eos"),
        format!("{c0b},1=VP|barked,2=.,2=eos"),
        format!("{c0},1*=VP,2=.,2=eos"),
        format!("{c0},1=VP|barked,2=.,2*=eos"),
        format!("{c0b},1*=VP,2=.,2=eos"),
        format!("{c0b},1=VP|barked,2=.,2*=eos"),
        format!("{c0},1*=VP,2=.,2*=eos"),
        format!("{c0b},1*=VP,2=.,2*=eos"),
    ]);
    // cons3(c_2, c_1, c0, null, null)
    f.extend([
        format!("-2=eos,-1=eos,{c0}"),
        format!("-2*=eos,-1=eos,{c0}"),
        format!("-2=eos,-1*=eos,{c0}"),
        format!("-2=eos,-1=eos,{c0b}"),
        format!("-2*=eos,-1*=eos,{c0}"),
        format!("-2*=eos,-1=eos,{c0b}"),
        format!("-2=eos,-1*=eos,{c0b}"),
        format!("-2*=eos,-1*=eos,{c0b}"),
    ]);
    // cons3(c_1, c0, c1, null, null)
    f.extend([
        format!("-1=eos,{c0},1=VP|barked"),
        format!("-1*=eos,{c0},1=VP|barked"),
        format!("-1=eos,{c0b},1=VP|barked"),
        format!("-1=eos,{c0},1*=VP"),
        format!("-1*=eos,{c0b},1=VP|barked"),
        format!("-1*=eos,{c0},1*=VP"),
        format!("-1=eos,{c0b},1*=VP"),
        format!("-1*=eos,{c0b},1*=VP"),
    ]);
    f.extend([format!("eos,{c0}"), format!("eos,{c0b}")]);
    f
}

#[test]
fn build_events_match_opennlp() {
    let events = events(ParserEventType::Build);
    let outcomes: Vec<&str> = events.iter().map(|(o, _)| o.as_str()).collect();
    // Build S over the NP chunk, done with it, then done with the VP chunk.
    assert_eq!(outcomes, ["S", "d", "d"]);
    assert_eq!(events[0].1, build_at_zero("NP"));
    assert_eq!(events[1].1, build_at_zero("S"));

    // The VP chunk at index 1. The left context comes from the right frontier of the parse built
    // so far, [NP, S]: both are headed by token 1, so getFrontierNodes keeps the NP alone and
    // the -2 slot is eos. Its punctuation (`.` after the VP) uses punctbo index 1.
    let expected = strs(&[
        "default",
        "-2=eos",
        "-2*=eos",
        "-1=NP|dog",
        "-1*=NP",
        "0=VP|barked",
        "0*=VP",
        "1=eos",
        "1*=eos",
        "2=eos",
        "2*=eos",
        // cons2(c0, c1, {.})
        "1=.",
        "0=VP|barked,1=.",
        "0*=VP,1=.",
        "0=VP|barked,1=.,1=eos",
        "0*=VP,1=.,1=eos",
        "0=VP|barked,1=.,1*=eos",
        "0*=VP,1=.,1*=eos",
        // cons2(c_1, c0, null)
        "-1=NP|dog,0=VP|barked",
        "-1*=NP,0=VP|barked",
        "-1=NP|dog,0*=VP",
        "-1*=NP,0*=VP",
        // cons3(c0, c1, c2, {.}, null)
        "0=VP|barked,1=.,1=eos,2=eos",
        "0*=VP,1=.,1=eos,2=eos",
        "0=VP|barked,1=.,1*=eos,2=eos",
        "0=VP|barked,1=.,1=eos,2*=eos",
        "0*=VP,1=.,1*=eos,2=eos",
        "0*=VP,1=.,1=eos,2*=eos",
        "0=VP|barked,1=.,1*=eos,2*=eos",
        "0*=VP,1=.,1*=eos,2*=eos",
        // cons3(c_2, c_1, c0, null, null)
        "-2=eos,-1=NP|dog,0=VP|barked",
        "-2*=eos,-1=NP|dog,0=VP|barked",
        "-2=eos,-1*=NP,0=VP|barked",
        "-2=eos,-1=NP|dog,0*=VP",
        "-2*=eos,-1*=NP,0=VP|barked",
        "-2*=eos,-1=NP|dog,0*=VP",
        "-2=eos,-1*=NP,0*=VP",
        "-2*=eos,-1*=NP,0*=VP",
        // cons3(c_1, c0, c1, null, null)
        "-1=NP|dog,0=VP|barked,1=eos",
        "-1*=NP,0=VP|barked,1=eos",
        "-1=NP|dog,0*=VP,1=eos",
        "-1=NP|dog,0=VP|barked,1*=eos",
        "-1*=NP,0*=VP,1=eos",
        "-1*=NP,0=VP|barked,1*=eos",
        "-1=NP|dog,0*=VP,1*=eos",
        "-1*=NP,0*=VP,1*=eos",
    ]);
    assert_eq!(events[2].1, expected);
}

#[test]
fn check_events_match_opennlp() {
    let events = events(ParserEventType::Check);
    let outcomes: Vec<&str> = events.iter().map(|(o, _)| o.as_str()).collect();
    // The S built over the NP is incomplete (the VP follows in the gold S); after the VP is
    // attached as its daughter it is complete.
    assert_eq!(outcomes, ["i", "c"]);
    assert_eq!(
        events[0].1,
        strs(&[
            "default",
            "cbegin=NP|dog|S",
            "cbegin*=NP|S",
            "clast=NP|dog|S",
            "clast*=NP|S",
            "p=S->NP",
            "pp=S->NP",
            "s-1=S|eos",
            "s-1*=S|eos",
            "s-2=S|eos",
            "s-2*=S|eos",
            "s1=barked|S|VP",
            "s1*=S|VP",
            "s2=S|eos|.",
            "s2*=S|eos|.",
            "s2*=S|.",
        ])
    );
    // Index 1 with the frontier trimmed below the S: nothing is left of it, and the `.` after the
    // VP is the punctuation of the eos at position 1.
    assert_eq!(
        events[1].1,
        strs(&[
            "default",
            "cbegin=NP|dog|S",
            "cbegin*=NP|S",
            "clast=VP|barked|S",
            "clast*=VP|S",
            "p=S->NP,VP",
            "pp=S->NP,VP",
            "s-1=S|eos",
            "s-1*=S|eos",
            "s-2=S|eos",
            "s-2*=S|eos",
            "s1=S|eos|.",
            "s1*=S|eos|.",
            "s1*=S|.",
            "s2=S|eos",
            "s2*=S|eos",
        ])
    );
}

/// `AttachContextGenerator` features for attaching the VP chunk to frontier node `rfi` of the
/// frontier [NP, S], `fnode`/`fp` being the cons strings of that node and the one above it.
fn attach_vp(rfi: usize, fnode: (&str, &str), fp: (&str, &str), prod: &str) -> Vec<String> {
    let (cf, cfb) = fnode;
    let (cfp, cfpb) = fp;
    let ftype = cfb.trim_start_matches("-2*=");
    let mut f = strs(&["default"]);
    f.extend(strs(&[cfp, cfpb, cf, cfb, "-1=NP|dog", "-1*=NP", "0=VP|barked", "0*=VP", "1=eos", "1*=eos"]));
    f.extend([format!("pn={prod}"), format!("pd={prod},VP"), format!("ps={ftype}->{ftype},VP")]);
    for (c, cb) in [(cfp, cfpb), (cf, cfb), ("-1=NP|dog", "-1*=NP")] {
        f.extend([
            format!("{c},0=VP|barked"),
            format!("{cb},0=VP|barked"),
            format!("{c},0*=VP"),
            format!("{cb},0*=VP"),
        ]);
    }
    f.extend(strs(&[
        "1=.",
        "0=VP|barked,1=.",
        "0*=VP,1=.",
        "0=VP|barked,1=.,1=eos",
        "0*=VP,1=.,1=eos",
        "0=VP|barked,1=.,1*=eos",
        "0*=VP,1=.,1*=eos",
    ]));
    // cons3(cf, c_1, c0, null, null)
    f.extend([
        format!("{cf},-1=NP|dog,0=VP|barked"),
        format!("{cfb},-1=NP|dog,0=VP|barked"),
        format!("{cf},-1*=NP,0=VP|barked"),
        format!("{cf},-1=NP|dog,0*=VP"),
        format!("{cfb},-1*=NP,0=VP|barked"),
        format!("{cfb},-1=NP|dog,0*=VP"),
        format!("{cf},-1*=NP,0*=VP"),
        format!("{cfb},-1*=NP,0*=VP"),
    ]);
    // cons3(cf, c0, c1, null, {.}): the `.` is the punctuation before c1 (index 1).
    f.extend([
        format!("{cf},0=VP|barked,1=.,1=eos"),
        format!("{cfb},0=VP|barked,1=.,1=eos"),
        format!("{cf},0*=VP,1=.,1=eos"),
        format!("{cf},0=VP|barked,1=.,1*=eos"),
        format!("{cfb},0*=VP,1=.,1=eos"),
        format!("{cfb},0=VP|barked,1=.,1*=eos"),
        format!("{cf},0*=VP,1=.,1*=eos"),
        format!("{cfb},0*=VP,1=.,1*=eos"),
    ]);
    // cons3(cfp, cf, c0, null, null)
    f.extend([
        format!("{cfp},{cf},0=VP|barked"),
        format!("{cfpb},{cf},0=VP|barked"),
        format!("{cfp},{cfb},0=VP|barked"),
        format!("{cfp},{cf},0*=VP"),
        format!("{cfpb},{cfb},0=VP|barked"),
        format!("{cfpb},{cf},0*=VP"),
        format!("{cfp},{cfb},0*=VP"),
        format!("{cfpb},{cfb},0*=VP"),
    ]);
    // Head distance: token 2 (barked) minus token 1 (dog).
    f.extend(["hd=1".to_owned(), format!("nd={rfi}"), format!("nd=VP.{rfi}"), "hd=VP.1".to_owned()]);
    f
}

#[test]
fn attach_events_match_opennlp() {
    let events = events(ParserEventType::Attach);
    let outcomes: Vec<&str> = events.iter().map(|(o, _)| o.as_str()).collect();
    // The daughter pass finds the S (frontier index 1) first; the sister pass then emits a
    // non-attach event for the NP (index 0) and skips the S.
    assert_eq!(outcomes, ["d", "n"]);
    assert_eq!(events[0].1, attach_vp(1, ("-2=S|dog", "-2*=S"), ("-3=eos", "-3*=eos"), "S->NP"));
    assert_eq!(events[1].1, attach_vp(0, ("-2=NP|dog", "-2*=NP"), ("-3=S|dog", "-3*=S"), "NP->DT,NN"));
}

#[test]
fn right_frontier_starts_at_first_non_punctuation_child_of_top() {
    let rules = english_head_rules();
    let mut p =
        Parse::parse("(TOP (`` ``) (S (NP (DT The) (NN dog)) (VP (VBD barked) (NP (NNS cats)))) ('' ''))").unwrap();
    p.update_heads(&rules);
    let rf = treeinsert::Parser::get_right_frontier(&p, turbo_parser::HeadRules::punctuation_tags(&rules));
    let kinds: Vec<&str> = rf.iter().map(|n| n.kind()).collect();
    // Deepest first, ending at the S, part-of-speech nodes left out.
    assert_eq!(kinds, ["NP", "VP", "S"]);
}

fn trained_parser() -> treeinsert::Parser {
    let rules = turbo_parser::HeadRulesArtifact::English(english_head_rules());
    let samples: Vec<Parse> = [
        GOLD,
        "(TOP (S (NP (DT The) (NN cat)) (VP (VBD sat) (PP (IN on) (NP (DT the) (NN mat)))) (. .)))",
        "(TOP (S (NP (NP (DT The) (NN dog)) (PP (IN in) (NP (DT the) (NN yard)))) (VP (VBD barked)) (. .)))",
    ]
    .iter()
    .map(|s| Parse::parse(s).unwrap())
    .collect();
    let params = turbo_parser::ParserTrainingParameters::with_iterations_and_cutoff(10, 0);
    let model = treeinsert::Parser::train("en", &samples, rules, &params).unwrap();
    treeinsert::Parser::new(model)
}

/// The `INC` node of a bracketed `(INC ...)` parse, with heads.
fn inc_parse(bracketed: &str) -> Parse {
    let mut p = Parse::parse(bracketed).unwrap().children()[0].clone();
    p.update_heads(&english_head_rules());
    p
}

#[test]
fn single_part_of_speech_node_is_left_to_the_search_loop() {
    // `advanceParses` returns null: the search loop completes the parse as it is.
    let parser = trained_parser();
    let mut p = inc_parse("(INC (NN dog) (. .))");
    assert_eq!(parser.advance_parses(&mut p, 0.95).unwrap(), None);
}

#[test]
fn single_constituent_takes_the_punctuation_around_it() {
    // `expandTopNode`: the `.` moves into the NP and the parse is complete.
    let parser = trained_parser();
    let mut p = inc_parse("(INC (`` ``) (NP (DT The) (NN dog)) (. .))");
    let next = parser.advance_parses(&mut p, 0.95).unwrap().unwrap();
    assert_eq!(next.len(), 1);
    assert!(next[0].complete());
    let np = &next[0].children()[0];
    let kinds: Vec<&str> = np.children().iter().map(Parse::kind).collect();
    assert_eq!(kinds, ["``", "DT", "NN", "."]);
}

#[test]
fn chunks_are_marked_complete_after_chunking() {
    let mut parses = vec![inc_parse("(INC (NP (DT The) (NN dog)) (VBD barked))")];
    treeinsert::Parser::mark_chunks_complete(&mut parses);
    let labels: Vec<Option<&str>> = parses[0].children().iter().map(Parse::label).collect();
    assert_eq!(labels, [Some("c"), Some("c")]);
}

#[test]
fn every_derivation_keeps_the_sentence_and_attaches_without_punctuation_left_behind() {
    // After the first constituent is built, an attach derivation moves the next constituent and
    // the punctuation before it into the parse built so far (`remove(originalZeroIndex + 1)` up
    // to the advanced node), so no derivation keeps a top-level mark between them.
    let parser = trained_parser();
    let mut p = inc_parse("(INC (NP (DT The) (NN dog)) (, ,) (VP (VBD barked)) (. .))");
    p.children_mut()[0].set_label(Some(treeinsert::BUILT.to_owned()));
    let derivations = parser.advance_parses(&mut p, 0.95).unwrap().unwrap();
    assert!(!derivations.is_empty());
    for d in derivations {
        let kinds: Vec<&str> = d.children().iter().map(Parse::kind).collect();
        if kinds.len() == 4 {
            // A build derivation: a new node over the VP, nothing removed.
            assert_eq!(d.children()[2].children()[0].kind(), "VP");
        } else {
            // An attach derivation: the `,` and the VP are now inside the NP subtree.
            assert_eq!(kinds, ["NP", "."]);
            assert_eq!(d.children()[0].covered_text(), "The dog , barked");
            assert!(d.children()[0].children().iter().any(|c| c.kind() == ","));
        }
    }
}
