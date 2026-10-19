//! Unit tests for `turbo-parser`'s `CHUNKING` and `TREEINSERT` context generators, the chunker's
//! sequence validator, punctuation collapsing and the n-gram dictionary. Training, the event
//! streams, `ParserEvaluator`, `ParserCrossValidator` and the parsers run on models trained from a
//! corpus are tested in `turbo-train-components`' `tests/parser.rs`.

use std::str::FromStr;

use turbo_parser::{
    ChunkContextGenerator, EnglishHeadRules, HeadRules, Parse, ParseSampleStream, ParserChunkerSequenceValidator,
    build_dictionary, chunking, collapse_punctuation, dictionary_key,
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

/// With two punctuation nodes after a constituent, which only happens when their probabilities
/// differ, the first surround feature of the second starts with the last feature of the first:
/// OpenNLP's `surround` does not clear its `StringBuilder` before it.
#[test]
fn surround_features_after_two_punctuation_nodes_are_written_as_java_writes_them() {
    let rules = english_head_rules();
    let p = Parse::parse("(TOP (S (NP (DT The) (NN soup)) (VP (VBZ is)) (, ,) ('' '') (NP (PRP she))))").unwrap();
    let mut chunks: Vec<Parse> = p.initial_chunks().into_iter().cloned().collect();
    // The tagger's probabilities: `''` above `,`, so the set holds both, `''` first.
    chunks[2].add_prob(0.5);
    chunks[3].add_prob(0.9);
    let mut chunks = collapse_punctuation(&chunks, rules.punctuation_tags());
    for (c, label) in chunks.iter_mut().zip(["S-S", "C-S", "S-NP"]) {
        c.set_label(Some(label.to_owned()));
    }
    let ctx = chunking::CheckContextGenerator::new().get_context(&chunks, "VP", 1, 1);
    let surround: Vec<&str> = ctx.iter().map(String::as_str).filter(|f| f.starts_with("s1")).collect();
    assert_eq!(
        surround,
        ["s1=she|VP|NP|''", "s1*=VP|NP|''", "s1*=VP|''", "s1*=VP|''she|VP|NP|,", "s1*=VP|NP|,", "s1*=VP|,"]
    );
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
