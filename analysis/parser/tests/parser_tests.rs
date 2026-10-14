//! Behaviour of the parse tree, its mutations and the head rules that the goldens
//! (`tests/goldens.rs`) do not reach: the tree operations the bottom-up parsers use, the head a
//! node keeps when the tree is rearranged, the punctuation sets, the regular expression subset of
//! the AnCora rules, and the errors.

use std::str::FromStr;

use turbo_analysis::Span;
use turbo_parser::{
    AncoraSpanishHeadRules, EnglishHeadRules, Head, HeadRules, HeadRulesError, INC_NODE, Parse, ParseError,
    ParseSampleStream, ParserChunkSample, TOK_NODE, TagPattern, decode_token, encode_token, space_untokenized_parens,
};
use turbo_unicode::WhitespaceMode;

fn english() -> EnglishHeadRules {
    let path = turbo_goldens::data_dir().join("fixtures/en-head-rules.txt");
    EnglishHeadRules::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn spanish() -> AncoraSpanishHeadRules {
    let path = turbo_goldens::data_dir().join("fixtures/es-head-rules.txt");
    AncoraSpanishHeadRules::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn show(p: &Parse) -> String {
    p.to_string_penn_treebank()
}

/// The node at `path`, by child indices from `p`.
fn at<'a>(p: &'a Parse, path: &[usize]) -> &'a Parse {
    path.iter().fold(p, |n, &i| &n.children()[i])
}

#[test]
fn reads_and_writes_a_tree() {
    let p = Parse::parse("(TOP (S (NP (DT The) (NN cat)) (VP (VBD sat)) (. .)))").unwrap();
    assert_eq!(p.text(), "The cat sat . ");
    assert_eq!(p.span(), Span::new(0, 14));
    assert_eq!(show(&p), "(TOP (S (NP (DT The) (NN cat)) (VP (VBD sat)) (. .)) )");
    assert_eq!(p.to_string(), "The cat sat . ");
    let again = Parse::parse(&show(&p)).unwrap();
    assert_eq!(again, p);
    assert_eq!(at(&p, &[0, 0, 1]).covered_text(), "cat");
    assert_eq!(at(&p, &[0, 0, 1]).span(), Span::new(4, 7));
    let tokens: Vec<i32> = p.token_nodes().iter().map(|t| t.head_index()).collect();
    assert_eq!(tokens, [0, 1, 2, 3]);
}

#[test]
fn reads_what_parse_parse_reads() {
    // The type runs to a space, `=` or `-`, so `((S` gives the type `(S`.
    let p = Parse::parse("((S (NP (NN dog))))").unwrap();
    assert_eq!(show(&p), "(TOP ((S (S (NP (NN dog)))) )");
    // A label followed by two spaces holds no token; text between brackets is ignored.
    let p = Parse::parse("(TOP (NN  spaced) stray (NN ok) more)").unwrap();
    assert_eq!(p.text(), "ok ");
    // A token runs to a space or bracket, so it may hold a tab or vertical tab.
    let p = Parse::parse("(TOP (NN a\tb) (NN c\u{b}) (NN d \u{b}))").unwrap();
    assert_eq!(p.text(), "a\tb c\u{b} d ");
    // Brackets still open at the end are dropped, and two roots both go under TOP.
    assert_eq!(show(&Parse::parse("(TOP (S (NN x)").unwrap()), "(TOP (NN x) )");
    assert_eq!(show(&Parse::parse("(A (B x)) (C (D y))").unwrap()), "(TOP (A (B x)) (C (D y)) )");
    // Escaped types and -NONE- are taken alone.
    let p = Parse::parse("(TOP (-NONE-1 *T*) (-LRB-X -LRB-))").unwrap();
    assert_eq!(show(&p), "(TOP (-NONE- *T*) (-LRB- -LRB-) )");
    assert_eq!(p.text(), "*T* ( ");
}

#[test]
fn keeps_the_first_function_tag() {
    let cases =
        [("NP-SBJ-1", "NP-SBJ"), ("NP-SBJ-TMP", "NP-SBJ"), ("WHNP-1", "WHNP-1"), ("NP=2", "NP"), ("NP--X", "NP")];
    for (label, expected) in cases {
        let p = Parse::parse_with_options(&format!("(TOP ({label} (NN word)))"), true, None).unwrap();
        assert_eq!(p.children()[0].kind(), expected, "{label}");
        let p = Parse::parse(&format!("(TOP ({label} (NN word)))")).unwrap();
        assert_eq!(p.children()[0].kind(), label.split(['-', '=']).next().unwrap(), "{label}");
    }
}

#[test]
fn fails_where_parse_parse_throws() {
    let missing = Parse::parse("( (S (NN rain)) )").unwrap_err();
    assert_eq!(missing, ParseError::MissingType { byte_offset: 17 });
    assert_eq!(missing.java_class(), Some("java.lang.NullPointerException"));
    let unmatched = Parse::parse("(TOP (NN a)))").unwrap_err();
    assert_eq!(unmatched, ParseError::UnmatchedClose { byte_offset: 12 });
    assert_eq!(unmatched.java_class(), Some("java.util.EmptyStackException"));
    // A bracket without a type that never closes around a token is harmless.
    assert_eq!(show(&Parse::parse("(TOP (NN a) ( ))").unwrap()), "(TOP (NN a) )");
}

#[test]
fn labels_gaps_and_fails_on_a_missing_label() {
    let en = english();
    let p = Parse::parse_with_gaps(
        "(TOP (SBAR (WHNP (WP what)) (S (NP (PRP she)) (VP (VBD saw) (NP (-NONE- *T*))))))",
        &en,
    )
    .unwrap();
    assert_eq!(show(&p), "(TOP (SBAR-G (WHNP (WP what)) (S-G (NP (PRP she)) (VP-G (VBD saw)))) )");
    // The labeler compares the second constituent from the top first; a bracket without a type
    // there is dereferenced, as in Java.
    let err = Parse::parse_with_gaps("(TOP (A (B (C ( (-NONE- *))))))", &en).unwrap_err();
    assert_eq!(err, ParseError::MissingType { byte_offset: 16 });
    let es = spanish();
    assert!(Parse::parse_with_gaps("(TOP (S ( (SN (-NONE- *))) (NN x)))", &es).is_err());
    assert!(Parse::parse_with_gaps("(TOP (S ( (NP (-NONE- *))) (NN x)))", &en).is_err());
    // The first comparison fails under the Spanish labels, so the missing type is never read.
    let p = Parse::parse_with_gaps("(TOP (S ( (NP (-NONE- *))) (NN x)))", &es).unwrap();
    assert_eq!(show(&p), "(TOP (S (NN x)) )");
    // Under the Spanish labels the chain of the English example is not a gap.
    let p = Parse::parse_with_gaps("(TOP (A (B (NP (-NONE- *)))) (NN x))", &es).unwrap();
    assert_eq!(show(&p), "(TOP (NN x) )");
}

#[test]
fn equality_ignores_the_type_as_java_does() {
    let a = Parse::parse("(TOP (NP (NN x)))").unwrap();
    let b = Parse::parse("(TOP (VP (NN x)))").unwrap();
    assert_eq!(a, b);
    let c = Parse::parse("(TOP (NP (NN y)))").unwrap();
    assert_ne!(a, c);
}

#[test]
fn creates_the_initial_tree_from_tokens() {
    let p = Parse::from_tokens(&["The", "cat", "sat"]).unwrap();
    assert_eq!(p.text(), "The cat sat");
    assert_eq!(p.kind(), INC_NODE);
    let kids: Vec<(&str, Span, i32)> = p.children().iter().map(|c| (c.kind(), c.span(), c.head_index())).collect();
    assert_eq!(kids, [(TOK_NODE, Span::new(0, 3), 0), (TOK_NODE, Span::new(4, 7), 1), (TOK_NODE, Span::new(8, 11), 2)]);
    assert_eq!(Parse::from_tokens(&[] as &[&str]).unwrap_err(), ParseError::EmptyTokens);
}

#[test]
fn bracket_tokens_round_trip() {
    for tok in ["(", ")", "{", "}", "[", "]", "word"] {
        assert_eq!(decode_token(encode_token(tok)), tok);
    }
    let p = Parse::parse("(TOP (-LRB- -LRB-) (NN 🐈) (-RRB- -RRB-))").unwrap();
    assert_eq!(p.text(), "( 🐈 ) ");
    assert_eq!(show(&p), "(TOP (-LRB- -LRB-) (NN 🐈) (-RRB- -RRB-) )");
}

/// Argon's review of #398: heads were child indices and went stale when children moved.
#[test]
fn heads_survive_rearranging_the_tree() {
    let en = english();
    let mut p = Parse::parse("(TOP (QQ (DT a) (NN b) (NN c)))").unwrap();
    p.update_heads(&en);
    Parse::add_names("PERSON", &[Span::new(0, 2)], &mut p).unwrap();
    assert_eq!(show(&p), "(TOP (QQ (PERSON (DT a) (NN b)) (NN c)) )");
    assert_eq!(p.children()[0].head_text(), "c");

    let mut p = Parse::parse("(TOP (S (NP (DT The) (NN cat)) (VP (VBD sat)) (. .)))").unwrap();
    p.update_heads(&en);
    assert_eq!(p.children()[0].head_text(), "sat");
    Parse::add_names("THING", &[Span::new(0, 3)], &mut p).unwrap();
    assert_eq!(show(&p), "(TOP (S (THING (NP (DT The) (NN cat)) (VP (VBD sat))) (. .)) )");
    assert_eq!(p.children()[0].head_text(), "sat");

    let mut p = Parse::parse("(TOP (NP (DT The) (NN cat)) (VP (VBD sat)) (. .))").unwrap();
    p.update_heads(&en);
    p.expand_top_node(1);
    assert_eq!(show(&p), "(TOP (VP (NP (DT The) (NN cat)) (VBD sat) (. .)) )");
    assert_eq!(p.children()[0].span(), Span::new(0, 13));
    assert_eq!(p.children()[0].head_text(), "sat");
}

#[test]
fn a_node_built_with_a_head_takes_that_node() {
    let p = Parse::parse("(TOP (NP (DT The) (NN cat)))").unwrap();
    let cat = at(&p, &[0, 1]);
    let np = Parse::with_head(p.text_arc().clone(), Span::new(0, 7), "NP", 1.0, Some(cat.as_head()));
    assert!(!np.is_own_head());
    assert_eq!(np.head(), Head { span: Span::new(4, 7), head_index: 1 });
    assert_eq!(np.head_text(), "cat");
    let own = Parse::with_head(p.text_arc().clone(), Span::new(0, 3), "DT", 1.0, None);
    assert!(own.is_own_head());
    assert_eq!(own.head_index(), 0);
}

#[test]
fn english_heads_follow_the_child_and_spanish_heads_stop_at_it() {
    let mut en = Parse::parse("(TOP (S (NP (DT the) (NN dog)) (VP (VBZ runs))))").unwrap();
    en.update_heads(&english());
    let s = at(&en, &[0]);
    assert_eq!(s.head_text(), "runs");
    assert_eq!(at(&en, &[0, 1]).head().span, at(&en, &[0, 1, 0]).span());

    let mut es = Parse::parse("(TOP (S (SN (GRUP.NOM (NCMS000 perro))) (GRUP.VERB (VMIP3S0 corre))))").unwrap();
    es.update_heads(&spanish());
    // S's rule finds GRUP.VERB (its regex is `GRUP\\.VERB` in OpenNLP's file, which never matches)
    // only through `SN`, the last tag, and takes SN itself, not SN's head.
    let s = at(&es, &[0]);
    assert_eq!(s.head().span, at(&es, &[0, 0]).span());
    assert!(!s.is_own_head());
    // A part-of-speech node is its own head.
    assert!(at(&es, &[0, 1, 0]).is_own_head());
}

#[test]
fn prunes_unary_chains_but_not_the_root() {
    let mut p = Parse::parse("(TOP (S (NP (NP (NP (NN tea)))) (VP (VP (VBZ cools)))))").unwrap();
    Parse::prune_parse(&mut p).unwrap();
    assert_eq!(show(&p), "(TOP (S (NP (NN tea)) (VP (VBZ cools))) )");
    let mut root = Parse::parse("(TOP (NN x))").unwrap();
    root.set_kind("NN");
    assert_eq!(Parse::prune_parse(&mut root), Err(ParseError::MissingParent));
}

#[test]
fn fixes_possessives_reading_parents_after_each_insert() {
    // After "mother 's" is wrapped in an NP, the second POS's grandparent is the middle NP, not
    // the outer one, so "dog" stays unwrapped.
    let mut p =
        Parse::parse("(TOP (S (NP (NP (NP (NNP John) (POS 's)) (NN mother) (POS 's)) (NN dog)) (VP (VBD ran))))")
            .unwrap();
    Parse::fix_possessives(&mut p);
    assert_eq!(
        show(&p),
        "(TOP (S (NP (NP (NP (NNP John) (POS 's)) (NP (NN mother) (POS 's))) (NN dog)) (VP (VBD ran))) )"
    );
    let np = at(&p, &[0, 0, 0, 1]);
    assert_eq!(np.head_text(), "mother");
    assert_eq!(np.head_index(), 2);
}

#[test]
fn adds_names() {
    let mut p = Parse::parse("(TOP (S (NP (NNP Ann) (NNP Lee)) (VP (VBD left))))").unwrap();
    Parse::add_names("PERSON", &[Span::new(0, 2), Span::new(2, 3)], &mut p).unwrap();
    assert_eq!(show(&p), "(TOP (S (NP (PERSON (NNP Ann) (NNP Lee))) (VP (PERSON (VBD left)))) )");
    // A name crossing a child of an NP that covers the last child of the NP's first child spans
    // the whole NP.
    let mut p = Parse::parse("(TOP (NP (NP (DT the) (NNP Big)) (NN apple)))").unwrap();
    Parse::add_names("CITY", &[Span::new(1, 3)], &mut p).unwrap();
    assert_eq!(show(&p), "(TOP (NP (CITY (NP (DT the) (NNP Big)) (NN apple))) )");
    // Crossing anything else skips the name.
    let mut p = Parse::parse("(TOP (S (NP (DT the) (NN dog)) (VP (VBD ran))))").unwrap();
    Parse::add_names("X", &[Span::new(1, 3)], &mut p).unwrap();
    assert_eq!(show(&p), "(TOP (S (NP (DT the) (NN dog)) (VP (VBD ran))) )");
    assert_eq!(Parse::add_names("X", &[Span::new(0, 4)], &mut p), Err(ParseError::TokenIndex { index: 3, len: 3 }));
    assert_eq!(Parse::add_names("X", &[Span::new(0, 0)], &mut p), Err(ParseError::TokenIndex { index: -1, len: 3 }));
    // An empty name past the first token ends before it starts; Java's `Span` constructor throws.
    let reversed = Parse::add_names("X", &[Span::new(1, 1)], &mut p).unwrap_err();
    assert_eq!(reversed, ParseError::ReversedName { name: Span::new(1, 1) });
    assert_eq!(reversed.java_class(), Some("java.lang.IllegalArgumentException"));
}

#[test]
fn insert_checks_the_span() {
    let mut p = Parse::parse("(TOP (NN été))").unwrap();
    let text = p.text_arc().clone();
    assert_eq!(
        p.insert(Parse::new(text.clone(), Span::new(0, 1), "X", 1.0, 0)),
        Err(ParseError::NotCharBoundary { span: Span::new(0, 1) })
    );
    assert!(matches!(
        p.insert(Parse::new(text.clone(), Span::new(0, 99), "X", 1.0, 0)),
        Err(ParseError::NotContained { .. })
    ));
    p.insert(Parse::new(text, Span::new(0, 5), "X", 1.0, 0)).unwrap();
    assert_eq!(show(&p), "(TOP (X (NN été)) )");
}

/// Punctuation that shares a probability with a mark already in the set is dropped, as Java's
/// `TreeSet` ordered by `compareTo` drops it.
#[test]
fn punctuation_sets_order_by_probability_and_drop_ties() {
    let p = Parse::parse("(TOP (NP (NN x)) (. .) ('' ''))").unwrap();
    let mut node = p.children()[0].clone();
    let mut low = p.children()[1].clone();
    low.add_prob(-0.5);
    node.add_next_punctuation(p.children()[1].clone());
    node.add_next_punctuation(p.children()[2].clone());
    node.add_next_punctuation(low);
    let kept: Vec<(&str, f64)> = node.next_punctuation().iter().map(|q| (q.kind(), q.prob())).collect();
    assert_eq!(kept, [(".", 1.0), (".", 0.5)]);
    node.add_previous_punctuation(p.children()[2].clone());
    assert_eq!(node.previous_punctuation().len(), 1);
}

#[test]
fn builds_nodes_as_the_bottom_up_parsers_do() {
    let en = english();
    let p = Parse::parse("(TOP (DT the) (, ,) (NN dog) (VBZ runs))").unwrap();
    let kids = p.children();
    let mut np = Parse::new(p.text_arc().clone(), kids[0].span(), "NP", 1.0, 0);
    np.add(kids[0].clone(), &en).unwrap();
    let mut noun = kids[2].clone();
    noun.add_previous_punctuation(kids[1].clone());
    np.add(noun, &en).unwrap();
    assert_eq!(show(&np), "(NP (DT the) (, ,) (NN dog))");
    assert_eq!(np.span(), Span::new(0, 9));
    assert_eq!(np.head_text(), "dog");
    // The daughter keeps its punctuation, as in Java.
    assert_eq!(np.children()[2].previous_punctuation().len(), 1);

    let mut s = Parse::new(p.text_arc().clone(), np.span(), "S", 1.0, 0);
    s.add(np, &en).unwrap();
    let vp = Parse::new(p.text_arc().clone(), kids[3].span(), "VP", 1.0, 3);
    let adjoined = s.adjoin(vp, &en).unwrap();
    assert_eq!(adjoined.kind(), "NP");
    assert_eq!(adjoined.span(), Span::new(0, 14));
    assert_eq!(s.span(), Span::new(0, 14));
    assert_eq!(show(&s), "(S (NP (NP (DT the) (, ,) (NN dog)) (VP runs)))");

    let mut root = Parse::parse("(TOP (NP (DT a)) (VP (VB b)))").unwrap();
    let vp = root.children()[1].clone();
    root.remove(1);
    assert_eq!(root.span(), Span::new(0, 1));
    let joined = root.adjoin_root(vp, &en, 0);
    assert_eq!(joined.span(), Span::new(0, 3));
    root.set_child(0, "built");
    assert_eq!(root.children()[0].label(), Some("built"));

    // A node whose first child is a token has no head, and `add` fails as Java's does.
    let mut tag = Parse::new(p.text_arc().clone(), Span::new(0, 3), "DT", 1.0, 0);
    let token = Parse::new(p.text_arc().clone(), Span::new(0, 3), TOK_NODE, 1.0, 0);
    assert_eq!(tag.add(token, &en), Err(ParseError::MissingHead));
}

#[test]
fn tag_sequence_probability_sums_logs() {
    let mut p = Parse::parse("(TOP (DT a) (NN b))").unwrap();
    assert_eq!(p.tag_sequence_prob(), 0.0);
    p.children_mut()[0].add_prob(-0.5);
    assert_eq!(p.tag_sequence_prob(), 0.5f64.ln());
}

#[test]
fn chunk_samples_count_childless_nodes_as_chunks() {
    let p = Parse::parse("(TOP (S (NP (DT the) (NN dog)) (VBZ runs)))").unwrap();
    let sample = ParserChunkSample::from_parse(&p);
    assert_eq!(sample.preds, ["S-NP", "C-NP", "O"]);
    let empty = Parse::parse("(TOP )").unwrap();
    assert_eq!(empty.initial_chunks().len(), 1);
}

#[test]
fn the_sample_stream_skips_blank_lines() {
    let input = "(TOP (NN a))\n \t\r\n(TOP (NN b)))\r(TOP (NN c))";
    let items: Vec<Result<String, ParseError>> =
        ParseSampleStream::new(input).map(|r| r.map(|p| p.to_string_penn_treebank())).collect();
    assert_eq!(
        items,
        [Ok("(TOP (NN a) )".into()), Err(ParseError::UnmatchedClose { byte_offset: 12 }), Ok("(TOP (NN c) )".into())]
    );
}

#[test]
fn spaces_untokenized_brackets() {
    let unicode = WhitespaceMode::Unicode;
    assert_eq!(space_untokenized_parens("no brackets", unicode), "no brackets");
    assert_eq!(space_untokenized_parens("a(b)c", unicode), "a ( b ) c");
    assert_eq!(space_untokenized_parens("(x){y}", unicode), "( x ) { y }");
    assert_eq!(space_untokenized_parens("a ( b ) c", unicode), "a ( b ) c");
    // U+001C is whitespace only in the legacy mode, U+0085 only in the Unicode mode.
    assert_eq!(space_untokenized_parens("a\u{1c}(b", unicode), "a\u{1c} ( b");
    assert_eq!(space_untokenized_parens("a\u{1c}(b", WhitespaceMode::Legacy), "a\u{1c}( b");
    assert_eq!(space_untokenized_parens("a\u{85}(b", unicode), "a\u{85}( b");
    assert_eq!(space_untokenized_parens("a\u{85}(b", WhitespaceMode::Legacy), "a\u{85} ( b");
}

#[test]
fn tag_patterns_follow_java_regular_expressions() {
    let m = |re: &str, s: &str| TagPattern::compile(re).unwrap().matches(s);
    assert!(m("AQA.*", "AQAMS0") && !m("AQA.*", "AQ0MS0"));
    assert!(m(r"GRUP\.A", "GRUP.A") && !m(r"GRUP\.A", "GRUPxA"));
    // OpenNLP's file doubles the backslash: a backslash, then any character.
    assert!(m(r"GRUP\\.VERB", r"GRUP\xVERB") && !m(r"GRUP\\.VERB", "GRUP.VERB"));
    assert!(m(r"\\$", r"\") && !m(r"\\$", "$"));
    assert!(m(r"\$", "$") && m("$", "") && !m("$", "$"));
    assert!(m("V[MAS]P.*", "VSP00") && !m("V[MAS]P.*", "VNP00"));
    assert!(m("SP[A-Z]0+", "SPS00") && !m("SP[A-Z]0+", "SPS") && !m("SP[A-Z]0+", "SPs0"));
    assert!(m("F[^a-o]", "Fp") && !m("F[^a-o]", "Fc"));
    assert!(m("SN?", "S") && m("SN?", "SN") && !m("SN?", "SNN"));
    assert!(m("^RG$", "RG") && m("[-M]G", "-G") && m("Z[m-]?", "Z-"));
    assert!(m(".", "é") && m(".", "😀") && !m(".", "\n") && !m(".", "\u{2028}"));
    assert!(m("[é-ê]x", "êx") && !m("[é-ê]x", "ex"));
    for unsupported in ["(A|B)", "A{2}", "A*?", "A*+", r"\d", "[[A]]", "[A&&B]", "*A", "A$B", "[]", "[A"] {
        assert!(TagPattern::compile(unsupported).is_err(), "{unsupported}");
    }
    let long = "A".repeat(64);
    assert!(TagPattern::compile(&long).is_err());
    assert!(TagPattern::compile(&long[..63]).unwrap().matches(&long[..63]));
}

#[test]
fn rule_files_fail_as_opennlp_fails() {
    let class = |text: &str| EnglishHeadRules::from_str(text).unwrap_err().java_class();
    assert_eq!(class("4 NP 1 NN NNS\n\n"), Some("java.util.NoSuchElementException"));
    assert_eq!(class("4 NP\n"), Some("java.util.NoSuchElementException"));
    assert_eq!(class("3 NP 1 NN NNS\n"), Some("java.lang.ArrayIndexOutOfBoundsException"));
    assert_eq!(class("5 NP 1 NN NNS\n"), Some("java.lang.NullPointerException"));
    for count in ["x", "1", "1003", "2147483648", "-2147483648"] {
        let err = EnglishHeadRules::from_str(&format!("{count} NP 1\n")).unwrap_err();
        assert_eq!(err.java_class(), Some("java.io.IOException"));
        assert_eq!(err.to_string(), format!("Invalid tag count in head rules: {count}"));
    }
    let header = AncoraSpanishHeadRules::from_str("# opennlp-ancora-head-rules 2\n3 SN 0 SN\n").unwrap_err();
    assert_eq!(header.to_string(), "Invalid tag count in head rules: #");
    let pattern = AncoraSpanishHeadRules::from_str("3 SN 0 (A|B)\n").unwrap_err();
    assert!(matches!(pattern, HeadRulesError::UnsupportedPattern { line: 1, .. }));
    assert_eq!(pattern.java_class(), None);
}

#[test]
fn rules_read_like_opennlp() {
    let rules = EnglishHeadRules::from_str("+5 NP 2 A B C\n3 NP 1 D\r4\tVP  0 E\tF").unwrap();
    assert_eq!(rules.len(), 2);
    let np = rules.rule("NP").unwrap();
    assert!(np.left_to_right);
    assert_eq!(np.tags, ["D"]);
    assert!(!rules.rule("VP").unwrap().left_to_right);
    // Written in the order of Java's HashMap: "A" hashes to bin 1 and "B" to bin 2 of 32.
    let rules = EnglishHeadRules::from_str("2 B 1\n2 A 1\n").unwrap();
    assert_eq!(rules.serialize_to_string(), "2 A 1\n2 B 1\n");
    let again = EnglishHeadRules::from_str(&rules.serialize_to_string()).unwrap();
    assert_eq!(again, rules);
    let es = spanish();
    assert_eq!(AncoraSpanishHeadRules::from_str(&es.serialize_to_string()).unwrap(), es);
    assert!(es.punctuation_tags().contains("``"));
}
