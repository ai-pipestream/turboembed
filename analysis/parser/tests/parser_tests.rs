use std::borrow::Cow;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::str::FromStr;

use turbo_analysis::Span;
use turbo_parser::{
    ANCORA_FORMAT_HEADER, AncoraSpanishHeadRules, EnglishHeadRules, HeadRules, INC_NODE, Parse, ParseSampleStream,
    ParserChunkSample, ParserPosSample, TOK_NODE, TagPattern, decode_token, encode_token, space_untokenized_parens,
};

const PARSE_STRING: &str = "(TOP  (S (S (NP-SBJ (PRP She)  )(VP (VBD was)  \
    (ADVP (RB just)  )(NP-PRD (NP (DT another)  (NN freighter)  )(PP (IN from)  (NP (DT the)  \
    (NNPS States)  )))))(, ,)  (CC and) (S (NP-SBJ (PRP she)  )(VP (VBD seemed)  \
    (ADJP-PRD (ADJP (RB as)  (JJ commonplace)  )(PP (IN as)  (NP (PRP$ her)  \
    (NN name)  )))))(. .)  ))";

const EN_HEAD_RULES_DATA: &str = include_str!("data/en_head_rules");
const ES_HEAD_RULES_DATA: &str = include_str!("data/es_head_rules");
const ES_HEAD_RULES_LEGACY_DATA: &str = include_str!("data/es_head_rules_legacy");
const TEST_PARSE_DATA: &str = include_str!("data/test.parse");
const PARSE_VECTORS_TSV: &str = include_str!("vectors/parse_vectors.tsv");

fn unescape_tsv(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\\' && i + 5 < bytes.len() && bytes[i + 1] == b'u' {
            let hex = &s[i + 2..i + 6];
            let cp = u32::from_str_radix(hex, 16).expect("valid hex escape");
            out.push(char::from_u32(cp).expect("valid char"));
            i += 6;
        } else {
            let ch = s[i..].chars().next().unwrap();
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    out
}

fn format_heads(root: &Parse) -> String {
    let mut sb = String::new();
    append_heads(root, &mut sb);
    sb
}

fn append_heads(node: &Parse, sb: &mut String) {
    if !sb.is_empty() {
        sb.push(';');
    }
    let head = node.head();
    let span = node.span();
    sb.push_str(&format!(
        "{}[{},{}]->{}:{}:{}",
        node.kind(),
        span.start,
        span.end,
        node.head_index(),
        head.kind(),
        head.covered_text()
    ));
    for child in node.children() {
        append_heads(child, sb);
    }
}

#[test]
fn jvm_vectors_match_opennlp() {
    let en_rules = EnglishHeadRules::from_str(EN_HEAD_RULES_DATA).expect("en_head_rules loads");
    let es_rules = AncoraSpanishHeadRules::from_str(ES_HEAD_RULES_DATA).expect("es_head_rules loads");

    let mut count = 0usize;
    for line in PARSE_VECTORS_TSV.lines() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        assert_eq!(fields.len(), 3, "expected 3 TSV columns: {line}");
        let kind = fields[0];
        let input = unescape_tsv(fields[1]);
        let expected = unescape_tsv(fields[2]);
        count += 1;

        match kind {
            "show" => {
                let p = Parse::parse(&input).expect("valid parse");
                assert_eq!(p.to_string_penn_treebank(), expected);
            }
            "text" => {
                let p = Parse::parse(&input).expect("valid parse");
                assert_eq!(p.text(), expected);
            }
            "pos" => {
                let p = Parse::parse(&input).expect("valid parse");
                let sample = ParserPosSample::from_parse(&p);
                let actual = sample
                    .sentence
                    .iter()
                    .zip(&sample.tags)
                    .map(|(w, t)| format!("{w}/{t}"))
                    .collect::<Vec<_>>()
                    .join(" ");
                assert_eq!(actual, expected);
            }
            "chunk" => {
                let p = Parse::parse(&input).expect("valid parse");
                let sample = ParserChunkSample::from_parse(&p);
                let actual = sample
                    .sentence
                    .iter()
                    .zip(&sample.tags)
                    .zip(&sample.preds)
                    .map(|((w, t), p)| format!("{w}/{t}/{p}"))
                    .collect::<Vec<_>>()
                    .join(" ");
                assert_eq!(actual, expected);
            }
            "en_heads" => {
                let mut p = Parse::parse(&input).expect("valid parse");
                p.update_heads(&en_rules);
                assert_eq!(format_heads(&p), expected);
            }
            "show_ftags" => {
                let p = Parse::parse_with_options(&input, true, None).expect("valid parse");
                assert_eq!(p.to_string_penn_treebank(), expected);
            }
            "es_heads" => {
                let mut p = Parse::parse(&input).expect("valid parse");
                p.update_heads(&es_rules);
                assert_eq!(format_heads(&p), expected);
            }
            "en_gap" => {
                let p = Parse::parse_with_options(&input, false, Some(&en_rules)).expect("valid gap parse");
                assert_eq!(p.to_string_penn_treebank(), expected);
            }
            "es_gap" => {
                let p = Parse::parse_with_options(&input, false, Some(&es_rules)).expect("valid gap parse");
                assert_eq!(p.to_string_penn_treebank(), expected);
            }
            "prune" => {
                let mut p = Parse::parse(&input).expect("valid parse");
                Parse::prune_parse(&mut p);
                assert_eq!(p.to_string_penn_treebank(), expected);
            }
            "possessives" => {
                let mut p = Parse::parse(&input).expect("valid parse");
                Parse::fix_possessives(&mut p);
                assert_eq!(p.to_string_penn_treebank(), expected);
            }
            "add_names" => {
                let mut p = Parse::parse(&input).expect("valid parse");
                Parse::add_names("location", &[Span::new(7, 8)], &mut p);
                assert_eq!(p.to_string_penn_treebank(), expected);
            }
            other => panic!("unknown vector kind: {other}"),
        }
    }
    assert_eq!(count, 45);
}

#[test]
fn parse_hash_display_eq_and_clone() {
    let p1 = Parse::parse(PARSE_STRING).unwrap();
    let p2 = p1.clone();
    assert_eq!(p1, p2);
    assert_eq!(p2, p1);
    assert_eq!(
        p1.to_string(),
        "She was just another freighter from the States , and she seemed as commonplace as her name . "
    );

    let mut h1 = DefaultHasher::new();
    let mut h2 = DefaultHasher::new();
    p1.hash(&mut h1);
    p2.hash(&mut h2);
    assert_eq!(h1.finish(), h2.finish());
}

#[test]
fn parse_get_text() {
    let p = Parse::parse(PARSE_STRING).unwrap();
    let expected = "She was just another freighter from the States , and she seemed as commonplace as her name . ";
    assert_eq!(p.text(), expected);
}

#[test]
fn parse_show_round_trip() {
    let p1 = Parse::parse(PARSE_STRING).unwrap();
    let shown = p1.to_string_penn_treebank();
    let p2 = Parse::parse(&shown).unwrap();
    assert_eq!(p1, p2);
}

#[test]
fn parse_token_replacement_round_trip() {
    let p1 = Parse::parse(
        "(TOP  (S-CLF (NP-SBJ (PRP It)  )(VP (VBD was) \
         (NP-PRD (NP (DT the)  (NN trial)  )(PP (IN of) \
         (NP (NP (NN oleomargarine)  (NN heir)  )(NP (NNP Minot) \
         (PRN (-LRB- -LRB-) (NNP Mickey) \
         (-RRB- -RRB-) )(NNP Jelke)  )))(PP (IN for) \
         (NP (JJ compulsory)  (NN prostitution) \
         ))(PP-LOC (IN in)  (NP (NNP New)  (NNP York) \
         )))(SBAR (WHNP-1 (WDT that)  )(S (VP (VBD put) \
         (NP (DT the)  (NN spotlight)  )(PP (IN on)  (NP (DT the) \
         (JJ international)  (NN play-girl)  ))))))(. .)  ))",
    )
    .unwrap();
    let shown = p1.to_string_penn_treebank();
    let p2 = Parse::parse(&shown).unwrap();
    assert_eq!(p1, p2);

    for tok in ["(", ")", "{", "}", "[", "]", "word"] {
        assert_eq!(decode_token(encode_token(tok)), tok);
    }
}

#[test]
fn parse_get_tag_nodes() {
    let p = Parse::parse(PARSE_STRING).unwrap();
    let tags = p.tag_nodes();
    for node in &tags {
        assert!(node.is_pos_tag());
    }
    let expected = [
        "PRP", "VBD", "RB", "DT", "NN", "IN", "DT", "NNPS", ",", "CC", "PRP", "VBD", "RB", "JJ", "IN", "PRP$", "NN",
        ".",
    ];
    let actual: Vec<&str> = tags.iter().map(|n| n.kind()).collect();
    assert_eq!(actual, expected);
}

#[test]
fn parse_create_from_tokens() {
    let tokens = ["The", "cat", "sat", "on", "the", "mat"];
    let p = Parse::from_tokens(&tokens).unwrap();
    assert_eq!(p.text(), "The cat sat on the mat");
    assert_eq!(p.span(), Span::new(0, 22));
    assert_eq!(p.kind(), INC_NODE);

    let children = p.children();
    assert_eq!(children.len(), tokens.len());
    let mut start = 0u32;
    for (i, &tok) in tokens.iter().enumerate() {
        assert_eq!(children[i].kind(), TOK_NODE);
        assert_eq!(children[i].span(), Span::new(start, start + tok.len() as u32));
        assert_eq!(children[i].covered_text(), tok);
        start += tok.len() as u32 + 1;
    }

    assert!(Parse::from_tokens(&[] as &[&str]).is_err());

    let single = Parse::from_tokens(&["Hello"]).unwrap();
    assert_eq!(single.text(), "Hello");
    assert_eq!(single.children().len(), 1);
    assert_eq!(single.children()[0].span(), Span::new(0, 5));
}

#[test]
fn parses_treebank_tree_across_whitespace() {
    let parse = Parse::parse("(TOP\n  (S\t(NP-SBJ-1 (NNP José))\n(VP (VBZ runs)) (. .)))").unwrap();
    assert_eq!(parse.text(), "José runs . ");
    let sentence = &parse.children()[0];
    assert_eq!(sentence.kind(), "S");
    assert_eq!(sentence.children()[0].kind(), "NP");
    assert_eq!(sentence.children()[0].covered_text(), "José");
}

#[test]
fn retains_function_tags_without_numeric_coindices() {
    let cases = [
        ("NP-SBJ-1", "NP-SBJ"),
        ("NP-SBJ-TMP-1", "NP-SBJ-TMP"),
        ("NP-1", "NP"),
        ("NP-SBJ=2", "NP-SBJ"),
        ("NP-SBJ-TMP=2", "NP-SBJ-TMP"),
    ];
    for (label, expected) in cases {
        let tree = format!("(TOP (S ({label} (NN word))))");
        let parse = Parse::parse_with_options(&tree, true, None).unwrap();
        assert_eq!(parse.children()[0].children()[0].kind(), expected);
    }
}

#[test]
fn unicode_whitespace_separates_labels_and_tokens() {
    let parse = Parse::parse("(TOP\u{00a0}(NN\u{2003}word))").unwrap();
    assert_eq!(parse.text(), "word ");
    assert_eq!(parse.children()[0].kind(), "NN");
}

#[test]
fn decodes_treebank_bracket_tokens_with_supplementary_chars() {
    let parse = Parse::parse("(TOP (S (-LRB- -LRB-) (NN 🐈) (-RRB- -RRB-)))").unwrap();
    assert_eq!(parse.text(), "( 🐈 ) ");
    let tags = parse.tag_nodes();
    assert_eq!(tags[0].covered_text(), "(");
    assert_eq!(tags[1].covered_text(), "🐈");
    assert_eq!(tags[2].covered_text(), ")");
}

#[test]
fn parses_treebank_file_wrapper() {
    let parse = Parse::parse("((S (NP-SBJ (PRP They)) (VP (VBP agree)) (. .)))").unwrap();
    assert_eq!(parse.text(), "They agree . ");
    assert_eq!(parse.children()[0].kind(), "S");
}

#[test]
fn rejects_malformed_treebank_trees() {
    let malformed = [
        ("", 0, "missing root constituent"),
        ("   ", 0, "missing root constituent"),
        ("TOP", 0, "content outside a constituent"),
        (")", 0, "unmatched closing parenthesis"),
        ("(TOP (NN token)", 15, "unclosed constituent"),
        ("(TOP () (NN token))", 5, "missing constituent label"),
        ("(TOP (NN two tokens))", 12, "terminal must end after one token"),
        ("(S (NN one)) (S (NN two))", 13, "multiple root constituents"),
        ("(TOP (NN token)) trailing", 17, "content outside a constituent"),
        ("(TOP (NN", 5, "unterminated constituent"),
        ("(TOP (NN token", 14, "terminal must end after one token"),
        ("(TOP (-bad token))", 5, "invalid constituent label"),
    ];
    for (tree, expected_idx, expected_reason) in malformed {
        let err = Parse::parse(tree).expect_err(&format!("expected error for: {tree:?}"));
        let expected_msg = format!("Malformed treebank parse at index {expected_idx}: {expected_reason}");
        assert_eq!(err.to_string(), expected_msg, "for input {tree:?}");
    }
}

#[test]
fn parse_sample_stream_reads_test_parse() {
    let parses: Vec<Parse> = ParseSampleStream::new(TEST_PARSE_DATA).collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(parses.len(), 4);
}

#[test]
fn chunk_sample_stream_matches_parse_string() {
    let p = Parse::parse(PARSE_STRING).unwrap();
    let sample = ParserChunkSample::from_parse(&p);
    let expected = [
        ("She", "PRP", "S-NP"),
        ("was", "VBD", "O"),
        ("just", "RB", "S-ADVP"),
        ("another", "DT", "S-NP"),
        ("freighter", "NN", "C-NP"),
        ("from", "IN", "O"),
        ("the", "DT", "S-NP"),
        ("States", "NNPS", "C-NP"),
        (",", ",", "O"),
        ("and", "CC", "O"),
        ("she", "PRP", "S-NP"),
        ("seemed", "VBD", "O"),
        ("as", "RB", "S-ADJP"),
        ("commonplace", "JJ", "C-ADJP"),
        ("as", "IN", "O"),
        ("her", "PRP$", "S-NP"),
        ("name", "NN", "C-NP"),
        (".", ".", "O"),
    ];
    assert_eq!(sample.sentence.len(), expected.len());
    for (i, (w, t, pr)) in expected.iter().enumerate() {
        assert_eq!(sample.sentence[i], *w);
        assert_eq!(sample.tags[i], *t);
        assert_eq!(sample.preds[i], *pr);
    }
}

#[test]
fn english_head_rules_validation_and_serialization() {
    let valid = EnglishHeadRules::from_str("5 NP 1 NN NNS NNP\n").unwrap();
    assert_eq!(valid.len(), 1);

    assert!(EnglishHeadRules::from_str("2147483647 NP 1\n").is_err());
    assert!(EnglishHeadRules::from_str("1 NP 1\n").is_err());
    assert!(EnglishHeadRules::from_str("1003 NP 1\n").is_err());
    assert!(EnglishHeadRules::from_str("NaN NP 1\n").is_err());

    let original = EnglishHeadRules::from_str(EN_HEAD_RULES_DATA).unwrap();
    let serialized = original.serialize_to_string();
    let recreated = EnglishHeadRules::from_str(&serialized).unwrap();
    assert_eq!(original, recreated);
}

fn es_rules(body: &str) -> Result<AncoraSpanishHeadRules, turbo_parser::HeadRulesError> {
    AncoraSpanishHeadRules::from_str(&format!("{ANCORA_FORMAT_HEADER}\n{body}"))
}

fn dummy_constituent(kind: &str, index: i32) -> Parse {
    Parse::new("text", Span::new(0, 4), kind, 1.0, index)
}

#[test]
fn ancora_version_two_glob_matches_literal_wildcard_and_choice() {
    let rules = es_rules("6 GRUP.VERB 0 GRUP.VERB V[MAS]* NC*S* $\n").unwrap();
    let false_dot = dummy_constituent("GRUPxVERB", 0);
    let wildcard = dummy_constituent("NCMS000", 1);
    let choice = dummy_constituent("VAIP3S0", 2);
    let exact = dummy_constituent("GRUP.VERB", 3);
    let constituents = [false_dot, wildcard.clone(), choice.clone(), exact.clone()];

    assert_eq!(rules.get_head(&constituents, "GRUP.VERB").unwrap().kind(), exact.kind());
    assert_eq!(
        es_rules("3 TEST 0 V[MAS]*\n")
            .unwrap()
            .get_head(&[dummy_constituent("OTHER", 0), choice.clone()], "TEST")
            .unwrap()
            .kind(),
        choice.kind()
    );
    assert_eq!(
        es_rules("3 TEST 0 NC*S*\n")
            .unwrap()
            .get_head(&[dummy_constituent("OTHER", 0), wildcard.clone()], "TEST")
            .unwrap()
            .kind(),
        wildcard.kind()
    );
}

#[test]
fn ancora_rule_and_constituent_priority_remain_stable() {
    let first_noun = dummy_constituent("NCMS", 0);
    let adjective = dummy_constituent("AQA0", 1);
    let last_noun = dummy_constituent("NCFP", 2);
    let constituents = [first_noun, adjective, last_noun];

    assert_eq!(es_rules("4 TEST 1 NC* AQA*\n").unwrap().get_head(&constituents, "TEST").unwrap().head_index(), 0);
    assert_eq!(es_rules("4 TEST 0 NC* AQA*\n").unwrap().get_head(&constituents, "TEST").unwrap().head_index(), 2);
    assert_eq!(es_rules("4 TEST 0 AQA* NC*\n").unwrap().get_head(&constituents, "TEST").unwrap().head_index(), 1);
}

#[test]
fn ancora_choice_does_not_match_nul() {
    let invalid = dummy_constituent("V\0", 0);
    let fallback = dummy_constituent("OTHER", 1);
    assert_eq!(es_rules("3 TEST 0 V[MAS]\n").unwrap().get_head(&[invalid, fallback], "TEST").unwrap().head_index(), 1);
}

#[test]
fn ancora_legacy_import_and_v2_equivalence() {
    let rules = AncoraSpanishHeadRules::from_str("3 TEST 1 GRUP.ADV\n").unwrap();
    let invalid = dummy_constituent("GRUPxADV", 0);
    let literal = dummy_constituent("GRUP.ADV", 1);
    assert_eq!(rules.get_head(&[invalid, literal], "TEST").unwrap().head_index(), 1);

    let legacy = AncoraSpanishHeadRules::from_str("5 TEST 0 GRUP\\\\.NOM \\\\$ SP[CS].*\n").unwrap();
    let false_dot = dummy_constituent("GRUPxNOM", 0);
    let literal_dollar = dummy_constituent("$", 1);
    let choice = dummy_constituent("SPS00", 2);
    let literal_dot = dummy_constituent("GRUP.NOM", 3);
    assert_eq!(
        legacy
            .get_head(&[false_dot, literal_dollar.clone(), choice.clone(), literal_dot], "TEST")
            .unwrap()
            .head_index(),
        3
    );

    let sn_rules = es_rules("").unwrap();
    assert_eq!(
        sn_rules
            .get_head(
                &[dummy_constituent("GRUPxA", 0), dummy_constituent("GRUP.A", 2), dummy_constituent("AQAC0", 1),],
                "SN"
            )
            .unwrap()
            .head_index(),
        2
    );

    let v2_shipped = AncoraSpanishHeadRules::from_str(ES_HEAD_RULES_DATA).unwrap();
    let legacy_shipped = AncoraSpanishHeadRules::from_str(ES_HEAD_RULES_LEGACY_DATA).unwrap();
    assert_eq!(v2_shipped, legacy_shipped);

    let serialized = v2_shipped.serialize_to_string();
    assert!(serialized.starts_with(ANCORA_FORMAT_HEADER));
    assert_eq!(v2_shipped, AncoraSpanishHeadRules::from_str(&serialized).unwrap());

    let mut parse = Parse::parse("(TOP (SN (NCMS000 coche) (AQ0MS0 rojo)))").unwrap();
    parse.update_heads(&v2_shipped);
    assert_eq!(parse.children()[0].head().kind(), "NCMS000");
}

#[test]
fn ancora_state_bounds_and_non_backtracking_matcher() {
    let exact62 = dummy_constituent(&"A".repeat(62), 0);
    let fallback = dummy_constituent("OTHER", 1);
    let r62 = es_rules(&format!("3 TEST 0 {}\n", "A".repeat(62))).unwrap();
    assert_eq!(r62.get_head(&[exact62, fallback.clone()], "TEST").unwrap().head_index(), 0);

    let no_gap = dummy_constituent("AB", 0);
    assert_eq!(
        es_rules("3 TEST 0 A*B\n").unwrap().get_head(&[no_gap, fallback.clone()], "TEST").unwrap().head_index(),
        0
    );
    assert!(es_rules("3 TEST 0 A**B\n").is_err());

    let err63 = es_rules(&format!("3 TEST 0 {}\n", "A".repeat(63))).unwrap_err();
    assert!(err63.to_string().contains("62 states"));

    let bounded = es_rules("3 TEST 0 A*B*C*Z\n").unwrap();
    let near_miss = dummy_constituent(&format!("A{}Y", "BC".repeat(50_000)), 0);
    assert_eq!(bounded.get_head(&[near_miss, fallback], "TEST").unwrap().head_index(), 1);
    assert!(TagPattern::compile("").is_err());
}

#[test]
fn ancora_rejects_unsupported_and_malformed_rules() {
    for bad_v2 in [
        "3 TEST 0 A+",
        "3 TEST 0 A|B",
        "3 TEST 0 (AB)",
        "3 TEST 0 A?",
        "3 TEST 0 [A-Z]",
        "3 TEST 0 []",
        "3 TEST 0 [XYZ]",
        "3 TEST 0 A\\B",
    ] {
        let err = es_rules(&format!("{bad_v2}\n")).unwrap_err().to_string();
        assert!(err.contains("line 2"), "missing line 2 in: {err}");
        assert!(err.contains("TEST"), "missing TEST in: {err}");
    }

    for bad_legacy in [
        "3 TEST 0 A+",
        "3 TEST 0 A|B",
        "3 TEST 0 (AB)",
        "3 TEST 0 [A-Z]",
        "3 TEST 0 A\\d",
        "3 TEST 0 A.B",
        "3 TEST 0 A$",
    ] {
        let err = AncoraSpanishHeadRules::from_str(&format!("{bad_legacy}\n")).unwrap_err().to_string();
        assert!(err.contains("line 1"), "missing line 1 in: {err}");
        assert!(err.contains("TEST"), "missing TEST in: {err}");
    }

    let bad_header = AncoraSpanishHeadRules::from_str("# opennlp-ancora-head-rules 3\n").unwrap_err().to_string();
    assert!(bad_header.contains("header"));
    assert!(bad_header.contains("line 1"));

    for malformed in [
        "3 TEST 2 A*",
        "4 TEST 0 A*",
        "2 TEST 0 A*",
        "3 TEST 0 A*\n3 TEST 1 B*",
        "not-a-count TEST 0 A*",
        "3 TEST 0 ",
        "",
    ] {
        let err = es_rules(&format!("{malformed}\n")).unwrap_err().to_string();
        assert!(err.contains("line"), "missing line in: {err}");
    }
}

#[test]
fn space_untokenized_parens_matches_opennlp() {
    let cases = [
        ("a(b)c", "a ( b ) c"),
        ("a (b) c", "a ( b ) c"),
        ("foo(bar){baz}", "foo ( bar ) { baz }"),
        ("((a)(b))", "( ( a ) ( b ) )"),
        ("a(b(c)d)e", "a ( b ( c ) d ) e"),
        ("a))b", "a ) ) b"),
        ("f(x)+g({y})", "f ( x ) +g ( { y } )"),
        ("x((", "x ( ("),
        ("((x", "( ( x"),
        ("()", "( )"),
        ("(", "("),
        ("", ""),
        ("no parens here", "no parens here"),
        ("«quoted»", "«quoted»"),
        ("[a]", "[a]"),
        ("tab\there(", "tab\there ("),
        ("a\t(\tb", "a\t(\tb"),
        ("a\u{00A0}(\u{00A0}b", "a\u{00A0}(\u{00A0}b"),
        ("a\n(", "a\n("),
        ("a  (b", "a  ( b"),
        ("caf\u{00E9}(\u{1F600})", "caf\u{00E9} ( \u{1F600} )"),
    ];
    for (input, expected) in cases {
        let actual = space_untokenized_parens(input);
        assert_eq!(actual.as_ref(), expected, "mismatch for input {input:?}");
    }

    assert!(matches!(space_untokenized_parens("no parens here"), Cow::Borrowed(_)));
}
