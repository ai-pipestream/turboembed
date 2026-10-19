use super::*;
use crate::lattice::{DictionaryFiles, DictionaryLimits};

/// OpenNLP's miniature dictionary from `LatticeTokenizerTest`.
const LEXICON: &str = "東京,0,0,3000,noun,proper\n京都,0,0,3000,noun,proper\n東,0,0,6000,noun,common\n\
    都,0,0,4000,noun,suffix\nに,0,0,1000,particle,case\n行く,0,0,3000,verb,base\n";
const CHAR_DEF: &str = "DEFAULT 0 1 0\nKANJI 0 0 2\nHIRAGANA 0 1 0\nLATIN 1 1 0\n\n\
    0x3041..0x3096 HIRAGANA\n0x4E00..0x9FFF KANJI\n0x0041..0x005A LATIN\n0x0061..0x007A LATIN\n";
const UNK_DEF: &str = "DEFAULT,0,0,10000,symbol,unknown\nLATIN,0,0,4000,noun,foreign\n\
    KANJI,0,0,8000,noun,unknown\nHIRAGANA,0,0,9000,particle,unknown\n";

fn tokenizer_of(matrix: &str, lexicon: &str, char_def: &str, unk_def: &str) -> LatticeTokenizer {
    let lexicons = [("lexicon.csv", lexicon.as_bytes())];
    let files = DictionaryFiles {
        matrix: matrix.as_bytes(),
        lexicons: &lexicons,
        char_def: char_def.as_bytes(),
        unk_def: unk_def.as_bytes(),
    };
    let d = MecabDictionary::from_files(files, WhitespaceMode::Unicode, DictionaryLimits::default()).unwrap();
    LatticeTokenizer::new(d, WhitespaceMode::Unicode)
}

fn tokenizer() -> LatticeTokenizer {
    tokenizer_of("1 1\n0 0 0\n", LEXICON, CHAR_DEF, UNK_DEF)
}

/// A morpheme as the tests compare it: surface, span, features and
/// whether it is unknown.
type Seen = (String, (u32, u32), Vec<String>, bool);

fn analyze(t: &LatticeTokenizer, text: &str) -> Vec<Seen> {
    let mut out = Vec::new();
    t.analyze(text, &mut out).unwrap();
    out.iter()
        .map(|m| {
            let f = t.dictionary().features(m.entry).map(str::to_string).collect();
            (text[m.span.range()].to_string(), (m.span.start, m.span.end), f, m.unknown)
        })
        .collect()
}

fn words(t: &LatticeTokenizer, text: &str) -> Vec<String> {
    analyze(t, text).into_iter().map(|m| m.0).collect()
}

// The cases of OpenNLP's LatticeTokenizerTest, with UTF-8 offsets
// where the Java has UTF-16 ones.

#[test]
fn prefers_the_cheaper_segmentation() {
    let t = tokenizer();
    let got = analyze(&t, "東京都に行く");
    assert_eq!(words(&t, "東京都に行く"), ["東京", "都", "に", "行く"]);
    assert_eq!(got.iter().map(|m| m.1).collect::<Vec<_>>(), [(0, 6), (6, 9), (9, 12), (12, 18)]);
    assert_eq!(got[0].2, ["noun", "proper"]);
    assert_eq!(got[2].2, ["particle", "case"]);
    assert!(!got[0].3);
}

#[test]
fn unknown_runs() {
    let t = tokenizer();
    let got = analyze(&t, "ABCに行く");
    assert_eq!(got[0].0, "ABC");
    assert!(got[0].3);
    assert_eq!(got[0].2, ["noun", "foreign"]);
    let got = analyze(&t, "峠道に行く");
    assert_eq!((got.len(), got[0].0.as_str(), got[0].3), (3, "峠道", true));
    // Never across a category boundary, but within one all lengths.
    assert_eq!(words(&t, "峠a"), ["峠", "a"]);
    assert_eq!(words(&t, "峠道a"), ["峠道", "a"]);
    let got = analyze(&t, "①②③");
    assert_eq!((got.len(), got[0].1, got[0].2.clone()), (1, (0, 9), vec!["symbol".into(), "unknown".into()]));
    let got = analyze(&t, "東京①に行く");
    assert_eq!(got.iter().map(|m| m.3).collect::<Vec<_>>(), [false, true, false, false]);
}

#[test]
fn whitespace_separates_and_is_never_a_morpheme() {
    let t = tokenizer();
    let got = analyze(&t, "東京 に 行く");
    assert_eq!(got.iter().map(|m| m.1).collect::<Vec<_>>(), [(0, 6), (7, 10), (11, 17)]);
    assert!(analyze(&t, "   ").is_empty());
    assert!(analyze(&t, "").is_empty());
    let got = analyze(&t, "  東京都に行く");
    assert_eq!(got[0].1, (2, 8));
}

#[test]
fn single_characters() {
    let t = tokenizer();
    let got = analyze(&t, "に");
    assert_eq!((got.len(), got[0].1, got[0].3), (1, (0, 3), false));
    let got = analyze(&t, "峠");
    assert_eq!((got.len(), got[0].1, got[0].3), (1, (0, 3), true));
}

#[test]
fn a_missing_default_template_fails_where_nothing_is_offered() {
    let t = tokenizer_of(
        "1 1\n0 0 0\n",
        "東,0,0,3000,noun\n",
        "DEFAULT 0 1 0\nKANJI 0 0 2\n0x4E00..0x9FFF KANJI\n",
        "KANJI,0,0,8000,noun\n",
    );
    let mut out = Vec::new();
    assert_eq!(t.analyze("峠 \u{2460}", &mut out), Err(NoCandidate { offset: 4 }));
    // The stretch before the failure was segmented.
    assert_eq!(out.len(), 1);
    assert!(t.analyze("東峠", &mut Vec::new()).is_ok());
}

#[test]
fn equal_paths_keep_the_one_the_java_keeps() {
    // Two entries of one surface with the same costs and context ids:
    // both candidates relax to the same cost; at the end the chain
    // holds the later one first, which the strict comparison keeps.
    let t =
        tokenizer_of("1 1\n0 0 0\n", "ab,0,0,10,first\nab,0,0,10,second\n", "DEFAULT 0 1 0\n", "DEFAULT,0,0,99,unk\n");
    let got = analyze(&t, "ab");
    assert_eq!(got[0].2, ["second"]);
    // a|b and ab cost the same: the end chain holds b (added after ab,
    // from a later position) first, so a|b wins.
    let t =
        tokenizer_of("1 1\n0 0 0\n", "ab,0,0,10,ab\na,0,0,5,a\nb,0,0,5,b\n", "DEFAULT 0 1 0\n", "DEFAULT,0,0,99,unk\n");
    assert_eq!(words(&t, "ab"), ["a", "b"]);
    // Two predecessors of equal cost: the later added one (a2, from the
    // second entry of "a") is first in the chain and is kept.
    let t =
        tokenizer_of("1 1\n0 0 0\n", "a,0,0,5,a1\na,0,0,5,a2\nb,0,0,5,b\n", "DEFAULT 0 1 0\n", "DEFAULT,0,0,99,unk\n");
    let got = analyze(&t, "ab");
    assert_eq!(got[0].2, ["a2"]);
}

#[test]
fn connection_costs_choose_between_entries() {
    // Context 1 then 2 is cheap, 1 then 1 dear: the particle reading of
    // the second word wins although its own cost is higher.
    let matrix = "3 3\n0 0 0\n0 1 0\n0 2 0\n1 0 0\n1 1 1000\n1 2 -1000\n2 0 0\n2 1 0\n2 2 0\n";
    let t = tokenizer_of(
        matrix,
        "x,1,1,10,noun\ny,1,1,10,noun\ny,2,2,500,particle\n",
        "DEFAULT 0 1 0\n",
        "DEFAULT,0,0,99,unk\n",
    );
    assert_eq!(analyze(&t, "xy")[1].2, ["particle"]);
    // The end boundary is charged too: 2 then the end is dear.
    let matrix = "3 3\n0 0 0\n0 1 0\n0 2 0\n1 0 0\n1 1 0\n1 2 0\n2 0 5000\n2 1 0\n2 2 0\n";
    let t = tokenizer_of(
        matrix,
        "x,1,1,10,noun\ny,1,1,10,noun\ny,2,2,5,particle\n",
        "DEFAULT 0 1 0\n",
        "DEFAULT,0,0,99,unk\n",
    );
    assert_eq!(analyze(&t, "xy")[1].2, ["noun"]);
}

#[test]
fn invoke_group_and_length() {
    // ALPHA invokes beside a lexicon match and groups; GREEK offers
    // lengths 1 to 3 only; KANA groups and offers lengths 1 and 2.
    let char_def = "DEFAULT 0 1 0\nALPHA 1 1 0\nGREEK 1 0 3\nKANA 0 1 2\n\
        0x0061..0x007A ALPHA\n0x03B1..0x03C9 GREEK\n0x30A1..0x30FF KANA\n";
    let unk = "DEFAULT,0,0,99,unk\nALPHA,0,0,10,alpha\nGREEK,0,0,10,greek\nKANA,0,0,10,kana\n";
    // "ab" listed dear: the grouped unknown "abc" (cost 10) beats it.
    let t = tokenizer_of("1 1\n0 0 0\n", "ab,0,0,5000,listed\n", char_def, unk);
    assert_eq!(words(&t, "abc"), ["abc"]);
    // GREEK: no group, at most three letters a morpheme.
    assert_eq!(words(&t, "αβγδε"), ["αβγ", "δε"]);
    // KANA groups; with a listed match and INVOKE 0 no unknown
    // candidate starts there.
    let t = tokenizer_of("1 1\n0 0 0\n", "アイ,0,0,5000,listed\n", char_def, unk);
    assert_eq!(
        analyze(&t, "アイウ").iter().map(|m| (m.0.clone(), m.3)).collect::<Vec<_>>(),
        [("アイ".to_string(), false), ("ウ".to_string(), true)]
    );
}

#[test]
fn a_run_continues_while_neighbours_share_a_category() {
    // The mark belongs to both scripts, so x ー y is one run: the run
    // from x reaches y. It does not reach z, which shares nothing with
    // y.
    let char_def = "DEFAULT 0 1 0\nA 0 1 0\nB 0 1 0\nC 0 1 0\n0x0078 A\n0x0079 B\n0x007A C\n0x30FC A B\n";
    let unk = "DEFAULT,0,0,99,unk\nA,0,0,10,a\nB,0,0,10,b\nC,0,0,10,c\n";
    let t = tokenizer_of("1 1\n0 0 0\n", "q,0,0,1,q\n", char_def, unk);
    assert_eq!(words(&t, "xーyz"), ["xーy", "z"]);
    assert_eq!(words(&t, "yーx"), ["yーx"]);
}

#[test]
fn supplementary_characters_are_whole() {
    let char_def = "DEFAULT 0 1 0\nKANJI 0 0 2\nEMOJI 0 1 0\n0x20000..0x2A6DF KANJI\n0x1F600..0x1F64F EMOJI\n";
    let unk = "DEFAULT,0,0,99,unk\nKANJI,0,0,10,kanji\nEMOJI,0,0,10,emoji\n";
    let t = tokenizer_of("1 1\n0 0 0\n", "𠮷野,0,0,1,listed\n", char_def, unk);
    let got = analyze(&t, "𠮷野😀😃𠀀𠀁𠀂");
    assert_eq!(got.iter().map(|m| m.1).collect::<Vec<_>>(), [(0, 7), (7, 15), (15, 23), (23, 27)]);
}

#[test]
fn scratch_reuse_gives_the_same_morphemes() {
    let t = tokenizer();
    let mut scratch = Scratch::new();
    let texts = ["東京都に行く", "x", "峠道 ABCに行く", "", "①②③ 東"];
    for text in texts.iter().chain(texts.iter().rev()) {
        let mut fresh = Vec::new();
        t.analyze(text, &mut fresh).unwrap();
        let mut reused = Vec::new();
        t.analyze_with(text, &mut scratch, &mut reused).unwrap();
        assert_eq!(reused, fresh, "{text}");
        let mut spans = vec![Span::new(1, 2)];
        t.tokenize_with(text, &mut scratch, &mut spans).unwrap();
        assert_eq!(spans[1..], fresh.iter().map(|m| m.span).collect::<Vec<_>>()[..]);
    }
}

#[test]
fn every_byte_is_covered_once_outside_whitespace() {
    let t = tokenizer();
    let text = "a\u{e9}東京\u{1f600}\u{20bb7}峠 \t\u{3000}に行くx\u{301}\u{85}①";
    let mut out = Vec::new();
    t.analyze(text, &mut out).unwrap();
    let mut at = 0usize;
    for m in &out {
        let (a, b) = (m.span.start as usize, m.span.end as usize);
        assert!(text.is_char_boundary(a) && text.is_char_boundary(b) && a < b);
        assert!(text[at..a].chars().all(|c| WhitespaceMode::Unicode.is_whitespace(c as u32)));
        assert!(!text[a..b].chars().any(|c| WhitespaceMode::Unicode.is_whitespace(c as u32)));
        at = b;
    }
    assert_eq!(at, text.len());
}
