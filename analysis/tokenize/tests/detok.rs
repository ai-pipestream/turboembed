//! The dictionary detokenizer: the cases of OpenNLP's
//! `DictionaryDetokenizerTest` and `DetokenizationDictionaryTest`, the
//! dictionary checks, and the English, French and Portuguese dictionaries
//! against what OpenNLP's `DictionaryDetokenizer` printed for the same
//! tokens (tests/dictionaries/README.md). Every golden file the JVM wrote
//! under tests/golden/detokenizer is also run, which holds the cases
//! written out here to the same values where they overlap.

mod common;

use turbo_analysis::Span;
use turbo_maxent::Entries;
use turbo_tokenize::{Detokenizer, DetokenizerDictionary, DictionaryError, DictionaryOperation, Operation};

const L: Operation = Operation::MergeToLeft;
const R: Operation = Operation::MergeToRight;
const B: Operation = Operation::MergeBoth;
const N: Operation = Operation::NoOperation;

const MOVE_LEFT: DictionaryOperation = DictionaryOperation::MoveLeft;
const MOVE_RIGHT: DictionaryOperation = DictionaryOperation::MoveRight;
const MOVE_BOTH: DictionaryOperation = DictionaryOperation::MoveBoth;
const MATCHING: DictionaryOperation = DictionaryOperation::RightLeftMatching;

const MARKER: &str = "<SPLIT>";

fn latin() -> DetokenizerDictionary {
    DetokenizerDictionary::read(include_bytes!("dictionaries/latin-detokenizer.xml")).unwrap()
}

/// Checks the operations, the text with and without the split marker, and
/// that the offset form writes the same text with every token where its
/// offset says.
fn check(dictionary: &DetokenizerDictionary, tokens: &[&str], operations: &[Operation], plain: &str, marked: &str) {
    let detokenizer = Detokenizer::new(dictionary);
    let mut ops = Vec::new();
    detokenizer.operations(tokens, &mut ops);
    assert_eq!(ops, operations, "{tokens:?}");

    let mut out = String::new();
    detokenizer.detokenize(tokens, None, &mut out);
    assert_eq!(out, plain);
    out.clear();
    detokenizer.detokenize(tokens, Some(MARKER), &mut out);
    assert_eq!(out, marked);

    for (marker, expected) in [(None, plain), (Some(MARKER), marked)] {
        let (text, spans) = spaced(tokens);
        let mut out = String::new();
        let mut starts = Vec::new();
        detokenizer.detokenize_spans(&text, &spans, marker, &mut out, &mut starts);
        assert_eq!(out, expected);
        assert_eq!(starts.len(), tokens.len());
        for ((&start, token), span) in starts.iter().zip(tokens).zip(&spans) {
            assert_eq!(&out[start as usize..(start + span.len()) as usize], *token);
        }
    }
}

/// The tokens joined by single spaces, and their spans in that text.
fn spaced(tokens: &[&str]) -> (String, Vec<Span>) {
    let text = tokens.join(" ");
    let mut spans = Vec::new();
    let mut start = 0;
    for token in tokens {
        spans.push(Span::new(start, start + token.len() as u32));
        start += token.len() as u32 + 1;
    }
    (text, spans)
}

// DictionaryDetokenizerTest.

#[test]
fn java_detokenizer_operations() {
    let mut d = DetokenizerDictionary::new();
    for (token, op) in
        [(".", MOVE_LEFT), ("!", MOVE_LEFT), ("(", MOVE_RIGHT), (")", MOVE_LEFT), ("\"", MATCHING), ("-", MOVE_BOTH)]
    {
        d.insert(token, op);
    }
    let mut ops = Vec::new();
    Detokenizer::new(&d).operations(&["Simple", "test", ".", "co", "-", "worker"], &mut ops);
    assert_eq!(ops, [N, N, L, N, B, N]);
}

#[test]
fn java_detokenize_to_string() {
    check(
        &latin(),
        &["A", "test", ",", "(", "string", ")", "."],
        &[N, N, L, R, N, L, L],
        "A test, (string).",
        "A test<SPLIT>, (<SPLIT>string<SPLIT>)<SPLIT>.",
    );
}

#[test]
fn java_detokenize_to_string_2() {
    check(
        &latin(),
        &["A", "co", "-", "worker", "helped", "."],
        &[N, N, B, N, N, L],
        "A co-worker helped.",
        "A co<SPLIT>-<SPLIT>worker helped<SPLIT>.",
    );
}

// DetokenizationDictionaryTest (the serialization case is not ported:
// this crate does not write dictionaries).

#[test]
fn java_simple_dict() {
    let mut d = DetokenizerDictionary::new();
    for (token, op) in [("\"", MATCHING), ("(", MOVE_RIGHT), (")", MOVE_LEFT), ("-", MOVE_BOTH)] {
        d.insert(token, op);
    }
    assert_eq!(d.get("\""), Some(MATCHING));
    assert_eq!(d.get("("), Some(MOVE_RIGHT));
    assert_eq!(d.get(")"), Some(MOVE_LEFT));
    assert_eq!(d.get("-"), Some(MOVE_BOTH));
    assert_eq!(d.get("x"), None);
}

// More cases on the latin dictionary, with the JVM's output.

#[test]
fn odd_number_of_quotes() {
    check(
        &latin(),
        &["\"", "a", "\"", "b", "\"", "c"],
        &[R, N, L, N, R, N],
        "\"a\" b \"c",
        "\"<SPLIT>a<SPLIT>\" b \"<SPLIT>c",
    );
}

#[test]
fn tokens_outside_the_bmp() {
    check(
        &latin(),
        &["\u{1F600}", "(", "\u{1D400}", ")", ".", "\"", "\u{1F600}", "\""],
        &[N, R, N, L, L, R, N, L],
        "\u{1F600} (\u{1D400}). \"\u{1F600}\"",
        "\u{1F600} (<SPLIT>\u{1D400}<SPLIT>)<SPLIT>. \"<SPLIT>\u{1F600}<SPLIT>\"",
    );
}

#[test]
fn empty_token_list() {
    check(&latin(), &[], &[], "", "");
}

#[test]
fn single_token() {
    // Nothing follows the last token, not even for one that merges right.
    check(&latin(), &["("], &[R], "(", "(");
    check(&latin(), &["word"], &[N], "word", "word");
}

#[test]
fn quote_flags_are_per_token() {
    let mut d = DetokenizerDictionary::new();
    d.insert("\"", MATCHING);
    d.insert("'", MATCHING);
    check(&d, &["\"", "'", "a", "\"", "'"], &[R, R, N, L, L], "\"'a\"'", "\"<SPLIT>'<SPLIT>a<SPLIT>\"<SPLIT>'");
}

#[test]
fn each_call_starts_with_every_quote_closed() {
    let d = latin();
    let detokenizer = Detokenizer::new(&d);
    let mut ops = Vec::new();
    detokenizer.operations(&["\"", "a"], &mut ops);
    detokenizer.operations(&["\"", "b"], &mut ops);
    assert_eq!(ops, [R, N, R, N]);
}

#[test]
fn offsets_of_every_token() {
    let d = latin();
    let text = "He said ( \u{1F600} ) , co - op .";
    let tokens = ["He", "said", "(", "\u{1F600}", ")", ",", "co", "-", "op", "."];
    let mut spans = Vec::new();
    let mut from = 0;
    for token in tokens {
        let start = from + text[from..].find(token).unwrap();
        spans.push(Span::new(start as u32, (start + token.len()) as u32));
        from = start + token.len();
    }
    let detokenizer = Detokenizer::new(&d);

    // Appends after what the buffers already hold.
    let mut out = String::from(">");
    let mut starts = vec![99];
    detokenizer.detokenize_spans(text, &spans, None, &mut out, &mut starts);
    assert_eq!(out, ">He said (\u{1F600}), co-op.");
    assert_eq!(starts, [99, 1, 4, 9, 10, 14, 15, 17, 19, 20, 22]);
    for (i, token) in tokens.iter().enumerate() {
        let start = starts[i + 1] as usize;
        assert_eq!(&out[start..start + token.len()], *token);
    }

    out.clear();
    starts.clear();
    detokenizer.detokenize_spans(text, &spans, Some("|"), &mut out, &mut starts);
    assert_eq!(out, "He said (|\u{1F600}|)|, co|-|op|.");
    assert_eq!(starts, [0, 3, 8, 10, 15, 17, 19, 22, 24, 27]);
    for (&start, token) in starts.iter().zip(tokens) {
        assert_eq!(&out[start as usize..start as usize + token.len()], token);
    }

    // The string form writes the same text.
    let mut plain = String::new();
    detokenizer.detokenize(&tokens, Some("|"), &mut plain);
    assert_eq!(plain, out);
}

// The dictionary.

#[test]
fn latin_dictionary_entries() {
    let d = latin();
    // Eighteen entries, with the straight quote twice.
    assert_eq!(d.len(), 17);
    assert_eq!(d.get("``"), Some(MOVE_RIGHT));
    assert_eq!(d.get("''"), Some(MOVE_LEFT));
    assert_eq!(d.get("\""), Some(MATCHING));
    assert_eq!(d.get("-"), Some(MOVE_BOTH));
}

#[test]
fn tokens_compare_exactly() {
    let xml =
        br#"<dictionary case_sensitive="false"><entry operation="MOVE_LEFT"><token>Xx</token></entry></dictionary>"#;
    let d = DetokenizerDictionary::read(xml).unwrap();
    assert_eq!(d.get("Xx"), Some(MOVE_LEFT));
    assert_eq!(d.get("xx"), None);
    assert_eq!(d.get("XX"), None);
}

#[test]
fn from_entries_matches_read() {
    let xml = include_bytes!("dictionaries/en-detokenizer.xml");
    let entries = Entries::read(xml).unwrap();
    let a = DetokenizerDictionary::from_entries(&entries).unwrap();
    let b = DetokenizerDictionary::read(xml).unwrap();
    assert_eq!(a.len(), b.len());
    for entry in entries.iter() {
        assert_eq!(a.get(&entry.tokens[0]), b.get(&entry.tokens[0]));
    }
}

#[test]
fn multi_token_entry_is_an_error() {
    let xml = br#"<dictionary>
        <entry operation="MOVE_LEFT"><token>.</token></entry>
        <entry operation="MOVE_LEFT"><token>New</token><token>York</token></entry>
    </dictionary>"#;
    match DetokenizerDictionary::read(xml) {
        Err(DictionaryError::TokenCount { index, tokens }) => {
            assert_eq!(index, 1);
            assert_eq!(&*tokens, [Box::from("New"), Box::from("York")]);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn unknown_operation_is_an_error() {
    let xml = br#"<dictionary><entry operation="MOVE_UP"><token>^</token></entry></dictionary>"#;
    let error = DetokenizerDictionary::read(xml).unwrap_err();
    match &error {
        DictionaryError::UnknownOperation { index: 0, token, value } => {
            assert_eq!(&**token, "^");
            assert_eq!(&**value, "MOVE_UP");
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(error.to_string(), "detokenizer dictionary entry 0: token \"^\" has unknown operation \"MOVE_UP\"");
}

#[test]
fn missing_operation_is_an_error() {
    let xml = br#"<dictionary><entry><token>^</token></entry></dictionary>"#;
    match DetokenizerDictionary::read(xml) {
        Err(DictionaryError::MissingOperation { index: 0, token }) => assert_eq!(&*token, "^"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn malformed_xml_is_an_error() {
    let xml = br#"<dictionary><entry operation="MOVE_LEFT"><token>.</token></dictionary>"#;
    assert!(matches!(DetokenizerDictionary::read(xml), Err(DictionaryError::Read(_))));
}

// The language dictionaries, against the JVM.

#[test]
fn english() {
    let d = DetokenizerDictionary::read(include_bytes!("dictionaries/en-detokenizer.xml")).unwrap();
    assert_eq!(d.len(), 28);
    check(
        &d,
        &["He", "said", ",", "\"", "I", "do", "n't", "know", "\"", "."],
        &[N, N, L, R, N, N, L, N, L, L],
        "He said, \"I don't know\".",
        "He said<SPLIT>, \"<SPLIT>I do<SPLIT>n't know<SPLIT>\"<SPLIT>.",
    );
    check(
        &d,
        &["It", "'s", "John", "'s", "book", "(", "the", "red", "one", ")", "!"],
        &[N, L, N, L, N, R, N, N, N, L, L],
        "It's John's book (the red one)!",
        "It<SPLIT>'s John<SPLIT>'s book (<SPLIT>the red one<SPLIT>)<SPLIT>!",
    );
    check(
        &d,
        &["Visit", "example", ".com", "or", "#", "tag", "for", "50", "%", "off", "."],
        &[N, N, L, N, R, N, N, N, L, N, L],
        "Visit example.com or #tag for 50% off.",
        "Visit example<SPLIT>.com or #<SPLIT>tag for 50<SPLIT>% off<SPLIT>.",
    );
    check(
        &d,
        &["'", "Quoted", "'", "and", "\"", "double", "\"", "and", "\"", "odd"],
        &[R, N, L, N, R, N, L, N, R, N],
        "'Quoted' and \"double\" and \"odd",
        "'<SPLIT>Quoted<SPLIT>' and \"<SPLIT>double<SPLIT>\" and \"<SPLIT>odd",
    );
}

#[test]
fn french() {
    let d = DetokenizerDictionary::read(include_bytes!("dictionaries/fr-detokenizer.xml")).unwrap();
    assert_eq!(d.len(), 62);
    check(
        &d,
        &["L'", "homme", "qu'", "il", "a", "vu", "est", "l\u{e0}", "."],
        &[R, N, R, N, N, N, N, N, L],
        "L'homme qu'il a vu est l\u{e0}.",
        "L'<SPLIT>homme qu'<SPLIT>il a vu est l\u{e0}<SPLIT>.",
    );
    check(
        &d,
        &["Aujourd'", "hui", ",", "c'", "est", "l'", "ex-", "pr\u{e9}sident", "!"],
        &[R, N, L, R, N, R, R, N, L],
        "Aujourd'hui, c'est l'ex-pr\u{e9}sident!",
        "Aujourd'<SPLIT>hui<SPLIT>, c'<SPLIT>est l'<SPLIT>ex-<SPLIT>pr\u{e9}sident<SPLIT>!",
    );
    check(
        &d,
        &["Il", "a", "dit", ":", "\"", "Jusqu'", "ici", ",", "tout", "va", "bien", "\"", "."],
        &[N, N, N, L, R, R, N, L, N, N, N, L, L],
        "Il a dit: \"Jusqu'ici, tout va bien\".",
        "Il a dit<SPLIT>: \"<SPLIT>Jusqu'<SPLIT>ici<SPLIT>, tout va bien<SPLIT>\"<SPLIT>.",
    );
}

#[test]
fn portuguese() {
    let d = DetokenizerDictionary::read(include_bytes!("dictionaries/pt-detokenizer.xml")).unwrap();
    assert_eq!(d.len(), 24);
    check(
        &d,
        &["O", "guarda", "-", "chuva", "est\u{e1}", "aqui", "."],
        &[N, N, B, N, N, N, L],
        "O guarda-chuva est\u{e1} aqui.",
        "O guarda<SPLIT>-<SPLIT>chuva est\u{e1} aqui<SPLIT>.",
    );
    check(
        &d,
        &["Ele", "disse", "\u{ab}", "ol\u{e1}", "\u{bb}", "(", "baixinho", ")", "."],
        &[N, N, R, N, L, R, N, L, L],
        "Ele disse \u{ab}ol\u{e1}\u{bb} (baixinho).",
        "Ele disse \u{ab}<SPLIT>ol\u{e1}<SPLIT>\u{bb} (<SPLIT>baixinho<SPLIT>)<SPLIT>.",
    );
    check(
        &d,
        &["\"", "Bom", "dia", "\"", ",", "disse", "o", "co", "-", "piloto", "."],
        &[R, N, N, L, L, N, N, N, B, N, L],
        "\"Bom dia\", disse o co-piloto.",
        "\"<SPLIT>Bom dia<SPLIT>\"<SPLIT>, disse o co<SPLIT>-<SPLIT>piloto<SPLIT>.",
    );
}

// The golden files of the generator's detokenizer run.

/// Converts an OpenNLP `DetokenizationOperation` name.
fn operation(name: &str) -> Operation {
    match name {
        "MERGE_TO_LEFT" => L,
        "MERGE_TO_RIGHT" => R,
        "MERGE_BOTH" => B,
        "NO_OPERATION" => N,
        other => panic!("operation {other}"),
    }
}

/// The value of a `key: value` line of a golden file.
fn value<'a>(lines: &[(&str, &'a str)], key: &str) -> &'a str {
    lines.iter().find(|(k, _)| *k == key).unwrap_or_else(|| panic!("no {key}")).1
}

#[test]
fn every_jvm_golden_file() {
    let generator = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../golden/jvm/variants/detokenizer");
    let dir = common::golden_dir().join("detokenizer");
    let mut paths: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().path()).collect();
    paths.sort();
    let mut tokens_seen = 0;
    for path in &paths {
        let text = std::fs::read_to_string(path).unwrap();
        let lines: Vec<(&str, &str)> =
            text.lines().map(|l| l.split_once(": ").unwrap_or_else(|| (l.trim_end_matches(':'), ""))).collect();
        assert_eq!(value(&lines, "component"), "detokenizer");
        assert_eq!(value(&lines, "split-marker"), MARKER);
        let name = value(&lines, "dictionary");
        let xml = std::fs::read(generator.join("dictionaries").join(name)).unwrap();
        assert_eq!(turbo_maxent::sha256(&xml).to_hex(), value(&lines, "dictionary-sha256"), "{name}");
        let dictionary = DetokenizerDictionary::read(&xml).unwrap();

        let mut tokens = Vec::new();
        let mut operations = Vec::new();
        for (_, line) in lines.iter().filter(|(k, _)| *k == "token") {
            let (token, op) = line.rsplit_once(' ').unwrap();
            tokens.push(common::unescape(token));
            operations.push(operation(op));
        }
        assert_eq!(tokens.len(), value(&lines, "tokens").parse::<usize>().unwrap());
        // The case file the JVM read has the same tokens, one per line.
        let case = std::fs::read_to_string(generator.join("cases").join(path.file_name().unwrap())).unwrap();
        assert_eq!(case.lines().collect::<Vec<_>>(), tokens, "{}", path.display());

        let tokens: Vec<&str> = tokens.iter().map(String::as_str).collect();
        let plain = common::unescape(value(&lines, "text"));
        let marked = common::unescape(value(&lines, "text-marked"));
        check(&dictionary, &tokens, &operations, &plain, &marked);
        tokens_seen += tokens.len();
    }
    assert_eq!(paths.len(), 19);
    assert_eq!(tokens_seen, 144);
}
