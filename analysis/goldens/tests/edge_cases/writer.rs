use super::*;

// ---------------------------------------------------------------------
// Round trips

fn f64_specials() -> Vec<f64> {
    vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        0.1,
        0.1 + 0.2,
        1.0 / 3.0,
        2.0 / 3.0,
        f64::NAN,
        -f64::NAN,
        f64::from_bits(0x7ff8000000000001),
        f64::from_bits(0xfff8000000000001),
        f64::from_bits(0x7ff0000000000001),
        f64::from_bits(0xfff0000000000001),
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::from_bits(0x000fffffffffffff),
        f64::MAX,
        f64::MIN,
        f64::EPSILON,
        1e-3,
        9.999999999999999e-4,
        1e7,
        9999999.0,
        123456.789,
        1e23,
        9.223372036854776e18,
        1e-5,
        100.0,
        1.0e300,
        1.0e-300,
    ]
}

fn escape_strings() -> Vec<String> {
    let mut v: Vec<String> = (0u32..0x20).map(|c| format!("<{}>", char::from_u32(c).unwrap())).collect();
    v.extend(
        [
            "\\",
            "\\\\",
            "\\n",
            "\\u0041",
            "\\ud83d\\ude00",
            "tab\t",
            "\tlead",
            "a\tb\tc",
            "line\r\nend",
            "\n",
            "\r",
            "\t",
            "",
            " ",
            "  ",
            "a b",
            "é",
            "日本語",
            "Ａ",
            "😀",
            "😀😁",
            "\u{10000}",
            "\u{10ffff}",
            "\u{ffff}",
            "\u{fffe}",
            "\u{e000}",
            "\u{d7ff}",
            "\u{7f}",
            "\u{80}",
            "\u{85}",
            "\u{a0}",
            "\u{2028}",
            "\u{2029}",
            "\u{feff}",
            "e\u{301}",
            "\"quote\"",
            "'",
            "/",
            "a\\",
            "\\a",
            "\0",
            "\0\0",
            "x\0y",
            "tab\\tnot",
            "<>",
            "&amp;",
            "\u{1}\u{2}\u{3}",
        ]
        .map(String::from),
    );
    v
}

fn strs(items: &[String]) -> Strs {
    let mut s = Strs::default();
    for i in items {
        s.push(i);
    }
    s
}

/// A golden holding every column type in several layers, with every
/// special f64 and every escape.
pub(crate) fn full_golden() -> Golden {
    let escapes = escape_strings();
    let floats = f64_specials();
    let n = escapes.len();
    let at = |i: usize| floats[i % floats.len()];
    let u32s: Vec<u32> = (0..n as u32).collect();
    let ends: Vec<u32> = u32s.iter().map(|&i| i + 3).collect();
    let layers = vec![
        GoldenLayer {
            id: "opennlp:tokens".into(),
            anchor: GoldenAnchor::Text,
            columns: vec![
                Column { name: "start".into(), data: ColumnData::U32(u32s.clone()) },
                Column { name: "end".into(), data: ColumnData::U32(ends.clone()) },
            ],
        },
        GoldenLayer {
            id: "turbo:confidence<opennlp:tokens>".into(),
            anchor: GoldenAnchor::Aligned("opennlp:tokens".into()),
            columns: vec![Column { name: "value".into(), data: ColumnData::F64((0..n).map(at).collect()) }],
        },
        GoldenLayer {
            id: "opennlp:pos".into(),
            anchor: GoldenAnchor::Aligned("opennlp:tokens".into()),
            columns: vec![
                Column { name: "tag".into(), data: ColumnData::Str(strs(&escapes)) },
                Column { name: "prob".into(), data: ColumnData::F64((0..n).map(|i| at(i + 7)).collect()) },
            ],
        },
        GoldenLayer {
            id: "opennlp:lemmas".into(),
            anchor: GoldenAnchor::Aligned("opennlp:tokens".into()),
            columns: vec![Column { name: "lemma".into(), data: ColumnData::Str(strs(&escapes)) }],
        },
        GoldenLayer {
            id: "opennlp:entities".into(),
            anchor: GoldenAnchor::Text,
            columns: vec![
                Column { name: "start".into(), data: ColumnData::U32(u32s.clone()) },
                Column { name: "end".into(), data: ColumnData::U32(ends.clone()) },
                Column { name: "token_start".into(), data: ColumnData::U32(vec![u32::MAX; n]) },
                Column {
                    name: "token_end".into(),
                    data: ColumnData::U32((0..n as u32).map(|i| u32::MAX - i).collect()),
                },
                Column { name: "kind".into(), data: ColumnData::Str(strs(&escapes)) },
                Column { name: "prob".into(), data: ColumnData::F64((0..n).map(|i| at(i + 3)).collect()) },
            ],
        },
        GoldenLayer {
            id: "opennlp:language".into(),
            anchor: GoldenAnchor::Document,
            columns: vec![
                Column { name: "code".into(), data: ColumnData::Str(strs(&escapes)) },
                Column { name: "confidence".into(), data: ColumnData::F64((0..n).map(|i| at(i + 11)).collect()) },
            ],
        },
        GoldenLayer {
            id: "turbo:view".into(),
            anchor: GoldenAnchor::Text,
            columns: vec![
                Column { name: "start".into(), data: ColumnData::U32(u32s) },
                Column { name: "end".into(), data: ColumnData::U32(ends) },
                Column { name: "with".into(), data: ColumnData::Str(strs(&escapes)) },
            ],
        },
    ];
    Golden {
        header: Header {
            generator: Generator { name: "turbo goldens".into(), version: u32::MAX },
            opennlp: OpenNlp { commit: [0xab; 20], version: "3.0.0-PRE-SNAPSHOT".into() },
            jdk: Jdk {
                version: "21.0.5".into(),
                vm_version: "21.0.5+11-LTS".into(),
                vendor: "Eclipse Adoptium".into(),
            },
            unicode: UnicodeVersion { major: 16, minor: 0 },
            input: Input { set: "s".into(), id: "an id".into(), utf8_len: 1000, utf16_len: 900, sha256: [7; 32] },
            models: vec![
                Model { file: "a.bin".into(), sha512: [0xff; 64] },
                Model { file: "b c.bin".into(), sha512: [0x00; 64] },
            ],
            annotator: AnnotatorRef { class: "a.b.C".into(), method: "m".into() },
            config: vec![
                ConfigEntry { key: String::new(), value: String::new() },
                ConfigEntry { key: "a".into(), value: "1".into() },
                ConfigEntry { key: "b\t".into(), value: "x\ny\\z\r".into() },
                ConfigEntry { key: "é".into(), value: "\u{1}\u{1f}".into() },
                ConfigEntry { key: "😀".into(), value: "😀\t😀".into() },
            ],
            error: None,
        },
        layers,
    }
}

#[test]
fn every_column_type_special_value_and_escape_round_trips() {
    let g = full_golden();
    let text = g.to_text().unwrap();
    let back = Golden::parse(&text).unwrap();
    assert_eq!(back, g);
    // And again: the second text is the first, byte for byte.
    assert_eq!(back.to_text().unwrap(), text);
    // Bytes, not only text.
    let mut bytes = Vec::new();
    g.write(&mut bytes).unwrap();
    assert_eq!(bytes, text.as_bytes());
    assert_eq!(Golden::parse_bytes(&bytes).unwrap(), g);

    // f64 values come back with the same bits (PartialEq on ColumnData
    // compares bits, and this checks it did).
    let floats = f64_specials();
    let conf = back.layer("turbo:confidence<opennlp:tokens>").unwrap().f64s("value").unwrap();
    for (i, v) in conf.iter().enumerate() {
        assert_eq!(v.to_bits(), floats[i % floats.len()].to_bits(), "row {i}");
    }
    assert!(conf.iter().any(|v| v.is_nan() && v.to_bits() == 0xfff8000000000001));
    assert!(conf.iter().any(|v| v.to_bits() == (-0.0f64).to_bits()));

    // Strings come back as they were.
    let tags = back.layer("opennlp:pos").unwrap().strs("tag").unwrap();
    for (i, s) in escape_strings().iter().enumerate() {
        assert_eq!(tags.get(i), s, "row {i}");
    }
    assert_eq!(back.header.config, g.header.config);
    assert_eq!(back.header.input.id, "an id");
    assert_eq!(back.header.generator.version, u32::MAX);
}

#[test]
fn the_written_form_has_no_stray_control_characters() {
    let text = full_golden().to_text().unwrap();
    assert!(text.ends_with('\n'));
    assert!(!text.contains('\r'));
    // Every line has only tabs as control characters, and each row has the
    // fields its layer declares.
    let mut columns = 0;
    for (i, line) in text.lines().enumerate() {
        assert!(!line.chars().any(|c| c < ' ' && c != '\t'), "line {}: {line:?}", i + 1);
        let fields: Vec<&str> = line.split('\t').collect();
        match fields[0] {
            "layer" => columns = fields.len() - 4,
            "end" | "golden 1" | "generator" | "opennlp" | "jdk" | "unicode" | "input" | "model" | "annotator"
            | "config" => {}
            _ => assert_eq!(fields.len(), columns, "line {}: {line:?}", i + 1),
        }
    }
    // Checks on a few of the escapes in the text.
    assert!(text.contains("\n<\\u0000>\t"));
    assert!(text.contains("\n<\\t>\t"));
    assert!(text.contains("\n<\\u001f>\t"));
    assert!(text.contains("\nline\\r\\nend\t"));
    assert!(text.contains("\n\\\\u0041\t"));
    assert!(text.contains("\n\\\\ud83d\\\\ude00\t"));
    assert!(text.contains("\n😀😁\t"));
    assert!(text.contains("\n\u{7f}\t"));
    assert!(text.contains("\nconfig\t\t\n"));
    assert!(text.contains("\nconfig\tb\\t\tx\\ny\\\\z\\r\n"));
}

#[test]
fn java_decimals_for_values_in_every_format() {
    let g = full_golden();
    let text = g.to_text().unwrap();
    // Java's Double.toString on JDK 19 and later, for the values above.
    for (bits, shown) in [
        (0.0f64.to_bits(), "0.0"),
        ((-0.0f64).to_bits(), "-0.0"),
        (1.0f64.to_bits(), "1.0"),
        (0.1f64.to_bits(), "0.1"),
        ((0.1f64 + 0.2).to_bits(), "0.30000000000000004"),
        (f64::NAN.to_bits(), "NaN"),
        (0x7ff8000000000001, "NaN"),
        (0xfff0000000000001, "NaN"),
        (f64::INFINITY.to_bits(), "Infinity"),
        (f64::NEG_INFINITY.to_bits(), "-Infinity"),
        (f64::MIN_POSITIVE.to_bits(), "2.2250738585072014E-308"),
        (1u64, "4.9E-324"),
        (f64::MAX.to_bits(), "1.7976931348623157E308"),
        (f64::MIN.to_bits(), "-1.7976931348623157E308"),
        (f64::EPSILON.to_bits(), "2.220446049250313E-16"),
        (1e-3f64.to_bits(), "0.001"),
        (1e7f64.to_bits(), "1.0E7"),
        (9999999.0f64.to_bits(), "9999999.0"),
        (123456.789f64.to_bits(), "123456.789"),
        (1e23f64.to_bits(), "1.0E23"),
        (9.223372036854776e18f64.to_bits(), "9.223372036854776E18"),
        (1e-5f64.to_bits(), "1.0E-5"),
        (100.0f64.to_bits(), "100.0"),
    ] {
        let line = format!("{bits:016x} {shown}\n");
        assert!(text.contains(&line), "{line:?} is not in the written golden");
    }
}

#[test]
fn a_surrogate_pair_in_a_file_reads_as_one_character_and_writes_as_utf8() {
    let g = Golden::parse(&lemma_file("\\ud83d\\ude00")).unwrap();
    let lemma = g.layer("opennlp:lemmas").unwrap().strs("lemma").unwrap();
    assert_eq!(lemma.get(0), "😀");
    let again = g.to_text().unwrap();
    assert!(again.contains("\n😀\n"), "{again}");
    assert!(!again.contains("\\ud83d"));
    assert_eq!(Golden::parse(&again).unwrap(), g);
    // In a config value, too.
    let lines: Vec<&str> = HEADER.lines().collect();
    let text = lines.join("\n") + "\nconfig\tk\t\\ud83d\\ude00\\ud83d\\ude01\n" + &sentences(&["0\t1"]);
    let g = Golden::parse(&text).unwrap();
    assert_eq!(g.header.config[0].value, "😀😁");
}

#[test]
fn a_cloned_golden_writes_the_same_bytes() {
    // Reading, cloning, writing and reading again changes nothing.
    let g = Golden::parse(&fixture_text()).unwrap();
    let c = g.clone();
    assert_eq!(c.to_text().unwrap(), fixture_text());
    assert_eq!(Golden::parse(&c.to_text().unwrap()).unwrap(), g);
}

// ---------------------------------------------------------------------
// Writer refusals

type Change = Box<dyn Fn(&mut Golden, &str)>;

fn refuses_write(g: &Golden, parts: &[&str]) {
    let e = g.to_text().expect_err("the writer should refuse");
    assert_eq!(e.kind(), std::io::ErrorKind::InvalidInput, "{e}");
    let shown = e.to_string();
    for p in parts {
        assert!(shown.contains(p), "{p:?} is not in {shown:?}");
    }
    // `write` refuses too, and writes nothing.
    let mut out = Vec::new();
    assert!(g.write(&mut out).is_err());
    assert!(out.is_empty());
}

#[test]
fn the_writer_refuses_a_tab_or_line_end_in_any_header_field() {
    for bad in ["a\tb", "a\nb", "a\rb", "\t", "\n"] {
        let fields: Vec<(&str, Change)> = vec![
            ("generator", Box::new(|g, s| g.header.generator.name = s.into())),
            ("opennlp", Box::new(|g, s| g.header.opennlp.version = s.into())),
            ("jdk", Box::new(|g, s| g.header.jdk.version = s.into())),
            ("jdk", Box::new(|g, s| g.header.jdk.vm_version = s.into())),
            ("jdk", Box::new(|g, s| g.header.jdk.vendor = s.into())),
            ("input", Box::new(|g, s| g.header.input.set = s.into())),
            ("input", Box::new(|g, s| g.header.input.id = s.into())),
            ("model", Box::new(|g, s| g.header.models[0].file = s.into())),
            ("annotator", Box::new(|g, s| g.header.annotator.class = s.into())),
            ("annotator", Box::new(|g, s| g.header.annotator.method = s.into())),
        ];
        for (record, set) in fields {
            let mut g = full_golden();
            set(&mut g, bad);
            refuses_write(&g, &[&format!("the {record} record"), "a tab or a line end"]);
        }
    }
    // Config keys and values are escaped, so they are fine.
    let g = full_golden();
    assert!(g.header.config.iter().any(|c| c.key.contains('\t')));
    g.to_text().unwrap();
}

#[test]
fn the_writer_refuses_a_slash_in_the_input_set() {
    let mut g = full_golden();
    g.header.input.set = "a/b".into();
    refuses_write(&g, &["the input set", "'/'"]);
    // A slash in the id is fine: the reader splits at the first one.
    let mut g = full_golden();
    g.header.input.id = "a/b".into();
    let back = Golden::parse(&g.to_text().unwrap()).unwrap();
    assert_eq!((back.header.input.set.as_str(), back.header.input.id.as_str()), ("s", "a/b"));
}

#[test]
fn the_writer_refuses_bad_layers() {
    // A tab or a line end in the layer id.
    for bad in ["a\tb", "a\nb", "a\rb"] {
        let mut g = full_golden();
        g.layers[0].id = bad.into();
        refuses_write(&g, &["the layer id", "a tab or a line end"]);
    }
    // No columns.
    let mut g = full_golden();
    g.layers[0].columns.clear();
    refuses_write(&g, &["opennlp:tokens has no columns"]);
    // Columns of different lengths, in any of the columns after the first,
    // naming the column.
    for (layer, col) in [(0, 1), (2, 1), (4, 5), (6, 2)] {
        let mut g = full_golden();
        let l = &mut g.layers[layer];
        let n = l.len();
        l.columns[col].data = match l.columns[col].data.ty() {
            ColumnType::U32 => ColumnData::U32(vec![0; n - 1]),
            ColumnType::F64 => ColumnData::F64(vec![0.0; n + 1]),
            ColumnType::Str => ColumnData::Str(Strs::default()),
        };
        let name = g.layers[layer].columns[col].name.clone();
        let id = g.layers[layer].id.clone();
        refuses_write(&g, &[&format!("the layer {id} has {n} rows in its first column"), &format!("in {name}")]);
    }
    // A column name with a tab, a line end or a colon.
    for bad in ["a\tb", "a\nb", "a\rb", "a:b"] {
        let mut g = full_golden();
        g.layers[0].columns[0].name = bad.into();
        refuses_write(&g, &["the column name"]);
    }
    // A shorter first column than the others is caught the same way.
    let mut g = full_golden();
    g.layers[0].columns[0].data = ColumnData::U32(vec![]);
    refuses_write(&g, &["has 0 rows in its first column"]);
}

#[test]
fn the_writer_refuses_config_keys_out_of_order_or_repeated() {
    let mut g = full_golden();
    g.header.config.swap(1, 2);
    refuses_write(&g, &["the config key", "keys are sorted, each once"]);
    let mut g = full_golden();
    g.header.config[2].key = "a".into();
    refuses_write(&g, &["the config key \"a\" comes after \"a\""]);
    // An empty list and one entry are fine.
    let mut g = full_golden();
    g.header.config.clear();
    g.to_text().unwrap();
    g.header.config.push(ConfigEntry { key: "k".into(), value: "v".into() });
    g.to_text().unwrap();
}

#[test]
fn the_writer_writes_a_layer_with_no_rows_and_a_file_with_no_layers() {
    let mut g = full_golden();
    for l in &mut g.layers {
        for c in &mut l.columns {
            c.data = ColumnData::empty(c.data.ty());
        }
    }
    let text = g.to_text().unwrap();
    assert!(text.contains("layer\topennlp:tokens\ttext\t0\tstart:u32\tend:u32\nend\topennlp:tokens\n"));
    assert_eq!(Golden::parse(&text).unwrap(), g);
    // A golden with no layers and no error record is refused by the writer.
    g.layers.clear();
    refuses_write(&g, &["has no layers"]);
}

#[test]
fn the_writer_refuses_a_tab_or_line_end_in_an_aligned_anchor() {
    for bad in ["a\tb", "a\nb", "a\rb"] {
        let mut g = full_golden();
        g.layers[1].anchor = GoldenAnchor::Aligned(bad.into());
        let e = g.to_text().expect_err("the writer should refuse an anchor that breaks the record");
        assert_eq!(e.kind(), std::io::ErrorKind::InvalidInput);
    }
}
