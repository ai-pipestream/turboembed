//! OpenNLP's `GermanUmlautCharSequenceNormalizerTest`.

use super::*;

// ---------------------------------------------------------------------------
// GermanUmlautCharSequenceNormalizerTest
// ---------------------------------------------------------------------------

#[test]
fn lowercase_umlauts() {
    assert_eq!(umlauts("M\u{FC}ller"), "Mueller");
    assert_eq!(umlauts("sch\u{F6}n"), "schoen");
    assert_eq!(umlauts("\u{E4}rger"), "aerger");
}

#[test]
fn capital_umlauts() {
    assert_eq!(umlauts("\u{D6}l"), "Oel");
    assert_eq!(umlauts("\u{C4}rger"), "Aerger");
    assert_eq!(umlauts("\u{DC}ber"), "Ueber");
}

#[test]
fn sharp_s() {
    assert_eq!(umlauts("Stra\u{DF}e"), "Strasse");
    assert_eq!(umlauts("STRA\u{1E9E}E"), "STRASSE");
}

#[test]
fn capital_sharp_s_offsets() {
    // UTF-8: "A" is byte 0, U+1E9E bytes 1..4; "SS" in the output is 1..3.
    let (out, a) = normalize(&GermanUmlauts::new(), "A\u{1E9E}B");
    assert_eq!(out, "ASSB");
    assert_eq!(a.span_to_parent(Span::new(1, 3)), Span::new(1, 4));
    assert_eq!(edits(&a), [(1, 4, 1, 3)]);
}

#[test]
fn ascii_and_other_characters_unchanged() {
    assert_eq!(umlauts("hello world 123"), "hello world 123");
    assert_eq!(umlauts("\u{E9}\u{F8}"), "\u{E9}\u{F8}");
}

#[test]
fn mixed_sentence() {
    assert_eq!(umlauts("M\u{FC}ller Stra\u{DF}e"), "Mueller Strasse");
}

#[test]
fn decomposed_umlaut_passes_through_unless_composed_first() {
    let decomposed = "Mu\u{308}ller";
    assert_eq!(umlauts(decomposed), decomposed);
    assert_eq!(umlauts(&nfc(decomposed)), "Mueller");
}

#[test]
fn uppercase_aware_umlauts() {
    let n = GermanUmlauts::uppercase_aware();
    let cases = [
        ("\u{DC}BER", "UEBER"),
        ("\u{C4}RGER", "AERGER"),
        ("M\u{DC}", "MUE"),
        ("GR\u{DC}N!", "GRUEN!"),
        ("\u{C4}\u{D6}", "AEOE"),
        ("\u{DC}ber", "Ueber"),
        ("\u{D6}l", "Oel"),
        ("\u{D6}", "Oe"),
        ("\u{D6} L", "Oe L"),
        ("M\u{FC}LLER", "MueLLER"),
        // A combining mark after the umlaut is passed over.
        ("\u{DC}\u{301}BER", "UE\u{301}BER"),
        ("STRA\u{1E9E}E", "STRASSE"),
        ("Stra\u{DF}e", "Strasse"),
    ];
    for (src, expected) in cases {
        let (out, a) = normalize(&n, src);
        assert_eq!(out, expected, "{src:?}");
        check_alignment(src, &out, &a, |_, _| true);
    }
    assert_ne!(GermanUmlauts::new().config_hash(), n.config_hash());
}

const UMLAUT_POOL: &[&str] = &[
    "a",
    "B",
    " ",
    "\u{E4}",
    "\u{F6}",
    "\u{FC}",
    "\u{C4}",
    "\u{D6}",
    "\u{DC}",
    "\u{DF}",
    "\u{1E9E}",
    "\u{308}",
    "\u{E9}",
    "\u{10400}",
];

#[test]
fn generated_umlaut_texts_match_the_reference() {
    let table = |c: char| -> Option<&'static str> {
        Some(match c {
            '\u{E4}' => "ae",
            '\u{F6}' => "oe",
            '\u{FC}' => "ue",
            '\u{C4}' => "Ae",
            '\u{D6}' => "Oe",
            '\u{DC}' => "Ue",
            '\u{DF}' => "ss",
            '\u{1E9E}' => "SS",
            _ => return None,
        })
    };
    let either = |c: char| -> Vec<&'static str> {
        match c {
            '\u{C4}' => vec!["Ae", "AE"],
            '\u{D6}' => vec!["Oe", "OE"],
            '\u{DC}' => vec!["Ue", "UE"],
            _ => table(c).into_iter().collect(),
        }
    };
    let mut rng = Rng::new(21);
    for _ in 0..ROUNDS {
        let src = rng.text(UMLAUT_POOL, 10);
        let (out, a) = normalize(&GermanUmlauts::new(), &src);
        assert_eq!(out, reference_umlauts(&src), "{src:?}");
        check_alignment(&src, &out, &a, per_char_edit(table));
        // The uppercase-aware fold differs only in the case of the second
        // letter of a capital umlaut.
        let (out, a) = normalize(&GermanUmlauts::uppercase_aware(), &src);
        assert_eq!(out.to_lowercase(), reference_umlauts(&src).to_lowercase(), "{src:?}");
        check_alignment(&src, &out, &a, |parent, child| {
            let mut rest = child;
            for c in parent.chars() {
                match either(c).into_iter().find(|w| rest.starts_with(w)) {
                    Some(w) => rest = &rest[w.len()..],
                    None => return false,
                }
            }
            rest.is_empty()
        });
    }
}
