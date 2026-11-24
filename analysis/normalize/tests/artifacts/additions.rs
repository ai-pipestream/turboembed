//! The two kinds that are off by default, mojibake sequences and repeated
//! lines, and property loops over generated text.

use super::*;

// ---------------------------------------------------------------------------
// MojibakeSequence (off by default)

fn sequences(text: &str) -> Vec<Artifact> {
    ArtifactDetector::builder().with(MojibakeSequence).build().detect(text)
}

#[test]
fn sequences_are_off_by_default() {
    assert!(!KindSet::OPENNLP.contains(MojibakeSequence));
    let text = format!("caf{}\u{2026}", cp(&[0xC3, 0xA9]));
    assert_eq!(detect(&text), vec![]);
}

#[test]
fn sequence_next_to_a_genuine_character() {
    // The typed ellipsis is byte 0x85 in Windows-1252, a bare continuation
    // after the image of "é", so the whole run does not round-trip.
    let text = format!("caf{}\u{2026} and na{}\u{2014}ve", cp(&[0xC3, 0xA9]), cp(&[0xC3, 0xAF]));
    assert_eq!(
        covered(&text, &sequences(&text)),
        vec![("\u{C3}\u{A9}", MojibakeSequence), ("\u{C3}\u{AF}", MojibakeSequence)]
    );
}

#[test]
fn sequence_with_a_byte_lost_to_a_replacement_character() {
    // "Á" is C3 81; Windows-1252 leaves 0x81 undefined.
    let text = format!("{}LVAREZ", cp(&[0xC3, 0xFFFD]));
    assert_eq!(detect(&text), vec![a(2, 5, ReplacementChar)]);
    assert_eq!(sequences(&text), vec![a(0, 5, MojibakeSequence)]);
}

#[test]
fn damaged_japanese_is_one_sequence_run() {
    // こんにちは as UTF-8 read as Windows-1252, the undefined 0x81 bytes
    // replaced: the round trip fails at every U+FFFD.
    let text = windows_1252_decode("こんにちは".as_bytes());
    assert!(text.contains('\u{FFFD}'));
    assert!(detect(&text).iter().all(|f| f.kind == ReplacementChar));
    assert_eq!(sequences(&text), vec![a(0, text.len(), MojibakeSequence)]);
}

#[test]
fn mac_roman_sequences() {
    // é is C3 A9: √© in MacRoman. ’ is E2 80 99: ‚Äô.
    for (text, image) in [("caf\u{221A}\u{A9}", "\u{221A}\u{A9}"), ("don\u{201A}\u{C4}\u{F4}t", "\u{201A}\u{C4}\u{F4}")]
    {
        assert_eq!(detect(text), vec![], "{text}");
        assert_eq!(covered(text, &sequences(text)), vec![(image, MojibakeSequence)], "{text}");
    }
}

#[test]
fn round_trip_mojibake_takes_precedence_over_sequences() {
    let text = format!("caf{} {}", cp(&[0xC3, 0xA9]), cp(&[0xE2, 0x20AC, 0x2122]));
    assert_eq!(sequences(&text), detect(&text));
    assert!(sequences(&text).iter().all(|f| f.kind == Mojibake));
}

#[test]
fn genuine_text_has_no_sequences() {
    let texts = [
        "\u{201C}\u{E9}lan\u{201D} and \u{AB}\u{E9}t\u{E9}\u{BB}",
        "CAF\u{C9} \u{2013} BAR, \u{2014}\u{E0} bient\u{F4}t\u{2026}",
        "\u{2018}\u{E0}\u{2019} \u{2026}\u{E0}",
        "S\u{C3}O PAULO, \u{C5}NGSTR\u{D6}M, \u{C9}COLE",
        "Stra\u{DF}e \u{2022} Gr\u{FC}\u{DF}e \u{2122} \u{A9} 2024",
    ];
    for text in texts.iter().copied().chain(CLEAN_TEXT) {
        assert_eq!(sequences(text), vec![], "{text}");
    }
}

// The round trip of OpenNLP reports an accented capital before a dash or
// quote: "É–" is C9 96, the UTF-8 of U+0256. The default keeps that for
// parity; the sequence rule alone would not report it (U+0256 is outside
// its two-byte target ranges).
#[test]
fn round_trip_reports_an_accented_capital_before_a_dash() {
    let text = "CAF\u{C9}\u{2013}BAR";
    assert_eq!(detect(text), vec![a(3, 8, Mojibake)]);
    assert_eq!(ArtifactDetector::builder().only(&[MojibakeSequence]).build().detect(text), vec![]);
}

#[test]
fn sequences_take_precedence_over_classes() {
    // U+0081 is a control and the continuation of "Á" after Ã; with the
    // genuine ellipsis after it the run does not round-trip.
    let text = format!("{}\u{2026}", cp(&[0xC3, 0x81]));
    assert_eq!(detect(&text), vec![a(2, 4, Control)]);
    assert_eq!(sequences(&text), vec![a(0, 4, MojibakeSequence)]);
}

// ---------------------------------------------------------------------------
// RepeatedLine (off by default)

fn report(pages: usize) -> String {
    let mut text = String::new();
    for page in 1..=pages {
        text.push_str("ACME Corporation  Annual Report 2024\n");
        text.push_str(&format!("Body text of page {page}, which differs: {}.\n", "x".repeat(page)));
        text.push_str(&format!("{page}\n"));
        text.push_str(&format!("   Page {page} of {pages}\r\n\n"));
    }
    text
}

fn repeated(text: &str, min: u32) -> Vec<Artifact> {
    ArtifactDetector::builder().repeated_lines(RepeatedLines { min_occurrences: min }).build().detect(text)
}

#[test]
fn repeated_lines_are_off_by_default() {
    assert_eq!(detect(&report(4)), vec![]);
}

#[test]
fn headers_and_page_labels_repeat() {
    let text = report(3);
    let found = repeated(&text, 3);
    let lines: Vec<&str> = found.iter().map(|f| &text[f.span.range()]).collect();
    assert_eq!(
        lines,
        [
            "ACME Corporation  Annual Report 2024",
            "Page 1 of 3",
            "ACME Corporation  Annual Report 2024",
            "Page 2 of 3",
            "ACME Corporation  Annual Report 2024",
            "Page 3 of 3",
        ]
    );
    assert!(found.iter().all(|f| f.kind == RepeatedLine));
    check_layer(&text, &ArtifactDetector::builder().repeated_lines(RepeatedLines::default()).build());
}

#[test]
fn two_occurrences_are_not_enough_by_default() {
    assert_eq!(repeated(&report(2), 3), vec![]);
    assert_eq!(repeated(&report(2), 2).len(), 4);
}

#[test]
fn inner_whitespace_and_digit_runs_are_folded() {
    let text = "Page 1 of 9\nPage  22 of 9\nPage\t333 of 10\nPage one of nine\n";
    let found = repeated(text, 3);
    assert_eq!(found.len(), 3);
    assert_eq!(&text[found[2].span.range()], "Page\t333 of 10");
}

#[test]
fn lines_without_letters_are_not_compared() {
    let text = "12.5\n13.7\n8.2\n- 4 -\n- 5 -\n- 6 -\n";
    assert_eq!(repeated(text, 3), vec![]);
}

#[test]
fn character_findings_inside_a_repeated_line_keep_their_spans() {
    let text = "Confidential\u{0}Draft\nConfidential\u{0}Draft\nConfidential\u{0}Draft\n";
    let found = repeated(text, 3);
    assert_eq!(found[..3], [a(0, 12, RepeatedLine), a(12, 13, Control), a(13, 18, RepeatedLine)]);
    assert_eq!(found.len(), 9);
}

// ---------------------------------------------------------------------------
// Property loops

/// A deterministic generator (xorshift64*), so a failure reproduces.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Characters of every class and of every context the rules look at.
const POOL: &[char] = &[
    'a',
    'b',
    'z',
    'Q',
    ' ',
    '.',
    '\n',
    '\t',
    '\r',
    '1', // ordinary
    '\u{FFFD}',
    '\u{0}',
    '\u{1B}',
    '\u{7F}',
    '\u{85}',
    '\u{9F}',
    '\u{1C}', // replacement, controls, NEL
    '\u{FDD0}',
    '\u{FFFE}',
    '\u{1FFFF}',
    '\u{10FFFF}', // noncharacters
    '\u{E000}',
    '\u{F8FF}',
    '\u{F0000}',
    '\u{100000}', // private use
    '\u{202E}',
    '\u{2068}',
    '\u{200E}',
    '\u{061C}', // bidi
    '\u{200B}',
    '\u{200C}',
    '\u{200D}',
    '\u{2060}',
    '\u{FEFF}', // zero width
    '\u{E0067}',
    '\u{E0062}',
    '\u{E0065}',
    '\u{E006E}',
    '\u{E007F}',
    '\u{E0031}',
    '\u{1F3F4}', // tags
    '\u{1F600}',
    '\u{FE0F}',
    '\u{2764}', // emoji context
    '\u{301}',
    '\u{E01}',
    '\u{628}',
    '\u{10400}', // marks and letters
    '\u{C3}',
    '\u{C2}',
    '\u{E2}',
    '\u{F0}',
    '\u{A9}',
    '\u{AF}',
    '\u{80}',
    '\u{20AC}',
    '\u{2122}',
    '\u{9F}',
    '\u{98}',
    '\u{2026}',
    '\u{221A}',
    '\u{201A}',
    '\u{C4}',
    '\u{F4}',
    '\u{E9}', // mojibake images
];

fn generate(rng: &mut Rng) -> String {
    let n = rng.below(40);
    (0..n).map(|_| POOL[rng.below(POOL.len())]).collect()
}

/// The class of a code point, written out apart from the detector.
fn reference_class(c: char) -> Option<ArtifactKind> {
    let p = c as u32;
    match p {
        0xFFFD => Some(ReplacementChar),
        0x09..=0x0D | 0x85 => None,
        0..=0x1F | 0x7F..=0x9F => Some(Control),
        0xFDD0..=0xFDEF => Some(Noncharacter),
        _ if p & 0xFFFE == 0xFFFE => Some(Noncharacter),
        0xE000..=0xF8FF | 0xF0000..=0xFFFFD | 0x100000..=0x10FFFD => Some(PrivateUse),
        0x202A..=0x202E | 0x2066..=0x2069 | 0x200E | 0x200F | 0x061C => Some(BidiControl),
        _ => None,
    }
}

/// Whether the characters' Windows-1252 bytes are valid multi-byte UTF-8.
fn round_trips(s: &str) -> bool {
    let bytes: Vec<u8> = s
        .chars()
        .map(|c| {
            let p = c as u32;
            if p <= 0xFF {
                p as u8
            } else {
                let decoded = (0x80..0xA0u8).find(|&b| windows_1252_decode(&[b]).starts_with(c));
                decoded.expect("a Windows-1252 character")
            }
        })
        .collect();
    std::str::from_utf8(&bytes).is_ok_and(|t| !t.is_ascii())
}

const CLASS_KINDS: [ArtifactKind; 5] = [ReplacementChar, Control, Noncharacter, PrivateUse, BidiControl];

fn check_structure(text: &str, found: &[Artifact]) {
    for f in found {
        assert!(!f.span.is_empty() && f.span.end as usize <= text.len(), "{text:?} {f:?}");
        assert!(text.is_char_boundary(f.span.start as usize) && text.is_char_boundary(f.span.end as usize));
    }
    assert!(found.windows(2).all(|w| w[0].span.end <= w[1].span.start), "{text:?} {found:?}");
}

fn covering(found: &[Artifact], at: usize) -> Option<ArtifactKind> {
    found.iter().find(|f| f.span.range().contains(&at)).map(|f| f.kind)
}

#[test]
fn generated_text_default_detector() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let detector = ArtifactDetector::default();
    for _ in 0..4000 {
        let text = generate(&mut rng);
        let found = detector.detect(&text);
        check_structure(&text, &found);
        check_layer(&text, &detector);
        for f in &found {
            let s = &text[f.span.range()];
            match f.kind {
                k if CLASS_KINDS.contains(&k) => assert!(s.chars().all(|c| reference_class(c) == Some(k)), "{s:?}"),
                ZeroWidth => assert!(s.chars().all(|c| "\u{200B}\u{200C}\u{200D}\u{2060}\u{FEFF}".contains(c))),
                UnicodeTag => assert!(s.chars().all(|c| ('\u{E0000}'..='\u{E007F}').contains(&c))),
                Mojibake => assert!(round_trips(s), "{s:?}"),
                other => panic!("{other:?} is not a default kind"),
            }
        }
        // Every classified character is reported, as its class or inside
        // mojibake, and a class run ends where its class ends.
        for (i, c) in text.char_indices() {
            if let Some(k) = reference_class(c) {
                assert!(matches!(covering(&found, i), Some(x) if x == k || x == Mojibake), "{text:?} at {i}");
            }
        }
        for f in found.iter().filter(|f| CLASS_KINDS.contains(&f.kind)) {
            let after = text[f.span.end as usize..].chars().next();
            if after.and_then(reference_class) == Some(f.kind) {
                assert_eq!(covering(&found, f.span.end as usize), Some(Mojibake), "{text:?}");
            }
        }
    }
}

#[test]
fn generated_text_each_kind_alone() {
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    let without_mojibake = ArtifactDetector::builder().without(Mojibake).build();
    let mojibake_only = ArtifactDetector::builder().only(&[Mojibake]).build();
    let default = ArtifactDetector::default();
    for _ in 0..3000 {
        let text = generate(&mut rng);
        let all = without_mojibake.detect(&text);
        for kind in [ReplacementChar, Control, Noncharacter, PrivateUse, BidiControl, ZeroWidth, UnicodeTag] {
            let alone = only(&[kind], &text);
            let filtered: Vec<Artifact> = all.iter().copied().filter(|f| f.kind == kind).collect();
            assert_eq!(alone, filtered, "{kind:?} in {text:?}");
        }
        let full: Vec<Artifact> = default.detect(&text).into_iter().filter(|f| f.kind == Mojibake).collect();
        assert_eq!(mojibake_only.detect(&text), full, "{text:?}");
    }
}

#[test]
fn generated_text_with_the_added_kinds() {
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let default = ArtifactDetector::default();
    let extended = ArtifactDetector::builder().with(MojibakeSequence).repeated_lines(RepeatedLines::default()).build();
    for _ in 0..3000 {
        let mut text = generate(&mut rng);
        // Repeat a generated line now and then so repeated lines occur.
        if rng.below(4) == 0 {
            let line = format!("Header {}\n", generate(&mut rng).replace(['\n', '\r'], " "));
            text = format!("{line}{text}\n{line}{line}");
        }
        let base = default.detect(&text);
        let found = extended.detect(&text);
        check_structure(&text, &found);
        check_layer(&text, &extended);
        // Round-trip findings are unchanged; every other default finding is
        // still covered, by its own kind or by a mojibake sequence.
        let moji = |v: &[Artifact]| v.iter().copied().filter(|f| f.kind == Mojibake).collect::<Vec<_>>();
        assert_eq!(moji(&found), moji(&base), "{text:?}");
        for f in &base {
            for (i, _) in text[f.span.range()].char_indices() {
                let at = f.span.start as usize + i;
                assert!(matches!(covering(&found, at), Some(k) if k == f.kind || k == MojibakeSequence), "{text:?}");
            }
        }
        for f in found.iter().filter(|f| f.kind == RepeatedLine) {
            assert!(!text[f.span.range()].contains('\n'), "{text:?}");
        }
    }
}

#[test]
fn every_kind_has_a_set_bit() {
    let mut set = KindSet::default();
    for k in ArtifactKind::ALL {
        assert!(!set.contains(k));
        set = set.with(k);
        assert!(set.contains(k));
    }
    assert_eq!(set, KindSet::ALL);
    assert!(!set.without(Mojibake).contains(Mojibake));
}
