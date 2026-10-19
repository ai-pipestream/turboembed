use super::*;

/// The features of the candidate at UTF-16 index `unit`, as strings
/// (lone surrogates shown as `\uXXXX`).
fn features(text: &str, unit: usize, dict: Option<&Dictionary>, mode: WhitespaceMode) -> Vec<String> {
    let eos = EndOfSentence::default();
    let ctx = SdContext::new(&eos, dict, mode, Limits::NONE);
    let mut positions = Vec::new();
    let scanned = eos.positions(text, &mut positions);
    let at = *positions.iter().find(|p| p.unit as usize == unit).expect("a candidate at that index");
    let mut out = Features::new();
    ctx.context(scanned, at, &mut Walks::new(), &mut out, &mut Lookup::default());
    out.iter().map(show).collect()
}

/// Modified UTF-8 back to a string, a surrogate written as an escape.
fn show(bytes: &[u8]) -> String {
    let mut s = String::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == 0xED && i + 2 < bytes.len() && bytes[i + 1] >= 0xA0 {
            let u = (u32::from(bytes[i] & 0x0F) << 12)
                | (u32::from(bytes[i + 1] & 0x3F) << 6)
                | u32::from(bytes[i + 2] & 0x3F);
            s.push_str(&format!("\\u{u:04X}"));
            i += 3;
        } else {
            let len = match bytes[i] {
                0..0x80 => 1,
                0xC0..0xE0 => 2,
                _ => 3,
            };
            s.push_str(std::str::from_utf8(&bytes[i..i + len]).unwrap());
            i += len;
        }
    }
    s
}

fn dictionary() -> Dictionary {
    Dictionary::read(
        br#"<dictionary case_sensitive="false"><entry><token>Mr.</token></entry><entry><token>U.S.</token></entry></dictionary>"#,
    )
    .unwrap()
}

const UNICODE: WhitespaceMode = WhitespaceMode::Unicode;

#[test]
fn plain_sentence_end() {
    // Worked from the Java: "Fine." at the end of the text.
    assert_eq!(
        features("It was fine. Then", 11, None, UNICODE),
        ["sn", "eos=.", "x=fine", "4", "v=was", "s=", "n=Then", "ncap"]
    );
    assert_eq!(features("sharp. Fine.", 11, None, UNICODE), ["eos=.", "x=Fine", "4", "xcap", "v=sharp.", "s=", "n="]);
}

#[test]
fn abbreviations_from_the_dictionary() {
    let dict = dictionary();
    // From the fixture context file: the prefix with the end character
    // is looked up, and the next word as it stands.
    assert_eq!(
        features("MR. SMITH", 2, Some(&dict), UNICODE),
        ["sn", "eos=.", "x=MR", "2", "xcap", "xabbrev", "v=", "s=", "n=SMITH", "ncap"]
    );
    assert_eq!(
        features("jones met. Mr. X", 9, Some(&dict), UNICODE),
        ["sn", "eos=.", "x=met", "3", "v=jones", "s=", "n=Mr.", "ncap", "nabbrev"]
    );
    // Without a dictionary there are no abbreviation flags.
    assert_eq!(
        features("MR. SMITH", 2, None, UNICODE),
        ["sn", "eos=.", "x=MR", "2", "xcap", "v=", "s=", "n=SMITH", "ncap"]
    );
}

#[test]
fn runs_of_end_characters() {
    // The prefix starts at the nearest end character before the
    // candidate, and includes it; the suffix stops at the next one.
    assert_eq!(
        features("at 9 A.M. sharp", 6, None, UNICODE),
        ["eos=.", "x=A", "1", "xcap", "v=9", "s=M", "scap", "n=sharp"]
    );
    assert_eq!(
        features("at 9 A.M. sharp", 8, None, UNICODE),
        ["sn", "eos=.", "x=.M", "2", "v=A", "vcap", "s=", "n=sharp"]
    );
    // The previous word is everything back to the whitespace before
    // the prefix, end characters and all.
    assert_eq!(
        features("A.M. sharp. Fine.", 10, None, UNICODE),
        ["sn", "eos=.", "x=sharp", "5", "v=A.M.", "vcap", "s=", "n=Fine.", "ncap"]
    );
}

#[test]
fn line_breaks_are_escaped() {
    let eos = EndOfSentence::from_chars("\n.").unwrap();
    let ctx = SdContext::new(&eos, None, UNICODE, Limits::NONE);
    let mut positions = Vec::new();
    let scanned = eos.positions("a\nb", &mut positions);
    let mut out = Features::new();
    ctx.context(scanned, positions[0], &mut Walks::new(), &mut out, &mut Lookup::default());
    // The Java computes the next word up to one unit before the end of
    // the text here, so the "b" is in neither the suffix nor the next word.
    assert_eq!(out.iter().map(show).collect::<Vec<_>>(), ["eos=<LF>", "x=a", "1", "v=", "s=", "n="]);
}

#[test]
fn a_lone_high_surrogate_after_the_last_period() {
    // From the fixture context file: the suffix stops one code unit
    // before the end of the text, inside the emoji.
    assert_eq!(
        features("here. Last.\u{1F600}", 10, None, UNICODE),
        ["eos=.", "x=Last", "4", "xcap", "v=here.", "s=\\uD83D", "n="]
    );
    // Further from the end the whole character is in the suffix, as its
    // two surrogates in modified UTF-8.
    assert_eq!(
        features("It ended.\u{1F600} Next", 8, None, UNICODE),
        ["eos=.", "x=ended", "5", "v=It", "vcap", "s=\\uD83D\\uDE00", "n=Next", "ncap"]
    );
}

#[test]
fn whitespace_mode_changes_words() {
    // U+001C is whitespace only in the legacy mode; U+0085 only in the
    // Unicode mode. From the fixture context files.
    let text = "Alpha.\u{1C}Beta.\u{1F} Gamma.";
    assert_eq!(
        features(text, 11, None, UNICODE),
        ["eos=.", "x=.\u{1C}Beta", "6", "v=Alpha", "vcap", "s=", "n=Gamma.", "ncap"]
    );
    assert_eq!(
        features(text, 11, None, WhitespaceMode::Legacy),
        ["sn", "eos=.", "x=Beta", "4", "xcap", "v=Alpha.", "vcap", "s=", "n=Gamma.", "ncap"]
    );
    let text = "First line.\u{85}Second line.";
    assert_eq!(
        features(text, 10, None, UNICODE),
        ["sn", "eos=.", "x=line", "4", "v=First", "vcap", "s=", "n=Second", "ncap"]
    );
    assert_eq!(
        features(text, 10, None, WhitespaceMode::Legacy),
        ["eos=.", "x=line", "4", "v=First", "vcap", "s=\u{85}Second", "n=line."]
    );
}

#[test]
fn both_walks_give_the_same_features() {
    // An ASCII text takes the byte walk; the general walk on the same
    // text must give the same features at every candidate.
    let dict = dictionary();
    let eos = EndOfSentence::default();
    for mode in [UNICODE, WhitespaceMode::Legacy] {
        let ctx = SdContext::new(&eos, Some(&dict), mode, Limits::NONE);
        let text = " Mr. Smith  met U.S. people.\tA.M. z.B.  x. \x1c? end!";
        let mut positions = Vec::new();
        let scanned = eos.positions(text, &mut positions);
        assert!(scanned.is_ascii());
        let (mut fast, mut general) = (Features::new(), Features::new());
        let mut lookup = Lookup::default();
        let (mut fast_walks, mut general_walks) = (Walks::new(), Walks::new());
        for &at in &positions {
            ctx.context(scanned, at, &mut fast_walks, &mut fast, &mut lookup);
            let cursor = Utf16Cursor::at(text, at.byte as usize, at.unit as usize);
            ctx.build(cursor, scanned, &mut general_walks, &mut general, &mut lookup);
            assert!(fast.iter().eq(general.iter()), "at {at:?}");
        }
    }
}

/// Texts on which the remembered whitespace searches are compared with
/// the plain walks: long stretches without whitespace, whitespace runs
/// at either end, end characters that are whitespace, characters
/// outside the BMP, and whitespace that only one mode has.
fn walk_texts() -> Vec<String> {
    let mut texts = vec![
        "A.".repeat(300),
        "x.Y".repeat(200),
        format!("  {}  ", "ab.c?D!".repeat(80)),
        format!("{} tail. end", "q.".repeat(150)),
        format!("Lead. {}. x", "w".repeat(400)),
        "He thought hard.Bekannt. Sie kam.\n\nNew. Line.\nA.B.C. d. e.".repeat(6),
        "\u{1F600}.\u{10400}x. \u{1F600}.\u{85}y.\u{1C}z.\u{3000}.".repeat(30),
        " . . ..  .\t.\n".repeat(20),
        ".".repeat(50),
        "\n.\n.x\n\n.".repeat(20),
    ];
    texts.push(texts.concat());
    texts
}

/// Every candidate of every text in order, with remembered searches and
/// with plain walks, in both modes and both cursor kinds, with and
/// without the dictionary.
#[test]
fn remembered_searches_match_the_plain_walks() {
    let dict = dictionary();
    for eos in [EndOfSentence::default(), EndOfSentence::from_chars("\n.!?").unwrap()] {
        for mode in [UNICODE, WhitespaceMode::Legacy] {
            for dict in [None, Some(&dict)] {
                let ctx = SdContext::new(&eos, dict, mode, Limits::NONE);
                let mut positions = Vec::new();
                let (mut remembered, mut plain) = (Features::new(), Features::new());
                let mut lookup = Lookup::default();
                for text in walk_texts() {
                    let scanned = eos.positions(&text, &mut positions);
                    let (mut walks, mut ascii_walks) = (Walks::new(), Walks::new());
                    for &at in &positions {
                        let cursor = Utf16Cursor::at(&text, at.byte as usize, at.unit as usize);
                        ctx.build(cursor, scanned, &mut Walks::plain(), &mut plain, &mut lookup);
                        ctx.build(cursor, scanned, &mut walks, &mut remembered, &mut lookup);
                        assert!(plain.iter().eq(remembered.iter()), "{mode:?} at {at:?} of {text:?}");
                        if scanned.is_ascii() {
                            let cursor = AsciiCursor::at(&text, at.byte as usize);
                            ctx.build(cursor, scanned, &mut ascii_walks, &mut remembered, &mut lookup);
                            assert!(plain.iter().eq(remembered.iter()), "{mode:?} at {at:?} of {text:?}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn limits_leave_out_only_words_that_cannot_match() {
    let dict = dictionary();
    let eos = EndOfSentence::default();
    let text = "Somewhat long. Mr. Smith left. Abbreviations.";
    let mut positions = Vec::new();
    let scanned = eos.positions(text, &mut positions);
    let at = positions[2];
    let mut out = Features::new();
    let mut get = |limits: Limits| {
        let ctx = SdContext::new(&eos, Some(&dict), UNICODE, limits);
        ctx.context(scanned, at, &mut Walks::new(), &mut out, &mut Lookup::default());
        out.iter().map(show).collect::<Vec<_>>()
    };
    assert_eq!(get(Limits::NONE), ["sn", "eos=.", "x=left", "4", "v=Smith", "vcap", "s=", "n=Abbreviations.", "ncap"]);
    // "n=Abbreviations." is 16 bytes and "v=Smith" 7: with 7 the first
    // is left out and its flags stay; with 6 the second goes too.
    assert_eq!(
        get(Limits { feature: 7, abbreviation: 3 }),
        ["sn", "eos=.", "x=left", "4", "v=Smith", "vcap", "s=", "ncap"]
    );
    assert_eq!(get(Limits { feature: 6, abbreviation: 3 }), ["sn", "eos=.", "x=left", "4", "vcap", "s=", "ncap"]);
    // The abbreviation limit counts the end character with the prefix:
    // "Mr." is three units, so a limit of 2 skips the lookup.
    let at = positions[1];
    let mut get = |abbreviation: usize| {
        let ctx = SdContext::new(&eos, Some(&dict), UNICODE, Limits { feature: usize::MAX, abbreviation });
        ctx.context(scanned, at, &mut Walks::new(), &mut out, &mut Lookup::default());
        out.iter().any(|f| f == b"xabbrev")
    };
    assert!(get(3));
    assert!(!get(2));
}

#[test]
fn a_lone_period() {
    assert_eq!(features(".", 0, None, UNICODE), ["eos=.", "x=", "v=", "s=", "n="]);
    assert_eq!(features(" . ", 1, None, UNICODE), ["sp", "sn", "eos=.", "x=", "v=", "s=", "n="]);
}
