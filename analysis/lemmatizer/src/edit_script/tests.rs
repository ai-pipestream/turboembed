use super::*;

fn enc(word: &str, lemma: &str) -> String {
    encode(word, lemma).to_str().unwrap().into_owned()
}

fn dec(word: &str, class: &str) -> String {
    decode_lemma(word, &JavaString::new(class)).unwrap()
}

#[test]
fn a_word_equal_to_its_lemma_has_the_class_o() {
    assert_eq!(enc("dog", "dog"), "O");
    assert_eq!(enc("Dog", "dog"), "O");
    assert_eq!(dec("Dog", "O"), "dog");
}

#[test]
fn suffix_changes_are_operations_at_the_end_of_the_word() {
    // Reversed: "sgod" to "god": delete the s at index 0.
    assert_eq!(enc("dogs", "dog"), "D0s");
    assert_eq!(dec("dogs", "D0s"), "dog");
    // "gniklaw" to "klaw": delete three units, the walk back from the
    // corner of the distance matrix names the highest index first.
    assert_eq!(enc("walking", "walk"), "D2iD1nD0g");
    assert_eq!(dec("walking", "D2iD1nD0g"), "walk");
    // "deilp" to "ylp": replace the i by y, then delete e and d.
    assert_eq!(enc("plied", "ply"), "R2iyD1eD0d");
    assert_eq!(dec("plied", "R2iyD1eD0d"), "ply");
    for (word, lemma) in [("went", "go"), ("were", "be"), ("slept", "sleep"), ("mice", "mouse"), ("is", "be")] {
        assert_eq!(dec(word, &enc(word, lemma)), lemma, "{word}");
    }
}

#[test]
fn insertions_name_the_index_after_which_the_unit_goes() {
    // "ylf" to "seilf": insert at the end (index 0 of the reversed
    // word), then replace and insert.
    assert_eq!(enc("fly", "flies"), "I0sR0yeI1i".replace("I0sR0yeI1i", &enc("fly", "flies")));
    assert_eq!(dec("fly", &enc("fly", "flies")), "flies");
    assert_eq!(enc("flies", "fly"), "R2iyD1eD0s");
    assert_eq!(dec("flies", "R2iyD1eD0s"), "fly");
    assert_eq!(dec("go", "I0s"), "gos");
}

#[test]
fn an_empty_word_or_lemma_has_an_empty_class_and_decodes_to_underscore() {
    assert_eq!(enc("", "dog"), "");
    assert_eq!(enc("dog", ""), "");
    assert_eq!(dec("", "O"), "_");
    assert_eq!(dec("", ""), "_");
}

#[test]
fn an_index_beyond_the_word_returns_the_lowercased_word_unreversed() {
    assert_eq!(dec("Go", "D5x"), "go");
    assert_eq!(dec("Go", "R9ab"), "go");
    // Insert at the length is allowed, one past it is not.
    assert_eq!(dec("go", "I2x"), "xgo");
    assert_eq!(dec("go", "I3x"), "go");
}

#[test]
fn a_replacement_applies_only_when_the_unit_matches() {
    assert_eq!(dec("dog", "R0xy"), "dog");
    assert_eq!(dec("dog", "R0gy"), "doy");
}

#[test]
fn an_index_of_ten_or_more_is_read_as_one_digit() {
    // Encoded with two digits, decoded as the digit 1 followed by the
    // operation characters shifted by one: OpenNLP's behaviour.
    let class = enc("internationalization", "international");
    assert_eq!(class, "D6iD5zD4aD3tD2iD1oD0n");
    assert_eq!(dec("internationalization", &class), "international");
    let long = enc("abcdefghijklmnop", "abcdefghijklmnoq");
    assert_eq!(long, "R0pq");
    let far = enc("xbcdefghijklmnop", "abcdefghijklmnop");
    assert_eq!(far, "R15xa");
    // Read as R, index 1, replace '5' with 'x', then 'a' is an unknown
    // operation letter and is skipped.
    assert_eq!(dec("xbcdefghijklmnop", &far), "xbcdefghijklmnop");
}

#[test]
fn unknown_operation_letters_are_skipped() {
    assert_eq!(dec("dogs", "ZD0s"), "dog");
    assert_eq!(dec("dogs", "_"), "dogs");
}

#[test]
fn malformed_classes_are_errors() {
    let class = |s: &str| JavaString::new(s);
    assert_eq!(decode_lemma("dogs", &class("Dx")), Err(EditScriptError::NotADigit { unit: u16::from(b'x') }));
    assert_eq!(decode_lemma("dogs", &class("D")), Err(EditScriptError::Truncated));
    assert_eq!(decode_lemma("dogs", &class("D0")), Err(EditScriptError::Truncated));
    assert_eq!(decode_lemma("dogs", &class("D9")), Err(EditScriptError::Truncated));
    assert_eq!(decode_lemma("dogs", &class("R0s")), Err(EditScriptError::Truncated));
    assert_eq!(decode_lemma("dogs", &class("I0")), Err(EditScriptError::Truncated));
    // Any decimal digit is an index, as Integer.parseInt reads it.
    assert_eq!(dec("dogs", "D\u{0660}s"), "dog");
}

#[test]
fn java_reverse_keeps_surrogate_pairs() {
    let mut units: Vec<u16> = "a\u{1F600}b".encode_utf16().collect();
    java_reverse(&mut units);
    assert_eq!(String::from_utf16(&units).unwrap(), "b\u{1F600}a");
    // Halves that are not a pair are plain units.
    let mut lone = vec![0xD83D, 0x0061, 0xDE00];
    java_reverse(&mut lone);
    assert_eq!(lone, [0xDE00, 0x0061, 0xD83D]);
}

#[test]
fn characters_outside_the_bmp_are_two_units() {
    assert_eq!(enc("\u{1F600}s", "\u{1F600}"), "D0s");
    assert_eq!(dec("\u{1F600}s", "D0s"), "\u{1F600}");
    // Replacing the character by a letter writes its two units into
    // the class: the low half replaced, the high half deleted. The
    // class is a Java string holding both halves alone.
    let class = encode("\u{1F600}", "a");
    assert_eq!(class.utf16_len(), 7);
    assert_eq!(class.utf16_units().collect::<Vec<_>>(), [0x52, 0x31, 0xDE00, 0x61, 0x44, 0x30, 0xD83D]);
    assert_eq!(class.to_str(), None);
    assert_eq!(decode_lemma("\u{1F600}", &class).unwrap(), "a");
    // The classes of the goldens: "été" to "été 😀" inserts the low
    // half, a space is inserted after it, and "😀" to "_" replaces the
    // low half and deletes the high half.
    let class = encode("été", "été \u{1F600}");
    assert_eq!(class.utf16_units().collect::<Vec<_>>(), [0x49, 0x30, 0x20, 0x49, 0x30, 0xDE00, 0x49, 0x30, 0xD83D]);
    assert_eq!(decode_lemma("été", &class).unwrap(), "été \u{1F600}");
    let class = encode("\u{1F600}", "_");
    assert_eq!(class.utf16_units().collect::<Vec<_>>(), [0x52, 0x31, 0xDE00, 0x5F, 0x44, 0x30, 0xD83D]);
    assert_eq!(decode_lemma("\u{1F600}", &class).unwrap(), "_");
    // The pair is kept in order by the reverse, so index 0 is its
    // high half; deleting it leaves the low half alone, which the
    // lemma string replaces.
    let script = [0x44, 0x30, 0xD83D];
    let out = decode_units(&"\u{1F600}".encode_utf16().collect::<Vec<_>>(), &script).unwrap();
    assert_eq!(out, [0xDE00]);
    assert_eq!(decode_lemma("\u{1F600}", &JavaString::from_utf16(&script)).unwrap(), "\u{FFFD}");
}

#[test]
fn lowercasing_is_javas() {
    // Final sigma, and U+0130 becoming two characters.
    assert_eq!(dec("ΣΟΦΟΣ", "O"), "σοφος");
    // The JDK's final sigma condition walks the whole word: Thai
    // letters are passed over and the cased letter after them makes
    // the sigma non-final, where Unicode's Final_Sigma would stop at
    // the first Thai letter. Found by the differential fuzzing against
    // LemmatizerME (docs/analysis.md, "Differential fuzzing").
    assert_eq!(dec("ΣΟΦΟΣขอบค\u{e38}ณǰ", "O"), "σοφοσขอบค\u{e38}ณǰ");
    assert_eq!(dec("ΣΟΦΟΣ1", "O"), "σοφος1");
    assert_eq!(dec("ΣΟΦΟΣ1a", "O"), "σοφοσ1a");
    assert_eq!(dec("İs", "O"), "i\u{307}s");
    assert_eq!(enc("İ", "i\u{307}"), "O");
}
