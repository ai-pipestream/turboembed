use super::*;

fn parse_err(text: &str) -> String {
    match parse(text) {
        Ok(_) => panic!("{text:?} parsed"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn blocks_and_settings_are_read() {
    let file = parse(
        "FLAG long\nCOMPOUNDFLAG Cc\nCOMPOUNDMIN 0\nLANG hu_HU\nSFX Aa Y 2\nSFX Aa y ies/Bb [^aeiou]y po:pl\nSFX Aa 0 s .\nPFX Bb N 1\nPFX Bb 0 re .\n",
    )
    .unwrap();
    assert_eq!(file.flag_mode, FlagMode::Long);
    assert_eq!(file.settings.compound_min, 1);
    assert_eq!(&*file.settings.language, "hu_HU");
    assert_eq!(file.suffixes.len(), 2);
    let s = &file.suffixes[0];
    assert_eq!((s.strip(), s.affix(), s.cross_product), ("y", "ies", true));
    assert!(s.allows(Flag(0x42_0062)));
    assert_eq!(s.fields(), [Box::<str>::from("po:pl")]);
    assert_eq!(file.prefixes[0].affix(), "re");
    assert!(!file.prefixes[0].cross_product);
}

#[test]
fn errors_name_their_line() {
    assert_eq!(parse_err("FLAG long\nFLAG num\n"), "multiple FLAG directives at line 2");
    assert_eq!(parse_err("FLAG chars\n"), "unsupported FLAG mode 'chars' at line 1");
    assert_eq!(parse_err("SFX A Y 2\nSFX A 0 s ."), "affix block truncated at line 3");
    assert_eq!(parse_err("SFX A Y 2\nSFX A 0 s .\n"), "malformed affix rule at line 3");
    assert_eq!(parse_err("SFX A Y 1\nPFX A 0 s .\n"), "malformed affix rule at line 2");
    assert_eq!(parse_err("SFX A Y 1\nSFX B 0 s .\n"), "affix rule flag does not match header at line 2");
    assert_eq!(parse_err("SFX A X 1\n"), "invalid cross-product marker at line 1");
    assert_eq!(parse_err("SFX A Y x\n"), "malformed affix rule count at line 1");
    assert_eq!(parse_err("AF 2\nAF A\n"), "AF header specifies 2 aliases but found 1");
    assert_eq!(
        parse_err("AF 1\nAF A\nSFX A Y 1\nSFX A 0 s/2 .\n"),
        "flag alias 2 at line 4 is outside the AF table of 1 aliases"
    );
    assert_eq!(parse_err("AM 1\nAM po:n\nSFX A Y 1\nSFX A 0 s . x\n"), "invalid AM alias at line 4");
    assert_eq!(parse_err("SFX A Y 1\nSFX A 0 s [ab\n"), "unterminated character class at line 2");
    assert_eq!(parse_err("COMPOUNDMIN\n"), "COMPOUNDMIN line without a value at line 1");
    assert_eq!(parse_err("COMPOUNDMIN x\n"), "malformed COMPOUNDMIN at line 1");
    assert_eq!(parse_err("COMPOUNDMIN 1073741824\n"), "COMPOUNDMIN exceeds 1073741823 at line 1");
    assert_eq!(parse_err("NEEDAFFIX\n"), "NEEDAFFIX line without a flag at line 1");
    assert_eq!(parse_err("IGNORE a b\n"), "invalid IGNORE at line 1");
    assert_eq!(parse_err("LANG\n"), "invalid LANG at line 1");
    assert_eq!(parse_err("ICONV 1\n"), "ICONV header specifies 1 entries but found 0");
}

#[test]
fn aliases_expand() {
    let file = parse("AF 1\nAF AB\nAM 1\nAM st:x po:y\nSFX A Y 1\nSFX A 0 s/1 . 1\n").unwrap();
    let s = &file.suffixes[0];
    assert_eq!(s.continuation.as_slice(), [Flag::of('A'), Flag::of('B')]);
    assert_eq!(s.fields().len(), 2);
}
