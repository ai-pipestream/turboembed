//! Conformance and regression tests for the phonetic keys: American
//! Soundex and Double Metaphone (key length 8), both allocation free.
//!
//! The vectors are the Apache Commons Codec implementation of Philips'
//! Double Metaphone at `maxCodeLen` 8 and its `US_ENGLISH` Soundex, run
//! over the reference inputs; the dialect notes on
//! [`turbo_matcher::soundex`] and [`turbo_matcher::double_metaphone`]
//! state the deliberate deviations. Soundex of a word with no ASCII
//! letter is `*b"0000"` here where Commons Codec returns an empty string,
//! and a non-ASCII letter is skipped where Commons Codec throws.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use turbo_matcher::{double_metaphone, soundex};

struct Counting;
thread_local! { static COUNT: Cell<usize> = const { Cell::new(0) }; }
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        COUNT.with(|c| c.set(c.get() + 1));
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        COUNT.with(|c| c.set(c.get() + 1));
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        COUNT.with(|c| c.set(c.get() + 1));
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// `(word, primary, alternate, soundex)`; the keys shorter than eight
/// bytes read NUL-padded, so the table keeps them bare. An empty soundex
/// marks a non-ASCII word, covered in [`non_ascii_letters_are_skipped`].
const VECTORS: &[(&str, &str, &str, &str)] = &[
    ("ll", "L", "L", "L000"),
    ("Smith", "SM0", "XMT", "S530"),
    ("Smyth", "SM0", "XMT", "S530"),
    ("Schmidt", "XMT", "SMT", "S530"),
    ("Schneider", "XNTR", "SNTR", "S536"),
    ("Gauss", "KS", "KS", "G200"),
    ("Robert", "RPRT", "RPRT", "R163"),
    ("Rupert", "RPRT", "RPRT", "R163"),
    ("Wroblewski", "RPLSK", "RPLFSK", "W614"),
    ("Ashcraft", "AXKRFT", "AXKRFT", "A261"),
    ("Tymczak", "TMSK", "TMXK", "T522"),
    ("Gutierrez", "KTRS", "KTRS", "G362"),
    ("Czuj", "SJ", "X ", "C200"),
    ("Czech", "SK", "XK", "C200"),
    ("Czerny", "SRN", "XRN", "C650"),
    ("Ruby", "RP", "RP", "R100"),
    ("O'Malley", "AML", "AML", "O540"),
    ("Diggory", "TKR", "TKR", "D260"),
    ("Tagliaro", "TKLR", "TLR", "T246"),
    ("Angelina", "ANJLN", "ANKLN", "A524"),
    ("Zhang", "JNK", "JNK", "Z520"),
    ("Jose", "HS", "HS", "J200"),
    ("San", "SN", "SN", "S500"),
    ("Juan", "JN", "AN", "J500"),
    ("Van", "FN", "FN", "V500"),
    ("Helsing", "HLSNK", "HLSNK", "H425"),
    ("McLaughlin", "MKLFLN", "MKLFLN", "M242"),
    ("McClelland", "MKLLNT", "MKLLNT", "M244"),
    ("Bach", "PK", "PK", "B200"),
    ("bacchus", "PKS", "PKS", "B220"),
    ("bellocchio", "PLX", "PLX", "B420"),
    ("accident", "AKSTNT", "AKSTNT", "A235"),
    ("accede", "AKST", "AKST", "A230"),
    ("succeed", "SKST", "SKST", "S230"),
    ("bacci", "PX", "PX", "B200"),
    ("bertucci", "PRTX", "PRTX", "B632"),
    ("Caesar", "SSR", "SSR", "C260"),
    ("focaccia", "FKX", "FKX", "F220"),
    ("Michael", "MKL", "MXL", "M240"),
    ("chemistry", "KMSTR", "KMSTR", "C523"),
    ("chorus", "KRS", "KRS", "C620"),
    ("orchid", "ARKT", "ARKT", "O623"),
    ("orchidaceae", "ARKTS", "ARKTS", "O623"),
    ("sugar", "XKR", "SKR", "S260"),
    ("island", "ALNT", "ALNT", "I245"),
    ("isle", "AL", "AL", "I240"),
    ("carlisle", "KRLL", "KRLL", "C642"),
    ("schlesinger", "XLSNKR", "SLSNJR", "S425"),
    ("school", "SKL", "SKL", "S400"),
    ("schooner", "SKNR", "SKNR", "S560"),
    ("schermerhorn", "XRMRRN", "SKRMRRN", "S656"),
    ("schenker", "XNKR", "SKNKR", "S526"),
    ("Thomas", "TMS", "TMS", "T520"),
    ("Thames", "TMS", "TMS", "T520"),
    ("Wasserman", "ASRMN", "FSRMN", "W265"),
    ("Vasserman", "FSRMN", "FSRMN", "V265"),
    ("Uomo", "AM", "AM", "U500"),
    ("Womo", "AM", "FM", "W500"),
    ("Arnow", "ARN", "ARNF", "A650"),
    ("Filipowicz", "FLPTS", "FLPFX", "F412"),
    ("breaux", "PR", "PR", "B620"),
    ("resnais", "RSN", "RSNS", "R252"),
    ("artois", "ART", "ARTS", "A632"),
    ("edge", "AJ", "AJ", "E320"),
    ("Edgar", "ATKR", "ATKR", "E326"),
    ("danger", "TNJR", "TNKR", "D526"),
    ("ranger", "RNJR", "RNKR", "R526"),
    ("manger", "MNJR", "MNKR", "M526"),
    ("biaggi", "PJ", "PK", "B200"),
    ("gnome", "NM", "NM", "G550"),
    ("knight", "NT", "NT", "K523"),
    ("pneumonia", "NMN", "NMN", "P555"),
    ("wrangle", "RNKL", "RNKL", "W652"),
    ("psychology", "SXLJ", "SKLK", "P224"),
    ("tougher", "TFR", "TFR", "T260"),
    ("laugh", "LF", "LF", "L200"),
    ("cough", "KF", "KF", "C200"),
    ("rough", "RF", "RF", "R200"),
    ("gyro", "KR", "JR", "G600"),
    ("gillies", "KLS", "JLS", "G420"),
    ("heiligen", "HLJN", "HLKN", "H425"),
    ("josef", "JSF", "HSF", "J210"),
    ("Joseph", "JSF", "HSF", "J210"),
    ("Jensen", "JNSN", "ANSN", "J525"),
    ("jenny", "JN", "AN", "J500"),
    ("liu", "L", "L", "L000"),
    ("lum", "LM", "LM", "L500"),
    ("bryan", "PRN", "PRN", "B650"),
    ("Bryant", "PRNT", "PRNT", "B653"),
    ("Xavier", "SF", "SFR", "X160"),
    ("xenophobia", "SNFP", "SNFP", "X511"),
    ("xerxes", "SRKSS", "SRKSS", "X622"),
    ("Zulu", "SL", "SL", "Z400"),
    ("zoo", "S", "S", "Z000"),
    ("Tsai", "TS", "TS", "T200"),
    ("tsar", "TSR", "TSR", "T260"),
    ("casserole", "KSRL", "KSRL", "C264"),
    ("kasimir", "KSMR", "KSMR", "K256"),
    ("auberge", "APRJ", "APRK", "A162"),
    ("renaud", "RNT", "RNT", "R530"),
    ("boucher", "PXR", "PKR", "B260"),
    ("bouvier", "PF", "PFR", "B160"),
    ("cathcart", "K0KRT", "KTKRT", "C326"),
    ("cherubic", "XRPK", "XRPK", "C612"),
    ("chiang", "KNK", "KNK", "C520"),
    ("chianti", "KNT", "KNT", "C530"),
    ("cuellar", "KLR", "KLR", "C460"),
    ("bouchard", "PXRT", "PKRT", "B263"),
    ("bourgeois", "PRJ", "PRKS", "B622"),
    ("Abercrombie", "APRKRMP", "APRKRMP", "A162"),
    ("Bohl", "PL", "PL", "B400"),
    ("Bublitz", "PPLTS", "PPLTS", "B143"),
    ("Pfister", "PFSTR", "PFSTR", "P236"),
    ("Rentschler", "RNTXLR", "RNTXLR", "R532"),
    ("Szymankiewicz", "SMNKTS", "XMNKFX", "S552"),
    ("Washington", "AXNKTN", "FXNKTN", "W252"),
    ("Caulk", "KLK", "KLK", "C420"),
    ("Colcolough", "KLKLF", "KLKLF", "C424"),
    ("Espinoza", "ASPNS", "ASPNS", "E215"),
    ("Fazio", "FS", "FS", "F200"),
    ("geoffrey", "JFR", "KFR", "G160"),
    ("Geoffrey", "JFR", "KFR", "G160"),
    ("Corwin", "KRN", "KRN", "C650"),
    ("Eriksen", "ARKSN", "ARKSN", "E625"),
    ("Feuerstein", "FRSTN", "FRSTN", "F623"),
    ("garcia", "KRS", "KRX", "G620"),
    ("gonzalez", "KNSLS", "KNSLS", "G524"),
    ("houghton", "HTN", "HTN", "H235"),
    ("hugh", "H", "H", "H200"),
    ("Hugh", "H", "H", "H200"),
    ("humble", "HMPL", "HMPL", "H514"),
    ("ibsen", "APSN", "APSN", "I125"),
    ("indigo", "ANTK", "ANTK", "I532"),
    ("jacques", "JKS", "AKS", "J220"),
    ("jarrett", "JRT", "ART", "J630"),
    ("khrushchev", "KRXXF", "KRXKF", "K621"),
    ("leicester", "LSSTR", "LSSTR", "L223"),
    ("ludwig", "LTK", "LTK", "L320"),
    ("macgregor", "MKRKR", "MKRKR", "M262"),
    ("mackenzie", "MKNS", "MKNTS", "M252"),
    ("maggie", "MJ", "MK", "M200"),
    ("mahler", "MLR", "MLR", "M460"),
    ("marchand", "MRXNT", "MRKNT", "M625"),
    ("mattias", "MTS", "MTS", "M320"),
    ("mediterranean", "MTTRNN", "MTTRNN", "M336"),
    ("mischa", "MX", "MX", "M200"),
    ("moreau", "MR", "MR", "M600"),
    ("mortensen", "MRTNSN", "MRTNSN", "M635"),
    ("najim", "NJM", "NJM", "N250"),
    ("nikolayev", "NKLF", "NKLF", "N241"),
    ("peshitta", "PXT", "PXT", "P230"),
    ("pietrzak", "PTRSK", "PTRTSK", "P362"),
    ("pythagoras", "P0KRS", "PTKRS", "P326"),
    ("raleigh", "RL", "RL", "R420"),
    ("reggie", "RK", "RK", "R200"),
    ("rodionov", "RTNF", "RTNF", "R351"),
    ("rogier", "RJ", "RJR", "R260"),
    ("salzwedel", "SLSTL", "SLTSTL", "S423"),
    ("scholtz", "XLTS", "XLTS", "S432"),
    ("schumann", "XMN", "XMN", "S550"),
    ("simon", "SMN", "SMN", "S550"),
    ("sykes", "SKS", "SKS", "S220"),
    ("tabitha", "TP0", "TPT", "T130"),
    ("thibault", "0PLT", "TPLT", "T143"),
    ("tichy", "TX", "TK", "T200"),
    ("tillman", "TLMN", "TLMN", "T455"),
    ("ullman", "ALMN", "ALMN", "U455"),
    ("unruh", "ANR", "ANR", "U560"),
    ("vinson", "FNSN", "FNSN", "V525"),
    ("wachter", "AKTR", "FKTR", "W236"),
    ("william", "ALM", "FLM", "W450"),
    ("yanez", "ANS", "ANS", "Y520"),
    ("yeager", "AKR", "AJR", "Y260"),
    ("zappa", "SP", "SP", "Z100"),
    ("zimmerman", "SMRMN", "SMRMN", "Z565"),
    ("allegro", "ALKR", "AKR", "A426"),
    ("kuczewski", "KSSK", "KXFSK", "K220"),
    ("Moskva", "MSKF", "MSKF", "M210"),
    ("Silva", "SLF", "SLF", "S410"),
    ("Rodriguez", "RTRKS", "RTRKS", "R362"),
    ("Martinez", "MRTNS", "MRTNS", "M635"),
    ("Hernandez", "HRNNTS", "HRNNTS", "H655"),
    ("Lopez", "LPS", "LPS", "L120"),
    ("lopez", "LPS", "LPS", "L120"),
    ("señor", "SNR", "SNR", ""),
    ("façade", "FST", "FST", ""),
    ("françois", "FRNS", "FRNSS", ""),
    ("mañana", "MNN", "MNN", ""),
    ("pinça", "PNS", "PNS", ""),
    ("çava", "SF", "SF", ""),
    ("ñandú", "NNT", "NNT", ""),
    ("naïve", "NF", "NF", ""),
    ("résumé", "RSM", "RSM", ""),
    ("xavier", "SF", "SFR", "X160"),
    ("XAVIER", "SF", "SFR", "X160"),
    ("smith", "SM0", "XMT", "S530"),
    ("SMITH", "SM0", "XMT", "S530"),
    ("b", "P", "P", "B000"),
    ("dt", "T", "T", "D000"),
    ("lr", "LR", "LR", "L600"),
    ("mn", "MN", "MN", "M000"),
    ("bcdfghjklmnpqrstvwxz", "PKTFKKLM", "PKTFKKLM", "B231"),
    ("a", "A", "A", "A000"),
    ("hh", "", "", "H000"),
    ("hw", "", "", "H000"),
    ("who", "A", "A", "W000"),
    ("why", "A", "A", "W000"),
    ("aha", "AH", "AH", "A000"),
    ("ax", "AKS", "AKS", "A200"),
    ("ashcraft", "AXKRFT", "AXKRFT", "A261"),
    ("ASHCRAFT", "AXKRFT", "AXKRFT", "A261"),
    ("dz", "TS", "TS", "D200"),
    ("cz", "S", "X", "C000"),
    ("sz", "S", "X", "S000"),
    ("squishy", "SKX", "SKX", "S200"),
    ("nguyen", "NKN", "NKN", "N250"),
];

fn key(s: &str) -> [u8; 8] {
    let mut k = [0; 8];
    k[..s.len()].copy_from_slice(s.as_bytes());
    k
}

#[test]
fn soundex_matches_the_reference_dialect() {
    for &(word, _, _, sx) in VECTORS {
        if sx.is_empty() {
            continue;
        }
        assert_eq!(soundex(word), *sx.as_bytes(), "{word}");
    }
}

#[test]
fn double_metaphone_matches_the_reference_dialect() {
    for &(word, primary, alternate, _) in VECTORS {
        assert_eq!(double_metaphone(word), (key(primary), key(alternate)), "{word}");
    }
}

#[test]
fn soundex_is_ascii_case_insensitive() {
    for &(word, _, _, sx) in VECTORS {
        if word.is_ascii() {
            assert_eq!(soundex(&word.to_lowercase()), *sx.as_bytes(), "{word}");
            assert_eq!(soundex(&word.to_uppercase()), *sx.as_bytes(), "{word}");
        }
    }
}

#[test]
fn double_metaphone_is_ascii_case_insensitive() {
    for &(word, primary, alternate, _) in VECTORS {
        if word.is_ascii() {
            let want = (key(primary), key(alternate));
            assert_eq!(double_metaphone(&word.to_lowercase()), want, "{word}");
            assert_eq!(double_metaphone(&word.to_uppercase()), want, "{word}");
        }
    }
}

#[test]
fn non_ascii_letters_are_skipped() {
    // Where Commons Codec throws on an unmapped letter, this dialect
    // skips it; the Double Metaphone keys still come from the table.
    for (word, sx) in [
        ("señor", *b"S600"),
        ("façade", *b"F300"),
        ("françois", *b"F652"),
        ("mañana", *b"M500"),
        ("pinça", *b"P500"),
        ("çava", *b"A100"),
        ("ñandú", *b"A530"),
        ("naïve", *b"N100"),
        ("résumé", *b"R250"),
    ] {
        assert_eq!(soundex(word), sx, "{word}");
    }
}

#[test]
fn empty_and_symbol_only_words_have_empty_keys() {
    // A word with nothing to encode: Soundex pads digits with zero, and
    // both Double Metaphone keys are all NUL.
    assert_eq!(soundex(""), *b"0000");
    assert_eq!(soundex("123"), *b"0000");
    assert_eq!(double_metaphone(""), ([0; 8], [0; 8]));
    assert_eq!(double_metaphone("123"), ([0; 8], [0; 8]));
}

#[test]
fn non_ascii_and_supplementary_input_never_panics() {
    // Non-ASCII letters without a rule are skipped; supplementary
    // scalars are one symbol each; keys stay NUL-padded.
    let (p, a) = double_metaphone("caf\u{e9} \u{1f600} na\u{ef}ve");
    assert_eq!(p[0], b'K');
    assert_eq!(p[1], b'F');
    assert_eq!(a, p);
    assert_eq!(soundex("caf\u{e9}"), *b"C100");
    assert_eq!(soundex("\u{1f600}abc"), *b"A120");
    let (p, a) = double_metaphone("\u{1f600}\u{1f601}");
    assert_eq!((p, a), ([0; 8], [0; 8]));
}

#[test]
fn phonetic_keys_do_not_allocate() {
    let words = [
        "Smith",
        "Schmidt",
        "Ashcraft",
        "se\u{f1}or",
        "fa\u{e7}ade",
        "na\u{ef}ve",
        "\u{4e2d}\u{6587} \u{1f600} mixed CASE",
        // Longer than the word window: the cap path must stay allocation free.
        &"abcdefghijklmnopqrstuvwxyz".repeat(8),
    ];
    let before = COUNT.get();
    let mut sink = 0usize;
    for _ in 0..1000 {
        for w in words {
            sink += usize::from(soundex(w)[0]);
            sink += usize::from(double_metaphone(w).0[0]);
        }
    }
    assert_eq!(COUNT.get() - before, 0, "{sink} key bytes hashed");
}

#[test]
fn double_metaphone_keeps_its_apache_attribution() {
    let notice = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../NOTICE"));
    assert!(notice.contains("Apache Commons Codec"));
    assert!(notice.contains("Copyright 2002-2026 The Apache Software Foundation"));
    assert!(notice.contains("analysis/matcher/src/phonetic.rs"));
    assert!(notice.contains("analysis/matcher/src/phonetic/metaphone.rs"));
}
