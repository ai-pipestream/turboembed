//! Parity of the normalizers with OpenNLP 3.x on the JVM.
//!
//! `analysis/goldens/data/normalize/<name>/norm/<id>.golden` holds, for every
//! text in `analysis/goldens/data/inputs/norm/`, the edits OpenNLP's normalizer
//! made to it (a `turbo:view` layer, see `analysis/goldens-tool/README.md`).
//! Each test below runs one normalizer over every text, puts the result on a
//! `Document` as a view and compares it with the golden using
//! `turbo_goldens::matches_golden_key::<View>`: the rebuilt text first, then
//! the edits one for one, Rust edit i against golden row i. Edits are not
//! merged on either side, so for a normalizer whose golden comes from an
//! aligned method (`Edits::Reported`) that is the whole check and any
//! difference fails the test.
//!
//! Where `docs/normalizers.md` records a deliberate difference from OpenNLP
//! ("Differences from OpenNLP" in the normalizer's section), the text is
//! listed in the test's `Divergence` table with the section and the output the
//! Rust normalizer is documented to give; the test asserts that output, and
//! that it differs from the golden's.
//!
//! Tolerance: OpenNLP's normalizers `emoji`, `nfc`, `nfkc`, `case-fold`,
//! `case-fold-tr`, `accent-fold`, `confusable-skeleton`, `url`,
//! `bounded-url`, `social-media`, `number`, `shrink`, `symbol-joiner` and the
//! default chain have no aligned method. Their goldens' edits were
//! reconstructed by a diff of the source and the result on code points, so an
//! edit boundary is the diff's choice (`Edits::Reconstructed`). For these the
//! output text must still equal the golden's exactly, and when the edit lists
//! differ they are accepted only if, for every region of the source that
//! either list changes, applying the Rust edits and applying the golden's
//! edits give the same text. Those texts are counted as `covered` in the line
//! each test prints. No other tolerance is applied.
//!
//! Normalizers without an OpenNLP counterpart: `case-fold-tr` has none.
//! `CaseFold` offers the Turkic folds of CaseFolding.txt instead of
//! locale-dependent lowercasing (docs/normalizers.md, "Case folding"), so its
//! test runs `turkic_simple` on the texts where the two agree and documents the
//! others. `emoji` is OpenNLP's removal of supplementary-plane code points, so
//! it runs `EmojiRemove::supplementary()`; the sequence mode is a documented
//! difference (docs/normalizers.md, "Emoji").
//!
//! Offsets: the goldens hold UTF-8 byte offsets, as the crates do.

use std::path::PathBuf;

use turbo_analysis::{Document, Key, Provenance, View};
use turbo_goldens::expect::{ViewEdit, apply_view_edits, view_edits_equivalent};
use turbo_goldens::{Golden, Mismatch, Tolerance, data_dir, matches_golden_key};
use turbo_normalize::{
    AccentFold, BoundedUrls, Bullets, CaseFold, Chain, ConfusableSkeleton, Dashes, Dehyphenate, Digits, Ellipsis,
    EmojiRemove, EmojiToEmoticon, EmoticonToEmoji, EnglishContractions, GermanUmlauts, Invisible, Normalizer, Numbers,
    Quotes, Shrink, SocialMedia, SymbolWords, UnicodeForm, Urls, Whitespace, normalize,
};

/// One edit of the golden: the source byte span it replaces and what it writes.
type Row = ViewEdit;

/// A text on which the Rust normalizer is documented to differ from OpenNLP.
struct Divergence {
    /// The input's id in `inputs/norm`.
    input: &'static str,
    /// The section of `docs/normalizers.md` that records the difference.
    section: &'static str,
    /// The output the Rust normalizer is documented to give.
    rust: &'static str,
}

/// Where the golden's edits come from.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Edits {
    /// OpenNLP's normalizer reports an alignment (`normalizeAligned`); the golden's edits are its.
    Reported,
    /// OpenNLP's normalizer has no aligned method; the generator rebuilt the edits by a diff of the
    /// source and the result on code points, so an edit boundary is the diff's choice.
    Reconstructed,
}

/// How many texts a test compared in each way.
#[derive(Default, Debug)]
struct Tally {
    exact: usize,
    covered: usize,
    diverged: usize,
}

const NORM: Key<View> = Key::named("norm");

/// The ids of the texts in `inputs/norm`, sorted.
fn input_ids() -> Vec<String> {
    let mut ids: Vec<String> = std::fs::read_dir(data_dir().join("inputs").join("norm"))
        .expect("the norm input set")
        .map(|e| {
            let name = e.expect("a directory entry").file_name().into_string().expect("a UTF-8 name");
            name.strip_suffix(".txt").expect("only .txt inputs").to_string()
        })
        .collect();
    ids.sort();
    ids
}

fn golden_path(name: &str, id: &str) -> PathBuf {
    data_dir().join("normalize").join(name).join("norm").join(format!("{id}.golden"))
}

/// The golden's `turbo:view` rows.
fn view_rows(golden: &Golden, name: &str, id: &str) -> Vec<Row> {
    let layer = golden.layer("turbo:view").unwrap_or_else(|| panic!("{name}/{id}: no turbo:view layer"));
    let start = layer.u32s("start").expect("a start column");
    let end = layer.u32s("end").expect("an end column");
    let with = layer.strs("with").expect("a with column");
    (0..start.len())
        .map(|i| Row { start: start[i] as usize, end: end[i] as usize, with: with.get(i).to_string() })
        .collect()
}

/// The source with the rows applied: the text OpenNLP produced.
fn rebuild(text: &str, rows: &[Row]) -> String {
    apply_view_edits(text, rows).unwrap_or_else(|e| panic!("{e}"))
}

/// A document holding `text` and, as the view `norm`, the text `n` makes of it.
fn normalized<N: Normalizer>(n: &N, text: &str) -> (Document, String, Vec<Row>) {
    let mut doc = Document::new(text);
    let (out, alignment) = normalize(n, text);
    let rows = alignment
        .edits()
        .iter()
        .map(|e| Row {
            start: e.parent.start as usize,
            end: e.parent.end as usize,
            with: out[e.child.range()].to_string(),
        })
        .collect();
    let view = View::derived(doc.get::<View>().expect("the source view"), out.clone(), alignment);
    doc.add(&NORM, view, Provenance::by("goldens test")).expect("the view is new");
    (doc, out, rows)
}

/// Whether the two edit lists change the same stretches of the source into the same text, whatever
/// their granularity: `turbo_goldens::expect::view_edits_equivalent`, the check for goldens whose
/// edits were reconstructed by a diff.
fn equivalent(text: &str, rust: &[Row], golden: &[Row]) -> bool {
    view_edits_equivalent(text, rust, golden)
}

/// Runs `n` over every text of the norm set and compares with the goldens of `name`.
fn check<N: Normalizer>(name: &str, edits: Edits, n: &N, divergences: &[Divergence]) -> Tally {
    let mut tally = Tally::default();
    let mut failures: Vec<String> = Vec::new();
    let ids = input_ids();
    for d in divergences {
        assert!(ids.iter().any(|i| i == d.input), "{name}: divergence for unknown input {}", d.input);
        assert!(d.section.starts_with("docs/normalizers.md, \""), "{name}/{}: name the doc section", d.input);
    }
    for id in &ids {
        let golden = Golden::read(golden_path(name, id)).unwrap_or_else(|e| panic!("{name}/{id}: {e}"));
        let text = golden.input_text(data_dir()).unwrap_or_else(|e| panic!("{name}/{id}: {e}"));
        let golden_rows = view_rows(&golden, name, id);
        let (doc, out, rust_rows) = normalized(n, &text);
        // The invariant every normalizer's alignment keeps, whatever the golden says.
        assert_eq!(rebuild(&text, &rust_rows), out, "{name}/{id}: the Rust edits do not rebuild the Rust output");

        if let Some(d) = divergences.iter().find(|d| d.input == id) {
            if rebuild(&text, &golden_rows) == d.rust {
                failures.push(format!("{id}: the documented output equals OpenNLP's; drop the entry"));
            } else if out != d.rust {
                failures.push(format!(
                    "{id}: differs from the output documented in {}\n rust: {out:?}\n doc:  {:?}",
                    d.section, d.rust
                ));
            }
            tally.diverged += 1;
            continue;
        }
        match matches_golden_key(&doc, &NORM, &golden, Tolerance::exact()) {
            Ok(()) => tally.exact += 1,
            Err(Mismatch::Item { .. })
                if edits == Edits::Reconstructed && equivalent(&text, &rust_rows, &golden_rows) =>
            {
                tally.covered += 1
            }
            Err(m) => failures.push(format!("{id}: does not match OpenNLP's\n{m}")),
        }
    }
    eprintln!("goldens {name}: {tally:?}");
    assert!(failures.is_empty(), "{name}: {} of {} texts differ\n{}", failures.len(), ids.len(), failures.join("\n"));
    assert_eq!(tally.exact + tally.covered + tally.diverged, ids.len());
    if edits == Edits::Reported {
        assert_eq!(tally.covered, 0, "{name}: an aligned golden is compared exactly");
    }
    tally
}

#[test]
fn every_normalizer_has_a_golden_for_every_input() {
    let ids = input_ids();
    assert!(ids.len() >= 30, "{} inputs", ids.len());
    let dir = data_dir().join("normalize");
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .expect("the normalize goldens")
        .map(|e| e.expect("entry").file_name().into_string().expect("name"))
        .collect();
    names.sort();
    assert_eq!(names.len(), 29, "{names:?}");
    for name in &names {
        for id in &ids {
            let golden = Golden::read(golden_path(name, id)).unwrap_or_else(|e| panic!("{name}/{id}: {e}"));
            let text = golden.input_text(data_dir()).unwrap_or_else(|e| panic!("{name}/{id}: {e}"));
            // The rows are in text order, inside the text, on character boundaries.
            let rows = view_rows(&golden, name, id);
            let mut at = 0;
            for r in &rows {
                assert!(r.start >= at && r.end >= r.start && r.end <= text.len(), "{name}/{id}: {r:?}");
                assert!(text.is_char_boundary(r.start) && text.is_char_boundary(r.end), "{name}/{id}: {r:?}");
                at = r.end;
            }
        }
    }
}

#[test]
fn whitespace_collapse() {
    check("whitespace", Edits::Reported, &Whitespace::collapse(), &[]);
}

#[test]
fn whitespace_keep_line_breaks() {
    check("whitespace-keep-lines", Edits::Reported, &Whitespace::keep_line_breaks(), &[]);
}

#[test]
fn whitespace_keep_paragraphs() {
    check("whitespace-keep-paragraphs", Edits::Reported, &Whitespace::keep_paragraphs(), &[]);
}

#[test]
fn dashes() {
    check("dash", Edits::Reported, &Dashes::new(), &[]);
}

#[test]
fn quotes() {
    check("quote", Edits::Reported, &Quotes, QUOTE);
}

#[test]
fn bullets() {
    check("bullet", Edits::Reported, &Bullets, BULLET);
}

#[test]
fn ellipsis() {
    check("ellipsis", Edits::Reported, &Ellipsis, ELLIPSIS);
}

#[test]
fn digits() {
    check("digit", Edits::Reported, &Digits, &[]);
}

#[test]
fn invisible() {
    check("invisible", Edits::Reported, &Invisible, INVISIBLE);
}

#[test]
fn dehyphenation() {
    check("dehyphenation", Edits::Reported, &Dehyphenate::new(), DEHYPHENATION);
}

#[test]
fn nfc() {
    check("nfc", Edits::Reconstructed, &UnicodeForm::nfc(), &[]);
}

#[test]
fn nfkc() {
    check("nfkc", Edits::Reconstructed, &UnicodeForm::nfkc(), &[]);
}

#[test]
fn case_fold_lowercase() {
    check("case-fold", Edits::Reconstructed, &CaseFold::lowercase(), &[]);
}

#[test]
fn case_fold_full() {
    check("full-case-fold", Edits::Reported, &CaseFold::full(), &[]);
}

#[test]
fn accent_fold() {
    check("accent-fold", Edits::Reconstructed, &AccentFold::new(), &[]);
}

#[test]
fn german_umlauts() {
    check("german-umlaut", Edits::Reported, &GermanUmlauts::new(), &[]);
}

#[test]
fn confusable_skeleton() {
    check("confusable-skeleton", Edits::Reconstructed, &ConfusableSkeleton, CONFUSABLE_SKELETON);
}

#[test]
fn urls() {
    check("url", Edits::Reconstructed, &Urls, &[]);
}

#[test]
fn bounded_urls() {
    check("bounded-url", Edits::Reconstructed, &BoundedUrls, BOUNDED_URL);
}

#[test]
fn social_media() {
    check("social-media", Edits::Reconstructed, &SocialMedia, &[]);
}

#[test]
fn numbers() {
    check("number", Edits::Reconstructed, &Numbers, &[]);
}

#[test]
fn shrink() {
    check("shrink", Edits::Reconstructed, &Shrink, &[]);
}

#[test]
fn english_contractions() {
    check("english-contraction", Edits::Reported, &EnglishContractions, ENGLISH_CONTRACTION);
}

#[test]
fn emoji() {
    check("emoji", Edits::Reconstructed, &EmojiRemove::supplementary(), EMOJI);
}

#[test]
fn emoji_to_emoticon() {
    check("emoji-to-emoticon", Edits::Reported, &EmojiToEmoticon, &[]);
}

#[test]
fn emoticon_to_emoji() {
    check("emoticon-to-emoji", Edits::Reported, &EmoticonToEmoji, &[]);
}

#[test]
fn symbol_words() {
    check("symbol-joiner", Edits::Reconstructed, &SymbolWords, SYMBOL_JOINER);
}

#[test]
fn default_chain() {
    let chain = Chain::new()
        .then(Invisible)
        .then(UnicodeForm::nfc())
        .then(Whitespace::collapse())
        .then(Quotes)
        .then(Dashes::new())
        .then(CaseFold::lowercase())
        .then(AccentFold::new());
    check("default-chain", Edits::Reconstructed, &chain, DEFAULT_CHAIN);
}

#[test]
fn case_fold_turkish() {
    // OpenNLP lowercases with the Turkish locale. `CaseFold` has no locale-dependent lowercasing;
    // the Turkic simple fold gives the same text wherever lowercasing and the simple fold agree, and
    // the four texts below are those where they do not (docs/normalizers.md, "Case folding").
    check("case-fold-tr", Edits::Reconstructed, &CaseFold::turkic_simple(), CASE_FOLD_TR);
}

// The texts on which a normalizer is documented to differ from OpenNLP, with the output it is
// documented to give. Every difference other than the Unicode version of the data is an addition
// that docs/normalizers.md lists under "Differences from OpenNLP"; the Unicode data differences are
// the 18.0 data against the JVM's.

const SECTION_QUOTES: &str = "docs/normalizers.md, \"Quotes\"";
const SECTION_BULLETS: &str = "docs/normalizers.md, \"Bullets\"";
const SECTION_ELLIPSIS: &str = "docs/normalizers.md, \"Ellipsis\"";
const SECTION_INVISIBLE: &str = "docs/normalizers.md, \"Invisible characters\"";
const SECTION_DEHYPHENATION: &str = "docs/normalizers.md, \"Dehyphenation\"";
const SECTION_URLS: &str = "docs/normalizers.md, \"URLs and e-mail addresses (`url/`)\"";
const SECTION_CONTRACTIONS: &str = "docs/normalizers.md, \"English contractions (`contractions/`)\"";
const SECTION_SYMBOLS: &str = "docs/normalizers.md, \"Symbols (`symbols/`)\"";
const SECTION_CONFUSABLES: &str = "docs/normalizers.md, \"Confusable skeleton (`confusables.rs`)\"";
const SECTION_CASEFOLD: &str = "docs/normalizers.md, \"Case folding (`case_fold.rs`)\"";
const SECTION_CHAIN_QUOTES: &str = SECTION_QUOTES;
const SECTION_CHAIN: &str = SECTION_INVISIBLE;

const QUOTE: &[Divergence] = &[Divergence {
    input: "quotes-reversed-9",
    section: SECTION_QUOTES,
    rust: "Das Wort \"Wort\" und \"Wort\" stehen nebeneinander.\n",
}];

const BULLET: &[Divergence] = &[Divergence {
    input: "bullets-extra",
    section: SECTION_BULLETS,
    rust: "  black left\n  black right\n  inverse bullet\n  circled white\n  circled bullet\n  the plain bullet\n",
}];

const ELLIPSIS: &[Divergence] = &[Divergence {
    input: "ellipsis-extra",
    section: SECTION_ELLIPSIS,
    rust: "One dot leader a.b, vertical ellipsis ... and vertical two dot .. and the usual ... and .. here.\n",
}];

const INVISIBLE: &[Divergence] = &[Divergence {
    input: "invisible",
    section: SECTION_INVISIBLE,
    rust: "zerowidthspace, joiner\u{200d}and non\u{200c}joiner, bominside, wordjoiner here.\nMarks  left-to-right and  right-to-left and  embedding  and isolate  .\nControls \u{1} \u{2} \u{7f} \u{80} \u{9f} and softhyphen, variation\u{fe0f} selectors, \u{180e} Mongolian,  .\n",
}];

const DEHYPHENATION: &[Divergence] = &[Divergence {
    input: "dehyphenation-marks",
    section: SECTION_DEHYPHENATION,
    rust: "The word cafe\u{301}teria is in decomposed form, and the Hindi किताबीका word ends in a vowel sign.\nPlain caféteria joins as before.\n",
}];

const EMOJI: &[Divergence] = &[];

const BOUNDED_URL: &[Divergence] = &[Divergence {
    input: "emails-international",
    section: SECTION_URLS,
    rust: "Mail   or   or   today.\nPlain   is the same in both.\n",
}];

const ENGLISH_CONTRACTION: &[Divergence] = &[
    Divergence {
        input: "contractions",
        section: SECTION_CONTRACTIONS,
        rust: "I do not know. We can not go and they will not come. I am sure you are right; he's here, she's there.\nThey have left, we have won, I will try, you will see, he'd go, I'd say. Shall not, ain't, you all, o'clock, ma'am.\nDO NOT SHOUT, Is not it? Would not, could not, should not, must not, need not, has not, have not, had not, are not, was not, were not, does not, did not.\n",
    },
    Divergence {
        input: "contractions-more",
        section: SECTION_CONTRACTIONS,
        rust: "Here are the keys, these are ours, those are yours. May have been, there have been, what have we done, where have you gone.\nHere will do, this will work, these will fit, those will go, which will one? YOU ALL come, you all come, You all.\n",
    },
];

const SYMBOL_JOINER: &[Divergence] = &[
    Divergence {
        input: "emoji-text-default",
        section: SECTION_SYMBOLS,
        rust: "Text default: ☺ ☺\u{fe0f} ❤ ❤\u{fe0f} ✔ ✔\u{fe0f} copyright ©\u{fe0f} trademark ™\u{fe0f} ☀ ☀\u{fe0f} ☂ ☂\u{fe0f}.\nWith text selector: ❤\u{fe0e} ☺\u{fe0e} ®\u{fe0e} and digits 5 5\u{fe0f} 5\u{fe0f}\u{20e3}.\nEmoji default: ✅ ❌ ⭐ ⌚ 😀 💩 🚀.\n",
    },
    Divergence {
        input: "symbols-lone",
        section: SECTION_SYMBOLS,
        rust: "AT&T and R&D but Q and A and a lone and here. 50% and 50 percent and a lone percent there.\ncopyright 2024 Example™ and Brand®, a lone copyright or trademark or ®. Temperature 20°C, section section 3, paragraph 4.\nx plus y = z, a < b > c, 5 * 6, 7 / 8, a ~ b, c ^ d, e | f, g # h, i at j, k $ l.\n",
    },
];

const CONFUSABLE_SKELETON: &[Divergence] = &[
    Divergence {
        input: "bullets",
        section: SECTION_CONFUSABLES,
        rust: "· first itern\n° second itern\n∎ third itern\n‣ fourth itern\n- fifth itern\n· sixth itern\n· seventh\n● eighth\n- dash itern\n* star itern\n",
    },
    Divergence {
        input: "casefold-special",
        section: SECTION_CONFUSABLES,
        rust: "STRASSE Straße ß l\u{307} i ffi 'n j\u{306} i\u{308}\u{301} u\u{308}\u{301} h\u{331} u\u{313} a\u{328}\u{303} p\u{313}.\nMixed Case: Hello World, MiXeD cAsE, E\u{301}COLE, N\u{303}ANDU, ABΓ, Ab\u{304}B, D ᴅ.\nTitlecase Dz\u{306} Lj Nj Dz and ligature st մն եi.\n",
    },
    Divergence {
        input: "cjk-mixed",
        section: SECTION_CONFUSABLES,
        rust: "ᄉーᄀ天天气很好˳ 我们去公园玩! ''你好''˳\n日本語のテキス卜 と 空白˳ABC l23˳\n하ᄂ구ᄀoᅥ 터丨ᄀᄉーᄐー、rnixed with English text... and ー dashes.\n",
    },
    Divergence {
        input: "confusables-misc",
        section: SECTION_CONFUSABLES,
        rust: "llllll lc, l o ᴀ i j, coʍ, paypal.\nMath: ABC abc and AB. Hebrew l vs ASCll l, l vs l.\nArrnenian o u and Cherokee A R S look like Latin letters.\n",
    },
    Divergence {
        input: "dehyphenation-marks",
        section: SECTION_CONFUSABLES,
        rust: "The word cafe\u{301}-\nteria is in decornposed forrn, and the Hindi कিत𑧑बी-\nक𑧑 word ends in a vowel sign.\nPlain cafe\u{301}-\nteria joins as before.\n",
    },
    Divergence {
        input: "emails-international",
        section: SECTION_CONFUSABLES,
        rust: "Mail Jose\u{301}@exarnple.corn or 用戶@例子.广告 or e\u{301}a@b.co today.\nPlain ascii@exarnple.org is the sarne in both.\n",
    },
    Divergence {
        input: "fullwidth-kana",
        section: SECTION_CONFUSABLES,
        rust: "Fullwidth ABC abc Ol2 !? and ideographic cornrna、full stop˳\nHalfwidth katakana 力\u{ff9e}キ\u{ff9e} 八\u{30a}匕\u{30a} ア亻ウ and voiced rnark sequences 力\u{ff9e} 八\u{30a}.\nSquare ㌀ ㌃ ㍿ and parenthesized (株), ㊤.\n",
    },
    Divergence {
        input: "hangul",
        section: SECTION_CONFUSABLES,
        rust: "하ᄂᄀーᄅ cornposed, and decornposed 하ᄂᄀーᄅ jarno.\nMixed: 가 가 가ᄀ and 다丨하ᄂᄆ丨ᄂ구ᄀ 다丨하ᄂᄆ丨ᄂ구ᄀ.\nCornpatibility jarno 가 and halfwidth 가 stay different under NFC.\n",
    },
    Divergence {
        input: "kana-small",
        section: SECTION_CONFUSABLES,
        rust: "ア亻ウ ー あいう ウ\u{ff9e} ウ\u{ff9e} か\u{ff9e}き❬\u{ff9e} か\u{ff9e}き\u{ff9e}❬\u{ff9e}\u{ff9e} y\u{308} Y\u{308}.\nCornbining dakuten after kana and ASCll voiced: 力\u{ff9e}キ\u{ff9e}ク\u{ff9e} and 八\u{30a}匕\u{30a}.\n",
    },
    Divergence {
        input: "ligatures-nfkc",
        section: SECTION_CONFUSABLES,
        rust: "The first flight offered an efficient office; st and ft are ligatures too.\nDutch ij: ij, Serbian dz\u{306} lj, units kg MHz ㎡, TM ™, SM ℠, No No, ➀➁.\nSuperscript xᒾ and E=rncᒾ, l/5 fifth, l/4 quarter, ... ellipsis,   nbsp.\n",
    },
    Divergence {
        input: "mixed-article",
        section: SECTION_CONFUSABLES,
        rust: "''The re-\nsults,'' wrote Dr. Mu\u{308}llerーa well-known expertー''were surprising...''\nSee https://exarnple.corn/paper.pdf. Cost: £l9.99 (≈ Ꞓ23). Contact: rn.rnueller@exarnple.org\n· itern one with ideographic space\n· itern two   with  extra   spaces\r\nlt's lOO°/₀ done :) 😀 #win\n",
    },
    Divergence {
        input: "numbers-basic",
        section: SECTION_CONFUSABLES,
        rust: "ln 2O24 there were l,234,567 visitors, 3.l4l59 ratio, l2:3O tirne, l/2 cup, -5 degrees, +7 points.\nPhone 555-l234 or +l (8OO) 555-Ol99. Price $l9.99, 2O°/₀ off, 5elO and OxlF, l,5 in Gerrnan style.\nVersion 2.O.l, lOth, 2lst, 3rd place, H2O, Bl2, 4K, 24/7, lOOrn.\n",
    },
    Divergence {
        input: "numbers-unicode",
        section: SECTION_CONFUSABLES,
        rust: "Arabic-lndic .l٢٣٤, extended o٦V, Devanagari ٩२3, Thai ๑๒๓.\nFullwidth l23 and 456, superscripts ᒾ³ᣳ, fractions l/2 l/3, Rornan lll and circled ➀.\nMixed l2٣ and 45 in one token, Bengali o১২ and Tarnil oக.\n",
    },
    Divergence { input: "symbol-percent", section: SECTION_CONFUSABLES, rust: "°/₀" },
    Divergence {
        input: "symbols-lone",
        section: SECTION_CONFUSABLES,
        rust: "AT&T and R&D but Q & A and a lone & here. 5O°/₀ and 5O °/₀ and a lone °/₀ there.\n© 2O24 Exarnple™ and Brand®, a lone © or ™ or ®. Ternperature 2O°C, section § 3, ¶ 4.\nx + y = z, a < b > c, 5 * 6, 7 / 8, a ~ b, c ^ d, e l f, g # h, i @ j, k $ l.\n",
    },
    Divergence {
        input: "whitespace-unicode",
        section: SECTION_CONFUSABLES,
        rust: "no break space, ideographic space, ern space, thin space, narrow no-break, hair space.\nFigure space and en space and four-per-ern six-per-ern.\n  leading and trailing   \n",
    },
];

const CASE_FOLD_TR: &[Divergence] = &[
    Divergence {
        input: "casefold-special",
        section: SECTION_CASEFOLD,
        rust: "strasse straße ß i ı ﬃ ŉ ǰ ΐ ΰ ẖ ὐ ᾷ ῤ.\nmixed case: hello world, mixed case, école, ñandu, αβγ, абв, Ꭰ Ꭰ.\ntitlecase ǆ ǉ ǌ ǳ and ligature ﬆ ﬓ և.\n",
    },
    Divergence {
        input: "confusables-misc",
        section: SECTION_CASEFOLD,
        rust: "ⅰⅱⅲ ⅼⅽ, ℓ ℴ ᴀ ı ȷ, ᴄᴏᴍ, ｐａｙｐａｌ.\nmath: 𝐀𝐁𝐂 𝐚𝐛𝐜 and 𝔸𝔹. hebrew ו vs ascıı |, ן vs l.\narmenian օ ս and cherokee Ꭺ Ꭱ Ꮪ look like latin letters.\n",
    },
    Divergence {
        input: "greek-sigma",
        section: SECTION_CASEFOLD,
        rust: "οδυσσευσ and σασ and σπιτι, ανθρωποσ.\nfinal sigma cases: αβσ ασ σ σ. σσ (ασ) ασα and αΐσ.\ntonos: άέήίόύώ, άέήίόύώ, ΐ ΰ, ἀ ἀ ᾳ ᾳ.\n",
    },
    Divergence {
        input: "ligatures-nfkc",
        section: SECTION_CASEFOLD,
        rust: "the ﬁrst ﬂight oﬀered an eﬃcient oﬃce; ﬆ and ﬆ are ligatures too.\ndutch ij: ĳ, serbian ǆ ǉ, units ㎏ ㎒ ㎡, tm ™, sm ℠, no №, ①②.\nsuperscript x² and e=mc², ⅕ fifth, ¼ quarter, … ellipsis, \u{a0} nbsp.\n",
    },
];

const DEFAULT_CHAIN: &[Divergence] = &[
    Divergence {
        input: "invisible",
        section: SECTION_CHAIN,
        rust: "zerowidthspace, joiner\u{200d}and non\u{200c}joiner, bominside, wordjoiner here. marks left-to-right and right-to-left and embedding and isolate . controls \u{1} \u{2} \u{7f} \u{80} \u{9f} and softhyphen, variation selectors, \u{180e} mongolian, .",
    },
    Divergence {
        input: "quotes-reversed-9",
        section: SECTION_CHAIN_QUOTES,
        rust: "das wort \"wort\" und \"wort\" stehen nebeneinander.",
    },
];
