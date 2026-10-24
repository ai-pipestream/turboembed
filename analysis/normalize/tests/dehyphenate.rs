//! The dehyphenator. The expectations are those of OpenNLP's
//! DehyphenationCharSequenceNormalizerTest, with UTF-16 offsets written as
//! UTF-8 byte offsets, followed by the compound mode and generated inputs
//! compared with a slow reference.

mod folds_support;

use folds_support::{Rng, back, check_alignment, covered, forward, run_checked};
use turbo_normalize::tables::{LINE_BREAKS, WHITESPACE};
use turbo_normalize::{Dehyphenate, DehyphenateMode, Normalizer, apply, normalize};

/// An edit the dehyphenator may make: delete an optional hyphen, one line
/// break and the indentation after it.
fn join_edit(parent: &str, child: &str) -> bool {
    let rest = parent.strip_prefix(['-', '\u{AD}', '\u{2010}']).unwrap_or(parent);
    let rest = match rest.strip_prefix("\r\n") {
        Some(after) => after,
        None => match rest.chars().next() {
            Some(c) if LINE_BREAKS.contains(&c) => &rest[c.len_utf8()..],
            _ => return false,
        },
    };
    child.is_empty() && rest.chars().all(|c| WHITESPACE.contains(&c) && !LINE_BREAKS.contains(&c))
}

fn join(s: &str) -> String {
    run_checked(&Dehyphenate::new(), s, join_edit)
}

fn join_keeping_compounds(s: &str) -> String {
    run_checked(&Dehyphenate::keep_compounds(), s, join_edit)
}

#[test]
fn id_and_modes() {
    assert_eq!(Dehyphenate::ID, "turbo:normalize:dehyphenate");
    assert_eq!(Dehyphenate::default().mode(), DehyphenateMode::Join);
    assert_eq!(Dehyphenate::keep_compounds().mode(), DehyphenateMode::KeepCompounds);
    assert_ne!(Dehyphenate::new().config_hash(), Dehyphenate::keep_compounds().config_hash());
}

// testJoinAcrossEveryLineBreakForm
#[test]
fn joins_across_every_line_break() {
    for text in [
        "litiga-\ntion",
        "litiga-\u{B}tion",
        "litiga-\u{C}tion",
        "litiga-\rtion",
        "litiga-\r\ntion",
        "litiga-\u{85}tion",
        "litiga-\u{2028}tion",
        "litiga-\u{2029}tion",
    ] {
        assert_eq!(join(text), "litigation", "{text:?}");
    }
}

// testFormFeedPageBreakDoesNotJoinAcrossThePage
#[test]
fn a_second_line_break_stops_the_join() {
    let text = "litiga-\n\u{C}Page 3\ntion";
    assert_eq!(join(text), text);
    assert_eq!(join("litiga-\n\ntion"), "litiga-\n\ntion");
    assert_eq!(join("litiga-\r\n\r\ntion"), "litiga-\r\n\r\ntion");
    // Two carriage returns are two breaks.
    assert_eq!(join("litiga-\r\rtion"), "litiga-\r\rtion");
}

// testJoinAcrossSoftHyphen, testJoinAcrossTypesetHyphen, testNonBreakingHyphenIsLeftAlone
#[test]
fn which_hyphens_join() {
    assert_eq!(join("litiga\u{AD}\ntion"), "litigation");
    assert_eq!(join("well\u{2010}\nknown"), "wellknown");
    assert_eq!(join("well\u{2011}\nknown"), "well\u{2011}\nknown");
    // Dashes are not hyphens.
    assert_eq!(join("well\u{2013}\nknown"), "well\u{2013}\nknown");
    assert_eq!(join("well\u{2212}\nknown"), "well\u{2212}\nknown");
}

// testHyphenatedCompoundSplitAtItsBreakJoinsAsDocumented
#[test]
fn a_compound_split_at_its_own_hyphen_joins() {
    assert_eq!(join("well-\nknown"), "wellknown");
}

// testJoinConsumesContinuationLineIndentation
#[test]
fn indentation_of_the_continuation_line_is_deleted() {
    for text in ["com-\r\n  plete", "com-\n\t\tplete", "com-\n \u{A0}plete", "com-\n\u{3000}\u{2009}plete"] {
        assert_eq!(join(text), "complete", "{text:?}");
    }
}

// testMultipleJoinsInOneText
#[test]
fn several_joins_in_one_text() {
    assert_eq!(join("litiga-\ntion com-\nplete"), "litigation complete");
    // A joined continuation can itself end in a hyphen at a break.
    assert_eq!(join("a-\nb-\nc"), "abc");
}

// testInlineHyphenIsLeftAlone, testHyphenBreakBeforeNonLetterIsLeftAlone,
// testHyphenAtEndOfTextIsLeftAlone, testHyphenWithoutPrecedingLetterIsLeftAlone,
// testLineBreakWithoutHyphenIsLeftAlone
#[test]
fn what_is_left_alone() {
    for text in [
        "well-known",
        "ver-\n3",
        "end-\n.",
        "word-",
        "word-\n",
        "word-\n  ",
        "-\nword",
        "3-\nword",
        " -\nword",
        "word\nnext",
        "",
    ] {
        assert_eq!(join(text), text, "{text:?}");
    }
}

// testNoOpReturnsSameInstanceAndIdentityAlignment
#[test]
fn ordinary_text_is_borrowed_with_an_identity_alignment() {
    let text = "an ordinary sentence.";
    assert!(matches!(apply(&Dehyphenate::new(), text), std::borrow::Cow::Borrowed(_)));
    let (out, a) = normalize(&Dehyphenate::new(), text);
    assert_eq!(out, text);
    assert!(a.edits().is_empty());
    assert_eq!(back(&a, 0, text.len()), (0, text.len() as u32));
    assert_eq!(back(&a, 3, 11), (3, 11));
    assert_eq!(forward(&a, 3, 11), (3, 11));
}

// testSerializationRestoresSharedInstance (the alignment part) and
// testJoinedWordSpanCoversBothHalvesAndTheBreakInTheOriginal
#[test]
fn the_joined_word_covers_both_halves_and_the_break() {
    let (out, a) = normalize(&Dehyphenate::new(), "litiga-\ntion");
    assert_eq!(out, "litigation");
    assert_eq!(back(&a, 0, 10), (0, 12));
    // The deleted hyphen maps forward to an empty span at the join.
    assert_eq!(forward(&a, 6, 7), (6, 6));
}

// testSpansAroundTheEditMapUnchanged
#[test]
fn spans_around_the_join_map_unchanged() {
    let text = "a litiga-\ntion b";
    let (out, a) = normalize(&Dehyphenate::new(), text);
    assert_eq!(out, "a litigation b");
    assert_eq!(back(&a, 0, 1), (0, 1));
    assert_eq!(back(&a, 2, 12), (2, 14));
    assert_eq!(back(&a, 13, 14), (15, 16));
}

// testCarriageReturnLineFeedWithIndentationAlignment
#[test]
fn crlf_and_indentation_alignment() {
    let (out, a) = normalize(&Dehyphenate::new(), "com-\r\n  plete");
    assert_eq!(out, "complete");
    assert_eq!(back(&a, 0, 8), (0, 13));
    assert_eq!(forward(&a, 3, 4), (3, 3));
}

// testJoinAfterSupplementaryPlaneLetter
#[test]
fn a_supplementary_letter_before_the_hyphen() {
    // U+10400 DESERET CAPITAL LETTER LONG I is four bytes.
    let text = "\u{10400}-\nbc";
    let (out, a) = normalize(&Dehyphenate::new(), text);
    assert_eq!(out, "\u{10400}bc");
    assert_eq!(back(&a, 0, 6), (0, 8));
}

// testSupplementaryCharactersKeepCodeUnitOffsets
#[test]
fn offsets_after_a_join_and_a_supplementary_character() {
    let text = "a-\nb\u{1F600}c";
    let (out, a) = normalize(&Dehyphenate::new(), text);
    assert_eq!(out, "ab\u{1F600}c");
    assert_eq!(back(&a, 6, 7), (8, 9));
    assert_eq!(back(&a, 0, 6), (0, 8));
    assert_eq!(covered(text, &a, 6, 7), "c");
}

#[test]
fn letters_of_other_scripts_join() {
    assert_eq!(join("\u{43F}\u{440}\u{438}-\n\u{43C}\u{435}\u{440}"), "\u{43F}\u{440}\u{438}\u{43C}\u{435}\u{440}");
    assert_eq!(join("Stra\u{DF}en-\nbahn"), "Stra\u{DF}enbahn");
    // Neither is a word that ends in a mark with no letter before it.
    assert_eq!(join("\u{301}-\nx"), "\u{301}-\nx");
    assert_eq!(join("3\u{301}-\nx"), "3\u{301}-\nx");
    // A continuation that starts with a mark does not join.
    assert_eq!(join("ab-\n\u{301}x"), "ab-\n\u{301}x");
}

// A difference from OpenNLP, which needs a letter right before the hyphen:
// a letter followed by combining marks also qualifies.
#[test]
fn a_word_ending_in_a_combining_mark_joins() {
    // Decomposed "caf\u{E9}-\nteria".
    assert_eq!(join("cafe\u{301}-\nteria"), "cafe\u{301}teria");
    // Devanagari: the first half ends in the vowel sign U+0940 (Mc).
    assert_eq!(
        join("\u{939}\u{93F}\u{928}\u{94D}\u{926}\u{940}-\n\u{92D}\u{93E}\u{937}\u{93E}"),
        "\u{939}\u{93F}\u{928}\u{94D}\u{926}\u{940}\u{92D}\u{93E}\u{937}\u{93E}"
    );
    // Several marks.
    assert_eq!(join("o\u{308}\u{301}-\r\n  b"), "o\u{308}\u{301}b");
}

#[test]
fn keep_compounds_keeps_the_hyphen_before_an_uppercase_letter() {
    assert_eq!(join_keeping_compounds("Wi-\nFi"), "Wi-Fi");
    assert_eq!(join_keeping_compounds("M\u{FC}ller-\r\n   L\u{FC}denscheidt"), "M\u{FC}ller-L\u{FC}denscheidt");
    assert_eq!(join_keeping_compounds("Baden-\n\u{C4}rger"), "Baden-\u{C4}rger");
    // Lowercase continuations are still joined.
    assert_eq!(join_keeping_compounds("litiga-\ntion"), "litigation");
    assert_eq!(join_keeping_compounds("well-\nknown"), "wellknown");
    // Only U+002D is kept; a soft hyphen or U+2010 is a typesetter's mark.
    assert_eq!(join_keeping_compounds("Wi\u{AD}\nFi"), "WiFi");
    assert_eq!(join_keeping_compounds("Wi\u{2010}\nFi"), "WiFi");
    // The parity mode joins them all.
    assert_eq!(join("Wi-\nFi"), "WiFi");
}

#[test]
fn keep_compounds_maps_the_kept_hyphen_to_itself() {
    let text = "Wi-\n  Fi";
    let (out, a) = normalize(&Dehyphenate::keep_compounds(), text);
    assert_eq!(out, "Wi-Fi");
    assert_eq!(covered(text, &a, 2, 3), "-");
    assert_eq!(covered(text, &a, 3, 5), "Fi");
    assert_eq!(a.edits().len(), 1);
    assert_eq!((a.edits()[0].parent.start, a.edits()[0].parent.end), (3, 6));
}

/// The dehyphenator written over a character vector. Letters are the
/// alphabetic characters that are not combining marks, which for the
/// characters the generator uses are the same as `Character.isLetter`; the
/// marks are the two mark ranges the generator draws from.
fn reference(src: &str, keep_compounds: bool) -> String {
    let chars: Vec<char> = src.chars().collect();
    let is_mark = |c: char| matches!(c, '\u{300}'..='\u{36F}' | '\u{93E}'..='\u{94D}');
    let is_letter = |c: char| c.is_alphabetic() && !is_mark(c);
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let hyphen = matches!(c, '-' | '\u{AD}' | '\u{2010}');
        let mut k = i;
        while k > 0 && is_mark(chars[k - 1]) {
            k -= 1;
        }
        if hyphen && k > 0 && is_letter(chars[k - 1]) && i + 1 < chars.len() && LINE_BREAKS.contains(&chars[i + 1]) {
            let mut j = i + 2;
            if chars[i + 1] == '\r' && j < chars.len() && chars[j] == '\n' {
                j += 1;
            }
            while j < chars.len() && WHITESPACE.contains(&chars[j]) && !LINE_BREAKS.contains(&chars[j]) {
                j += 1;
            }
            if j < chars.len() && is_letter(chars[j]) {
                if keep_compounds && c == '-' && chars[j].is_uppercase() {
                    out.push('-');
                }
                i = j;
                continue;
            }
        }
        out.push(c);
        i += 1;
    }
    out
}

#[test]
fn generated_texts_match_the_reference() {
    let pool = [
        "a",
        "Z",
        "word",
        "\u{FC}",
        "\u{C4}",
        "\u{4E2D}",
        "\u{10400}",
        "-",
        "-",
        "\u{AD}",
        "\u{2010}",
        "\u{2011}",
        "\n",
        "\r",
        "\r\n",
        "\u{B}",
        "\u{C}",
        "\u{85}",
        "\u{2028}",
        "\u{2029}",
        " ",
        "\t",
        "\u{A0}",
        "\u{3000}",
        "7",
        ".",
        "\u{1F600}",
        "\u{301}",
        "\u{940}",
        "\u{926}",
    ];
    let mut rng = Rng::new(11);
    for round in 0..4000 {
        let src = rng.text(&pool, 16);
        let keep = round % 2 == 1;
        let n = if keep { Dehyphenate::keep_compounds() } else { Dehyphenate::new() };
        let (out, a) = normalize(&n, &src);
        assert_eq!(out, reference(&src, keep), "{:?} on {src:?}", n.mode());
        check_alignment(&src, &out, &a, join_edit);
    }
}

#[test]
fn lexicon_mode_with_hunspell_and_morfologik_on_extracted_text() {
    use std::path::Path;
    use std::sync::Arc;
    use turbo_hunspell::{Dictionary as HunspellDictionary, LoadMode};
    use turbo_morfologik::{MorfologikDictionary, MorfologikDictionaryBuilder};

    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../goldens/data/fixtures/hunspell");
    let compound_aff = std::fs::read(fixtures.join("compound.aff")).unwrap();
    let compound_dic = std::fs::read(fixtures.join("compound.dic")).unwrap();
    let hunspell_compound =
        Arc::new(HunspellDictionary::from_bytes(&compound_aff, &compound_dic, LoadMode::Strict).unwrap());
    let dehyp_compound = Dehyphenate::with_hunspell(hunspell_compound);
    assert_eq!(dehyp_compound.mode(), DehyphenateMode::Lexicon);
    assert_ne!(dehyp_compound.config_hash(), Dehyphenate::new().config_hash());
    assert_ne!(dehyp_compound.config_hash(), Dehyphenate::keep_compounds().config_hash());

    // In compound.dic, "foot/YS" + "ball/YS" forms the compound "football",
    // whereas "bad/YF" carries COMPOUNDFORBIDFLAG F so "badball" is forbidden
    // while both "bad" and "ball" are known words.
    assert_eq!(run_checked(&dehyp_compound, "foot-\nball and bad-\nball", join_edit), "football and bad-ball");

    let affixes_aff = std::fs::read(fixtures.join("affixes.aff")).unwrap();
    let affixes_dic = std::fs::read(fixtures.join("affixes.dic")).unwrap();
    let hunspell_affixes =
        Arc::new(HunspellDictionary::from_bytes(&affixes_aff, &affixes_dic, LoadMode::Strict).unwrap());
    let dehyp_affixes = Dehyphenate::with_hunspell(hunspell_affixes);
    assert_ne!(dehyp_compound.config_hash(), dehyp_affixes.config_hash());
    // "ponies" is derived from "pony/SM", whereas "co-op" is listed with its hyphen.
    assert_eq!(run_checked(&dehyp_affixes, "pon-\nies at the co-\nop", join_edit), "ponies at the co-op");

    // Distinct lexicons produce distinct provenance config hashes and views on a Document.
    use turbo_analysis::{Analyzer, Bound, Document, Key, View};
    use turbo_normalize::NormalizeStep;
    let analyzer = Analyzer::builder()
        .step(Bound::new(NormalizeStep::new(dehyp_compound)).write(0, "v1"))
        .step(Bound::new(NormalizeStep::new(dehyp_affixes)).write(0, "v2"))
        .build()
        .unwrap();
    let mut docs = vec![Document::new("foot-\nball and bad-\nball")];
    assert!(analyzer.run(&mut docs)[0].is_ok());
    assert_eq!(docs[0].get_key(&Key::<View>::named("v1")).unwrap().text(), "football and bad-ball");
    assert_eq!(docs[0].get_key(&Key::<View>::named("v2")).unwrap().text(), "football and badball");
    assert_ne!(
        docs[0].provenance(Key::<View>::named("v1").id()).unwrap().config,
        docs[0].provenance(Key::<View>::named("v2").id()).unwrap().config
    );

    // Multi-line dictionary compound ("mother-\nin-\nlaw") and mixed compound + syllable split
    // ("well-\nin-\nformed") with a Hunspell dictionary.
    let multi_aff = b"SET UTF-8\n";
    let multi_dic = b"3\nmother-in-law\nwell\ninformed\n";
    let hunspell_multi = Arc::new(HunspellDictionary::from_bytes(multi_aff, multi_dic, LoadMode::Strict).unwrap());
    let dehyp_multi = Dehyphenate::with_hunspell(hunspell_multi);
    assert_eq!(
        run_checked(&dehyp_multi, "mother-\nin-\nlaw and well-\nin-\nformed", join_edit),
        "mother-in-law and well-informed"
    );

    // Morfologik FSA dictionary on extracted prose with "litiga-\ntion",
    // "well-\nknown", "Mc-\nDonald", "state-of-the-\nart", and multi-line "inter-\nnation-\nal".
    let entries = b"litigation,litigation,NN\n\
well,well,RB\n\
known,known,JJ\n\
McDonald,McDonald,NNP\n\
state,state,NN\n\
of,of,IN\n\
the,the,DT\n\
art,art,NN\n\
international,international,JJ\n";
    let info = b"fsa.dict.separator=,\nfsa.dict.encoding=UTF-8\nfsa.dict.encoder=SUFFIX\n";
    let dict_bytes = MorfologikDictionaryBuilder::compile_bytes(entries, info).unwrap();
    let morfologik = Arc::new(MorfologikDictionary::from_bytes(&dict_bytes, info).unwrap());
    let dehyp_morf = Dehyphenate::with_morfologik(morfologik);

    let extracted = "The well-\n  known firm of Mc-\nDonald handled the litiga-\ntion with state-of-the-\nart inter-\nnation-\nal care.";
    let (out, alignment) = normalize(&dehyp_morf, extracted);
    check_alignment(extracted, &out, &alignment, join_edit);
    assert_eq!(out, "The well-known firm of McDonald handled the litigation with state-of-the-art international care.");
    let well_known_out = out.find("well-known").unwrap();
    let well_known_src = extracted.find("well-\n  known").unwrap();
    assert_eq!(
        back(&alignment, well_known_out, well_known_out + "well-known".len()),
        (well_known_src as u32, (well_known_src + "well-\n  known".len()) as u32)
    );
    let mcdonald_out = out.find("McDonald").unwrap();
    let mcdonald_src = extracted.find("Mc-\nDonald").unwrap();
    assert_eq!(
        back(&alignment, mcdonald_out, mcdonald_out + "McDonald".len()),
        (mcdonald_src as u32, (mcdonald_src + "Mc-\nDonald".len()) as u32)
    );
    let litigation_out = out.find("litigation").unwrap();
    let litigation_src = extracted.find("litiga-\ntion").unwrap();
    assert_eq!(
        back(&alignment, litigation_out, litigation_out + "litigation".len()),
        (litigation_src as u32, (litigation_src + "litiga-\ntion".len()) as u32)
    );
}
