//! `SocialMedia`: the expectations of OpenNLP's
//! SocialMediaCharSequenceNormalizerCharacterizationTest, the one-pass
//! scan against a four-pass reference, and the placeholders.

mod scanners_support;

use scanners_support::{Rng, placeholders, run};
use turbo_normalize::SocialMedia;
use turbo_normalize_layers::PlaceholderKind::{Emoticon, Hashtag, Mention, RetweetMarker};

fn check(input: &str, expected: &str) {
    assert_eq!(run(&SocialMedia, input), expected, "{input:?}");
}

/// OpenNLP's four passes, ported one to one over chars.
mod reference {
    use turbo_unicode::category::{GeneralCategory, general_category};

    fn space(c: char) -> bool {
        matches!(c, '\t'..='\r' | ' ')
    }

    fn tags(t: &[char]) -> Vec<char> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < t.len() {
            if (t[i] == '#' || t[i] == '@') && i + 1 < t.len() && !space(t[i + 1]) {
                let mut end = i + 2;
                while end < t.len() && !space(t[end]) {
                    end += 1;
                }
                out.push(' ');
                i = end;
            } else {
                out.push(t[i]);
                i += 1;
            }
        }
        out
    }

    fn blocked(t: &[char], i: usize) -> bool {
        let Some(&before) = i.checked_sub(1).map(|k| &t[k]) else { return false };
        if before.is_ascii_alphanumeric() || before == '_' {
            return true;
        }
        if general_category(before) == GeneralCategory::Mn {
            for &c in t[..i].iter().rev() {
                if u32::from(c) > 0xFFFF {
                    return false;
                }
                if turbo_unicode::java::is_letter_or_digit(c) {
                    return true;
                }
                if general_category(c) != GeneralCategory::Mn {
                    return false;
                }
            }
        }
        false
    }

    fn retweets(t: &[char]) -> Vec<char> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < t.len() {
            if matches!(t[i], 'r' | 'R') && !blocked(t, i) {
                let mut end = i;
                while end + 3 <= t.len()
                    && matches!(t[end], 'r' | 'R')
                    && matches!(t[end + 1], 't' | 'T')
                    && matches!(t[end + 2], ' ' | ':')
                {
                    end += 3;
                }
                if end > i {
                    out.push(' ');
                    i = end;
                    continue;
                }
            }
            out.push(t[i]);
            i += 1;
        }
        out
    }

    fn emoticons(t: &[char]) -> Vec<char> {
        let mouth = |c: char| "()dDoOpP".contains(c);
        let mut out = Vec::new();
        let mut i = 0;
        while i < t.len() {
            if ":;xX".contains(t[i]) {
                if i + 2 < t.len() && t[i + 1] == '-' && mouth(t[i + 2]) {
                    out.push(' ');
                    i += 3;
                    continue;
                }
                if i + 1 < t.len() && mouth(t[i + 1]) {
                    out.push(' ');
                    i += 2;
                    continue;
                }
            }
            out.push(t[i]);
            i += 1;
        }
        out
    }

    fn laughter(t: &[char]) -> Vec<char> {
        let consonant = |c: char| "hHjJ".contains(c);
        let vowel = |c: char| "aeiouAEIOU".contains(c);
        let eq = |a: char, b: char| a.eq_ignore_ascii_case(&b);
        let mut out = Vec::new();
        let mut i = 0;
        while i < t.len() {
            if consonant(t[i]) {
                let mut p = i + 1;
                while p < t.len() && consonant(t[p]) {
                    p += 1;
                }
                if p < t.len() && vowel(t[p]) {
                    let c = t[p - 1];
                    let mut q = p + 1;
                    while q < t.len() && vowel(t[q]) {
                        q += 1;
                    }
                    let v = t[q - 1];
                    let mut end = q;
                    loop {
                        let mut ce = end;
                        while ce < t.len() && eq(t[ce], c) {
                            ce += 1;
                        }
                        if ce == end {
                            break;
                        }
                        let mut ve = ce;
                        while ve < t.len() && eq(t[ve], v) {
                            ve += 1;
                        }
                        if ve == ce {
                            break;
                        }
                        end = ve;
                    }
                    if end > q {
                        out.extend([c, v, c, v]);
                        i = end;
                        continue;
                    }
                }
            }
            out.push(t[i]);
            i += 1;
        }
        out
    }

    pub fn four_pass(text: &str) -> String {
        let t: Vec<char> = text.chars().collect();
        laughter(&emoticons(&retweets(&tags(&t)))).into_iter().collect()
    }
}

#[test]
fn hashtags_and_handles_become_a_single_space() {
    check("", "");
    check("asdf #hasdk23 2nnfdf", "asdf   2nnfdf");
    check("asdf @hasdk23 2nnfdf", "asdf   2nnfdf");
    check("#", "#");
    check("##", " ");
    check("# #", "# #");
    check("#a#b", " ");
    check("abc#def", "abc ");
    check("@user hi", "  hi");
    check("@ home", "@ home");
    check("#\u{1F600}", " ");
    check("#\u{A0}x", " ");
    check("email me @home now", "email me   now");
    check("\u{1F600} #yes \u{1F600}", "\u{1F600}   \u{1F600}");
}

#[test]
fn retweet_markers_collapse_behind_a_jdk_style_word_boundary() {
    check("RT RT RT 2nnfdf", " 2nnfdf");
    check("rt this", " this");
    check("rt rt rt", " rt");
    check("RT: hello", "  hello");
    check("rt:rt:go", " go");
    check("RT @user: hello", "   hello");
    check("art here", "art here");
    check("smart tv", "smart tv");
    check("cart rt cart", "cart  cart");
    check("hrt x", "hrt x");
    check("1rt x", "1rt x");
    check("_rt x", "_rt x");
    check("-rt x", "- x");
}

#[test]
fn word_boundary_treats_non_ascii_letters_as_non_word_but_honors_combining_marks() {
    check("\u{E9}rt x", "\u{E9} x");
    check("e\u{301}rt x", "e\u{301}rt x");
    check("\u{301}rt x", "\u{301} x");
    check("a\u{101FD}rt x", "a\u{101FD} x");
}

#[test]
fn emoticons_become_a_single_space_even_inside_words() {
    check("hello :-) hello", "hello   hello");
    check("hello ;) hello", "hello   hello");
    check(":) hello", "  hello");
    check("hello :P", "hello  ");
    check("x-d ok", "  ok");
    check(":-( sad", "  sad");
    check("::-)", ": ");
    check("expo", "e o");
    check("xoxo", "  ");
    check("taxi", "taxi");
    check(":-x", ":-x");
    check("X-P", " ");
    check(":o", " ");
    check(":d", " ");
    check("x-o", " ");
    check("xp", " ");
    check("max power", "max power");
}

#[test]
fn laughter_shrinks_to_the_last_consonant_vowel_pair_twice() {
    for (input, expected) in [
        ("ahahahah", "ahahah"),
        ("hahha", "haha"),
        ("hahaa", "haha"),
        ("ahahahahhahahhahahaaaa", "ahaha"),
        ("jajjajajaja", "jaja"),
        ("hahaha", "haha"),
        ("haha", "haha"),
        ("HaHaHa", "HaHa"),
        ("hAHa", "hAhA"),
        ("HAHAHA", "HAHA"),
        ("hhaahhaa", "haha"),
        ("hjaja", "jaja"),
        ("ha", "ha"),
        ("haaaa", "haaaa"),
        ("hahe", "hahe"),
        ("hajaha", "hajaha"),
        ("jejeje", "jeje"),
        ("Bahahaha", "Bahaha"),
        ("hihihi", "hihi"),
        ("hohoho", "hoho"),
        ("huhuhu", "huhu"),
        ("hyhy", "hyhy"),
        ("jujuju", "juju"),
        ("hahaha!", "haha!"),
        ("hjhj", "hjhj"),
    ] {
        check(input, expected);
    }
}

#[test]
fn the_passes_compose_in_order() {
    check("RT @user: check #cool :-) hahaha", "   check     haha");
    check("x #tag rt go", "x    go");
}

/// The one place the passes interact: the space a hashtag becomes ends a
/// retweet unit, so the marker and the tag become one space.
#[test]
fn a_tag_can_end_a_retweet_unit() {
    check("rt#tag", " ");
    check("rt rt@user hi", "  hi");
    check("rt#", "rt#");
    assert_eq!(reference::four_pass("rt#tag"), " ");
    let (out, found) = placeholders(SocialMedia, "go rt rt@user hi");
    assert_eq!(out, "go   hi");
    assert_eq!(found, [("rt rt".to_string(), RetweetMarker), ("@user".to_string(), Mention)]);
}

#[test]
fn placeholders_name_what_was_replaced() {
    let (out, found) = placeholders(SocialMedia, "RT @ana: #win :-) hahaha");
    assert_eq!(out, "       haha");
    assert_eq!(
        found,
        [
            ("RT ".to_string(), RetweetMarker),
            ("@ana:".to_string(), Mention),
            ("#win".to_string(), Hashtag),
            (":-)".to_string(), Emoticon),
        ]
    );
}

/// The one-pass scan against OpenNLP's four passes, on the pool of
/// SocialMediaCharSequenceNormalizerCharacterizationTest (without its lone
/// surrogates).
#[test]
fn one_pass_matches_four_passes_on_random_inputs() {
    let pool = [
        "a",
        "A",
        "e",
        "h",
        "H",
        "j",
        "J",
        "o",
        "u",
        "y",
        "x",
        "X",
        "d",
        "p",
        "P",
        ")",
        "(",
        "-",
        ":",
        ";",
        " ",
        "  ",
        "\t",
        "\n",
        "#",
        "@",
        "_",
        "1",
        "rt",
        "RT",
        "rt:",
        "Rt ",
        "t",
        "r",
        "ha",
        "Ha",
        "ah",
        "hh",
        "aa",
        "jj",
        "\u{E9}",
        "\u{301}",
        "\u{101FD}",
        "\u{1F600}",
        "\u{A0}",
    ];
    let mut rng = Rng::new(42);
    for _ in 0..50_000 {
        let input = rng.input(&pool);
        assert_eq!(run(&SocialMedia, &input), reference::four_pass(&input), "{input:?}");
    }
}

mod bounded {
    use super::scanners_support::{placeholders, run};
    use turbo_normalize::BoundedSocialMedia;
    use turbo_normalize_layers::PlaceholderKind::{Emoticon, Hashtag, Mention, RetweetMarker};

    fn check(input: &str, expected: &str) {
        assert_eq!(run(&BoundedSocialMedia, input), expected, "{input:?}");
    }

    /// Hashtags end where the word ends, in any script; OpenNLP's run to the
    /// next ASCII whitespace swallows the punctuation after them.
    #[test]
    fn hashtags_end_at_the_word() {
        check("#win!", " !");
        check("#caf\u{E9}, ok", " , ok");
        check("#\u{65E5}\u{672C}\u{3002}", " \u{3002}");
        check("#\u{0939}\u{093F}\u{0928}\u{094D}\u{0926}\u{0940} ok", "  ok");
        check("#100DaysOfCode", " ");
        check("\u{FF03}tag", " ");
        check("#1 and #2024", "#1 and #2024");
        check("it&#39;s", "it&#39;s");
        check("abc#def", "abc#def");
        check("#", "#");
        check("##", "##");
    }

    #[test]
    fn mentions_follow_word_and_handle_rules() {
        check("@ana hi", "  hi");
        check("hi @Jos\u{E9}_1.", "hi  .");
        check("mail a@b.com", "mail a@b.com");
        check("@ana@social.example says", "  says");
        check("@ana@social.example.", " .");
        check("@ana@x", "@ana@x");
        check("!@ana", "!@ana");
        check("(@ana)", "( )");
        check("\u{FF20}ana", " ");
    }

    #[test]
    fn retweet_markers_need_a_word_boundary() {
        check("RT @ana: hello", "  : hello");
        check("RT@ana hi", "   hi");
        check("rt rt go", " go");
        check("\u{E9}rt x", "\u{E9}rt x");
        check("art here", "art here");
        check("-rt x", "- x");
    }

    /// Emoticons do not match inside words; OpenNLP turns `expo` into
    /// `"e o"` and `xoxo` into two spaces.
    #[test]
    fn emoticons_stand_alone() {
        check("expo", "expo");
        check("taxi", "taxi");
        check("xoxo", "xoxo");
        check("max power", "max power");
        check(":people", ":people");
        check("hello :-) hello", "hello   hello");
        check("great:)", "great ");
        check("xD", " ");
        check("lol xD!", "lol  !");
        check(":-x", ":-x");
    }

    #[test]
    fn laughter_shrinks_as_in_opennlp() {
        check("hahaha", "haha");
        check("Bahahaha", "Bahaha");
        check("hAHa", "hAhA");
    }

    #[test]
    fn placeholders_cover_exactly_the_markup() {
        let (out, found) = placeholders(BoundedSocialMedia, "RT @ana: #win :-) yes");
        assert_eq!(out, "  :     yes");
        let found: Vec<(&str, _)> = found.iter().map(|(s, k)| (s.as_str(), *k)).collect();
        assert_eq!(found, [("RT ", RetweetMarker), ("@ana", Mention), ("#win", Hashtag), (":-)", Emoticon)]);
    }
}

/// A long run of laughter consonants is scanned once, not once per
/// position: a million of them finish well within a debug test run.
#[test]
fn long_consonant_runs_are_linear() {
    let text = "h".repeat(1_000_000) + "a";
    assert_eq!(run(&SocialMedia, &text), text);
    let text = "hahaha".to_string() + &"h".repeat(1_000_000);
    assert_eq!(run(&SocialMedia, &text).len(), 4 + 1_000_000);
}

/// The annotators in a pipeline: URLs on the source, social media markup
/// on the URL-free view, and every placeholder mapped back to the source.
#[test]
fn placeholder_annotators_in_a_pipeline() {
    use turbo_analysis::{Analyzer, Bound, Document, Key, View};
    use turbo_normalize::{BoundedSocialMedia, BoundedUrls};
    use turbo_normalize_layers::PlaceholderKind::Url;
    use turbo_normalize_layers::Placeholders;

    let analyzer = Analyzer::builder()
        .step(Bound::new(BoundedUrls).write(0, "no-urls").write(1, "urls"))
        .step(Bound::new(BoundedSocialMedia).read(0, "no-urls").write(0, "clean").write(1, "social"))
        .build()
        .unwrap();
    let src = "see https://x.example/a?b=1, #rust @ana";
    let mut docs = vec![Document::new(src)];
    assert!(analyzer.run(&mut docs)[0].is_ok());
    let doc = &docs[0];
    let clean = doc.get_key(&Key::<View>::named("clean")).unwrap();
    assert_eq!(clean.text(), "see  ,    ");
    let social = doc.get_key(&Key::<Placeholders>::named("social")).unwrap();
    let in_source: Vec<&str> =
        social.spans.iter().map(|(_, s)| &src[doc.to_source(social.spans.view(), s).range()]).collect();
    assert_eq!(in_source, ["#rust", "@ana"]);
    let urls = doc.get_key(&Key::<Placeholders>::named("urls")).unwrap();
    let (_, url) = urls.spans.iter().next().unwrap();
    assert_eq!(&src[url.range()], "https://x.example/a?b=1");
    assert_eq!(urls.kind, [Url]);
}
