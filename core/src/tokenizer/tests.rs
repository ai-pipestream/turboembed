use super::*;

/// The normalizer's steps, each on the whole text in turn.
fn steps(n: &Normalizer, text: &str) -> String {
    let mut s: String = if n.clean_text {
        text.chars()
            .filter(|&c| !(c == '\0' || c == '\u{fffd}' || (!matches!(c, '\t' | '\n' | '\r') && c.is_other())))
            .map(|c| if is_whitespace(c) { ' ' } else { c })
            .collect()
    } else {
        text.to_owned()
    };
    if n.split_cjk {
        let mut t = String::with_capacity(s.len());
        for c in s.chars() {
            if is_cjk(c) {
                t.push(' ');
                t.push(c);
                t.push(' ');
            } else {
                t.push(c);
            }
        }
        s = t;
    }
    if n.strip_accents {
        s = s.nfd().map(|(c, _)| c).filter(|c| !c.is_mark_nonspacing()).collect();
    }
    if n.lowercase {
        s = s.chars().flat_map(char::to_lowercase).collect();
    }
    s
}

/// A row as encode made it before rows were appended: the text's ids
/// on their own, cut, then put in the template.
fn row(tok: &Tokenizer, text: &str, opts: Encode) -> Result<Vec<i32>> {
    let specials = if opts.add_special_tokens { tok.specials_per_sequence() } else { 0 };
    if opts.max_tokens < specials {
        return Err(Error::new(INVALID_ARGUMENT, "no room"));
    }
    let budget = (opts.max_tokens - specials) as usize;
    let ids = |t: &str| {
        let mut v = Vec::new();
        tok.text_ids(t, &mut v);
        v
    };
    let mut body = match &tok.stat {
        None => ids(text),
        Some(st) => {
            let mut v = match opts.truncation {
                Truncation::Right => {
                    let mut v = ids(char_prefix(text, budget.saturating_mul(st.median_chars)));
                    v.truncate(budget);
                    v
                }
                Truncation::Left => {
                    let mut v = ids(text);
                    v.drain(..v.len().saturating_sub(budget));
                    v
                }
                Truncation::None => ids(text),
            };
            v.retain(|&id| id != tok.unk_id);
            v
        }
    };
    if body.len() > budget {
        match opts.truncation {
            Truncation::None => return Err(Error::new(CAPACITY, "over")),
            Truncation::Right => body.truncate(budget),
            Truncation::Left => {
                body.drain(..body.len() - budget);
            }
        }
    }
    if !opts.add_special_tokens {
        return Ok(body);
    }
    let mut row = Vec::new();
    for piece in &tok.template {
        match piece {
            Some(id) => row.push(*id),
            None => row.extend_from_slice(&body),
        }
    }
    Ok(row)
}

/// Rows appended one after another into one buffer are the rows made
/// alone, under every cut, and a row that fails leaves the buffer as
/// it was.
#[test]
fn appended_rows_are_the_rows_alone() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata");
    let texts: Vec<String> = std::fs::read_to_string(root.join("tokenizer-texts.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str::<Value>(l).unwrap()["text"].as_str().unwrap().to_owned())
        .chain(["".into(), "word ".repeat(300)])
        .collect();
    for bundle in ["tiny-bert-bundle", "tiny-static-bundle"] {
        let tok = Tokenizer::unchecked(&Bundle::open(&root.join(bundle)).unwrap()).unwrap();
        for add_special_tokens in [true, false] {
            for truncation in [Truncation::Right, Truncation::Left, Truncation::None] {
                for max_tokens in [0, 2, 3, 7, 64, 8192] {
                    let opts = Encode { add_special_tokens, truncation, max_tokens, prompt: PromptRole::None };
                    let mut out = vec![-7];
                    for t in &texts {
                        let before = out.clone();
                        let got = tok.encode_into(t, opts, &mut out);
                        let what = format!("{bundle} {t:?} {opts:?}");
                        match row(&tok, t, opts) {
                            Ok(r) => {
                                assert!(got.is_ok(), "{what}: {got:?}");
                                assert_eq!(out[before.len()..], r, "{what}");
                                assert_eq!(out[..before.len()], before, "{what}");
                            }
                            Err(_) => {
                                assert!(got.is_err(), "{what}");
                                assert_eq!(out, before, "{what}");
                            }
                        }
                    }
                }
            }
        }
    }
}

/// The table gives each character's categories as unicode_categories
/// does.
#[test]
fn the_category_table_is_the_unicode_categories() {
    for c in (0..=0x10ffffu32).filter(|&u| u < 0x10000 || u % 7 == 0).filter_map(char::from_u32) {
        assert_eq!(category(c, OTHER), c.is_other(), "{c:?}");
        assert_eq!(category(c, MARK_NONSPACING), c.is_mark_nonspacing(), "{c:?}");
        assert_eq!(category(c, PUNCTUATION), c.is_punctuation(), "{c:?}");
    }
}

/// WordPiece as it reads with no second table and no bound on a
/// piece: every piece looked up with its prefix in the one vocabulary.
fn pieces(vocab: &Vocab, prefix: &str, max_chars: usize, unk: i32, word: &str) -> Vec<i32> {
    if word.chars().count() > max_chars {
        return vec![unk];
    }
    let (mut out, mut start) = (Vec::new(), 0);
    while start < word.len() {
        let mut end = word.len();
        let found = loop {
            if start == end {
                break None;
            }
            let key = if start == 0 { word[..end].to_owned() } else { format!("{prefix}{}", &word[start..end]) };
            if let Some(&id) = vocab.get(&key) {
                break Some(id as i32);
            }
            end -= word[..end].chars().next_back().unwrap().len_utf8();
        };
        let Some(id) = found else { return vec![unk] };
        out.push(id);
        start = end;
    }
    out
}

/// The test bundle's WordPiece splits words as the plain algorithm
/// does: words in the vocabulary, words made of pieces, words with a
/// piece missing, and words at and over max_chars_per_word.
#[test]
fn word_pieces_are_the_plain_algorithm_s() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/tiny-bert-bundle");
    let tok = Tokenizer::unchecked(&Bundle::open(&dir).unwrap()).unwrap();
    let Kind::WordPiece { vocab, continuing_prefix, max_chars_per_word, .. } = &tok.kind else {
        panic!("the test bundle is WordPiece");
    };
    let mut words: Vec<String> = vocab.keys().cloned().collect();
    let parts: Vec<&str> = vocab.keys().map(|k| k.trim_start_matches(continuing_prefix.as_str())).take(500).collect();
    let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
    for _ in 0..5000 {
        let mut w = String::new();
        for _ in 0..(x >> 62) + 1 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            w.push_str(parts[(x % parts.len() as u64) as usize]);
        }
        words.push(w);
    }
    // The longest pieces inside a word, after a few first pieces.
    let mut rests: Vec<&str> = vocab.keys().filter_map(|k| k.strip_prefix(continuing_prefix.as_str())).collect();
    rests.sort_by_key(|r| std::cmp::Reverse(r.len()));
    for r in rests.iter().take(30) {
        words.extend(["a", "the", "un", "x", "q"].map(|first| format!("{first}{r}")));
    }
    for n in [*max_chars_per_word - 1, *max_chars_per_word, *max_chars_per_word + 1] {
        words.push("a".repeat(n));
        words.push("é".repeat(n));
    }
    words.extend(["東京", "😀x", "qqqzzz", "ab\u{301}c"].map(String::from));
    for w in words.iter().filter(|w| !w.is_empty()) {
        let mut got = Vec::new();
        tok.word_pieces(w, |_, _| NOWHERE, &mut got);
        assert_eq!(got, pieces(vocab, continuing_prefix, *max_chars_per_word, tok.unk_id, w), "{w:?}");
    }
}

/// normalize() gives what its steps give, each on the whole text, for every
/// combination of options: ASCII control and whitespace characters,
/// combining marks after ASCII letters and in runs that NFD reorders,
/// CJK, characters whose lowercase is longer, and random mixes of them.
#[test]
fn normalizing_by_runs_is_normalizing_the_whole_text() {
    let pool: Vec<char> = "aZ09 .,!\t\n\r\x00\x01\x0b\x0c\x1f\x7f\u{85}\u{a0}\u{3000}\u{fffd}\u{200b}\
         \u{301}\u{300}\u{327}\u{316}\u{93f}\u{94d}\u{903}\u{20dd}éÅñçİΣσςß東京한국\u{1100}\u{1161}ＡＢ１😀\u{e000}\u{feff}"
        .chars()
        .collect();
    let mut texts: Vec<String> = [
        "",
        "plain ascii, UPPER and lower!",
        "e\u{301}\u{316}\u{327}x",
        "a\u{316}\u{301}\u{903}\u{327}b",
        "Ｃａｆé 東京 İstanbul ΣΑΣ\u{85}end",
        "\x00\x7f\t\x0b\x0c\r\n",
    ]
    .iter()
    .map(|t| t.to_string())
    .collect();
    let mut x: u64 = 0x2545_f491_4f6c_dd1d;
    for _ in 0..2000 {
        let len = (x >> 59) as usize + 1;
        let mut t = String::new();
        for _ in 0..len {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            t.push(pool[(x % pool.len() as u64) as usize]);
        }
        texts.push(t);
    }
    for bits in 0..16u8 {
        let n = Normalizer {
            clean_text: bits & 1 != 0,
            split_cjk: bits & 2 != 0,
            strip_accents: bits & 4 != 0,
            lowercase: bits & 8 != 0,
            unicode_form: crate::manifest::UnicodeForm::None,
        };
        let (mut s, mut sources) = (String::new(), Vec::new());
        for t in &texts {
            let want = steps(&n, t);
            assert_eq!(normalized(&n, &ascii_map(&n), t), want, "{t:?} with {n:?}");
            // The aligned normalizer writes the same text, and each
            // byte's source is a whole character of the text.
            normalize_aligned(&n, &ascii_map(&n), t, 3, &mut s, &mut sources);
            assert_eq!(s, want, "aligned: {t:?} with {n:?}");
            assert_eq!(sources.len(), s.len());
            for &[a, b] in &sources {
                let (a, b) = (a as usize - 3, b as usize - 3);
                assert_eq!(t[a..b].chars().count(), 1, "{t:?} with {n:?}: {a}..{b}");
            }
        }
    }
}

#[test]
fn pre_words_part_punctuation_and_ideographs() {
    let t = "(Paris) Merkel's\u{3000}\u{6771}\u{4eac}x \u{ff0c}y";
    let got: Vec<&str> = pre_words(t).into_iter().map(|(a, b)| &t[a..b]).collect();
    assert_eq!(got, ["(", "Paris", ")", "Merkel", "'", "s", "\u{6771}", "\u{4eac}", "x", "\u{ff0c}", "y"]);
}
