use super::*;

/// MiniLM's tokenizer with its [PAD] matched as Model2Vec's code models
/// add it: in the normalized text, as a whole word, taking the whitespace
/// around it.
#[test]
fn a_normalized_special_token_matches_upstream() {
    let mut json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(upstream_tokenizer_json()).unwrap()).unwrap();
    let pad = json["added_tokens"].as_array_mut().unwrap().iter_mut().find(|a| a["content"] == "[PAD]").unwrap();
    for flag in ["normalized", "single_word", "lstrip", "rstrip"] {
        pad[flag] = json!(true);
    }
    let path = std::env::temp_dir().join(format!("turbo-normalized-pad-{}.json", std::process::id()));
    std::fs::write(&path, serde_json::to_vec(&json).unwrap()).unwrap();
    let mut m = manifest();
    let specials = m["tokenizer"]["special_tokens"].as_array_mut().unwrap();
    let pad = specials.iter_mut().find(|s| s["content"] == "[PAD]").unwrap();
    for flag in ["normalized", "single_word", "lstrip", "rstrip"] {
        pad[flag] = json!(true);
    }
    let f = Fixture::with_tokenizer("normalized-pad", m, &path);
    let tok = f.open().expect("tokenizer loads");
    let up = upstream_at(&path);
    let mut all = texts();
    all.extend(
        [
            "x [PAD] y",
            "x [pad] y",
            "[PAD]",
            "[Pad][pAd]",
            "a[pad]b",
            "a [pad]b",
            "[pad]. and [pad],",
            "\u{e9}[pad] \u{e9} [pad]_ _[pad]",
            "[UNK] [pad] [CLS]",
        ]
        .map(str::to_owned),
    );
    let core = core_tokenizer(&f);
    let mut whole = upstream_at(&path);
    whole.with_truncation(None).unwrap();
    for text in all {
        assert_eq!(tok.row(&text, None).unwrap(), upstream_ids(&up, &text), "text {text:?}");
        let (ours, theirs) = both_spans(&core, &whole, &text);
        assert_eq!(ours, theirs, "spans of {text:?}");
    }
    std::fs::remove_file(path).unwrap();
}

/// The core's tokenizer on the bundle `f` describes, through the Rust
/// interface, which has the spans.
pub fn core_tokenizer(f: &Fixture) -> turbo::tokenizer::Tokenizer {
    f.write();
    turbo::tokenizer::Tokenizer::load(&turbo::bundle::Bundle::open(&f.dir).unwrap()).unwrap()
}

fn whole_text_opts() -> turbo::tokenizer::Encode {
    turbo::tokenizer::Encode {
        add_special_tokens: true,
        truncation: turbo::manifest::Truncation::None,
        max_tokens: u32::MAX,
        prompt: turbo::manifest::PromptRole::None,
    }
}

/// `span` with the whitespace at either end left out, unless that is all.
fn trimmed(text: &str, span: [u32; 2]) -> [u32; 2] {
    // Upstream's offset can be inside a character; it is compared as it is.
    let Some(piece) = text.get(span[0] as usize..span[1] as usize) else { return span };
    let inner = piece.trim();
    if inner.is_empty() {
        return span;
    }
    let start = span[0] + (inner.as_ptr() as usize - piece.as_ptr() as usize) as u32;
    [start, start + inner.len() as u32]
}

/// Each token's id and span.
pub type Tokens = Vec<(i32, [u32; 2])>;

/// Each token's ids and span, the core's and upstream's, for `text`:
/// upstream's spans trimmed of whitespace as the core's are.
pub fn both_spans(tok: &turbo::tokenizer::Tokenizer, up: &tokenizers::Tokenizer, text: &str) -> (Tokens, Tokens) {
    let (mut ids, mut spans) = (Vec::new(), Vec::new());
    tok.encode_spans(text, whole_text_opts(), &mut ids, &mut spans).unwrap();
    let e = up.encode(text, true).unwrap();
    let theirs = e
        .get_ids()
        .iter()
        .zip(e.get_offsets())
        .map(|(&id, &(a, b))| (id as i32, trimmed(text, [a as u32, b as u32])))
        .collect();
    (ids.into_iter().zip(spans).collect(), theirs)
}

/// Every token's span is inside the text and on character boundaries,
/// and the spans of the text's tokens start in order.
pub fn check_well_formed(text: &str, ours: &[(i32, [u32; 2])]) {
    let mut last = 0;
    for &(id, [a, b]) in ours {
        assert!(a <= b && b as usize <= text.len(), "{text:?}: token {id} span {a}..{b}");
        assert!(text.is_char_boundary(a as usize) && text.is_char_boundary(b as usize), "{text:?}: {a}..{b}");
        if [a, b] != turbo::tokenizer::NOWHERE {
            assert!(a >= last, "{text:?}: token {id} at {a} starts before {last}");
            last = a;
        }
    }
}

/// WordPiece's spans are upstream's offsets, trimmed of whitespace, on
/// every text, the special tokens of the template at [0, 0].
#[test]
fn spans_match_upstream_offsets() {
    let f = Fixture::standard("spans");
    let tok = core_tokenizer(&f);
    let mut up = upstream();
    up.with_truncation(None).unwrap();
    for text in texts() {
        let (ours, theirs) = both_spans(&tok, &up, &text);
        assert_eq!(ours, theirs, "text {text:?}");
        check_well_formed(&text, &ours);
    }
}

/// A row cut to fewer tokens keeps the spans of the tokens it keeps; a
/// row that fails leaves the ids and spans as they were.
#[test]
fn spans_follow_truncation() {
    let f = Fixture::standard("spans-truncation");
    let tok = core_tokenizer(&f);
    let text = long_text();
    let (mut ids, mut spans) = (Vec::new(), Vec::new());
    tok.encode_spans(&text, whole_text_opts(), &mut ids, &mut spans).unwrap();
    let n = ids.len();
    let body = &spans[1..n - 1];
    for (truncation, max) in [(turbo::manifest::Truncation::Right, 32), (turbo::manifest::Truncation::Left, 40)] {
        let opts = turbo::tokenizer::Encode { truncation, max_tokens: max, ..whole_text_opts() };
        let (mut i2, mut s2) = (vec![7], vec![[1, 2]]);
        tok.encode_spans(&text, opts, &mut i2, &mut s2).unwrap();
        assert_eq!(i2[1..], tok.encode(&text, opts).unwrap()[..], "{truncation:?}");
        assert_eq!(s2[0], [1, 2], "what was there stays");
        let kept = &s2[2..s2.len() - 1];
        let want = match truncation {
            turbo::manifest::Truncation::Left => &body[body.len() - kept.len()..],
            _ => &body[..kept.len()],
        };
        assert_eq!(kept, want, "{truncation:?}");
        assert_eq!((s2[1], s2[s2.len() - 1]), (turbo::tokenizer::NOWHERE, turbo::tokenizer::NOWHERE));
    }
    let none =
        turbo::tokenizer::Encode { truncation: turbo::manifest::Truncation::None, max_tokens: 32, ..whole_text_opts() };
    let (mut i3, mut s3) = (vec![7], vec![[1, 2]]);
    assert!(tok.encode_spans(&text, none, &mut i3, &mut s3).is_err());
    assert_eq!((i3, s3), (vec![7], vec![[1, 2]]));
    let (mut i4, mut s4) = (vec![7], Vec::new());
    assert!(tok.encode_spans(&text, whole_text_opts(), &mut i4, &mut s4).is_err(), "a span for each id");
}

/// With a prompt's prefix in front, spans are into the caller's text and
/// the prefix's tokens come from nowhere.
#[test]
fn spans_are_into_the_text_without_the_prefix() {
    let mut m = manifest();
    m["embed"]["prefix_query"] = json!("query: ");
    let f = Fixture::new("spans-prompt", m);
    let tok = core_tokenizer(&f);
    let mut up = upstream();
    up.with_truncation(None).unwrap();
    for text in ["reset a password", "", "  Ünïcödé text", "[SEP] at the start"] {
        let opts = turbo::tokenizer::Encode { prompt: turbo::manifest::PromptRole::Query, ..whole_text_opts() };
        let (mut ids, mut spans) = (Vec::new(), Vec::new());
        tok.encode_spans(text, opts, &mut ids, &mut spans).unwrap();
        let prompted = format!("query: {text}");
        let (_, theirs) = both_spans(&tok, &up, &prompted);
        let want: Vec<(i32, [u32; 2])> =
            theirs.into_iter().map(|(id, [a, b])| (id, if b <= 7 { [0, 0] } else { [a - 7, b - 7] })).collect();
        assert_eq!(ids.into_iter().zip(spans).collect::<Vec<_>>(), want, "{text:?}");
    }
}

/// The core's spans are upstream's offsets, trimmed of whitespace,
/// except where upstream's offset ends inside a grapheme cluster that the
/// character map replaced as a whole: the core's span covers the cluster.
/// The number of such spans.
pub fn spans_near_upstream(text: &str, ours: &[(i32, [u32; 2])], theirs: &[(i32, [u32; 2])]) -> usize {
    use unicode_segmentation::UnicodeSegmentation;
    assert_eq!(ours.len(), theirs.len(), "{text:?}");
    let bounds: Vec<usize> = text.grapheme_indices(true).map(|(i, _)| i).chain([text.len()]).collect();
    let inside = |x: u32| !bounds.contains(&(x as usize));
    let mut widened = 0;
    for (o, t) in ours.iter().zip(theirs) {
        assert_eq!(o.0, t.0, "{text:?}");
        if o.1 != t.1 {
            let (a, b) = (o.1, t.1);
            assert!(a[0] <= b[0] && a[1] >= b[1], "{text:?}: ours {a:?} theirs {b:?}");
            assert!(inside(b[1]) && !inside(a[1]), "{text:?}: upstream's {b:?} ends on a cluster");
            widened += 1;
        }
    }
    check_well_formed(text, ours);
    widened
}

#[test]
fn unigram_spans_match_upstream_offsets() {
    let path = testdata().join("bge-m3/tokenizer.json");
    let f = Fixture::with_tokenizer("m3-spans", unigram::manifest_m3(), &path);
    let tok = core_tokenizer(&f);
    let mut up = upstream_at(&path);
    up.with_truncation(None).unwrap();
    let mut widened = 0;
    for text in unigram::texts() {
        let (ours, theirs) = both_spans(&tok, &up, &text);
        widened += spans_near_upstream(&text, &ours, &theirs);
    }
    assert!(widened > 0, "the texts have a cluster the map composes");
}

/// A static model's spans are its tokens' spans with the unknown token
/// dropped, as its ids are.
#[test]
fn static_spans_drop_the_unknown_token() {
    let bundle = turbo::bundle::Bundle::open(&testdata().join("tiny-static-bundle")).unwrap();
    let tok = turbo::tokenizer::Tokenizer::unchecked(&bundle).unwrap();
    for text in texts() {
        let (mut ids, mut spans) = (Vec::new(), Vec::new());
        tok.encode_spans(&text, whole_text_opts(), &mut ids, &mut spans).unwrap();
        assert_eq!(ids, tok.encode(&text, whole_text_opts()).unwrap(), "{text:?}");
        assert!(!ids.contains(&tok.unk_id));
        check_well_formed(&text, &ids.iter().copied().zip(spans.iter().copied()).collect::<Vec<_>>());
    }
}

/// The core's decode of `text`'s ids is upstream's, with the special
/// tokens and without.
pub fn decode_matches_upstream(tok: &turbo::tokenizer::Tokenizer, up: &tokenizers::Tokenizer, text: &str) {
    let ids = tok.encode(text, whole_text_opts()).unwrap();
    let raw: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
    for skip in [true, false] {
        assert_eq!(tok.decode(&ids, skip).unwrap(), up.decode(&raw, skip).unwrap(), "{text:?} skip {skip}");
    }
}

#[test]
fn decode_matches_upstream_wordpiece() {
    let f = Fixture::standard("decode");
    let tok = core_tokenizer(&f);
    let up = upstream();
    for text in texts() {
        decode_matches_upstream(&tok, &up, &text);
    }
}

#[test]
fn decode_matches_upstream_unigram() {
    let path = testdata().join("bge-m3/tokenizer.json");
    let f = Fixture::with_tokenizer("m3-decode", unigram::manifest_m3(), &path);
    let tok = core_tokenizer(&f);
    let up = upstream_at(&path);
    for text in unigram::texts() {
        decode_matches_upstream(&tok, &up, &text);
    }
}

#[test]
fn decode_refuses_an_id_outside_the_vocabulary() {
    let f = Fixture::standard("decode-bad");
    let tok = core_tokenizer(&f);
    for id in [-1, 30522] {
        let e = tok.decode(&[101, id], false).unwrap_err();
        assert!(e.code == turbo::status::INVALID_ARGUMENT && e.message.contains("ids[1]"), "{e:?}");
    }
    assert_eq!(tok.decode(&[], false).unwrap(), "");
}

/// Every id's piece is upstream's token for it, and there is none past
/// the vocabulary.
#[test]
fn pieces_are_upstream_tokens() {
    let path = testdata().join("bge-m3/tokenizer.json");
    let f = Fixture::with_tokenizer("m3-pieces", unigram::manifest_m3(), &path);
    let tok = core_tokenizer(&f);
    let up = upstream_at(&path);
    let n = tok.vocab_size() as i32;
    for id in 0..n {
        assert_eq!(tok.piece(id), up.id_to_token(id as u32).as_deref(), "id {id}");
    }
    assert_eq!(tok.piece(6), Some("\u{2581}"));
    assert_eq!(tok.piece(-1), None);
    assert_eq!(tok.piece(n), None);
}
