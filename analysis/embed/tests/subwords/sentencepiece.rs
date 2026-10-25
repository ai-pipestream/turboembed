//! The SentencePiece pieces against what Google's sentencepiece library
//! gave for the same texts (testdata/sentencepiece), id for id and span
//! for span.

use super::*;

/// What the library gave for one text (testdata/sentencepiece).
struct Golden {
    text: String,
    norm_to_orig: Vec<u32>,
    ids: Vec<i32>,
    offsets: Vec<[u32; 2]>,
    decoded: String,
}

fn goldens(model: &str) -> Vec<Golden> {
    let file = std::fs::read_to_string(sentencepiece_dir().join(format!("{model}.goldens.jsonl"))).unwrap();
    let nums = |v: &serde_json::Value| -> Vec<u32> {
        v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
    };
    let mut out = Vec::new();
    for line in file.lines().skip(1) {
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        if v.get("long").is_some() {
            continue;
        }
        out.push(Golden {
            text: v["text"].as_str().unwrap().to_owned(),
            norm_to_orig: nums(&v["norm_to_orig"]),
            ids: nums(&v["ids"]).into_iter().map(|x| x as i32).collect(),
            offsets: v["offsets"].as_array().unwrap().iter().map(|o| [nums(o)[0], nums(o)[1]]).collect(),
            decoded: v["decoded"].as_str().unwrap().to_owned(),
        });
    }
    out
}

/// The span the layer gives a piece the library put at `[b, e)`, by the
/// rules every kind's spans follow (docs/tokenizer.md): a piece the
/// library leaves empty (a byte piece before the last of its character,
/// or a piece written for part of one input character) runs to where
/// the input it came from ends, the next source in norm_to_orig; then
/// whitespace at either end is left out unless that is all there is.
fn layer_span(g: &Golden, [b, e]: [u32; 2]) -> [u32; 2] {
    let e = if e > b { e } else { g.norm_to_orig.iter().copied().find(|&o| o > b).unwrap_or(b) };
    trimmed(&g.text, [b, e])
}

/// The layer holds the library's ids, and its spans are the library's
/// offsets by the rules of layer_span, on every text of the goldens; the
/// ids decode to the library's text.
#[test]
fn sentencepiece_pieces_match_the_library() {
    for (model, (_dir, a)) in [("t5", t5()), ("bpe-bytes", bpe_bytes()), ("unigram-bytes", unigram_bytes())] {
        let goldens = goldens(model);
        let mut docs: Vec<Document> = goldens.iter().map(|g| Document::new(g.text.as_str())).collect();
        let decoded: Vec<String> = goldens.iter().map(|g| a.tokenizer().decode(&g.ids, false).unwrap()).collect();
        for s in run(a, &mut docs, false) {
            s.unwrap();
        }
        let mut differ = 0;
        for ((g, doc), decoded) in goldens.iter().zip(&docs).zip(decoded) {
            let want: Vec<(i32, [u32; 2])> =
                g.ids.iter().zip(&g.offsets).map(|(&id, &o)| (id, layer_span(g, o))).collect();
            assert_eq!(pairs(doc.get::<Subwords>().unwrap()), want, "{model}: {:?}", g.text);
            assert_eq!(decoded, g.decoded, "{model}: decode of {:?}", g.text);
            differ += usize::from(want.iter().zip(&g.offsets).any(|((_, s), o)| s != o));
        }
        // The rules change the library's offsets on most texts: a piece
        // after a space starts with it there.
        assert!(differ > goldens.len() / 2, "{model}: {differ} of {} texts", goldens.len());
    }
}

/// A character a byte-fallback model has no piece for is its bytes'
/// pieces, every one spanning the whole character, where the library
/// gives the last its offsets and the others an empty one at its start.
#[test]
fn byte_fallback_pieces_of_one_character_share_its_span() {
    let (_dir, a) = bpe_bytes();
    let text = "a \u{13000} b";
    let glyph = text.find('\u{13000}').unwrap() as u32;
    let mut docs = vec![Document::new(text)];
    run(a, &mut docs, false).remove(0).unwrap();
    let sub = docs[0].get::<Subwords>().unwrap();
    let on_glyph: Vec<Span> = sub.spans.iter().map(|(_, s)| s).filter(|s| s.start == glyph).collect();
    assert_eq!(on_glyph.len(), 4, "{:?}", sub.spans);
    assert!(on_glyph.iter().all(|s| s.end == glyph + 4), "{:?}", sub.spans);
}

/// The library matches no special token in the text: T5's `</s>` written
/// there is its characters, not the end-of-sequence id.
#[test]
fn a_special_token_written_in_the_text_is_its_characters() {
    let (_dir, a) = t5();
    let ids = a.tokenizer().encode("a </s> b", bare()).unwrap();
    assert!(!ids.contains(&1), "{ids:?}");
    let g = goldens("t5");
    let line = g.iter().find(|g| g.text.contains("</s>")).unwrap();
    assert_eq!(a.tokenizer().encode(&line.text, bare()).unwrap(), line.ids);
}
