//! `JwtHeaderTest`: a JWT needs a base64url JSON object header with one
//! string `alg` member, and three well formed segments.

mod support;

use support::*;
use turbo_pii::rewrite::mask_all;
use turbo_pii::{Extractor, Pack, PiiDetector, PiiKind};

const HEADER: &str = "{\"alg\":\"HS256\"}";
const PAYLOAD: &str = "eyJzdWIiOiJ0ZXN0In0";
const SIGNATURE: &str = "dGVzdA";

fn jwt() -> PiiDetector {
    only(Extractor::Secrets, &[PiiKind::Jwt])
}

fn token(json: &str) -> String {
    token_of(json.as_bytes())
}

fn token_of(header: &[u8]) -> String {
    format!("{}.{PAYLOAD}.{SIGNATURE}", base64url(header))
}

#[test]
fn valid_headers() {
    for json in [
        HEADER,
        "{ \"alg\":\"HS256\"}",
        " \r\n\t{\"alg\":\"HS256\"} \r\n\t",
        "{\"\\u0061lg\":\"HS256\"}",
        "{\"a\\u006cg\":\"HS256\"}",
        "{\"al\\u0067\":\"HS256\"}",
        "{\"alg\":\"H\\u0053256\"}",
        "{\"kid\":\"é😀\",\"alg\":\"HS256\"}",
        "{\"other\":{\"alg\":null},\"alg\":\"RS256\"}",
        "{\"alg\":\"RS256\",\"meta\":[{},[],true,false,null,0,-1,1.5,1e1000000]}",
        "{\"meta\":\"\\\"alg\\\":null\",\"alg\":\"HS256\"}",
        "{\"kid\":\"\\b\\f\\n\\r\\t\\/\\\\\\\"\",\"alg\":\"HS256\"}",
    ] {
        let value = token(json);
        let prefix = "😀 Bearer ";
        let start = prefix.len() as u32;
        assert_found(&jwt(), &format!("{prefix}{value};"), &[(start, start + value.len() as u32, "jwt", &value)]);
    }
}

#[test]
fn a_late_algorithm() {
    for length in [0, 32, 64, 65, 66, 67, 88, 128, 4096, 65536] {
        let value = token(&format!("{{\"kid\":\"{}\",\"alg\":\"HS256\"}}", "x".repeat(length)));
        assert_found(&jwt(), &value, &[(0, value.len() as u32, "jwt", &value)]);
    }
}

#[test]
fn leading_characters() {
    for c in 0u8..128 {
        let whitespace = matches!(c, b' ' | b'\t' | b'\r' | b'\n');
        let json = format!("{}{HEADER}", c as char);
        assert_eq!(rows(&jwt(), &token(&json)).len(), usize::from(whitespace), "{c}");
    }
}

#[test]
fn invalid_headers() {
    for json in [
        "{}",
        "[]",
        "null",
        "true",
        "123",
        "{\"meta\":{\"alg\":\"HS256\"}}",
        "{\"alg\":true}",
        "{\"alg\":false}",
        "{\"alg\":null}",
        "{\"alg\":123}",
        "{\"alg\":[]}",
        "{\"alg\":{}}",
        "{\"alg\":\"\"}",
        "{\"alg\":\"é\"}",
        "{\"alg\":\"\\u00e9\"}",
        "{\"alg\":\"HS256\"",
        "{\"alg\":\"HS256}",
        "{\"alg\":\"HS256\",}",
        "{\"alg\":\"HS256\"}garbage",
        "{\"alg\":\"HS256\"}{}",
        "{\"alg\":\"HS256\",\"alg\":\"RS256\"}",
        "{\"alg\":\"HS256\",\"\\u0061lg\":\"HS256\"}",
        "{\"alg\":\"HS256\" \"kid\":0}",
        "{\"alg\" \"HS256\"}",
        "{\"alg\":\"HS256\",\"x\":[1,]}",
        "{\"alg\":\"HS256\",\"x\":{\"a\":1,}}",
        "{\"alg\":\"HS256\",\"x\":[1 2]}",
        "{\"alg\":\"HS256\",\"x\":{a:1}}",
        "{\"alg\":\"HS256\",\"x\":\"\\x20\"}",
        "{\"alg\":\"HS256\",\"x\":\"\\u12\"}",
        "{\"alg\":\"HS256\",\"x\":\"\\u12g0\"}",
        "{\"alg\":\"HS256\",\"x\":NaN}",
        "{\"alg\":\"HS256\",\"x\":Infinity}",
        "{\"alg\":\"HS256\",\"x\":01}",
        "{\"alg\":\"HS256\",\"x\":-01}",
        "{\"alg\":\"HS256\",\"x\":+1}",
        "{\"alg\":\"HS256\",\"x\":1.}",
        "{\"alg\":\"HS256\",\"x\":.1}",
        "{\"alg\":\"HS256\",\"x\":1e}",
        "{\"alg\":\"HS256\",\"x\":1e+}",
        "{\"alg\":\"HS256\",\"x\":-}",
        "{\"alg\":\"HS256\",\"x\":TRUE}",
        "{\"alg\":\"HS256\",\"x\":nul}",
        "{\"alg\":\"HS256\",\"x\":falsee}",
    ] {
        assert_none(&jwt(), &token(json));
    }
    for control in 0u8..32 {
        assert_none(&jwt(), &token(&format!("{{\"alg\":\"HS256\",\"x\":\"{}\"}}", control as char)));
    }
}

#[test]
fn deep_nesting() {
    for depth in [1, 32, 256, 4096, 32768] {
        let value = token(&format!("{{\"x\":{}0{},\"alg\":\"HS256\"}}", "[".repeat(depth), "]".repeat(depth)));
        assert_eq!(rows(&jwt(), &value).len(), 1, "{depth}");
    }
    for depth in [1, 32, 1024, 8192] {
        let nested = format!("{}{HEADER}{}", "{\"x\":".repeat(depth), "}".repeat(depth));
        assert_none(&jwt(), &token(&nested));
        assert_eq!(rows(&jwt(), &token(&format!("{{\"x\":{nested},\"alg\":\"HS256\"}}"))).len(), 1, "{depth}");
    }
}

#[test]
fn invalid_utf8_in_the_header() {
    for invalid in [&[0xffu8][..], &[0x80], &[0xc0, 0xaf], &[0xc2], &[0xed, 0xa0, 0x80], &[0xf4, 0x90, 0x80, 0x80]] {
        let mut header = b"{\"alg\":\"HS256\",\"x\":\"".to_vec();
        header.extend_from_slice(invalid);
        header.extend_from_slice(b"\"}");
        assert_none(&jwt(), &token_of(&header));
    }
}

#[test]
fn segments() {
    let header = base64url(HEADER.as_bytes());
    for length in [1, 2, 3, 4, 5, 32, 64, 256] {
        assert_eq!(rows(&jwt(), &format!("{header}.e30.{}", base64url(&vec![0; length]))).len(), 1);
    }
    for segment in 0..3 {
        for bad in ["", "A", "AAAAA", "AB", "AAB", "AAAA=", "AAAA+", "AAAA/"] {
            let mut parts = [header.as_str(), PAYLOAD, SIGNATURE];
            parts[segment] = bad;
            assert_none(&jwt(), &parts.join("."));
        }
    }
    assert_none(&jwt(), &format!("{header}A.{PAYLOAD}.{SIGNATURE}"));
    assert_none(&jwt(), &format!("{header}.{PAYLOAD}.{SIGNATURE}="));
}

#[test]
fn continuations_and_punctuation() {
    let value = token(HEADER);
    for continuation in ["a", "eyJx", "é", "𐐀", "_", "-", ".part"] {
        assert_none(&jwt(), &format!("{value}.{continuation}"));
        assert_none(&jwt(), &format!("{continuation}.{value}"));
    }
    for suffix in ["", ".", "...", ". ", ")", "]", ",", ";", "?", "!"] {
        assert_found(&jwt(), &format!("{value}{suffix}"), &[(0, value.len() as u32, "jwt", &value)]);
    }
}

#[test]
fn secrets_pack_masks_a_spaced_header() {
    let value = "eyAiYWxnIjogIkhTMjU2IiB9.e30.dGVzdA";
    let text = format!("Bearer {value}");
    let d = pack(Pack::Secrets);
    assert_eq!(mask_all(&text, &layer(&d, &text), '*').unwrap(), format!("Bearer {}", "*".repeat(value.len())));
    assert_eq!(normalized(&d, &text), [value]);
}

#[test]
fn headers_on_every_thread() {
    let d = jwt();
    std::thread::scope(|s| {
        for worker in 0..8 {
            let d = &d;
            s.spawn(move || {
                let value = token(&format!("{{\"kid\":\"{}\",\"alg\":\"HS256\"}}", "x".repeat(100 + worker)));
                for _ in 0..32 {
                    assert_eq!(rows(d, &value).len(), 1);
                    assert_none(d, &token("{\"alg\":null}"));
                }
            });
        }
    });
}
