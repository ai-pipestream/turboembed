//! `SecretsPiiExtractorTest` and `SecretsBoundaryTest`: AWS access keys,
//! GitHub tokens, JWTs and URL credentials. The Java counts the reads of
//! a `CharSequence` to show the scans are linear; here the same inputs
//! are scanned at full size.

mod support;

use support::*;
use turbo_pii::rewrite::mask_all;
use turbo_pii::{Extractor, Pack, PiiDetector, PiiKind};

const HS256: &str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9";
const RS256: &str = "eyJhbGciOiJSUzI1NiIsImtpZCI6ImFiYzEyMyJ9";
const NO_ALGORITHM: &str = "eyJ0eXAiOiJKV1QifQ";
const NOTALG: &str = "eyJub3RhbGciOiJIUzI1NiJ9";
const ALGORITHM: &str = "eyJhbGdvcml0aG0iOiJIUzI1NiJ9";
const SPACED: &str = "eyJhbGciIDogIkhTMjU2In0";
const PAYLOAD: &str = "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ";
const SIGNATURE: &str = "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c";
const JWT: &str = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ0ZXN0In0.dGVzdA";
const CREDENTIAL: &str = "user:pw%40:demo";
const AUTHORITY_END: &str = "@example.invalid";

fn secrets() -> PiiDetector {
    pack(Pack::Secrets)
}

fn urls() -> PiiDetector {
    only(Extractor::Secrets, &[PiiKind::UrlCredential])
}

fn has(text: &str, kind: &str) -> bool {
    kinds(&secrets(), text).contains(&kind)
}

#[test]
fn prefix_heavy_near_misses() {
    for text in ["eyJ_".repeat(1024), "ghp_".repeat(1024), "s3://".repeat(1024), "1http://".repeat(1024)] {
        assert_none(&secrets(), &text);
    }
    assert_none(&secrets(), &"https://u:p%g0@example.invalid ".repeat(1024));
    for prefix in ["e.e.", "I.I.", "C.C.", "D.D."] {
        assert_none(&secrets(), &prefix.repeat(4096));
    }
    for group in ["ghs_.", "ghs_-", "ghs_a."] {
        assert_none(&secrets(), &format!("{}é", group.repeat(4096)));
    }
}

#[test]
fn long_values() {
    for length in [64, 1024, 65536] {
        assert_eq!(normalized(&secrets(), &format!("{}1://u:pw@example.invalid", "a".repeat(length))), ["u:pw"]);
    }
    for length in [64, 4096, 65536] {
        let json = format!("{{\"field\":\"{}\",\"alg\":\"HS256\"}}", "x".repeat(length));
        let token = format!("{}.{PAYLOAD}.{SIGNATURE}", base64url(json.as_bytes()));
        assert_eq!(kinds(&secrets(), &token), ["jwt"]);
    }
    for length in [128, 4096, 65536] {
        let value = format!("ghs_{}", "a-_b.c".repeat(length));
        assert_eq!(normalized(&secrets(), &value), [value.as_str()]);
    }
}

#[test]
fn aws_access_keys() {
    for text in [
        "AKIAIOSFODNN7EXAMPLE",
        "ASIAIOSFODNN7EXAMPLE",
        "AKIA1234567890ABCDEF",
        "ASIAZZZZZZZZZZZZZZZZ",
        "AKIA0000000000000000",
    ] {
        assert_found(&secrets(), text, &[(0, 20, "aws-access-key", text)]);
    }
    for text in [
        "AKIAIOSFODNN7EXAMPL",
        "AKIAIOSFODNN7EXAMPLEX",
        "akiaiosfodnn7example",
        "AKIAiosfodnn7EXAMPLE",
        "AIDAIOSFODNN7EXAMPLE",
        "AROAIOSFODNN7EXAMPLE",
        "ANPAIOSFODNN7EXAMPLE",
        "APKAIOSFODNN7EXAMPLE",
        "XAKIAIOSFODNN7EXAMPLE",
        "AKIA-IOSFODNN7EXAMPL",
        "AKIA_IOSFODNN7EXAMPL",
    ] {
        assert!(!has(text, "aws-access-key"), "{text}");
    }
    assert_eq!(covered(&secrets(), "Rotate AKIAIOSFODNN7EXAMPLE right away."), ["AKIAIOSFODNN7EXAMPLE"]);
}

#[test]
fn github_tokens() {
    for text in [
        "ghp_1234567890abcdefghijklmnopqrstuvwxyz",
        "gho_1234567890abcdefghijklmnopqrstuvwxyz",
        "ghu_1234567890abcdefghijklmnopqrstuvwxyz",
        "ghs_1234567890abcdefghijklmnopqrstuvwxyz",
        "ghr_1234567890abcdefghijklmnopqrstuvwxyz",
        "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    ] {
        assert_found(&secrets(), text, &[(0, text.len() as u32, "github-token", text)]);
    }
    let token = "github_pat_11ABCDEFG0abcdefghijkl_0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVW";
    assert_eq!(token.len(), 93);
    assert_found(&secrets(), &format!("token: {token}"), &[(7, 100, "github-token", token)]);
    for (prefix, minimum) in [("ghp_", 36), ("gho_", 36), ("ghu_", 36), ("ghr_", 36), ("github_pat_", 82)] {
        let value = format!("{prefix}{}_{}", "a".repeat(minimum), "B".repeat(24));
        assert_eq!(normalized(&secrets(), &value), [value.as_str()]);
    }
    for prefix in ["ghp_", "gho_", "ghu_", "ghr_", "github_pat_"] {
        let accepted = format!("{prefix}{}", "a".repeat(255 - prefix.len()));
        assert_eq!(normalized(&secrets(), &accepted), [accepted.as_str()]);
        assert_none(&secrets(), &format!("{accepted}a"));
    }
    for text in [
        "ghp_1234567890abcdefghijklmnopqrstuvwxy",
        "ghx_1234567890abcdefghijklmnopqrstuvwxyz",
        "GHP_1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "ghp1234567890abcdefghijklmnopqrstuvwxyz",
        "ghp_1234567890abcdefghijklmnopqrstuvwx-z",
        "xghp_1234567890abcdefghijklmnopqrstuvwxyz",
        "github_pat_tooshort",
    ] {
        assert!(!has(text, "github-token"), "{text}");
    }
}

#[test]
fn json_web_tokens() {
    let token = format!("{HS256}.{PAYLOAD}.{SIGNATURE}");
    assert_eq!(normalized(&secrets(), &format!("Authorization: Bearer {token}")), [token.as_str()]);
    for header in [HS256, RS256, SPACED] {
        let token = format!("{header}.{PAYLOAD}.{SIGNATURE}");
        assert_found(&secrets(), &token, &[(0, token.len() as u32, "jwt", &token)]);
    }
    for text in [
        format!("{NO_ALGORITHM}.eyJzdWIiOiIxIn0.abcdefgh"),
        format!("{NOTALG}.{PAYLOAD}.{SIGNATURE}"),
        format!("{ALGORITHM}.{PAYLOAD}.{SIGNATURE}"),
        format!("notbase64url.{PAYLOAD}.{SIGNATURE}"),
        format!("{HS256}.{PAYLOAD}"),
        format!("{HS256}.{PAYLOAD}."),
        format!("{HS256}..{SIGNATURE}"),
        format!("eyJ.{PAYLOAD}.{SIGNATURE}"),
        format!("x{HS256}.{PAYLOAD}.{SIGNATURE}"),
    ] {
        assert!(!has(&text, "jwt"), "{text}");
    }
}

#[test]
fn url_credentials() {
    for (text, credential) in [
        ("https://user:secret@example.com/path", "user:secret"),
        ("http://admin:s3cr3t@10.0.0.1:8080/", "admin:s3cr3t"),
        ("ftp://jane.doe:pw%21@files.example.org", "jane.doe:pw%21"),
        ("postgres://app:hunter2@db.internal:5432/main", "app:hunter2"),
        ("redis://default:abc-123_x@cache.example.net", "default:abc-123_x"),
        ("amqp://guest:guest@broker/", "guest:guest"),
    ] {
        let start = text.find("://").unwrap() as u32 + 3;
        assert_found(&secrets(), text, &[(start, start + credential.len() as u32, "url-credential", credential)]);
    }
    for text in [
        "https://example.com/path",
        "https://user@example.com",
        "https://:secret@example.com",
        "https://user:@example.com",
        "https://example.com/a?b=c@d",
        "mailto:jane@example.com",
        "://user:secret@example.com",
        "user:secret@example.com",
    ] {
        assert!(!has(text, "url-credential"), "{text}");
    }
}

#[test]
fn a_credential_holding_a_token_keeps_both() {
    let credential = "oauth2:ghp_1234567890abcdefghijklmnopqrstuvwxyz";
    let token = "ghp_1234567890abcdefghijklmnopqrstuvwxyz";
    assert_found(
        &secrets(),
        &format!("https://{credential}@example.com/x.git"),
        &[
            (8, 8 + credential.len() as u32, "url-credential", credential),
            (15, 15 + token.len() as u32, "github-token", token),
        ],
    );
}

#[test]
fn several_secrets_in_one_text() {
    let text = format!(
        "key AKIAIOSFODNN7EXAMPLE token ghp_1234567890abcdefghijklmnopqrstuvwxyz jwt {HS256}.{PAYLOAD}.{SIGNATURE} url https://u:p@example.com/"
    );
    assert_eq!(kinds(&secrets(), &text), ["aws-access-key", "github-token", "jwt", "url-credential"]);
    assert_eq!(
        rows(&secrets(), "ghp_1234567890abcdefghijklmnopqrstuvwxyz ghs_abcdefghijklmnopqrstuvwxyz1234567890").len(),
        2
    );
    let text = "AKIAIOSFODNN7EXAMPLE and https://u:p@example.com/";
    assert_eq!(kinds(&only(Extractor::Secrets, &[PiiKind::AwsAccessKey]), text), ["aws-access-key"]);
    assert_eq!(kinds(&urls(), text), ["url-credential"]);
    for text in [
        "nothing to see here",
        "AKIA",
        "ghp_",
        "eyJ",
        "https://",
        "",
        "The word algorithm contains alg but is no token",
    ] {
        assert_none(&secrets(), text);
    }
}

fn tokens() -> Vec<(&'static str, String)> {
    let mut out = Vec::new();
    for prefix in ["AKIA", "ASIA"] {
        out.push(("aws-access-key", format!("{prefix}{}", "A".repeat(16))));
    }
    for prefix in ["ghp_", "gho_", "ghu_", "ghs_", "ghr_"] {
        out.push(("github-token", format!("{prefix}{}", "a".repeat(36))));
    }
    out.push(("github-token", format!("github_pat_{}", "a".repeat(82))));
    out.push(("jwt", JWT.to_owned()));
    out
}

#[test]
fn token_boundaries() {
    for (kind, token) in tokens() {
        for prefix in ["a", "Z", "0", "é", "中", "𐐀", "١", "𝟙", "_"] {
            assert_none(&secrets(), &format!("{prefix}{token}"));
        }
        for suffix in ["é", "中", "𐐀", "١", "𝟙"] {
            assert_none(&secrets(), &format!("{token}{suffix}"));
        }
        for d in ["", " ", "\t", "\n", "\u{a0}", "😀", "\"", "'", ",", ";", ":", "(", ")", "[", "]", "\0"] {
            let start = d.len() as u32;
            assert_found(&secrets(), &format!("{d}{token}{d}"), &[(start, start + token.len() as u32, kind, &token)]);
        }
        assert_eq!(rows(&secrets(), &format!("-{token}")).len(), usize::from(kind != "jwt"), "{token}");
        assert_found(&secrets(), &format!("value={token} "), &[(6, 6 + token.len() as u32, kind, &token)]);
    }
}

#[test]
fn schemes() {
    for scheme in
        ["s3", "h2", "HTTP", "a", "a+", "a-", "a.", "a0", "a1b2", "git+ssh", "web+demo2", "a.b-3+4", "x-custom-7"]
    {
        let prefix = "😀 (";
        let start = (prefix.len() + scheme.len() + 3) as u32;
        assert_found(
            &urls(),
            &format!("{prefix}{scheme}://{CREDENTIAL}{AUTHORITY_END})"),
            &[(start, start + CREDENTIAL.len() as u32, "url-credential", CREDENTIAL)],
        );
    }
    for scheme in [
        "", "1http", "0ftp", "123abc", "9", "+http", "-http", ".http", "éhttp", "中http", "𐐀http", "١http", "𝟙http",
        "_http", "ht_tp",
    ] {
        assert_none(&urls(), &format!("{scheme}://{CREDENTIAL}{AUTHORITY_END}"));
    }
}

#[test]
fn percent_escapes() {
    for octet in 0..=255u8 {
        let userinfo = format!("u%{octet:02x}:p%{octet:02X}");
        assert_eq!(normalized(&urls(), &format!("https://{userinfo}{AUTHORITY_END}")), [userinfo]);
    }
    for escape in ["%", "%0", "%x", "%gg", "%0g", "%g0", "%000%", "%%00", "%é0", "%𝟙0", "%١0"] {
        for userinfo in [format!("u{escape}:pw"), format!("user:pw{escape}")] {
            assert_none(&urls(), &format!("https://{userinfo}{AUTHORITY_END}"));
        }
    }
}

#[test]
fn the_scan_resumes_after_an_invalid_url() {
    for malformed in
        ["1http://u:p@example.invalid", "https://u:p%@example.invalid", "https://u:p%", "https://u:p", "https://"]
    {
        let prefix = format!("{malformed} 😀 ");
        let start = (prefix.len() + 5) as u32;
        assert_found(
            &urls(),
            &format!("{prefix}s3://{CREDENTIAL}{AUTHORITY_END}"),
            &[(start, start + CREDENTIAL.len() as u32, "url-credential", CREDENTIAL)],
        );
    }
}

#[test]
fn secrets_pack_masks_the_user_information() {
    let text = "Fetch s3://user:demo%40pass@example.invalid/model.bin";
    assert_eq!(
        mask_all(text, &layer(&secrets(), text), '*').unwrap(),
        "Fetch s3://****************@example.invalid/model.bin"
    );
}

#[test]
fn secrets_on_every_thread() {
    let d = secrets();
    std::thread::scope(|s| {
        for worker in 0..8 {
            let d = &d;
            s.spawn(move || {
                let prefix = format!("{} s3://", "😀".repeat(worker));
                let userinfo = format!("u{worker}:p%40{worker}");
                let text = format!("{prefix}{userinfo}{AUTHORITY_END} {JWT}");
                let start = prefix.len() as u32;
                let end = text.len() as u32;
                for _ in 0..32 {
                    assert_found(
                        d,
                        &text,
                        &[
                            (start, start + userinfo.len() as u32, "url-credential", &userinfo),
                            (end - JWT.len() as u32, end, "jwt", JWT),
                        ],
                    );
                }
            });
        }
    });
}
