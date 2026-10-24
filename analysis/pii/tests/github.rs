//! `GithubInstallationTokenTest`: `ghs_` installation tokens, opaque or
//! dotted, of any length from 36 characters.

mod support;

use support::*;
use turbo_pii::rewrite::mask_all;
use turbo_pii::{Extractor, Pack, PiiDetector, PiiKind};

const PREFIX: &str = "ghs_";
const JWT_PREFIX: &str = "ghs_12345_eyJhbGciOiJIUzI1NiJ9.e30.";

fn github() -> PiiDetector {
    only(Extractor::Secrets, &[PiiKind::GithubToken])
}

fn token(repetitions: usize) -> String {
    format!("{JWT_PREFIX}{}", "A-_".repeat(repetitions))
}

#[track_caller]
fn assert_token(value: &str) {
    let prefix = "😀 Authorization: Bearer ";
    let start = prefix.len() as u32;
    let text = format!("{prefix}{value}\n");
    for d in [github(), pack(Pack::Secrets)] {
        assert_found(&d, &text, &[(start, start + value.len() as u32, "github-token", value)]);
    }
}

#[test]
fn lengths() {
    for length in [36, 251, 252, 516, 1024, 65536] {
        assert_token(&format!("{PREFIX}{}", "a".repeat(length)));
    }
    for repetitions in [12, 64, 160, 1024, 21846] {
        assert_token(&token(repetitions));
    }
    for (length, accepted) in [(0, false), (1, false), (35, false), (36, true), (37, true)] {
        assert_eq!(rows(&github(), &format!("{PREFIX}{}", "-".repeat(length))).len(), usize::from(accepted));
    }
    assert_none(&github(), &format!("{PREFIX}{}...", "a".repeat(35)));
}

#[test]
fn alphabet() {
    for c in 0u8..128 {
        let allowed = c.is_ascii_alphanumeric() || matches!(c, b'.' | b'-' | b'_');
        let value = format!("{PREFIX}{}{}{}", "a".repeat(18), c as char, "B".repeat(18));
        if allowed {
            assert_found(&github(), &value, &[(0, value.len() as u32, "github-token", &value)]);
        } else {
            assert_none(&github(), &value);
        }
    }
}

#[test]
fn opaque_contents() {
    for body in [
        "not_a_numeric_app_id_or_a_decodable_JWT",
        "app_noJsonHeader.noJsonClaims.notASignature",
        "letters.with.more.than.two.dots.and-hyphens",
    ] {
        assert_token(&format!("{PREFIX}{body}"));
    }
}

#[test]
fn boundaries_and_prefixes() {
    let value = token(160);
    for adjoining in ["é", "中", "𐐀", "١", "𝟙"] {
        assert_none(&github(), &format!("{adjoining}{value}"));
        assert_none(&github(), &format!("{value}{adjoining}"));
        assert_none(&github(), &format!("{value}.{adjoining}"));
    }
    for prefix in ["xghs_", "_ghs_", "0ghs_", "GHS_", "Ghs_", "ghS_", "ghx_"] {
        assert_none(&github(), &format!("{prefix}{}", "a".repeat(516)));
    }
    for suffix in ["", ".", "...", ". ", ";", ",", ")", "]", "\"", "'", "!", "?", "\n"] {
        assert_found(&github(), &format!("{value}{suffix}"), &[(0, value.len() as u32, "github-token", &value)]);
    }
}

#[test]
fn other_token_limits() {
    for prefix in ["ghp_", "gho_", "ghu_", "ghr_", "github_pat_"] {
        assert_none(&github(), &format!("{prefix}{}", "a".repeat(256)));
        assert_none(&github(), &format!("{prefix}{}-{}", "a".repeat(18), "B".repeat(18)));
    }
    for kind in [PiiKind::AwsAccessKey, PiiKind::Jwt, PiiKind::UrlCredential] {
        assert_none(&only(Extractor::Secrets, &[kind]), &token(160));
    }
}

#[test]
fn inside_a_url_credential() {
    let value = token(160);
    let credential = format!("user:{value}");
    let text = format!("https://{credential}@example.invalid/repo.git");
    assert_found(&pack(Pack::Secrets), &text, &[(8, 8 + credential.len() as u32, "url-credential", &credential)]);
    assert_found(
        &pack(Pack::Secrets).keeping_nested(),
        &text,
        &[
            (8, 8 + credential.len() as u32, "url-credential", &credential),
            (13, 8 + credential.len() as u32, "github-token", &value),
        ],
    );
    assert_eq!(normalized(&github(), &text), [value]);
}

#[test]
fn the_next_token() {
    let (first, second) = (token(160), token(161));
    let text = format!("{PREFIX}{}é; {first} {second}", "a".repeat(516));
    assert_eq!(normalized(&github(), &text), [first, second]);
}

#[test]
fn secrets_pack_masks_an_installation_token() {
    let value = format!("ghs_12345_eyJhbGciOiJIUzI1NiJ9.e30.{}", "A-_".repeat(160));
    let text = format!("Bearer {value}");
    let d = pack(Pack::Secrets);
    assert_eq!(mask_all(&text, &layer(&d, &text), '*').unwrap(), format!("Bearer {}", "*".repeat(value.len())));
    assert_eq!(normalized(&d, &text), [value]);
}

#[test]
fn tokens_on_every_thread() {
    std::thread::scope(|s| {
        for worker in 0..8 {
            s.spawn(move || {
                let value = token(160 + worker);
                for _ in 0..32 {
                    assert_token(&value);
                    assert_none(&github(), &format!("{value}é"));
                }
            });
        }
    });
}
