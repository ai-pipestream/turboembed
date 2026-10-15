//! Credentials, `SecretsPiiExtractor`: AWS access key ids, GitHub tokens
//! and the userinfo of URLs. JSON Web Tokens are in `jwt`. The scans run
//! in the Java's order: URL credentials, JWTs, AWS keys, GitHub tokens.

use turbo_analysis_layers::{PiiConfidence, PiiKind};

use crate::hits::Hits;
use crate::text::{on_end, on_word_start, starts_with};

/// No letter, digit or underscore after `end`.
pub(crate) fn on_token_end(text: &str, end: usize) -> bool {
    on_end(text, end) && text.as_bytes().get(end) != Some(&b'_')
}

/// No letter, digit or underscore before `start`, nor a hyphen when the
/// token is base64url.
pub(crate) fn on_token_start(text: &str, start: usize, base64url: bool) -> bool {
    if start == 0 {
        return true;
    }
    let previous = text.as_bytes()[start - 1];
    on_word_start(text, start) && previous != b'_' && (!base64url || previous != b'-')
}

const AWS_PREFIXES: [&[u8]; 2] = [b"AKIA", b"ASIA"];
const AWS_BODY: usize = 16;

/// `AKIA` or `ASIA` and 16 capitals or digits.
pub(crate) fn scan_aws(text: &str, hits: &mut Hits) {
    let b = text.as_bytes();
    let mut i = 0;
    while i < b.len() {
        let found = b[i] == b'A'
            && on_token_start(text, i, false)
            && AWS_PREFIXES.iter().any(|prefix| starts_with(b, i, prefix))
            && i + 4 + AWS_BODY <= b.len()
            && b[i + 4..i + 4 + AWS_BODY].iter().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit())
            && on_token_end(text, i + 4 + AWS_BODY);
        if found {
            hits.add_text(text, i, i + 4 + AWS_BODY, PiiKind::AwsAccessKey, PiiConfidence::Format);
            i += 4 + AWS_BODY;
        } else {
            i += 1;
        }
    }
}

const GITHUB_PREFIXES: [&[u8]; 4] = [b"ghp_", b"gho_", b"ghu_", b"ghr_"];
const GITHUB_INSTALLATION: &[u8] = b"ghs_";
const GITHUB_FINE_GRAINED: &[u8] = b"github_pat_";
const GITHUB_BODY: usize = 36;
const GITHUB_FINE_GRAINED_BODY: usize = 82;
const GITHUB_MAX: usize = 255;

/// `ghp_`, `gho_`, `ghu_` or `ghr_` and 36 letters, digits or underscores
/// or more, `github_pat_` and 82 or more, at most 255 in all; `ghs_` and
/// 36 or more that may also be dots and hyphens, trailing dots left out.
pub(crate) fn scan_github(text: &str, hits: &mut Hits) {
    let b = text.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] != b'g' || !on_token_start(text, i, false) {
            i += 1;
            continue;
        }
        let installation = starts_with(b, i, GITHUB_INSTALLATION);
        let (body_start, minimum) = if installation {
            (i + GITHUB_INSTALLATION.len(), GITHUB_BODY)
        } else if starts_with(b, i, GITHUB_FINE_GRAINED) {
            (i + GITHUB_FINE_GRAINED.len(), GITHUB_FINE_GRAINED_BODY)
        } else if let Some(p) = GITHUB_PREFIXES.iter().find(|p| starts_with(b, i, p)) {
            (i + p.len(), GITHUB_BODY)
        } else {
            i += 1;
            continue;
        };
        let mut end = body_start;
        while end < b.len()
            && (b[end].is_ascii_alphanumeric() || b[end] == b'_' || (installation && matches!(b[end], b'.' | b'-')))
        {
            end += 1;
        }
        let mut token_end = end;
        if installation {
            while token_end > body_start && b[token_end - 1] == b'.' {
                token_end -= 1;
            }
        }
        if token_end - body_start >= minimum && (installation || token_end - i <= GITHUB_MAX) && on_token_end(text, end)
        {
            hits.add_text(text, i, token_end, PiiKind::GithubToken, PiiConfidence::Format);
        }
        // The scanned run is skipped even when refused, so a prefix inside
        // it is not read again.
        i = end.max(i + 1);
    }
}

/// `user:password` before `@` in a URL whose scheme starts with an ASCII
/// letter at a token start; percent escapes need two hexadecimal digits.
pub(crate) fn scan_url_credentials(text: &str, hits: &mut Hits) {
    let b = text.as_bytes();
    let mut i = 1;
    while i + 3 < b.len() {
        if !starts_with(b, i, b"://") || !has_scheme(text, i) {
            i += 1;
            continue;
        }
        let start = i + 3;
        let mut p = start;
        let mut colon = None;
        while p < b.len() && is_userinfo(b[p]) {
            if b[p] == b'%' {
                if b.len() - p < 3 || !b[p + 1].is_ascii_hexdigit() || !b[p + 2].is_ascii_hexdigit() {
                    break;
                }
                p += 3;
                continue;
            }
            if b[p] == b':' && colon.is_none() {
                colon = Some(p);
            }
            p += 1;
        }
        let found = b.get(p) == Some(&b'@') && colon.is_some_and(|c| c > start && c + 1 < p);
        if found {
            hits.add_text(text, start, p, PiiKind::UrlCredential, PiiConfidence::Format);
            i = p;
        } else {
            i += 1;
        }
    }
}

fn has_scheme(text: &str, end: usize) -> bool {
    let b = text.as_bytes();
    let mut start = end;
    while start > 0 && (b[start - 1].is_ascii_alphanumeric() || matches!(b[start - 1], b'+' | b'-' | b'.')) {
        start -= 1;
    }
    start < end && b[start].is_ascii_alphabetic() && on_token_start(text, start, false)
}

fn is_userinfo(c: u8) -> bool {
    c.is_ascii_alphanumeric()
        || matches!(
            c,
            b'-' | b'.'
                | b'_'
                | b'~'
                | b'%'
                | b'!'
                | b'$'
                | b'&'
                | b'\''
                | b'('
                | b')'
                | b'*'
                | b'+'
                | b','
                | b';'
                | b'='
                | b':'
        )
}
