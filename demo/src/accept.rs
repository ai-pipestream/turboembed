//! Negotiation for the gateway's two UTF-8 stream representations.

use axum::http::{HeaderMap, header};

#[derive(Clone, Copy)]
struct Preference {
    specificity: u8,
    quality: u16,
}

/// NDJSON is the absent-header/default wildcard choice. An explicit SSE
/// range wins a tie; an exact exclusion overrides a broader wildcard.
pub(crate) fn stream_sse(headers: &HeaderMap) -> Option<bool> {
    if !headers.contains_key(header::ACCEPT) {
        return Some(false);
    }
    let mut best: [Option<Preference>; 2] = [None, None];
    for value in headers.get_all(header::ACCEPT) {
        let Ok(value) = value.to_str() else { continue };
        let (mut quoted, mut escaped) = (false, false);
        // A comma inside a quoted parameter is not another media range.
        for entry in value.split(|c| {
            if escaped {
                escaped = false;
            } else if quoted && c == '\\' {
                escaped = true;
            } else if c == '"' {
                quoted = !quoted;
            } else if c == ',' && !quoted {
                return true;
            }
            false
        }) {
            for (slot, media) in best.iter_mut().zip(["application/x-ndjson", "text/event-stream"]) {
                let Some(p) = preference(entry, media) else { continue };
                if slot.is_none_or(|old| (p.specificity, p.quality) > (old.specificity, old.quality)) {
                    *slot = Some(p);
                }
            }
        }
    }
    let rank = |p: Option<Preference>| p.map(|p| (p.quality, p.specificity)).unwrap_or((0, 0));
    let (ndjson, sse) = (rank(best[0]), rank(best[1]));
    if ndjson.0 == 0 && sse.0 == 0 { None } else { Some(sse > ndjson || (sse == ndjson && sse.1 >= 4)) }
}

fn preference(entry: &str, media: &str) -> Option<Preference> {
    let mut parts = entry.trim().split(';');
    let range = parts.next()?.trim();
    let mut specificity = if range.eq_ignore_ascii_case(media) {
        4
    } else if range.strip_suffix("/*").is_some_and(|t| t.eq_ignore_ascii_case(media.split('/').next().unwrap())) {
        2
    } else if range == "*/*" {
        0
    } else {
        return None;
    };
    let (mut quality, mut charset) = (None, false);
    for part in parts {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let (name, value) = part.split_once('=')?;
        let (name, value) = (name.trim(), value.trim());
        if name.eq_ignore_ascii_case("q") && quality.is_none() {
            quality = Some(qvalue(value)?);
        } else if name.eq_ignore_ascii_case("charset") && !charset {
            let value = value.strip_prefix('"').and_then(|v| v.strip_suffix('"')).unwrap_or(value);
            if !value.eq_ignore_ascii_case("utf-8") {
                return None;
            }
            charset = true;
            specificity += 1;
        } else {
            // No other media parameters are offered by these representations.
            return None;
        }
    }
    Some(Preference { specificity, quality: quality.unwrap_or(1000) })
}

fn qvalue(value: &str) -> Option<u16> {
    let (whole, fraction) = value.split_once('.').unwrap_or((value, ""));
    if fraction.len() > 3 || !fraction.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    match whole {
        "0" => Some(fraction.bytes().zip([100, 10, 1]).map(|(b, scale)| u16::from(b - b'0') * scale).sum()),
        "1" if fraction.bytes().all(|b| b == b'0') => Some(1000),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    #[test]
    fn defaults_repeated_fields_and_parameter_boundaries() {
        let mut headers = HeaderMap::new();
        assert_eq!(stream_sse(&headers), Some(false));
        headers.append(header::ACCEPT, "text/event-stream;q=0".parse().unwrap());
        headers.append(header::ACCEPT, "*/*".parse().unwrap());
        assert_eq!(stream_sse(&headers), Some(false));
        for (value, expected) in [
            ("", None),
            ("*/*", Some(false)),
            ("application/x-ndjson, text/event-stream", Some(true)),
            ("text/event-stream; charset=\"UTF-8\"; Q=0.8, */*;q=0.1", Some(true)),
            ("text/event-stream; charset=latin1, application/x-ndjson", Some(false)),
            ("text/event-stream;q=0.8, text/event-stream;q=0.2", Some(true)),
            ("application/x-ndjson;profile=\"a,text/event-stream,b\"", None),
            ("application/x-ndjson;profile=\"a\\\",text/event-stream,b\"", None),
            ("text/event-stream;q=NaN", None),
            ("text/event-stream;q=1.001", None),
            ("text/event-stream;q=0.1234", None),
            ("text/event-stream;q=0;q=1", None),
            ("text/event-stream;q=0.001", Some(true)),
            ("text/event-stream;q=1.000", Some(true)),
            ("text/event-stream;", Some(true)),
            ("text/event-stream ;", Some(true)),
            ("text/event-stream; q=0.5;", Some(true)),
            ("text/event-stream;;q=1", Some(true)),
            ("application/x-ndjson;", Some(false)),
        ] {
            headers.insert(header::ACCEPT, value.parse().unwrap());
            assert_eq!(stream_sse(&headers), expected, "{value}");
        }
    }
    #[test]
    fn valid_quoted_pairs_in_utf8_charset_match_the_offered_representation() {
        for value in [
            r#"text/event-stream;charset="utf\-8""#,
            r#"text/event-stream;charset="u\tf-8";q=0.5;;"#,
            r#"application/x-ndjson;charset="UTF\-8";"#,
        ] {
            let mut headers = HeaderMap::new();
            headers.insert(header::ACCEPT, HeaderValue::from_str(value).unwrap());
            let expected = Some(value.starts_with("text/"));
            assert_eq!(stream_sse(&headers), expected, "{value}");
        }
    }

    #[test]
    fn empty_parameters_keep_repeated_header_quality_and_wildcard_controls() {
        let mut headers = HeaderMap::new();
        headers.append(header::ACCEPT, HeaderValue::from_static("text/event-stream;;;q=0;;"));
        headers.append(header::ACCEPT, HeaderValue::from_static("application/x-ndjson; ; q=0.7;"));
        assert_eq!(stream_sse(&headers), Some(false));
        headers.insert(header::ACCEPT, HeaderValue::from_static("text/event-stream;;q=0;;, */*;q=1;"));
        assert_eq!(stream_sse(&headers), Some(false));
    }

    #[test]
    fn opaque_quoted_bytes_do_not_hide_a_later_supported_range() {
        let values: &[&[u8]] = &[
            b"application/x-ndjson;profile=\"\x80\", text/event-stream",
            "application/x-ndjson;profile=\"é\", text/event-stream".as_bytes(),
        ];
        for value in values {
            let mut headers = HeaderMap::new();
            headers.insert(header::ACCEPT, HeaderValue::from_bytes(value).unwrap());
            assert_eq!(stream_sse(&headers), Some(true), "{value:?}");
        }
    }
}
