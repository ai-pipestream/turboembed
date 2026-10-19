//! Hand-written byte scanners for `MGRS` and `DEGREES_MIN_SEC_LAT_LON`.

use super::net::{count_digits, is_java_space};

#[inline]
fn skip_spaces(b: &[u8], mut i: usize) -> usize {
    while i < b.len() && is_java_space(b[i]) {
        i += 1;
    }
    i
}

/// `\d{1,2}[A-Za-z]\s*[A-Za-z]{2}\s*\d{1,5}\s*\d{1,5}` at `pos`.
pub(super) fn match_mgrs(b: &[u8], pos: usize) -> Option<usize> {
    if !b[pos].is_ascii_digit() {
        return None;
    }
    let after_zone =
        if b.get(pos + 1).is_some_and(u8::is_ascii_digit) && b.get(pos + 2).is_some_and(u8::is_ascii_alphabetic) {
            pos + 3
        } else if b.get(pos + 1).is_some_and(u8::is_ascii_alphabetic) {
            pos + 2
        } else {
            return None;
        };
    let i = skip_spaces(b, after_zone);
    if !b.get(i).is_some_and(u8::is_ascii_alphabetic) || !b.get(i + 1).is_some_and(u8::is_ascii_alphabetic) {
        return None;
    }
    let i = skip_spaces(b, i + 2);
    let d1 = count_digits(b, i, 5);
    if d1 == 0 {
        return None;
    }
    let after_d1 = i + d1;
    let after_ws = skip_spaces(b, after_d1);
    let d2 = count_digits(b, after_ws, 5);
    if d2 > 0 {
        Some(after_ws + d2)
    } else if d1 >= 2 {
        // `\s*` matched 0 spaces and the two `\d{1,5}` groups split `d1`.
        Some(after_d1)
    } else {
        None
    }
}

/// `[d|D|\u00B0|\s]` at `i`.
#[inline]
fn step_deg_unit(b: &[u8], i: usize) -> Option<usize> {
    let c = *b.get(i)?;
    if matches!(c, b'd' | b'D' | b'|') || is_java_space(c) {
        Some(i + 1)
    } else if b.get(i..i + 2) == Some(&[0xC2, 0xB0]) {
        // U+00B0 DEGREE SIGN in UTF-8.
        Some(i + 2)
    } else {
        None
    }
}

/// `\s*\d{1,2}['|\u2019|\s]` at `i`.
fn match_dms_minutes(b: &[u8], i: usize) -> Option<usize> {
    let j = skip_spaces(b, i);
    let d = count_digits(b, j, 2);
    if d == 0 {
        return None;
    }
    let k = j + d;
    let c = *b.get(k)?;
    if matches!(c, b'\'' | b'|') || is_java_space(c) {
        Some(k + 1)
    } else if b.get(k..k + 3) == Some(&[0xE2, 0x80, 0x99]) {
        // U+2019 RIGHT SINGLE QUOTATION MARK in UTF-8.
        Some(k + 3)
    } else {
        None
    }
}

/// `\s*\d{1,2}[\"|\u201d]` at `i`.
fn match_dms_seconds(b: &[u8], i: usize) -> Option<usize> {
    let j = skip_spaces(b, i);
    let d = count_digits(b, j, 2);
    if d == 0 {
        return None;
    }
    let k = j + d;
    let c = *b.get(k)?;
    if matches!(c, b'"' | b'|') {
        Some(k + 1)
    } else if b.get(k..k + 3) == Some(&[0xE2, 0x80, 0x9D]) {
        // U+201D RIGHT DOUBLE QUOTATION MARK in UTF-8.
        Some(k + 3)
    } else {
        None
    }
}

/// Matches the longitude half starting at `pos`:
/// `[-|\+]?\d{1,3}[d|D|\u00B0|\s](\s*\d{1,2}['|\u2019|\s])?(\s*\d{1,2}[\"|\u201d])?\s*[E|e|W|w]?`.
fn match_dms_lon(b: &[u8], mut i: usize) -> Option<usize> {
    if b.get(i).is_some_and(|&c| matches!(c, b'-' | b'|' | b'+')) {
        i += 1;
    }
    let deg = count_digits(b, i, 3);
    if deg == 0 {
        return None;
    }
    let mut i = step_deg_unit(b, i + deg)?;
    if let Some(after_min) = match_dms_minutes(b, i) {
        i = after_min;
    }
    if let Some(after_sec) = match_dms_seconds(b, i) {
        i = after_sec;
    }
    i = skip_spaces(b, i);
    if b.get(i).is_some_and(|&c| matches!(c, b'E' | b'e' | b'W' | b'w' | b'|')) {
        i += 1;
    }
    Some(i)
}

/// Matches `\s*[N|n|S|s]?(\s*|,|,\s*)` followed by `match_dms_lon` from
/// `after_lat_body`.
fn match_dms_after_lat(b: &[u8], after_lat_body: usize) -> Option<usize> {
    let ws = skip_spaces(b, after_lat_body);
    if b.get(ws).is_some_and(|&c| matches!(c, b'N' | b'n' | b'S' | b's' | b'|')) {
        let after_hemi = ws + 1;
        // Separator `(\s*|,|,\s*)` after hemisphere.
        let after_sep_ws = skip_spaces(b, after_hemi);
        if let Some(end) = match_dms_lon(b, after_sep_ws) {
            return Some(end);
        }
        if b.get(after_hemi) == Some(&b',') {
            if let Some(end) = match_dms_lon(b, after_hemi + 1) {
                return Some(end);
            }
            let after_comma_ws = skip_spaces(b, after_hemi + 1);
            if let Some(end) = match_dms_lon(b, after_comma_ws) {
                return Some(end);
            }
        }
    }
    // Without `[N|n|S|s]`: `ws` already consumed all `\s*`.
    if let Some(end) = match_dms_lon(b, ws) {
        return Some(end);
    }
    if b.get(ws) == Some(&b',') {
        if let Some(end) = match_dms_lon(b, ws + 1) {
            return Some(end);
        }
        let after_comma_ws = skip_spaces(b, ws + 1);
        if let Some(end) = match_dms_lon(b, after_comma_ws) {
            return Some(end);
        }
    }
    None
}

/// `BuiltIn::DegreesMinSecLatLon` at `pos`.
pub(super) fn match_dms_lat_lon(b: &[u8], pos: usize) -> Option<usize> {
    let mut i = pos;
    if matches!(b[i], b'-' | b'|' | b'+') {
        i += 1;
    }
    let deg = count_digits(b, i, 3);
    if deg == 0 {
        return None;
    }
    let after_deg = step_deg_unit(b, i + deg)?;

    if let Some(after_min) = match_dms_minutes(b, after_deg) {
        if let Some(after_sec) = match_dms_seconds(b, after_min)
            && let Some(end) = match_dms_after_lat(b, after_sec)
        {
            return Some(end);
        }
        if let Some(end) = match_dms_after_lat(b, after_min) {
            return Some(end);
        }
    }
    if let Some(after_sec) = match_dms_seconds(b, after_deg)
        && let Some(end) = match_dms_after_lat(b, after_sec)
    {
        return Some(end);
    }
    match_dms_after_lat(b, after_deg)
}
