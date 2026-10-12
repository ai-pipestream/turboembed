//! Readers for the Unicode Character Database text formats.
//!
//! Most UCD files share one line format: a code point or a range
//! `start..end`, a semicolon, one or more `;`-separated fields, and an
//! optional `#` comment. `UnicodeData.txt` is the exception: one code
//! point per line and fifteen fields, read by [`unicode_data`].

use std::fs;
use std::path::Path;

/// Code points from U+0000 to U+10FFFF.
pub const CODE_POINTS: usize = 0x11_0000;

/// One data line: an inclusive code point range and the first field
/// after it, trimmed.
pub struct Range<'a> {
    pub start: u32,
    pub end: u32,
    pub value: &'a str,
}

/// Reads a file from the UCD directory.
pub fn read(dir: &Path, name: &str) -> Result<String, String> {
    fs::read_to_string(dir.join(name)).map_err(|error| format!("{}: {error}", dir.join(name).display()))
}

/// Parses a hexadecimal code point and checks that it is at most U+10FFFF.
fn code_point(text: &str) -> Option<u32> {
    u32::from_str_radix(text.trim(), 16).ok().filter(|&cp| (cp as usize) < CODE_POINTS)
}

/// The data lines of a file in the common format. `name` is used in errors.
pub fn ranges<'a>(name: &str, text: &'a str) -> Result<Vec<Range<'a>>, String> {
    let mut out = Vec::new();
    for (number, line) in text.lines().enumerate() {
        let content = line.split('#').next().unwrap_or("").trim();
        if content.is_empty() {
            continue;
        }
        let error = || format!("{name} line {}: cannot read `{line}`", number + 1);
        let mut fields = content.split(';');
        let points = fields.next().ok_or_else(error)?;
        let value = fields.next().ok_or_else(error)?.trim();
        let (start, end) = match points.split_once("..") {
            Some((start, end)) => (code_point(start).ok_or_else(error)?, code_point(end).ok_or_else(error)?),
            None => {
                let cp = code_point(points).ok_or_else(error)?;
                (cp, cp)
            }
        };
        if end < start {
            return Err(error());
        }
        out.push(Range { start, end, value });
    }
    Ok(out)
}

/// Sets `bit` in `flags` for every code point whose value in the file is
/// `property`. Fails when the file has no such line, which catches a
/// misspelt property name.
pub fn set_flag(name: &str, text: &str, property: &str, flags: &mut [u8], bit: u8) -> Result<(), String> {
    let mut found = false;
    for range in ranges(name, text)?.iter().filter(|range| range.value == property) {
        found = true;
        for cp in range.start..=range.end {
            flags[cp as usize] |= bit;
        }
    }
    if found { Ok(()) } else { Err(format!("{name} has no {property} lines")) }
}

/// The fields of each `UnicodeData.txt` line, with the line's code point.
/// The `<..., First>` and `<..., Last>` range lines are returned as two
/// lines like any other; the fields this generator reads are empty on them.
pub fn unicode_data(text: &str) -> Result<Vec<(u32, Vec<&str>)>, String> {
    let mut out = Vec::new();
    for (number, line) in text.lines().enumerate() {
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(';').collect();
        let cp = fields.first().and_then(|cp| code_point(cp));
        match cp {
            Some(cp) if fields.len() == 15 => out.push((cp, fields)),
            _ => return Err(format!("UnicodeData.txt line {}: cannot read `{line}`", number + 1)),
        }
    }
    Ok(out)
}

/// The version a file's header names: `# PropList-17.0.0.txt` gives
/// `17.0.0`, `# Version: 17.0` gives `17.0`.
pub fn version(name: &str, text: &str) -> Result<String, String> {
    for line in text.lines().take_while(|line| line.starts_with('#')) {
        let line = line.trim_start_matches('#').trim();
        if let Some(version) = line.strip_prefix("Version:") {
            return Ok(version.trim().to_string());
        }
        if let Some(stem) = line.strip_suffix(".txt")
            && let Some((_, version)) = stem.rsplit_once('-')
            && version.starts_with(|c: char| c.is_ascii_digit())
        {
            return Ok(version.to_string());
        }
    }
    Err(format!("{name} names no version in its header"))
}
