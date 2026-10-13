//! Reading one member of a gzip compressed tar archive in memory: enough
//! for an archive of a few regular files, as Qasper publishes.
//!
//! The tar reader takes ustar headers (512-byte blocks, octal sizes, the
//! `prefix` field joined to the name with `/`), returns regular files
//! (type `0` or NUL) and steps over every other entry, pax headers
//! included. A pax header that renames the following entry is not
//! applied, so such a member is found under its ustar name only.

use std::io::Read;

use flate2::read::MultiGzDecoder;

use crate::Result;

/// The bytes of member `name` of the gzip tar `archive`, which is named
/// `file` in errors.
pub fn gzip_member(file: &str, archive: &[u8], name: &str) -> Result<Vec<u8>> {
    let mut tar = Vec::new();
    MultiGzDecoder::new(archive).read_to_end(&mut tar).map_err(|e| format!("{file}: gzip: {e}"))?;
    member(file, &tar, name)
}

/// The bytes of member `name` of the tar `tar`.
pub fn member(file: &str, tar: &[u8], name: &str) -> Result<Vec<u8>> {
    let mut at = 0;
    while at + 512 <= tar.len() {
        let h = &tar[at..at + 512];
        if h.iter().all(|&b| b == 0) {
            break;
        }
        let size = octal(&h[124..136]).ok_or_else(|| format!("{file}: tar header at byte {at}: bad size"))?;
        let data = at + 512;
        let end = data.checked_add(size).filter(|&e| e <= tar.len());
        let end = end.ok_or_else(|| format!("{file}: tar entry at byte {at} runs past the end"))?;
        let kind = h[156];
        if (kind == b'0' || kind == 0) && entry_name(h) == name {
            return Ok(tar[data..end].to_vec());
        }
        at = data + size.div_ceil(512) * 512;
    }
    Err(format!("{file}: no member {name}"))
}

fn field(b: &[u8]) -> &[u8] {
    let n = b.iter().position(|&c| c == 0).unwrap_or(b.len());
    &b[..n]
}

fn entry_name(h: &[u8]) -> String {
    let name = String::from_utf8_lossy(field(&h[0..100]));
    let prefix = if &h[257..262] == b"ustar" { String::from_utf8_lossy(field(&h[345..500])) } else { "".into() };
    if prefix.is_empty() { name.into_owned() } else { format!("{prefix}/{name}") }
}

/// An octal number padded with spaces or NULs.
fn octal(b: &[u8]) -> Option<usize> {
    let s = std::str::from_utf8(field(b)).ok()?.trim();
    if s.is_empty() {
        return Some(0);
    }
    usize::from_str_radix(s, 8).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(name: &str, kind: u8, size: usize) -> Vec<u8> {
        let mut h = vec![0u8; 512];
        h[..name.len()].copy_from_slice(name.as_bytes());
        let s = format!("{size:011o}");
        h[124..135].copy_from_slice(s.as_bytes());
        h[156] = kind;
        h[257..263].copy_from_slice(b"ustar\0");
        h
    }

    fn entry(tar: &mut Vec<u8>, name: &str, kind: u8, body: &[u8]) {
        tar.extend(header(name, kind, body.len()));
        tar.extend_from_slice(body);
        tar.resize(tar.len().div_ceil(512) * 512, 0);
    }

    #[test]
    fn finds_a_member_past_others() {
        let mut tar = Vec::new();
        entry(&mut tar, "pax", b'x', b"30 path=something/else.json\n");
        entry(&mut tar, "a.txt", b'0', &[b'a'; 600]);
        entry(&mut tar, "b.json", b'0', b"{}");
        tar.extend([0u8; 1024]);
        assert_eq!(member("t.tar", &tar, "b.json").unwrap(), b"{}");
        assert_eq!(member("t.tar", &tar, "a.txt").unwrap().len(), 600);
        assert_eq!(member("t.tar", &tar, "c").unwrap_err(), "t.tar: no member c");
    }

    #[test]
    fn refuses_an_entry_past_the_end() {
        let mut tar = header("a", b'0', 10_000);
        tar.extend([0u8; 512]);
        assert_eq!(member("t.tar", &tar, "a").unwrap_err(), "t.tar: tar entry at byte 0 runs past the end");
    }
}
