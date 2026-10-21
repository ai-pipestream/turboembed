//! The files of a zip archive whose entries are stored, not compressed, as
//! `torch.save` writes them. Only what that writer produces is read: one
//! disk, method 0 (stored), zip64 records where a size or offset needs
//! them, each entry's CRC-32 checked. Anything else is refused by name.

use crate::Result;

/// One stored file: its name and its bytes inside the archive.
pub struct Entry<'a> {
    pub name: &'a str,
    pub data: &'a [u8],
}

fn u16_at(b: &[u8], at: usize) -> Result<u16> {
    b.get(at..at + 2).map(|s| u16::from_le_bytes([s[0], s[1]])).ok_or_else(|| "zip: truncated".into())
}

fn u32_at(b: &[u8], at: usize) -> Result<u32> {
    b.get(at..at + 4).map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]])).ok_or_else(|| "zip: truncated".into())
}

fn u64_at(b: &[u8], at: usize) -> Result<usize> {
    let s = b.get(at..at + 8).ok_or("zip: truncated")?;
    usize::try_from(u64::from_le_bytes(s.try_into().expect("eight bytes"))).map_err(|_| "zip: offset overflows".into())
}

const END: u32 = 0x0605_4b50;
const CENTRAL: u32 = 0x0201_4b50;
const LOCAL: u32 = 0x0403_4b50;
const ZIP64_LOCATOR: u32 = 0x0706_4b50;
const ZIP64_END: u32 = 0x0606_4b50;
const FULL16: usize = u16::MAX as usize;
const FULL32: usize = u32::MAX as usize;

/// The central directory's entry count, size and offset, from the zip64
/// end record when the archive has one.
fn directory(bytes: &[u8], end: usize) -> Result<(usize, usize, usize)> {
    let (disk, cd_disk) = (u16_at(bytes, end + 4)?, u16_at(bytes, end + 6)?);
    let (here, count) = (u16_at(bytes, end + 8)?, u16_at(bytes, end + 10)? as usize);
    let (size, offset) = (u32_at(bytes, end + 12)? as usize, u32_at(bytes, end + 16)? as usize);
    if end < 20 || u32_at(bytes, end - 20)? != ZIP64_LOCATOR {
        if disk != 0 || cd_disk != 0 || usize::from(here) != count {
            return Err("zip: multi-disk archives are not read".into());
        }
        return Ok((count, size, offset));
    }
    let at = u64_at(bytes, end - 20 + 8)?;
    if u32_at(bytes, end - 20 + 4)? != 0 || u32_at(bytes, end - 20 + 16)? != 1 || u32_at(bytes, at)? != ZIP64_END {
        return Err("zip: bad zip64 end record".into());
    }
    let (here, count) = (u64_at(bytes, at + 24)?, u64_at(bytes, at + 32)?);
    if u32_at(bytes, at + 16)? != 0 || u32_at(bytes, at + 20)? != 0 || here != count {
        return Err("zip: multi-disk archives are not read".into());
    }
    Ok((count, u64_at(bytes, at + 40)?, u64_at(bytes, at + 48)?))
}

/// An entry's sizes and local offset, each taken from its zip64 extra field
/// (id 1) when the 32-bit field is full, in the order zip lists them.
fn widen(extra: &[u8], mut fields: [usize; 3]) -> Result<[usize; 3]> {
    let mut at = 0;
    while at + 4 <= extra.len() {
        let (id, n) = (u16_at(extra, at)?, u16_at(extra, at + 2)? as usize);
        let body = extra.get(at + 4..at + 4 + n).ok_or("zip: truncated extra field")?;
        if id == 1 {
            let mut k = 0;
            for f in fields.iter_mut().filter(|f| **f == FULL32) {
                *f = u64_at(body, k)?;
                k += 8;
            }
            return Ok(fields);
        }
        at += 4 + n;
    }
    if fields.contains(&FULL32) {
        return Err("zip: a full size or offset with no zip64 field".into());
    }
    Ok(fields)
}

/// The end-of-central-directory record: the last one in the final 64 KiB
/// plus its 22 bytes, as a comment may follow it.
fn end_record(bytes: &[u8]) -> Result<usize> {
    let lowest = bytes.len().saturating_sub(22 + 65_535);
    (lowest..=bytes.len().saturating_sub(22))
        .rev()
        .find(|&i| u32_at(bytes, i).is_ok_and(|s| s == END))
        .ok_or_else(|| "zip: no end of central directory".into())
}

/// Every entry of the archive, in central-directory order.
pub fn entries(bytes: &[u8]) -> Result<Vec<Entry<'_>>> {
    let end = end_record(bytes)?;
    let (count, size, offset) = directory(bytes, end)?;
    if count == FULL16 || size == FULL32 || offset == FULL32 {
        return Err("zip: a full count, size or offset with no zip64 end record".into());
    }
    if offset.checked_add(size).is_none_or(|e| e > end) {
        return Err("zip: central directory outside the archive".into());
    }
    let mut out = Vec::with_capacity(count.min(size / 46));
    let mut at = offset;
    for _ in 0..count {
        if u32_at(bytes, at)? != CENTRAL {
            return Err("zip: bad central directory entry".into());
        }
        let method = u16_at(bytes, at + 10)?;
        let crc = u32_at(bytes, at + 16)?;
        let (n, extra, comment) =
            (u16_at(bytes, at + 28)? as usize, u16_at(bytes, at + 30)? as usize, u16_at(bytes, at + 32)? as usize);
        let fields =
            [u32_at(bytes, at + 24)? as usize, u32_at(bytes, at + 20)? as usize, u32_at(bytes, at + 42)? as usize];
        let [length, stored, local] =
            widen(bytes.get(at + 46 + n..at + 46 + n + extra).ok_or("zip: truncated")?, fields)?;
        let name = bytes.get(at + 46..at + 46 + n).ok_or("zip: truncated")?;
        let name = std::str::from_utf8(name).map_err(|_| "zip: an entry name is not UTF-8")?;
        if method != 0 || stored != length {
            return Err(format!("zip: {name} is compressed (method {method}); only stored entries are read"));
        }
        if u32_at(bytes, local)? != LOCAL || bytes.get(local + 30..local + 30 + n) != Some(name.as_bytes()) {
            return Err(format!("zip: {name}: local header does not match the central directory"));
        }
        let start = local + 30 + n + u16_at(bytes, local + 28)? as usize;
        let data = start
            .checked_add(length)
            .filter(|&e| e <= offset)
            .and_then(|e| bytes.get(start..e))
            .ok_or_else(|| format!("zip: {name} lies outside the archive's files"))?;
        if crc32(data) != crc {
            return Err(format!("zip: {name}: CRC-32 mismatch"));
        }
        out.push(Entry { name, data });
        at += 46 + n + extra + comment;
    }
    Ok(out)
}

/// The CRC-32 zip uses (IEEE, reflected, polynomial 0xEDB88320).
fn crc32(data: &[u8]) -> u32 {
    const TABLE: [u32; 256] = {
        let mut t = [0u32; 256];
        let mut i = 0;
        while i < 256 {
            let mut c = i as u32;
            let mut k = 0;
            while k < 8 {
                c = if c & 1 == 1 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
                k += 1;
            }
            t[i] = c;
            i += 1;
        }
        t
    };
    !data.iter().fold(!0u32, |c, &b| TABLE[((c ^ u32::from(b)) & 0xff) as usize] ^ (c >> 8))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_of_the_check_string() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }
}
