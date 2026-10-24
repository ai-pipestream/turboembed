use super::{DIRECTORY, HEADER, MAGIC, PrebuiltError, PrebuiltImage, SECTIONS, Table, Text, VERSION, validate};
use crate::lattice::chardef::{Assignment, Category, CategoryId, CharDef};
use crate::lattice::entries::FeatureStore;
use crate::lattice::matrix::Matrix;
use crate::lattice::trie::SurfaceTrie;
use crate::lattice::{DictionaryLimits, MecabDictionary};
use turbo_analysis_layers::TagSet;
use turbo_maxent::{Digest, sha256};
use turbo_unicode::WhitespaceMode;

pub(super) fn word(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}
fn size(bytes: &[u8], offset: usize) -> Result<usize, PrebuiltError> {
    usize::try_from(u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()))
        .map_err(|_| PrebuiltError::Invalid("offset exceeds address space"))
}
fn strings(mut bytes: &[u8], categories: bool) -> Result<Vec<(&str, [u8; 4])>, PrebuiltError> {
    let mut out = Vec::new();
    while !bytes.is_empty() {
        if bytes.len() < 4 || out.len() >= if categories { 17 } else { 65536 } {
            return Err(PrebuiltError::Invalid("string table count or length"));
        }
        let len = word(bytes, 0) as usize;
        bytes = &bytes[4..];
        let s = bytes.get(..len).ok_or(PrebuiltError::Invalid("short string"))?;
        let s = std::str::from_utf8(s).map_err(|_| PrebuiltError::Invalid("string UTF-8"))?;
        bytes = &bytes[len..];
        let mut flags = [0; 4];
        if categories {
            flags.copy_from_slice(bytes.get(..4).ok_or(PrebuiltError::Invalid("short category flags"))?);
            bytes = &bytes[4..];
            if flags[0] > 1 || flags[1] > 1 || flags[2] > 15 || flags[3] != 0 {
                return Err(PrebuiltError::Invalid("category flags"));
            }
        }
        out.push((s, flags));
    }
    Ok(out)
}

impl MecabDictionary {
    /// Checks a prebuilt image, then shares its large tables without copying
    /// or parsing entries. Small category and tag-label metadata is owned.
    pub fn from_prebuilt(image: PrebuiltImage, limits: DictionaryLimits) -> Result<Self, PrebuiltError> {
        let bytes = image.bytes();
        if bytes.len() < HEADER + 32 || &bytes[..8] != MAGIC || word(bytes, 8) != VERSION {
            return Err(PrebuiltError::Invalid("magic, version or short header"));
        }
        if word(bytes, 12) != 0x01020304 || cfg!(target_endian = "big") {
            return Err(PrebuiltError::Invalid("unsupported endian"));
        }
        if sha256(&bytes[..bytes.len() - 32]).0 != bytes[bytes.len() - 32..] {
            return Err(PrebuiltError::Invalid("image checksum"));
        }
        let whitespace = match word(bytes, 16) {
            0 => WhitespaceMode::Legacy,
            1 => WhitespaceMode::Unicode,
            _ => return Err(PrebuiltError::Invalid("whitespace mode")),
        };
        if bytes[DIRECTORY + SECTIONS * 16..HEADER].iter().any(|&b| b != 0) {
            return Err(PrebuiltError::Invalid("reserved header"));
        }
        let mut sections = [(0, 0); SECTIONS];
        let mut expected = HEADER;
        for (i, range) in sections.iter_mut().enumerate() {
            let at = DIRECTORY + i * 16;
            let (start, len) = (size(bytes, at)?, size(bytes, at + 8)?);
            let end = start.checked_add(len).ok_or(PrebuiltError::Invalid("section overflow"))?;
            let padded = end.checked_add(7).ok_or(PrebuiltError::Invalid("section overflow"))? & !7;
            if start != expected || padded > bytes.len() - 32 || bytes[end..padded].iter().any(|&b| b != 0) {
                return Err(PrebuiltError::Invalid("section order, length or padding"));
            }
            *range = (start, len);
            expected = padded;
        }
        if expected != bytes.len() - 32 || sections[14].1 != 0 {
            return Err(PrebuiltError::Invalid("trailing or reserved section"));
        }
        let section = |i: usize| {
            let (a, n) = sections[i];
            &bytes[a..a + n]
        };
        let u32table = |i: usize| {
            let (a, n) = sections[i];
            Table::<u32>::mapped(&image, a, n)
        };
        let words = |i: usize, width: usize, max: usize| {
            let v = u32table(i)?;
            if !v.len().is_multiple_of(width) || v.len() / width > max {
                return Err(PrebuiltError::Invalid("metadata row width or limit"));
            }
            Ok(v)
        };
        let (a, n) = sections[0];
        let matrix = Matrix { costs: Table::mapped(&image, a, n)?, rows: word(bytes, 24), cols: word(bytes, 28) };
        let (a, n) = sections[1];
        let trie = SurfaceTrie { units: Table::mapped(&image, a, n)? };
        let (a, n) = sections[3];
        let entries = Table::mapped(&image, a, n)?;
        if section(3).as_chunks::<16>().0.iter().any(|e| e[10] != 0 || e[11] != 0) {
            return Err(PrebuiltError::Invalid("entry padding"));
        }
        let unknown: Vec<_> = words(4, 2, 17)?.as_chunks::<2>().0.iter().map(|p| (p[0], p[1])).collect();
        let ranges = words(5, 3, 0x100000)?.as_chunks::<3>().0.iter().map(|p| (p[0], p[1], p[2])).collect();
        let (a, n) = sections[7];
        let features = FeatureStore { text: Text::mapped(&image, a, n)?, ends: u32table(8)?, first: u32table(9)? };
        let (a, n) = sections[10];
        let tags = Table::mapped(&image, a, n)?;
        let label_strings = strings(section(11), false)?;
        let labels: Vec<_> = label_strings.iter().map(|x| x.0).collect();
        let pos = TagSet::new("mecab", &labels);
        let assignments = words(12, 2, 17 * (1 << 17))?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| {
                let primary = u8::try_from(p[0]).map_err(|_| PrebuiltError::Invalid("category index"))?;
                Ok(Assignment { primary: CategoryId(primary), mask: p[1] })
            })
            .collect::<Result<_, PrebuiltError>>()?;
        let categories = strings(section(13), true)?
            .into_iter()
            .map(|(s, f)| Category { name: s.to_owned(), invoke: f[0] != 0, group: f[1] != 0, length: f[2] })
            .collect();
        let default = u8::try_from(word(bytes, 32)).map_err(|_| PrebuiltError::Invalid("default category"))?;
        let chars = CharDef {
            categories,
            default: CategoryId(default),
            assignments,
            fallback: word(bytes, 36),
            bmp: u32table(6)?,
            ranges,
        };
        let dict = Self {
            matrix,
            trie,
            groups: u32table(2)?,
            entries,
            lexicon_len: word(bytes, 20),
            unknown: unknown.into_boxed_slice(),
            chars,
            features,
            pos,
            tags,
            whitespace,
            sha256: Digest(bytes[40..72].try_into().unwrap()),
        };
        validate::dictionary(&dict, limits)?;
        Ok(dict)
    }
}
