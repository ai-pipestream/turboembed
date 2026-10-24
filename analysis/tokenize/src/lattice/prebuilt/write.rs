use super::{DIRECTORY, HEADER, MAGIC, PrebuiltError, SECTIONS, VERSION};
use crate::lattice::MecabDictionary;
use turbo_analysis_layers::Tag;
use turbo_maxent::sha256;
use turbo_unicode::WhitespaceMode;

fn u32s(values: impl Iterator<Item = u32>) -> Vec<u8> {
    values.flat_map(u32::to_le_bytes).collect()
}
fn string(out: &mut Vec<u8>, s: &str) -> Result<(), PrebuiltError> {
    let len = u32::try_from(s.len()).map_err(|_| PrebuiltError::Invalid("string too long"))?;
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(s.as_bytes());
    Ok(())
}

impl MecabDictionary {
    /// Compiles the loaded dictionary to format version 1. This does not
    /// serialize Rust padding, pointers or platform-dependent integers.
    pub fn to_prebuilt(&self) -> Result<Vec<u8>, PrebuiltError> {
        let mut sections: [Vec<u8>; SECTIONS] = std::array::from_fn(|_| Vec::new());
        sections[0] = self.matrix.costs.iter().flat_map(|n| n.to_le_bytes()).collect();
        sections[1] = u32s(self.trie.units.iter().flat_map(|u| [u.base, u.check]));
        sections[2] = u32s(self.groups.iter().copied());
        for e in self.entries.iter() {
            let out = &mut sections[3];
            out.extend_from_slice(&e.left.to_le_bytes());
            out.extend_from_slice(&e.right.to_le_bytes());
            out.extend_from_slice(&e.cost.to_le_bytes());
            out.extend_from_slice(&[0, 0]);
            out.extend_from_slice(&e.features.to_le_bytes());
        }
        sections[4] = u32s(self.unknown.iter().flat_map(|&(a, b)| [a, b]));
        // Supplementary ranges and assignments are small metadata tables.
        sections[5] = u32s(self.chars.ranges.iter().flat_map(|&(a, b, c)| [a, b, c]));
        sections[6] = u32s(self.chars.bmp.iter().copied());
        sections[7] = self.features.text.as_bytes().to_vec();
        sections[8] = u32s(self.features.ends.iter().copied());
        sections[9] = u32s(self.features.first.iter().copied());
        sections[10] = self.tags.iter().flat_map(|n| n.to_le_bytes()).collect();
        for i in 0..self.pos.len() {
            string(&mut sections[11], self.pos.label(Tag(i as u16)))?;
        }
        sections[12] = u32s(self.chars.assignments.iter().flat_map(|a| [u32::from(a.primary.0), a.mask]));
        for c in &self.chars.categories {
            string(&mut sections[13], &c.name)?;
            sections[13].extend_from_slice(&[u8::from(c.invoke), u8::from(c.group), c.length, 0]);
        }
        // Section 14 is reserved and empty in version 1.
        let mut out = vec![0; HEADER];
        out[..8].copy_from_slice(MAGIC);
        let mode = match self.whitespace {
            WhitespaceMode::Legacy => 0,
            WhitespaceMode::Unicode => 1,
        };
        let header = [
            VERSION,
            0x01020304,
            mode,
            self.lexicon_len,
            self.matrix.rows(),
            self.matrix.cols(),
            u32::from(self.chars.default.0),
            self.chars.fallback,
        ];
        out[8..40].copy_from_slice(&u32s(header.into_iter()));
        out[40..72].copy_from_slice(&self.sha256.0);
        for (i, section) in sections.iter().enumerate() {
            let start = out.len() as u64;
            let len = section.len() as u64;
            let at = DIRECTORY + i * 16;
            out[at..at + 8].copy_from_slice(&start.to_le_bytes());
            out[at + 8..at + 16].copy_from_slice(&len.to_le_bytes());
            out.extend_from_slice(section);
            out.resize(out.len().next_multiple_of(8), 0);
        }
        out.extend_from_slice(&sha256(&out).0);
        Ok(out)
    }
}
