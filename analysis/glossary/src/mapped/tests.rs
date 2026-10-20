//! Corrupted images: refused at load with a typed error, or, where the
//! change still makes a well-formed automaton, matched without a panic.

use super::*;
use crate::{Boundaries, Glossary, MappedGlossaryWriter, Scratch};

fn glossary() -> Glossary {
    let mut g = Glossary::new();
    for (i, t) in ["New York", "York", "ork", "東京", "Größe", "🙂 ok", "he", "she", "hers", "his"].iter().enumerate()
    {
        g.push(&format!("id{i}"), t).unwrap();
    }
    g
}

fn image() -> Vec<u8> {
    MappedGlossaryWriter::write(&glossary(), Case::Insensitive).unwrap()
}

/// `bytes` copied into an aligned buffer, checksum resealed.
fn sealed(bytes: &[u8]) -> ImageBuf {
    let mut b = bytes.to_vec();
    if b.len() >= HEADER && b.len().is_multiple_of(8) {
        let sum = checksum(&b);
        b[48..56].copy_from_slice(&sum.to_le_bytes());
    }
    ImageBuf::copy_from(&b)
}

const TEXTS: [&str; 3] = ["ushers in NEW YORK and 東京", "GRÖSSE größe 🙂 OK his hers", ""];

/// Every hit inside the text and naming an entry of the glossary.
fn exercise(m: &MappedGlossaryMatcher<'_>) {
    let mut scratch = Scratch::new();
    for text in TEXTS {
        m.scan(text, &mut scratch, |s, e| {
            assert!(s.start <= s.end && s.end as usize <= text.len());
            assert!((e.0 as usize) < m.entries());
            assert!(m.id(e).is_some() && m.term(e).is_some());
        });
        let mut out = Vec::new();
        m.find(text, &Boundaries::from_offsets(text.len() as u32, 0..=text.len() as u32), &mut scratch, &mut out);
    }
}

#[test]
fn every_truncation_is_refused() {
    let img = image();
    for n in 0..img.len() {
        let b = ImageBuf::copy_from(&img[..n]);
        let e = MappedGlossaryMatcher::from_bytes(b.bytes()).unwrap_err();
        assert!(matches!(e, ImageError::Short { .. } | ImageError::Length { .. }), "{n}: {e:?}");
    }
    let mut longer = img.clone();
    longer.extend_from_slice(&[0; 8]);
    assert!(matches!(MappedGlossaryMatcher::from_bytes(sealed(&longer).bytes()), Err(ImageError::Length { .. })));
}

#[test]
fn every_flipped_byte_is_refused() {
    let img = image();
    for i in 0..img.len() {
        for bit in [0x01u8, 0x80] {
            let mut b = img.clone();
            b[i] ^= bit;
            let b = ImageBuf::copy_from(&b);
            assert!(
                MappedGlossaryMatcher::from_bytes(b.bytes()).is_err(),
                "byte {i} bit {bit:#x} flipped and accepted"
            );
        }
    }
}

#[test]
fn resealed_changes_to_any_word_are_refused_or_safe() {
    let img = image();
    let (mut refused, mut accepted) = (0, 0);
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    for at in (HEADER..img.len()).step_by(4) {
        let old = u32::from_le_bytes(img[at..at + 4].try_into().unwrap());
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        for new in [0, 1, 2, old.wrapping_add(1), old.wrapping_sub(1), u32::MAX, u32::MAX - 1, (seed >> 32) as u32] {
            if new == old {
                continue;
            }
            let mut b = img.clone();
            b[at..at + 4].copy_from_slice(&new.to_le_bytes());
            let b = sealed(&b);
            match MappedGlossaryMatcher::from_bytes(b.bytes()) {
                Ok(m) => {
                    exercise(&m);
                    accepted += 1;
                }
                Err(ImageError::Table { .. }) => refused += 1,
                Err(e) => panic!("word at {at} set to {new}: {e:?}"),
            }
        }
    }
    assert!(refused > 1000 && accepted > 1000, "{refused} refused, {accepted} accepted");
}

#[test]
fn header_words_are_checked() {
    let img = image();
    let set = |at: usize, v: u32| {
        let mut b = img.clone();
        b[at..at + 4].copy_from_slice(&v.to_le_bytes());
        MappedGlossaryMatcher::from_bytes(sealed(&b).bytes()).err()
    };
    assert_eq!(set(0, 0), Some(ImageError::Magic));
    assert_eq!(set(8, 2), Some(ImageError::Version(2)));
    assert_eq!(set(12, 0x0403_0201), Some(ImageError::Endian));
    assert_eq!(set(16, 2), Some(ImageError::Case(2)));
    for (at, v) in [(20, 1u32), (24, 99), (32, 7), (40, 3), (20, u32::MAX), (24, u32::MAX)] {
        assert!(matches!(set(at, v), Some(ImageError::Length { .. })), "header word at {at} = {v}");
    }
    assert_eq!(set(44, 1), Some(ImageError::Table { table: "reserved header word", at: 44 }));
    // Fewer codes than the tables use.
    assert!(matches!(set(36, 1), Some(ImageError::Table { table: "ASCII code" | "page code", .. })));
    // The longest pattern: shorter than a cell's depth is refused, longer is
    // a wider ring and still matches the same.
    assert!(matches!(set(28, 2), Some(ImageError::Table { table: "cell base or depth", .. })));
    assert_eq!(set(28, 1000), None);
}

#[test]
fn broken_links_are_refused() {
    let img = image();
    let buf = ImageBuf::copy_from(&img);
    let m = MappedGlossaryMatcher::from_bytes(buf.bytes()).unwrap();
    let cells_at = HEADER + 4 * (128 + PAGE_INDEX + 256 * m.counts.pages as usize);
    // The deepest used cell, and one at depth 1.
    let deep = (0..m.cells.len()).max_by_key(|&i| (m.cells[i].check != FREE, m.cells[i].depth)).unwrap();
    let shallow = (1..m.cells.len()).find(|&i| m.cells[i].check != FREE && m.cells[i].depth == 1).unwrap();
    let with = |cell: usize, field: usize, v: u32| {
        let mut b = img.clone();
        let at = cells_at + 4 * (CELL_WORDS * cell + field);
        b[at..at + 4].copy_from_slice(&v.to_le_bytes());
        match MappedGlossaryMatcher::from_bytes(sealed(&b).bytes()) {
            Err(ImageError::Table { table, .. }) => table,
            other => panic!("cell {cell} field {field} = {v}: {other:?}"),
        }
    };
    // A failure link to a deeper cell would let a walk climb forever.
    assert_eq!(with(shallow, 2, deep as u32), "cell failure link");
    assert_eq!(with(deep, 2, deep as u32), "cell failure link");
    assert_eq!(with(deep, 3, m.entries() as u32), "cell entry");
    assert_eq!(with(shallow, 4, deep as u32), "cell output link");
    assert_eq!(with(deep, 5, 1), "cell edge");
    assert_eq!(with(deep, 1, m.cells.len() as u32), "cell parent");
    assert_eq!(with(0, 1, 0), "root cell");
    assert_eq!(with(0, 2, 1), "root cell");
    // An entry at the root would be a hit of no characters.
    assert_eq!(with(0, 3, 0), "root cell");
}
