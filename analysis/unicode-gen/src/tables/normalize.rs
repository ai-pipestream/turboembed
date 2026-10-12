//! `normalize.rs`: the data of the Unicode normalization forms (UAX #15).
//! Combining classes, quick check values and decomposition flags are
//! packed into one `u32` per code point; decompositions and compositions
//! are sorted maps.

use std::collections::BTreeMap;

use super::{Inputs, emit_map, emit_trie};
use crate::emit::{RustFile, char_lit, dec};
use crate::properties::{binary, valued};
use crate::trie::Trie;
use crate::ucd::{CODE_POINTS, to_char};

// Hangul syllables decompose algorithmically (Unicode section 3.12).
const S_BASE: u32 = 0xAC00;
const L_BASE: u32 = 0x1100;
const V_BASE: u32 = 0x1161;
const T_BASE: u32 = 0x11A7;
const T_COUNT: u32 = 28;
const N_COUNT: u32 = 21 * T_COUNT;
const S_COUNT: u32 = 19 * N_COUNT;

/// Bit positions of the packed per-code-point value.
const QC_SHIFTS: [(&str, &str, u32); 4] = [
    ("NFC_QC", "NFC_QC_SHIFT", 8),
    ("NFD_QC", "NFD_QC_SHIFT", 10),
    ("NFKC_QC", "NFKC_QC_SHIFT", 12),
    ("NFKD_QC", "NFKD_QC_SHIFT", 14),
];
const HAS_CANONICAL: u32 = 1 << 16;
const HAS_COMPATIBILITY: u32 = 1 << 17;
const COMPOSES_FIRST: u32 = 1 << 18;
const COMPOSES_SECOND: u32 = 1 << 19;

pub fn render(inputs: &Inputs) -> String {
    let props = inputs.file("DerivedNormalizationProps.txt");
    let excluded = binary(&props, &["Full_Composition_Exclusion"]).remove(0);
    let compositions = compositions(inputs, &excluded);
    let mut values: Vec<u32> = (0..CODE_POINTS as u32).map(|cp| u32::from(inputs.data.combining_class(cp))).collect();
    for (property, _, shift) in QC_SHIFTS {
        for (range, value) in valued(&props, property) {
            let code = match value.as_str() {
                "N" => 1,
                "M" => 2,
                _ => panic!("DerivedNormalizationProps.txt: unknown {property} value {value}"),
            };
            for cp in range {
                values[cp as usize] |= code << shift;
            }
        }
    }
    for (cp, value) in values.iter_mut().enumerate() {
        match inputs.data.decomposition(cp as u32) {
            Some(d) if d.compatibility => *value |= HAS_COMPATIBILITY,
            Some(_) => *value |= HAS_CANONICAL,
            None => {}
        }
    }
    for &(first, second, _) in &compositions {
        values[first as usize] |= COMPOSES_FIRST;
        values[second as usize] |= COMPOSES_SECOND;
    }
    let trie = Trie::build(&values);

    let mut file = RustFile::new(
        "Normalization data. Each code point has a u32: bits 0 to 7 the canonical combining class, then two bits \
         of quick check value per form (0 Yes, 1 No, 2 Maybe), then flags for a single-level canonical or \
         compatibility decomposition and for being the first or second character of a primary composite. \
         Decompositions are stored once in POOL and referred to by slots (offset << 8 | length). Hangul syllables \
         are not in these tables: they decompose and compose algorithmically.",
        &["UnicodeData.txt", "DerivedNormalizationProps.txt"],
        inputs.version,
    );
    emit_constants(&mut file);
    let mut pool = Pool::default();
    let single = |compatibility: bool| -> Vec<(u32, Vec<u32>)> {
        (0..CODE_POINTS as u32)
            .filter_map(|cp| {
                let d = inputs.data.decomposition(cp)?;
                (d.compatibility == compatibility).then(|| (cp, d.mapping.clone()))
            })
            .collect()
    };
    let full = |compatibility: bool| -> Vec<(u32, Vec<u32>)> {
        (0..CODE_POINTS as u32)
            .filter(|&cp| !is_hangul_syllable(cp))
            .filter_map(|cp| {
                let decomposed = full_decomposition(inputs, cp, compatibility);
                (decomposed != [cp]).then_some((cp, decomposed))
            })
            .collect()
    };
    let maps = [
        (
            "CANONICAL",
            ["The code points with a canonical decomposition mapping.", "Their canonical mappings, one level deep."],
            single(false),
        ),
        (
            "COMPATIBILITY",
            [
                "The code points with a compatibility (tagged) decomposition mapping.",
                "Their compatibility mappings, one level deep.",
            ],
            single(true),
        ),
        (
            "FULL_CANONICAL",
            [
                "The code points whose full canonical decomposition is not themselves (Hangul syllables aside).",
                "Their full canonical decompositions, before canonical ordering.",
            ],
            full(false),
        ),
        (
            "FULL_COMPATIBILITY",
            [
                "The code points whose full compatibility decomposition is not themselves (Hangul syllables aside).",
                "Their full compatibility decompositions, before canonical ordering.",
            ],
            full(true),
        ),
    ];
    for (prefix, docs, entries) in &maps {
        let slots: Vec<(u32, String)> = entries.iter().map(|(cp, mapping)| (*cp, dec(pool.slot(mapping)))).collect();
        emit_map(&mut file, prefix, *docs, "u32", &slots);
    }
    let chars: Vec<String> = pool.chars.iter().map(|&cp| char_lit(to_char(cp))).collect();
    file.blank();
    file.array("Every decomposition of the four maps above, each stored once.", "POOL", "char", &chars);
    emit_compositions(&mut file, &compositions);
    emit_trie(&mut file, "the packed normalization value", "u32", &trie);
    file.finish()
}

fn emit_constants(file: &mut RustFile) {
    file.blank();
    file.constant("The bits of the canonical combining class.", "CCC_MASK", "u32", "0xFF");
    for (property, constant, shift) in QC_SHIFTS {
        file.blank();
        file.constant(&format!("The position of the two {property} bits."), constant, "u32", &dec(shift));
    }
    for (name, doc, value) in [
        ("QC_NO", "The quick check value No.", "1"),
        ("QC_MAYBE", "The quick check value Maybe; 0 is Yes.", "2"),
        ("HAS_CANONICAL", "The code point has an entry in CANONICAL and FULL_CANONICAL.", "1 << 16"),
        ("HAS_COMPATIBILITY", "The code point has an entry in COMPATIBILITY.", "1 << 17"),
        ("COMPOSES_FIRST", "The code point is the first character of a pair in COMPOSITION_KEYS.", "1 << 18"),
        ("COMPOSES_SECOND", "The code point is the second character of a pair in COMPOSITION_KEYS.", "1 << 19"),
    ] {
        file.blank();
        file.constant(doc, name, "u32", value);
    }
    debug_assert_eq!(HAS_CANONICAL | HAS_COMPATIBILITY | COMPOSES_FIRST | COMPOSES_SECOND, 0xF << 16);
}

fn emit_compositions(file: &mut RustFile, compositions: &[(u32, u32, u32)]) {
    let keys: Vec<String> = compositions
        .iter()
        .map(|&(first, second, _)| format!("0x{:X}", (u64::from(first) << 32) | u64::from(second)))
        .collect();
    let values: Vec<String> = compositions.iter().map(|&(_, _, composite)| char_lit(to_char(composite))).collect();
    file.blank();
    file.array(
        "The pairs that compose to a primary composite, as first << 32 | second, in ascending order. Pairs of \
         characters with Full_Composition_Exclusion are left out.",
        "COMPOSITION_KEYS",
        "u64",
        &keys,
    );
    file.blank();
    file.array("The primary composite of each pair of COMPOSITION_KEYS.", "COMPOSITION_VALUES", "char", &values);
}

/// The canonical decompositions of two characters that are not excluded
/// from composition: (first, second, composite), sorted by the pair.
fn compositions(inputs: &Inputs, excluded: &[bool]) -> Vec<(u32, u32, u32)> {
    let mut pairs = BTreeMap::new();
    for cp in 0..CODE_POINTS as u32 {
        let Some(d) = inputs.data.decomposition(cp) else { continue };
        if d.compatibility || excluded[cp as usize] {
            continue;
        }
        match d.mapping[..] {
            [first, second] => {
                if pairs.insert((first, second), cp).is_some() {
                    panic!("two primary composites decompose to U+{first:04X} U+{second:04X}");
                }
            }
            _ => panic!("U+{cp:04X} is a primary composite whose decomposition is not a pair"),
        }
    }
    pairs.into_iter().map(|((first, second), composite)| (first, second, composite)).collect()
}

fn is_hangul_syllable(cp: u32) -> bool {
    (S_BASE..S_BASE + S_COUNT).contains(&cp)
}

/// The full decomposition of `cp`: its mapping, applied again to each
/// result until nothing decomposes. Compatibility mappings are applied
/// only when `compatibility` is true; Hangul syllables use the algorithm.
fn full_decomposition(inputs: &Inputs, cp: u32, compatibility: bool) -> Vec<u32> {
    if is_hangul_syllable(cp) {
        let index = cp - S_BASE;
        let mut jamo = vec![L_BASE + index / N_COUNT, V_BASE + (index % N_COUNT) / T_COUNT];
        if !index.is_multiple_of(T_COUNT) {
            jamo.push(T_BASE + index % T_COUNT);
        }
        return jamo;
    }
    match inputs.data.decomposition(cp) {
        Some(d) if compatibility || !d.compatibility => {
            d.mapping.iter().flat_map(|&part| full_decomposition(inputs, part, compatibility)).collect()
        }
        _ => vec![cp],
    }
}

/// The shared store of decomposition strings.
#[derive(Default)]
struct Pool {
    chars: Vec<u32>,
    offsets: BTreeMap<Vec<u32>, usize>,
}

impl Pool {
    /// The slot of `mapping`, adding it to the pool on first use.
    fn slot(&mut self, mapping: &[u32]) -> u32 {
        let offset = match self.offsets.get(mapping) {
            Some(&offset) => offset,
            None => {
                let offset = self.chars.len();
                self.chars.extend_from_slice(mapping);
                self.offsets.insert(mapping.to_vec(), offset);
                offset
            }
        };
        assert!(mapping.len() < 256 && offset < 1 << 24, "decomposition slot out of range");
        ((offset as u32) << 8) | mapping.len() as u32
    }
}
