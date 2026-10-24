//! Real checked-in MeCab dictionaries, including malformed-image probes.
mod common;
use std::sync::Arc;
use turbo_analysis::Annotator;
use turbo_tokenize::lattice::{DictionaryLimits, PrebuiltImage};
use turbo_tokenize::{LatticeTokenizer, MecabDictionary, WhitespaceMode};

fn source() -> MecabDictionary {
    MecabDictionary::load(common::fixtures_dir().join("mecab-fixture-ja"), WhitespaceMode::Unicode).unwrap()
}
fn read(bytes: &[u8]) -> Result<MecabDictionary, turbo_tokenize::lattice::PrebuiltError> {
    MecabDictionary::from_prebuilt(PrebuiltImage::copy_from(bytes), Default::default())
}
fn section(bytes: &[u8], id: usize) -> usize {
    u64::from_le_bytes(bytes[72 + id * 16..80 + id * 16].try_into().unwrap()) as usize
}
fn signed(mut bytes: Vec<u8>) -> Vec<u8> {
    let n = bytes.len() - 32;
    let sum = turbo_maxent::sha256(&bytes[..n]);
    bytes[n..].copy_from_slice(&sum.0);
    bytes
}
fn change(bytes: &[u8], offset: usize, value: u32) -> Vec<u8> {
    let mut changed = bytes.to_vec();
    changed[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    signed(changed)
}

#[test]
fn round_trip_identity_and_shared_storage() {
    let dictionary = source();
    let bytes = dictionary.to_prebuilt().unwrap();
    let image = PrebuiltImage::copy_from(&bytes);
    let loaded = MecabDictionary::from_prebuilt(image.clone(), Default::default()).unwrap();
    assert_eq!(loaded.sha256(), dictionary.sha256());
    assert_eq!(loaded.whitespace(), dictionary.whitespace());
    assert_eq!(loaded.lexicon_len(), dictionary.lexicon_len());
    assert_eq!(loaded.to_prebuilt().unwrap(), bytes);
    let tokenizer = LatticeTokenizer::new(loaded.clone(), WhitespaceMode::Unicode);
    let expected = LatticeTokenizer::new(dictionary, WhitespaceMode::Unicode);
    assert_eq!(tokenizer.model(), expected.model());
    assert_eq!(tokenizer.config_hash(), expected.config_hash());
    let start = image.bytes().as_ptr() as usize;
    let end = start + image.bytes().len();
    drop(image);
    drop(loaded);
    for text in ["東京都に行く。", "𠮷野家でABC１２３を食べた。", "", "\u{85}東京\u{a0}大阪"] {
        let (mut a, mut b) = (Vec::new(), Vec::new());
        assert_eq!(tokenizer.analyze(text, &mut a), expected.analyze(text, &mut b));
        assert_eq!(a, b);
    }
    let mut found = Vec::new();
    tokenizer.analyze("東京", &mut found).unwrap();
    // Feature strings stay inside the image, including after the caller drops
    // its image and dictionary clones; there is no per-entry String copy.
    assert!(!found.is_empty());
    for m in found {
        for column in tokenizer.dictionary().features(m.entry).filter(|s| !s.is_empty()) {
            let at = column.as_ptr() as usize;
            assert!(at >= start && at + column.len() <= end);
        }
    }
}

#[test]
fn load_accepts_both_forms_and_checks_the_packed_mode() {
    let path = std::env::temp_dir().join(format!("turbo-lattice-{}.bin", std::process::id()));
    let dictionary = source();
    dictionary.write_prebuilt(&path).unwrap();
    let loaded = MecabDictionary::load(&path, WhitespaceMode::Unicode).unwrap();
    assert_eq!(loaded.sha256(), dictionary.sha256());
    assert!(MecabDictionary::load(&path, WhitespaceMode::Legacy).is_err());
    std::fs::remove_file(path).unwrap();
}

#[test]
fn short_images_bad_headers_and_checksums_are_errors() {
    let bytes = source().to_prebuilt().unwrap();
    for end in (0..544).chain([bytes.len() - 33, bytes.len() - 32, bytes.len() - 1]) {
        assert!(read(&bytes[..end]).is_err(), "truncated at {end}");
    }
    for offset in [0, 8, 12, 16, 20, 24, 28, 32, 36, 72, 80, 312] {
        assert!(read(&change(&bytes, offset, u32::MAX)).is_err(), "header {offset}");
    }
    let mut corrupt = bytes.clone();
    corrupt[600] ^= 1;
    assert!(read(&corrupt).is_err());
    let mut trailing = bytes.clone();
    trailing.extend_from_slice(&[0; 8]);
    assert!(read(&signed(trailing)).is_err());
}

#[test]
fn table_indices_and_utf8_are_checked_even_with_a_valid_checksum() {
    let bytes = source().to_prebuilt().unwrap();
    for (id, offset, value) in [
        (1, 4, u32::MAX),  // root parent
        (2, 0, 1),         // first group must start at entry zero
        (3, 0, u32::MAX),  // left context
        (3, 12, u32::MAX), // feature record
        (4, 0, u32::MAX),  // unknown range
        (6, 0, u32::MAX),  // BMP assignment
        (8, 0, u32::MAX),  // feature column end
        (9, 0, 1),         // first feature record
        (10, 0, u32::MAX), // tag ids
        (12, 0, u32::MAX), // assignment primary
        (12, 4, 0),        // assignment mask
    ] {
        assert!(read(&change(&bytes, section(&bytes, id) + offset, value)).is_err(), "section {id} offset {offset}");
    }
    let mut invalid_text = bytes.clone();
    invalid_text[section(&bytes, 7)] = 0xff;
    assert!(read(&signed(invalid_text)).is_err());
    let image = PrebuiltImage::copy_from(&bytes);
    assert!(
        MecabDictionary::from_prebuilt(image.clone(), DictionaryLimits { max_entries: 1, max_matrix_cells: 1 })
            .is_err()
    );
    assert!(
        MecabDictionary::from_prebuilt(image, DictionaryLimits { max_entries: usize::MAX, max_matrix_cells: 0 })
            .is_err()
    );
}

#[test]
fn mutated_runtime_tables_never_panic() {
    let bytes = source().to_prebuilt().unwrap();
    // Mutations of actual runtime records, re-signed to reach structural checks.
    let mut state = 0x92123u64;
    for i in 0..128 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let at = if i < 64 {
            section(&bytes, 1) + (state as usize % 64) * 4
        } else {
            512 + state as usize % (bytes.len() - 544)
        };
        let mut changed = bytes.clone();
        changed[at] ^= 1 << (i % 8);
        if let Ok(dictionary) = read(&signed(changed)) {
            let tokenizer = LatticeTokenizer::new(dictionary, WhitespaceMode::Unicode);
            let _ = tokenizer.analyze("東京都に行く。ABC１２３𠮷野家", &mut Vec::new());
        }
    }
}

#[test]
fn unaligned_external_storage_is_checked() {
    struct Offset(Vec<u8>, usize);
    impl AsRef<[u8]> for Offset {
        fn as_ref(&self) -> &[u8] {
            &self.0[self.1..]
        }
    }
    let bytes = source().to_prebuilt().unwrap();
    let mut storage = vec![0; bytes.len() + 8];
    let offset = (0..8).find(|&i| !(storage.as_ptr() as usize + i).is_multiple_of(8)).unwrap();
    storage.truncate(bytes.len() + offset);
    storage[offset..].copy_from_slice(&bytes);
    // SAFETY: this owner retains one immutable allocation and a fixed slice.
    let image = unsafe { PrebuiltImage::from_owner(Arc::new(Offset(storage, offset))) };
    assert!(MecabDictionary::from_prebuilt(image, Default::default()).is_err());
}

#[test]
fn signed_trie_cycles_and_invalid_utf8_edges_are_rejected() {
    let bytes = source().to_prebuilt().unwrap();
    let start = section(&bytes, 1);
    let stop = section(&bytes, 2);
    let word = |at| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    let free = (2..(stop - start) / 8).find(|&i| word(start + i * 8 + 4) == u32::MAX).unwrap();
    let mut cycle = bytes.clone();
    cycle[start + free * 8..start + free * 8 + 4].copy_from_slice(&(free as u32 - 1).to_le_bytes());
    cycle[start + free * 8 + 4..start + free * 8 + 8].copy_from_slice(&(free as u32).to_le_bytes());
    assert!(read(&signed(cycle)).unwrap_err().to_string().contains("trie cycle"));

    // A UTF-8 root cannot have the byte FF as a child. Reuse that unused slot
    // of the real dictionary, while keeping every existing surface group.
    let at = start + (word(start) as usize + 256) * 8;
    assert_eq!(word(at + 4), u32::MAX);
    let mut invalid = bytes.clone();
    invalid[at..at + 8].fill(0);
    assert!(read(&signed(invalid)).unwrap_err().to_string().contains("trie UTF-8"));
}
