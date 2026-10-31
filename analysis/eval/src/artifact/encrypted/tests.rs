use super::*;
use std::io::{Cursor, Read, Write};

struct Keys;
impl KeyProvider for Keys {
    fn active(&self) -> Result<(KeyId, KeyMaterial), Error> {
        Ok((KeyId([7; 16]), KeyMaterial::new([19; 32])))
    }
    fn resolve(&self, id: KeyId) -> Result<KeyMaterial, Error> {
        if id == KeyId([7; 16]) { Ok(KeyMaterial::new([19; 32])) } else { Err(Error::UnavailableKey) }
    }
}

fn encoded(text: &[u8]) -> Vec<u8> {
    let mut w = Encoder::new(Vec::new(), &Keys, Binding([3; 32])).unwrap();
    w.write_all(text).unwrap();
    w.finish().unwrap()
}

fn decode(bytes: Vec<u8>, binding: Binding) -> Result<Vec<u8>, Error> {
    let mut r = Decoder::new(Cursor::new(bytes), &Keys, binding)?;
    let mut text = Vec::new();
    r.read_to_end(&mut text).map_err(|_| Error::InvalidArtifact)?;
    Ok(text)
}

#[test]
fn round_trip_empty_unicode_and_multiple_frames() {
    for text in [Vec::new(), "秘密 café 🦀".as_bytes().to_vec(), vec![42; 200_000]] {
        assert_eq!(decode(encoded(&text), Binding([3; 32])).unwrap(), text);
    }
}

#[test]
fn every_truncation_and_single_byte_tamper_is_rejected() {
    let bytes = encoded(b"sensitive document source text");
    for end in 0..bytes.len() {
        assert!(decode(bytes[..end].to_vec(), Binding([3; 32])).is_err(), "truncation {end}");
    }
    for pos in 0..bytes.len() {
        let mut changed = bytes.clone();
        changed[pos] ^= 1;
        assert!(decode(changed, Binding([3; 32])).is_err(), "tamper {pos}");
    }
}

#[test]
fn bindings_randomization_and_trailing_data_fail_closed() {
    let a = encoded(b"sensitive source text");
    let b = encoded(b"sensitive source text");
    assert_ne!(a, b);
    assert!(!a.windows(21).any(|w| w == b"sensitive source text"));
    assert!(decode(a.clone(), Binding([4; 32])).is_err());
    let mut suffix = a;
    suffix.push(0);
    assert!(decode(suffix, Binding([3; 32])).is_err());
}

#[test]
fn reorder_duplicate_and_omitted_frames_are_rejected() {
    let bytes = encoded(&vec![42; 150_000]);
    // Header: 8-byte version, 16-byte key ID, 32-byte random salt.
    let first = 56;
    let second = first + 5 + 65_536 + 16;
    let third = second + 5 + 65_536 + 16;
    let mut reordered = bytes[..first].to_vec();
    reordered.extend_from_slice(&bytes[second..third]);
    reordered.extend_from_slice(&bytes[first..second]);
    reordered.extend_from_slice(&bytes[third..]);
    assert!(decode(reordered, Binding([3; 32])).is_err());
    let mut duplicated = bytes[..second].to_vec();
    duplicated.extend_from_slice(&bytes[first..]);
    assert!(decode(duplicated, Binding([3; 32])).is_err());
    let mut omitted = bytes[..first].to_vec();
    omitted.extend_from_slice(&bytes[second..]);
    assert!(decode(omitted, Binding([3; 32])).is_err());
}

#[test]
fn verified_reader_withholds_incomplete_artifacts_and_cleans_ciphertext_snapshot() {
    let root = std::env::temp_dir().join(format!("artifact-verified-test-{}", std::process::id()));
    std::fs::create_dir(&root).unwrap();
    let text = b"source data never written to snapshot";
    let bytes = encoded(text);
    assert!(
        VerifiedReader::open(Cursor::new(&bytes[..bytes.len() - 1]), &Keys, Binding([3; 32]), &root, 10_000).is_err()
    );
    assert_eq!(std::fs::read_dir(&root).unwrap().count(), 0);
    assert!(VerifiedReader::open(Cursor::new(&bytes), &Keys, Binding([3; 32]), &root, bytes.len() as u64 - 1).is_err());
    assert_eq!(std::fs::read_dir(&root).unwrap().count(), 0);
    let mut reader =
        VerifiedReader::open(Cursor::new(&bytes), &Keys, Binding([3; 32]), &root, bytes.len() as u64).unwrap();
    let stage = std::fs::read_dir(&root).unwrap().next().unwrap().unwrap().path().join("ciphertext");
    let staged = std::fs::read(stage).unwrap();
    assert_eq!(staged, bytes);
    assert!(!staged.windows(text.len()).any(|w| w == text));
    let mut got = Vec::new();
    reader.read_to_end(&mut got).unwrap();
    assert_eq!(got, text);
    drop(reader);
    assert_eq!(std::fs::read_dir(&root).unwrap().count(), 0);
    std::fs::remove_dir(root).unwrap();
}

struct WrongKeys;
impl KeyProvider for WrongKeys {
    fn active(&self) -> Result<(KeyId, KeyMaterial), Error> {
        Err(Error::UnavailableKey)
    }
    fn resolve(&self, _: KeyId) -> Result<KeyMaterial, Error> {
        Ok(KeyMaterial::new([20; 32]))
    }
}

#[test]
fn invalid_and_unavailable_keys_never_select_plaintext() {
    assert!(Encoder::new(Vec::new(), &WrongKeys, Binding([3; 32])).is_err());
    let bytes = encoded(b"secret");
    let mut wrong = Decoder::new(Cursor::new(bytes.clone()), &WrongKeys, Binding([3; 32])).unwrap();
    assert!(wrong.read_to_end(&mut Vec::new()).is_err());
    let mut missing = bytes;
    missing[8] ^= 1;
    assert!(matches!(Decoder::new(Cursor::new(missing), &Keys, Binding([3; 32])), Err(Error::UnavailableKey)));
}

struct FailsAfterHeader;
impl Write for FailsAfterHeader {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() == 56 { Ok(56) } else { Err(std::io::Error::other("injected source text must be redacted")) }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn a_failed_write_cannot_be_finalized_or_retried_with_a_reused_nonce() {
    let mut writer = Encoder::new(FailsAfterHeader, &Keys, Binding([3; 32])).unwrap();
    let error = writer.write_all(&vec![42; 65_536]).unwrap_err();
    assert!(!error.to_string().contains("injected"));
    assert!(writer.write_all(b"later").is_err());
    assert!(matches!(writer.finish(), Err(Error::Poisoned)));
}

#[test]
fn oversized_frame_lengths_are_refused_before_payload_allocation() {
    let mut bytes = encoded(b"small");
    bytes[57..61].copy_from_slice(&u32::MAX.to_be_bytes());
    assert!(decode(bytes, Binding([3; 32])).is_err());
}

#[test]
#[ignore = "explicit bounded filesystem measurement"]
fn bounded_filesystem_measurement() {
    let mib: u64 = std::env::var("ARTIFACT_MEASURE_MIB").unwrap().parse().unwrap();
    let root = std::env::temp_dir().join(format!("artifact-memory-test-{}", std::process::id()));
    std::fs::create_dir(&root).unwrap();
    let path = root.join("published");
    let file = std::fs::OpenOptions::new().write(true).create_new(true).open(&path).unwrap();
    let mut writer = Encoder::new(file, &Keys, Binding([3; 32])).unwrap();
    let block = [42; 65_536];
    for _ in 0..mib * 16 {
        writer.write_all(&block).unwrap();
    }
    writer.finish().unwrap().sync_all().unwrap();
    let file = std::fs::File::open(&path).unwrap();
    let max_bytes = file.metadata().unwrap().len();
    let mut reader = VerifiedReader::open(file, &Keys, Binding([3; 32]), &root, max_bytes).unwrap();
    // Provider replacement cannot alter the verified private snapshot.
    std::fs::remove_file(&path).unwrap();
    std::fs::write(&path, b"replacement").unwrap();
    let mut count = 0_u64;
    let mut buffer = [0; 32_768];
    loop {
        let n = reader.read(&mut buffer).unwrap();
        if n == 0 {
            break;
        }
        assert!(buffer[..n].iter().all(|&b| b == 42));
        count += n as u64;
    }
    assert_eq!(count, mib * 1024 * 1024);
    drop(reader);
    assert_eq!(std::fs::read_dir(&root).unwrap().count(), 1);
    std::fs::remove_file(path).unwrap();
    std::fs::remove_dir(root).unwrap();
}
