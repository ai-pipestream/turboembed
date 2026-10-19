use super::*;
use crate::entry::RegionCode;
use crate::gazetteer::Gazetteer;

const TABLE: &str = "q1\tQ\tAlt One|Alt Two\t1\t2\tDE\tCITY\t10\t\tBY\tka=1\tkb=2\n\
                         q2\tBerlin\t\t52.5\t13.4\tDE\tCITY\t3000000\n";

fn snapshot() -> Vec<u8> {
    let g = UserGazetteer::from_bytes(TABLE.as_bytes(), "sites").unwrap();
    let mut bytes = Vec::new();
    g.write_snapshot(&mut bytes).unwrap();
    bytes
}

/// `bytes` with its checksum written again, as an altered file whose
/// author fixed the checksum would be.
fn resum(mut bytes: Vec<u8>) -> Vec<u8> {
    let body = bytes.len() - 8;
    let sum = checksum(&bytes[..body]);
    bytes[body..].copy_from_slice(&sum.to_le_bytes());
    bytes
}

/// Replaces the only occurrence of `from` in `bytes` by `to`.
fn replace(bytes: &[u8], from: &[u8], to: &[u8]) -> Vec<u8> {
    let at = bytes.windows(from.len()).position(|w| w == from).unwrap();
    assert!(bytes[at + 1..].windows(from.len()).all(|w| w != from), "{from:?} occurs once");
    let mut out = bytes.to_vec();
    out[at..at + from.len()].copy_from_slice(to);
    out
}

#[test]
fn a_damaged_byte_fails_the_checksum() {
    let bytes = snapshot();
    assert!(UserGazetteer::from_snapshot(&bytes).is_ok());
    let mut damaged = bytes.clone();
    damaged[bytes.len() / 2] ^= 1;
    assert!(matches!(UserGazetteer::from_snapshot(&damaged), Err(SnapshotError::Corrupt(_))));
    assert!(matches!(UserGazetteer::from_snapshot(&bytes[..20]), Err(SnapshotError::Truncated)));
}

#[test]
fn records_a_gazetteer_entry_could_not_hold_are_refused() {
    let bytes = snapshot();
    let blank_name = resum(replace(&bytes, b"q1Q", b"q1 "));
    assert!(matches!(
        UserGazetteer::from_snapshot(&blank_name),
        Err(SnapshotError::Corrupt("blank record strings or repeated attribute keys"))
    ));
    let blank_alternate = resum(replace(&bytes, b"Alt Two", b"       "));
    assert!(UserGazetteer::from_snapshot(&blank_alternate).is_err());
    let repeated_key = resum(replace(&bytes, b"kb", b"ka"));
    assert!(matches!(
        UserGazetteer::from_snapshot(&repeated_key),
        Err(SnapshotError::Corrupt("blank record strings or repeated attribute keys"))
    ));
    let blank_value = resum(replace(&bytes, b"ka1", b"ka "));
    assert!(matches!(
        UserGazetteer::from_snapshot(&blank_value),
        Err(SnapshotError::Corrupt("blank attribute keys or values"))
    ));
    let blank_class = resum(replace(&bytes, b"CITY", b"    "));
    assert!(matches!(
        UserGazetteer::from_snapshot(&blank_class),
        Err(SnapshotError::Corrupt("blank sources or feature classes"))
    ));
}

/// Every single byte change with a matching checksum is refused or
/// gives a gazetteer whose queries and records read without panicking.
#[test]
fn altered_bytes_with_a_matching_checksum_never_panic() {
    let bytes = snapshot();
    for i in 13..bytes.len() - 8 {
        for flip in [0x01, 0x5A, 0xFF] {
            let mut altered = bytes.clone();
            altered[i] ^= flip;
            let Ok(g) = UserGazetteer::from_snapshot(&resum(altered)) else { continue };
            for q in ["Q", "alt one", "Berlin", "nowhere"] {
                for e in g.lookup(q) {
                    let _ = e.to_entry();
                }
            }
            for id in ["q1", "q2"] {
                let _ = g.by_id("sites", id).map(|e| e.to_entry());
            }
            let _ = g.by_region(RegionCode::parse("DE").unwrap()).map(|e| e.to_entry());
        }
    }
}
