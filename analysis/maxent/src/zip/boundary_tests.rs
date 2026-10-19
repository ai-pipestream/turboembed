use super::*;

#[test]
fn central_directory_size_is_enforced() {
    let mut z = stored_archive("a", b"x", b"", b"");
    let end = z.len() - END_LEN;
    z[end + 12..end + 16].copy_from_slice(&0u32.to_le_bytes());
    assert!(Zip::new(z).is_err());
}

#[test]
fn central_directory_disk_start_is_enforced() {
    let mut z = stored_archive("a", b"x", b"", b"");
    let end = z.len() - END_LEN;
    let central = u32_at(&z, end + 16) as usize;
    z[central + 34..central + 36].copy_from_slice(&1u16.to_le_bytes());
    assert!(Zip::new(z).is_err());
}

#[test]
fn local_header_metadata_must_match_directory() {
    for (at, value) in [(6, 1), (8, 8), (14, 0), (18, 0), (22, 0), (30, b'b')] {
        let mut z = stored_archive("a", b"x", b"", b"");
        z[at] = value;
        assert!(Zip::new(z).unwrap().entry("a").is_err(), "accepted mismatch at {at}");
    }
}

#[test]
fn descriptor_entries_can_omit_local_crc_and_sizes() {
    let mut z = stored_archive("a", b"x", b"", b"");
    let end = z.len() - END_LEN;
    let central = u32_at(&z, end + 16) as usize;
    z[6] = 8;
    z[central + 8] = 8;
    z[14..26].fill(0);
    assert_eq!(&*Zip::new(z).unwrap().entry("a").unwrap(), b"x");
}

#[test]
fn entry_payload_must_end_before_the_directory() {
    let mut z = stored_archive("a", b"x", b"", b"");
    let end = z.len() - END_LEN;
    let central = u32_at(&z, end + 16) as usize;
    let crc = crc32(&z[31..33]);
    for base in [0, central] {
        let offset = if base == 0 { 14 } else { 16 };
        z[base + offset..base + offset + 4].copy_from_slice(&crc.to_le_bytes());
        z[base + offset + 4..base + offset + 8].copy_from_slice(&2u32.to_le_bytes());
        z[base + offset + 8..base + offset + 12].copy_from_slice(&2u32.to_le_bytes());
    }
    assert!(Zip::new(z).unwrap().entry("a").is_err());
}
