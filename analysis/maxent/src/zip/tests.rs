/// Builds a one-entry stored archive by hand, for tests here and in the
/// modules that read archives.
#[cfg(test)]
pub(crate) fn stored_archive(name: &str, data: &[u8], local_extra: &[u8], comment: &[u8]) -> Vec<u8> {
    let mut z = Vec::new();
    z.extend_from_slice(&LOCAL_SIGNATURE.to_le_bytes());
    z.extend_from_slice(&[20, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    z.extend_from_slice(&crc32(data).to_le_bytes());
    z.extend_from_slice(&(data.len() as u32).to_le_bytes());
    z.extend_from_slice(&(data.len() as u32).to_le_bytes());
    z.extend_from_slice(&(name.len() as u16).to_le_bytes());
    z.extend_from_slice(&(local_extra.len() as u16).to_le_bytes());
    z.extend_from_slice(name.as_bytes());
    z.extend_from_slice(local_extra);
    z.extend_from_slice(data);
    let dir = z.len();
    z.extend_from_slice(&CENTRAL_SIGNATURE.to_le_bytes());
    z.extend_from_slice(&[20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    z.extend_from_slice(&crc32(data).to_le_bytes());
    z.extend_from_slice(&(data.len() as u32).to_le_bytes());
    z.extend_from_slice(&(data.len() as u32).to_le_bytes());
    z.extend_from_slice(&(name.len() as u16).to_le_bytes());
    z.extend_from_slice(&[0; 12]);
    z.extend_from_slice(&0u32.to_le_bytes());
    z.extend_from_slice(name.as_bytes());
    let dir_len = z.len() - dir;
    z.extend_from_slice(&END_SIGNATURE.to_le_bytes());
    z.extend_from_slice(&[0, 0, 0, 0, 1, 0, 1, 0]);
    z.extend_from_slice(&(dir_len as u32).to_le_bytes());
    z.extend_from_slice(&(dir as u32).to_le_bytes());
    z.extend_from_slice(&(comment.len() as u16).to_le_bytes());
    z.extend_from_slice(comment);
    z
}

use super::*;

/// The offsets of the central record and the end record in an archive
/// `stored_archive` built without a comment.
fn offsets(z: &[u8]) -> (usize, usize) {
    let end = z.len() - END_LEN;
    (u32_at(z, end + 16) as usize, end)
}

fn put_u16(z: &mut [u8], at: usize, v: u16) {
    z[at..at + 2].copy_from_slice(&v.to_le_bytes());
}

fn put_u32(z: &mut [u8], at: usize, v: u32) {
    z[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

/// The message of the zip error opening `z` gives.
fn open_error(z: Vec<u8>) -> &'static str {
    match Zip::new(z) {
        Err(Error::Zip { message, .. }) => message,
        other => panic!("expected a zip error, got {other:?}"),
    }
}

/// The message of the zip error reading entry "a" of `z` gives.
fn entry_error(z: Vec<u8>) -> &'static str {
    match Zip::new(z).unwrap().entry("a") {
        Err(Error::Entry { entry, error }) => match *error {
            Error::Zip { message, .. } if &*entry == "a" => message,
            other => panic!("expected a zip error, got {other:?}"),
        },
        other => panic!("expected an entry error, got {other:?}"),
    }
}

#[test]
fn two_entries_with_the_same_name() {
    let mut z = stored_archive("a", b"x", b"", b"");
    let (central, end) = offsets(&z);
    let record = z[central..end].to_vec();
    z.splice(end..end, record.iter().copied());
    let end = end + record.len();
    put_u16(&mut z, end + 8, 2);
    put_u16(&mut z, end + 10, 2);
    put_u32(&mut z, end + 12, 2 * record.len() as u32);
    assert_eq!(open_error(z), "two entries have the same name");
}

#[test]
fn split_archives() {
    for (field, value) in [(4, 1), (6, 1), (8, 0)] {
        let mut z = stored_archive("a", b"x", b"", b"");
        let (_, end) = offsets(&z);
        put_u16(&mut z, end + field, value);
        assert_eq!(open_error(z), "archives split over several disks are not supported", "field {field}");
    }
}

#[test]
fn encrypted_entries() {
    let mut z = stored_archive("a", b"x", b"", b"");
    let (central, _) = offsets(&z);
    put_u16(&mut z, central + 8, 1);
    assert_eq!(open_error(z), "encrypted entries are not supported");
}

#[test]
fn unsupported_compression_methods() {
    let mut z = stored_archive("a", b"x", b"", b"");
    let (central, _) = offsets(&z);
    put_u16(&mut z, central + 10, 12); // bzip2
    assert_eq!(open_error(z), "compression methods other than stored and deflated are not supported");
}

#[test]
fn a_central_directory_outside_the_archive() {
    let mut z = stored_archive("a", b"x", b"", b"");
    let (_, end) = offsets(&z);
    put_u32(&mut z, end + 16, end as u32 + 1);
    assert_eq!(open_error(z), "the central directory lies outside the archive");
    let mut z = stored_archive("a", b"x", b"", b"");
    let (_, end) = offsets(&z);
    put_u32(&mut z, end + 12, 0xFFFF_FFF0);
    assert_eq!(open_error(z), "the central directory lies outside the archive");
}

#[test]
fn a_central_record_cut_off_by_the_end_record() {
    // A second record is counted but the directory ends after the
    // first: its fixed part would overlap the end record.
    let mut z = stored_archive("a", b"x", b"", b"");
    let (_, end) = offsets(&z);
    put_u16(&mut z, end + 8, 2);
    put_u16(&mut z, end + 10, 2);
    assert_eq!(open_error(z), "a central directory record is cut off");
}

#[test]
fn a_central_record_whose_name_runs_into_the_end_record() {
    let mut z = stored_archive("a", b"x", b"", b"");
    let (central, _) = offsets(&z);
    put_u16(&mut z, central + 28, 200);
    assert_eq!(open_error(z), "a central directory record is cut off");
}

#[test]
fn a_local_header_cut_off() {
    let mut z = stored_archive("a", b"x", b"", b"");
    let (central, _) = offsets(&z);
    let at = z.len() as u32 - 10;
    put_u32(&mut z, central + 42, at);
    assert_eq!(entry_error(z), "a local header is cut off");
    // An offset near u32::MAX, where the header's end would wrap on a
    // 32-bit target.
    let mut z = stored_archive("a", b"x", b"", b"");
    put_u32(&mut z, central + 42, 0xFFFF_FFF0);
    assert_eq!(entry_error(z), "a local header is cut off");
}

#[test]
fn no_local_header_signature() {
    let mut z = stored_archive("a", b"x", b"", b"");
    let (central, _) = offsets(&z);
    put_u32(&mut z, central + 42, 1);
    assert_eq!(entry_error(z), "no local header signature where the directory points");
}

#[test]
fn data_past_the_end_of_the_archive() {
    let mut z = stored_archive("a", b"x", b"", b"");
    let (central, _) = offsets(&z);
    put_u32(&mut z, central + 20, 1000);
    assert_eq!(entry_error(z), "an entry's data runs past the end of the archive");
    let mut z = stored_archive("a", b"x", b"", b"");
    put_u32(&mut z, central + 20, 0xFFFF_FFF0);
    assert_eq!(entry_error(z), "an entry's data runs past the end of the archive");
}

#[test]
fn a_stored_entry_whose_sizes_differ() {
    let mut z = stored_archive("a", b"xy", b"", b"");
    let (central, _) = offsets(&z);
    put_u32(&mut z, central + 20, 1);
    assert_eq!(entry_error(z), "a stored entry's compressed and uncompressed sizes differ");
}

#[test]
fn the_local_extra_field_length_is_honoured() {
    let z = stored_archive("a.txt", b"hello", b"\x99\x99\x02\x00xy", b"");
    let zip = Zip::new(&z[..]).unwrap();
    assert_eq!(&*zip.entry("a.txt").unwrap(), b"hello");
    assert!(matches!(zip.entry("a.txt").unwrap(), Cow::Borrowed(_)));
}

#[test]
fn the_bytes_are_the_archive_as_given() {
    let z = stored_archive("a", b"x", b"", b"");
    let zip = Zip::new(z.clone()).unwrap();
    assert_eq!(zip.bytes(), &z[..]);
}

#[test]
fn a_comment_holding_the_end_signature_is_skipped() {
    let mut comment = END_SIGNATURE.to_le_bytes().to_vec();
    comment.extend_from_slice(&[0; 30]);
    let z = stored_archive("a", b"x", b"", &comment);
    let zip = Zip::new(z).unwrap();
    assert_eq!(&*zip.entry("a").unwrap(), b"x");
}

#[test]
fn a_missing_entry_is_named() {
    let z = stored_archive("a", b"x", b"", b"");
    let zip = Zip::new(z).unwrap();
    assert!(matches!(zip.entry("b"), Err(Error::MissingEntry(n)) if &*n == "b"));
}

#[test]
fn a_wrong_crc_is_refused() {
    let mut z = stored_archive("a", b"x", b"", b"");
    z[30 + 1] = b'y'; // the data byte after the 30-byte header and the name
    let zip = Zip::new(z).unwrap();
    assert!(matches!(zip.entry("a"), Err(Error::Crc { .. })));
}

#[test]
fn zip64_markers_are_refused() {
    let mut z = stored_archive("a", b"x", b"", b"");
    let end = z.len() - END_LEN;
    z[end + 16..end + 20].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
    assert!(matches!(Zip::new(z), Err(Error::Zip { message: "zip64 archives are not supported", .. })));

    let mut z = stored_archive("a", b"x", b"", b"");
    let end = z.len() - END_LEN;
    let central = u32_at(&z, end + 16) as usize;
    z[central + 24..central + 28].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
    assert!(matches!(Zip::new(z), Err(Error::Zip { message: "zip64 entries are not supported", .. })));

    let mut z = stored_archive("a", b"x", b"", b"");
    let end = z.len() - END_LEN;
    let mut locator = ZIP64_LOCATOR_SIGNATURE.to_le_bytes().to_vec();
    locator.extend_from_slice(&[0; 16]);
    z.splice(end..end, locator);
    assert!(matches!(Zip::new(z), Err(Error::Zip { message: "zip64 archives are not supported", .. })));
}

#[test]
fn not_a_zip() {
    assert!(matches!(Zip::new(&b"short"[..]), Err(Error::Zip { message: "too short to be a zip archive", .. })));
    assert!(matches!(Zip::new(&[0u8; 100][..]), Err(Error::Zip { message: "no end of central directory record", .. })));
}
