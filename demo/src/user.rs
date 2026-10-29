//! The user a catalog call is made for. Every catalog method that names a
//! user carries it in one string field; the demo sets that field to the
//! identity it was started with (`--user`), or clears it when it has none,
//! whatever the page sent. So a page can neither accept a licence for
//! someone else nor read another user's acceptances. This is the one
//! place the demo looks inside a message: it walks the top-level fields
//! of the wire format, drops the user field and appends its own.

use bytes::{BufMut, Bytes, BytesMut};

/// Each method that names a user, and the field number of that user.
pub const USER_FIELDS: [(&str, u32); 8] = [
    ("/turbo.training.v1.TurboCatalogService/ListModelCatalog", 1),
    ("/turbo.training.v1.TurboCatalogService/RecordLicenceAcceptance", 1),
    ("/turbo.training.v1.TurboCatalogService/InstallModel", 1),
    ("/turbo.training.v1.TurboCatalogService/DescribeModel", 1),
    ("/turbo.training.v1.TurboCatalogService/ResolveModel", 3),
    ("/turbo.training.v1.TurboCatalogService/ReadModelFile", 1),
    ("/turbo.training.v1.TurboCatalogService/RevokeLicenceAcceptance", 1),
    ("/turbo.training.v1.TurboCatalogService/ListLicenceAcceptances", 1),
];

/// The user field of the method at `path`, if it names a user.
pub fn field(path: &str) -> Option<u32> {
    USER_FIELDS.iter().find(|(p, _)| *p == path).map(|&(_, f)| f)
}

/// The demo's identity: a user name as the zoo takes it, 1 to 256 bytes,
/// no control characters and no leading or trailing whitespace.
pub fn check(user: &str) -> Result<(), String> {
    let ok = !user.is_empty() && user.len() <= 256 && !user.chars().any(char::is_control) && user.trim() == user;
    if ok { Ok(()) } else { Err(format!("--user {user:?}: not a user name")) }
}

fn varint(b: &[u8], at: &mut usize) -> Result<u64, String> {
    let mut v = 0u64;
    for shift in (0..64).step_by(7) {
        let byte = *b.get(*at).ok_or("the message ends inside a varint")?;
        *at += 1;
        v |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(v);
        }
    }
    Err("a varint is over 10 bytes".into())
}

fn put_varint(out: &mut BytesMut, mut v: u64) {
    while v >= 0x80 {
        out.put_u8((v as u8) | 0x80);
        v >>= 7;
    }
    out.put_u8(v as u8);
}

/// `message` with every occurrence of `field` dropped and, with a user,
/// that user appended as a string in `field`. A message that is not well
/// formed protobuf, or that sends `field` as anything but a string, is an
/// error.
pub fn set(message: &[u8], field: u32, user: Option<&str>) -> Result<Bytes, String> {
    let mut out = BytesMut::with_capacity(message.len() + user.map_or(0, |u| u.len() + 8));
    let mut at = 0;
    while at < message.len() {
        let start = at;
        let key = varint(message, &mut at)?;
        let (number, wire) = (key >> 3, key & 7);
        if number == 0 || number > u64::from(u32::MAX >> 3) {
            return Err(format!("field number {number} is not valid"));
        }
        match wire {
            0 => {
                varint(message, &mut at)?;
            }
            1 => at += 8,
            2 => {
                let len = varint(message, &mut at)?;
                at = usize::try_from(len).ok().and_then(|l| at.checked_add(l)).ok_or("a length is out of range")?;
            }
            5 => at += 4,
            _ => return Err(format!("field {number} has wire type {wire}, which the demo does not accept")),
        }
        if at > message.len() {
            return Err(format!("field {number} runs past the end of the message"));
        }
        if number == u64::from(field) {
            if wire != 2 {
                return Err(format!("field {field}, the user, is not a string"));
            }
            continue;
        }
        out.extend_from_slice(&message[start..at]);
    }
    if let Some(u) = user {
        put_varint(&mut out, u64::from(field) << 3 | 2);
        put_varint(&mut out, u.len() as u64);
        out.extend_from_slice(u.as_bytes());
    }
    Ok(out.freeze())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_user_field_is_replaced_and_other_fields_kept_in_order() {
        // user = "mallory" (1), licence = "CC-BY-SA-4.0" (2), user again.
        let mut m = vec![0x0a, 7];
        m.extend_from_slice(b"mallory");
        m.extend_from_slice(&[0x12, 12]);
        m.extend_from_slice(b"CC-BY-SA-4.0");
        m.extend_from_slice(&[0x0a, 1, b'x']);
        let out = set(&m, 1, Some("alice")).unwrap();
        let mut want = vec![0x12, 12];
        want.extend_from_slice(b"CC-BY-SA-4.0");
        want.extend_from_slice(&[0x0a, 5]);
        want.extend_from_slice(b"alice");
        assert_eq!(&out[..], &want[..]);
        assert_eq!(&set(&m, 1, None).unwrap()[..], &want[..14]);
        assert_eq!(&set(&[], 3, Some("é")).unwrap()[..], &[0x1a, 2, 0xc3, 0xa9]);
    }

    #[test]
    fn scalars_and_long_fields_pass_through() {
        // A varint field 2, a fixed64 field 3, a fixed32 field 4, then a
        // 200 byte string in field 5, whose length takes two bytes.
        let mut m = vec![0x10, 0x96, 0x01, 0x19, 1, 2, 3, 4, 5, 6, 7, 8, 0x25, 1, 2, 3, 4, 0x2a, 0xc8, 0x01];
        m.extend(std::iter::repeat_n(b'a', 200));
        assert_eq!(&set(&m, 1, None).unwrap()[..], &m[..]);
    }

    #[test]
    fn a_malformed_message_is_refused() {
        for bad in [
            &[0x0a, 5, b'a'][..],
            &[0x08][..],
            &[0x08, 0x80][..],
            &[0x0b][..],
            &[0x00, 0x01][..],
            &[0x19, 1, 2][..],
            &[0x08, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01][..],
        ] {
            assert!(set(bad, 1, Some("a")).is_err(), "{bad:?}");
        }
        assert!(set(&[0x08, 1], 1, Some("a")).is_err(), "a user sent as a number");
    }

    #[test]
    fn identities_follow_the_zoos_user_names() {
        assert!(check("alice").is_ok() && check("Ålice Ü").is_ok());
        for bad in ["", " alice", "alice ", "a\nb", &"a".repeat(257)] {
            assert!(check(bad).is_err(), "{bad:?}");
        }
    }
}
