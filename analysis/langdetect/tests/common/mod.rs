pub fn modified_utf8(units: &[u16]) -> Vec<u8> {
    let mut out = Vec::new();
    for &u in units {
        turbo_unicode::modified_utf8::push_modified_utf8(&mut out, u32::from(u));
    }
    out
}

pub fn mutf8(s: &str) -> Vec<u8> {
    modified_utf8(&s.encode_utf16().collect::<Vec<u16>>())
}
