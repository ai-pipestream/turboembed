use super::*;

#[test]
fn ranges_are_ordered_disjoint_unicode_scalars() {
    for table in [LETTERS, NUMBERS] {
        for &(a, b) in table {
            assert!(a <= b && char::from_u32(a).is_some() && char::from_u32(b).is_some());
            assert!(b < 0xd800 || a > 0xdfff);
        }
        assert!(table.windows(2).all(|w| w[0].1 < w[1].0));
    }
    for &(a, b) in LETTERS {
        assert!(NUMBERS.iter().all(|&(c, d)| b < c || d < a));
    }
}

pub(crate) fn write_classes() {
    let ranges = |wanted: fn(char) -> bool| {
        let mut out: Vec<(u32, u32)> = Vec::new();
        for c in (0..=0x10ffff).filter_map(char::from_u32) {
            if wanted(c) {
                match out.last_mut() {
                    Some(r) if r.1 + 1 == c as u32 => r.1 = c as u32,
                    _ => out.push((c as u32, c as u32)),
                }
            }
        }
        out
    };
    let letters = ranges(|c| turbo_unicode::category::general_category(c).is_letter());
    let numbers = ranges(|c| turbo_unicode::category::general_category(c).is_number());
    let middle = letters.len().div_ceil(2);
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/classes");
    for (file, table) in
        [("letters_a.rs", &letters[..middle]), ("letters_b.rs", &letters[middle..]), ("numbers.rs", &numbers[..])]
    {
        assert!(table.len() < 490, "split the class table further before regenerating");
        let mut out = format!("pub(super) const RANGES: [(u32, u32); {}] = [\n", table.len());
        for (a, b) in table {
            out.push_str(&format!("    (0x{a:x}, 0x{b:x}),\n"));
        }
        out.push_str("];\n");
        std::fs::write(root.join(file), out).unwrap();
    }
}
