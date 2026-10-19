use super::*;

/// SI prefixes written as words, with their powers of ten: `kilo meter`,
/// `mega bytes`.
pub(super) const PREFIXES: &[(&str, i64)] = &[
    ("kilo", 3),
    ("hecto", 2),
    ("deca", 1),
    ("deka", 1),
    ("deci", -1),
    ("centi", -2),
    ("milli", -3),
    ("micro", -6),
    ("nano", -9),
    ("pico", -12),
    ("mega", 6),
    ("giga", 9),
    ("tera", 12),
    ("peta", 15),
];

/// Time words: what a speed is per and what an age is counted in. The
/// value is a [`crate::lang::en::unit::Time`] as a number.
pub(super) const TIMES: &[(&str, crate::lang::en::unit::Time, u8)] = {
    use crate::lang::en::unit::Time::*;
    &[
        ("ms", Millisecond, ABBR),
        ("millisecond", Millisecond, 0),
        ("milliseconds", Millisecond, 0),
        ("s", Second, ABBR),
        ("sec", Second, ABBR),
        ("second", Second, 0),
        ("seconds", Second, 0),
        ("min", Minute, ABBR),
        ("minute", Minute, 0),
        ("minutes", Minute, 0),
        ("h", Hour, ABBR),
        ("hr", Hour, ABBR),
        ("hour", Hour, 0),
        ("hours", Hour, 0),
        ("day", Day, 0),
        ("days", Day, 0),
        ("week", Week, 0),
        ("weeks", Week, 0),
        ("month", Month, 0),
        ("months", Month, 0),
        ("year", Year, 0),
        ("years", Year, 0),
        ("yr", Year, ABBR),
        ("yrs", Year, ABBR),
    ]
};

/// Temperature scales: names, and letters that name a scale only after a
/// degree (`deg c`, `°F`).
pub(super) const SCALES: &[(&str, Temperature, u8)] = &[
    ("celsius", Temperature::Celsius, 0),
    ("celcius", Temperature::Celsius, 0),
    ("centigrade", Temperature::Celsius, 0),
    ("centigrate", Temperature::Celsius, 0),
    ("fahrenheit", Temperature::Fahrenheit, 0),
    ("farenheit", Temperature::Fahrenheit, 0),
    ("kelvin", Temperature::Kelvin, 0),
    ("kelvins", Temperature::Kelvin, 0),
    ("rankine", Temperature::Rankine, 0),
    ("delisle", Temperature::Delisle, 0),
    ("c", Temperature::Celsius, ABBR),
    ("f", Temperature::Fahrenheit, ABBR),
    ("k", Temperature::Kelvin, ABBR),
];

/// The temperature scale of a [`TSCALE`] entry's value.
pub(in crate::lang) fn scale(i: i64) -> Temperature {
    SCALES[i as usize].1
}

/// Multipliers glued to an amount after a currency: `$3.5m`, `€2bn`.
pub(super) const MULTIPLIERS: &[(&str, i64)] =
    &[("m", 6), ("mn", 6), ("mm", 6), ("bn", 9), ("b", 9), ("tn", 12), ("k", 3)];

pub(super) const fn e(word: &'static str, dim: Dim, value: i64, flags: u8) -> Entry {
    Entry { word, dim, value, flags }
}

/// Every entry of the unit words, to add to the number grammar's.
pub(in crate::lang) fn entries() -> Vec<Entry> {
    let mut out = Vec::new();
    out.extend(UNITS.iter().enumerate().map(|(i, x)| e(x.word, UWORD, i as i64, x.flags)));
    out.extend(MONEY.iter().enumerate().map(|(i, x)| e(x.word, CURW, i as i64, x.flags)));
    for (i, (c, _)) in CODES.iter().enumerate() {
        out.push(e(c, CODE, i as i64, if ambiguous_code(c) { AMBIG } else { PREFIX }));
    }
    for (m, c) in MILLION_CODES {
        let i = CODES.binary_search_by(|(k, _)| k.cmp(c)).expect("a code in the table");
        debug_assert!(code(c).is_some());
        out.push(e(m, MCODE, i as i64, 0));
    }
    for (i, r) in NATIONS.iter().enumerate() {
        let first = i == 0 || NATIONS[i - 1].word != r.word;
        if first && !r.word.contains([' ', '.']) {
            out.push(e(r.word, NATION, i as i64, r.flags));
        }
    }
    out.extend(PREFIXES.iter().map(|&(p, v)| e(p, SIPRE, v, 0)));
    out.extend(TIMES.iter().map(|&(t, v, f)| e(t, TIMEW, v as i64, f)));
    out.extend(SCALES.iter().enumerate().map(|(i, &(s, _, f))| e(s, TSCALE, i as i64, f)));
    for (d, f) in [("degree", 0), ("degrees", 0), ("deg", DOT), ("degs", 0), ("°", 0)] {
        out.push(e(d, DEG, 0, f));
    }
    out.extend(MULTIPLIERS.iter().map(|&(m, v)| e(m, MULT, v, 0)));
    // `in` only after feet (`5 ft 2 in`); `"` after a number outside a
    // quotation.
    out.push(e("in", INCHW, 0, AMBIG));
    out.push(e("\"", INCHW, 1, AMBIG));
    // U+2033 DOUBLE PRIME.
    out.push(e("″", INCHW, 2, 0));
    out
}

#[cfg(test)]
mod tests {
    use super::super::money::AMBIGUOUS_CODES;
    use super::*;

    #[test]
    fn words_are_lowercase_and_codes_resolve() {
        for x in UNITS.iter().chain(MONEY) {
            assert_eq!(x.word.to_lowercase(), x.word, "{}", x.word);
        }
        for r in NATIONS {
            assert!(code(r.code).is_some(), "{}", r.code);
        }
        for (_, c) in MILLION_CODES {
            assert!(code(c).is_some(), "{c}");
        }
        for c in AMBIGUOUS_CODES {
            assert!(code(c).is_some(), "{c}");
        }
    }

    #[test]
    fn a_nation_word_has_its_rows_together() {
        for (i, r) in NATIONS.iter().enumerate() {
            let first = nation(r.word).unwrap();
            assert!(nation_rows(first).len() > i - first, "{} is split", r.word);
        }
        assert_eq!(nation_rows(nation("us").unwrap()).len(), 1);
        assert_eq!(nation("atlantis"), None);
    }

    #[test]
    fn entries_cover_every_table() {
        let all = entries();
        assert!(all.iter().any(|x| x.word == "musd" && x.dim == MCODE));
        assert!(all.iter().any(|x| x.word == "try" && x.dim == CODE && x.flags & AMBIG != 0));
        assert!(all.iter().any(|x| x.word == "usd" && x.dim == CODE && x.flags & PREFIX != 0));
        assert!(!all.iter().any(|x| x.word.contains(' ')));
        assert_eq!(scale(0), Temperature::Celsius);
    }
}
