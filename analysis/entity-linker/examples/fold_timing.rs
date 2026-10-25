//! Timing receipt for `fold_alias` (#876): the shared accent fold
//! against a copy of the per-crate implementation it replaced,
//! over representative place names. Run with:
//! `cargo run --release -p turbo-entity-linker --example fold_timing`.

use std::time::Instant;

use turbo_entity_linker::fold_alias;
use turbo_unicode::GeneralCategory;
use turbo_unicode::casefold;
use turbo_unicode::category::general_category;
use turbo_unicode::normalize::{Form, normalize};

/// The pre-#876 entity-linker fold, kept only as the timing
/// reference.
fn fold_alias_before(input: &str) -> String {
    if input.is_empty() {
        return String::new();
    }
    let mut nfd = String::with_capacity(input.len());
    normalize(Form::Nfd, input, &mut nfd);
    let mut stripped = String::with_capacity(nfd.len());
    for ch in nfd.chars() {
        if general_category(ch) == GeneralCategory::Mn {
            continue;
        }
        match ch {
            '\u{00DF}' | '\u{1E9E}' => stripped.push_str("ss"),
            '\u{00E6}' | '\u{00C6}' => stripped.push_str("ae"),
            '\u{0153}' | '\u{0152}' => stripped.push_str("oe"),
            '\u{00F8}' | '\u{00D8}' => stripped.push('o'),
            '\u{0142}' | '\u{0141}' => stripped.push('l'),
            '\u{0111}' | '\u{0110}' | '\u{00F0}' | '\u{00D0}' => stripped.push('d'),
            '\u{0131}' => stripped.push('i'),
            _ => stripped.push(ch),
        }
    }
    let mut folded = String::with_capacity(stripped.len());
    casefold::fold(&stripped, &mut folded);
    let chars: Vec<char> = folded.chars().collect();
    let mut collapsed = String::with_capacity(folded.len());
    let mut i = 0;
    while i < chars.len() {
        if i + 3 <= chars.len()
            && chars[i].is_alphanumeric()
            && chars[i + 1] == '.'
            && (i + 2 >= chars.len() || chars[i + 2].is_alphanumeric() || chars[i + 2] == ' ')
        {
            collapsed.push(chars[i]);
            i += 2;
        } else {
            collapsed.push(chars[i]);
            i += 1;
        }
    }
    collapsed.split(|c: char| !c.is_alphanumeric()).filter(|t| !t.is_empty()).collect::<Vec<_>>().join(" ")
}

fn main() {
    let names: Vec<String> = [
        "S\u{00E3}o Paulo",
        "\u{0141}\u{00F3}d\u{017A}",
        "\u{00C6}r\u{00F8}sk\u{00F8}bing",
        "Gro\u{00DF}beeren",
        "\u{0130}stanbul",
        "Diyarbak\u{0131}r",
        "United States of America",
        "C\u{00F4}te d'Ivoire",
        "Myanmar (Burma)",
        "M\u{00FC}nchen",
        "Z\u{00FC}rich Hauptbahnhof",
        "\u{00DE}ingvellir",
    ]
    .iter()
    .cycle()
    .take(12_000)
    .map(|s| s.to_string())
    .collect();
    for (label, f) in
        [("shared", fold_alias as fn(&str) -> String), ("before", fold_alias_before as fn(&str) -> String)]
    {
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t = Instant::now();
            let mut total = 0usize;
            for n in &names {
                total += f(n).len();
            }
            best = best.min(t.elapsed().as_nanos() as f64 / names.len() as f64);
            std::hint::black_box(total);
        }
        println!("{label}: {best:.0} ns/name over {} names", names.len());
    }
}
