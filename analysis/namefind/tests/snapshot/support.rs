//! Shared fixture machinery for the snapshot tests: the corpus, the
//! one-time converter from OpenNLP dictionary XML to the line format, and
//! the snapshot compare-or-update helper. Nothing here reads the golden
//! crates; the XML fixtures and the hand-made sentences are the crate's
//! own copies under `tests/snapshot-data`.

use std::path::Path;

use turbo_namefind::{Dictionary, Error, TokenNameFinder};

/// SplitMix64: the corpus generation is seeded, so a failure repeats.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }

    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    pub fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.below(items.len())]
    }
}

/// The hand-made sentences and a seeded generated corpus. Each entry is
/// one sentence's tokens; the finders join them with single spaces.
pub fn corpus() -> Vec<String> {
    let mut sentences: Vec<String> =
        include_str!("../snapshot-data/sentences.txt").lines().map(str::to_owned).collect();
    let mut rng = Rng::new(0x5EED);
    let starters = ["see", "call", "mail", "at", "grid", "write", "visit", "dial", "check", "open"];
    let stoppers = ["today", "now", "please", ".", "soon", "again"];
    let words = [
        "a",
        "b",
        "x",
        "the",
        "and",
        "or",
        "in",
        "café",
        "😀",
        "face",
        "FACE",
        "straße",
        "STRASSE",
        "İstanbul",
        "www.example.com",
        "http://x.io/p?q=1",
        "a@b.co",
        "JANE@EXAMPLE.ORG",
        "(555)",
        "123-4567",
        "555-123-4567",
        "4QFJ12345678",
        "40°26'46\"N",
        "79°58'56\"W",
        "new",
        "york",
        "city",
        "hall",
        "john",
        "smith",
        "dr.",
        "who",
        "san",
        "francisco",
        "são",
        "paulo",
    ];
    for _ in 0..120 {
        let n = 3 + rng.below(9);
        let mut sentence = vec![rng.pick(&starters).to_string()];
        for _ in 0..n {
            sentence.push(rng.pick(&words).to_string());
        }
        sentence.push(rng.pick(&stoppers).to_string());
        sentences.push(sentence.join(" "));
    }
    sentences
}

/// Compares `actual` with the checked-in snapshot `name`, or refreshes it
/// when `TURBO_SNAPSHOT_UPDATE` is set.
pub fn check(name: &str, actual: &str) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/snapshot-data").join(name);
    if std::env::var_os("TURBO_SNAPSHOT_UPDATE").is_some() {
        std::fs::write(&path, actual).unwrap();
        return;
    }
    let want = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(actual, want, "snapshot {name} differs; re-run with TURBO_SNAPSHOT_UPDATE=1 to refresh it");
}

/// The one-time converter from an OpenNLP dictionary XML file to the
/// line format [`Dictionary::read`] takes: the `case_sensitive` attribute
/// becomes a directive line and every entry becomes `type token ...` with
/// the given `default_type`. Checked-in XML fixtures convert through here.
pub fn xml_to_lines(xml: &str, default_type: &str) -> Result<String, Error> {
    let mut out = String::new();
    let case_sensitive = !xml.contains("case_sensitive=\"false\"") && !xml.contains("case_sensitive='false'");
    out.push_str(if case_sensitive { "case-sensitive\n" } else { "case-insensitive\n" });
    for entry in xml.split("<entry>").skip(1) {
        let Some(end) = entry.find("</entry>") else {
            return Err(Error::Dictionary { at: 0, message: "an entry is not closed" });
        };
        let tokens: Vec<&str> =
            entry[..end].split("<token>").skip(1).map(|t| t.split("</token>").next().unwrap()).collect();
        if tokens.is_empty() {
            return Err(Error::Dictionary { at: 0, message: "an entry has no token" });
        }
        out.push_str(default_type);
        for token in tokens {
            if token.split_whitespace().count() != 1 {
                return Err(Error::Dictionary { at: 0, message: "a converted token holds whitespace" });
            }
            out.push(' ');
            out.push_str(token);
        }
        out.push('\n');
    }
    Ok(out)
}

/// A dictionary fixture converted to the line format, refusing the empty
/// fixture with the error the finder construction gives it.
pub fn dictionary(name: &str, default_type: &str) -> Result<turbo_namefind::DictionaryNameFinder, Error> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/snapshot-data").join(format!("{name}.xml"));
    let xml = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    turbo_namefind::DictionaryNameFinder::new(Dictionary::read(&xml_to_lines(&xml, default_type)?)?)
}

/// The names `finder` finds in each corpus sentence, one sentence per
/// line as `start:end type`, empty for none.
pub fn find_all(finder: &dyn TokenNameFinder) -> String {
    let mut out = String::new();
    for sentence in corpus() {
        let tokens: Vec<&str> = sentence.split(' ').collect();
        let mut names = Vec::new();
        finder.find(&tokens, &mut names).unwrap();
        let mut line = String::new();
        for n in &names {
            line.push_str(&format!("{}:{} {} ", n.start, n.end, finder.kinds().label(n.kind)));
        }
        out.push_str(line.trim_end());
        out.push('\n');
    }
    out
}
