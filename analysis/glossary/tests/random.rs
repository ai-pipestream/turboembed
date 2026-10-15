//! Seeded random checks: the readers on mutated inputs never panic and
//! give spans inside the file, and the character automaton finds exactly
//! what a naive search over every term finds.

use turbo_analysis::Span;
use turbo_glossary::{Case, DelimitedReader, EntryIx, FirstRecord, Glossary, GlossaryFile, TbxReader, TextMatcher};
use turbo_unicode::java::to_lower_char;

struct Lcg(u64);

impl Lcg {
    fn below(&mut self, n: usize) -> usize {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) % n as u64) as usize
    }
}

/// Pieces the mutations insert: markup, quotes, line ends, references and
/// characters of one to four bytes.
const PIECES: &[&str] = &[
    "<",
    ">",
    "</",
    "/>",
    "\"",
    "'",
    "&",
    ";",
    "&amp;",
    "&#x1F642;",
    "&#0;",
    "<!--",
    "-->",
    "<![CDATA[",
    "]]>",
    "<!DOCTYPE a [<!ENTITY e \"x\">]>",
    "&e;",
    "<term>",
    "</term>",
    "xml:lang=\"en\"",
    "id=\"\"",
    ",",
    "\t",
    "\n",
    "\r",
    "\r\n",
    " ",
    "\u{a0}",
    "é",
    "東",
    "🙂",
    "\u{feff}",
    "\u{0}",
];

fn mutate(rng: &mut Lcg, bytes: &[u8]) -> Vec<u8> {
    let mut out = bytes.to_vec();
    for _ in 0..1 + rng.below(4) {
        let at = rng.below(out.len() + 1);
        match rng.below(3) {
            0 if at < out.len() => {
                let end = (at + 1 + rng.below(8)).min(out.len());
                out.drain(at..end);
            }
            1 if at < out.len() => out[at] = rng.below(256) as u8,
            _ => {
                let piece = PIECES[rng.below(PIECES.len())].as_bytes();
                out.splice(at..at, piece.iter().copied());
            }
        }
    }
    out
}

fn check_spans(bytes: &[u8], f: &GlossaryFile) {
    assert_eq!(f.term_at.len(), f.glossary.len());
    for s in &f.term_at {
        assert!(s.end as usize <= bytes.len());
        assert!(std::str::from_utf8(&bytes[s.range()]).is_ok(), "a span cuts a character");
    }
}

#[test]
fn readers_survive_mutated_inputs() {
    let dir = turbo_goldens::data_dir().join("inputs").join("glossary");
    let mut files: Vec<Vec<u8>> =
        std::fs::read_dir(&dir).unwrap().map(|e| std::fs::read(e.unwrap().path()).unwrap()).collect();
    files.sort();
    let readers = [DelimitedReader::csv(), DelimitedReader::tsv(FirstRecord::Header)];
    let tbx = TbxReader::new("en").unwrap();
    let mut rng = Lcg(202);
    let mut read = 0;
    for bytes in &files {
        for _ in 0..300 {
            let m = mutate(&mut rng, bytes);
            for r in &readers {
                if let Ok(f) = r.read(&m) {
                    check_spans(&m, &f);
                }
            }
            if let Ok(f) = tbx.read(&m) {
                check_spans(&m, &f);
                read += 1;
            }
        }
    }
    assert!(read > 100, "{read}");
}

/// Every occurrence of every term, by trying each term at each character.
fn naive(g: &Glossary, case: Case, text: &str) -> Vec<(Span, EntryIx)> {
    let fold = |s: &str| -> Vec<char> {
        s.chars().map(|c| if case == Case::Insensitive { to_lower_char(c) } else { c }).collect()
    };
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let folded = fold(text);
    let mut out = Vec::new();
    for (ix, e) in g.iter() {
        let p = fold(e.term);
        // The first entry with the same folded term reports it.
        if g.iter().take(ix.get()).any(|(_, f)| fold(f.term) == p) {
            continue;
        }
        for i in 0..chars.len() {
            if folded[i..].starts_with(&p) {
                let end = chars.get(i + p.len()).map_or(text.len(), |c| c.0);
                out.push((Span::new(chars[i].0 as u32, end as u32), ix));
            }
        }
    }
    out.sort_by_key(|&(s, ix)| (s.start, s.end, ix));
    out
}

#[test]
fn the_automaton_finds_what_a_naive_search_finds() {
    const ALPHABET: &[&str] = &["a", "b", "A", "ab", " ", "é", "É", "東", "🙂", "İ", "i", "ß"];
    let mut rng = Lcg(7);
    for round in 0..500 {
        let mut g = Glossary::new();
        for i in 0..1 + rng.below(12) {
            let term: String = (0..1 + rng.below(4)).map(|_| ALPHABET[rng.below(ALPHABET.len())]).collect();
            if term.trim().is_empty() {
                continue;
            }
            g.push(&format!("t{i}"), &term).unwrap();
        }
        if g.is_empty() {
            continue;
        }
        let text: String = (0..rng.below(40)).map(|_| ALPHABET[rng.below(ALPHABET.len())]).collect();
        for case in [Case::Sensitive, Case::Insensitive] {
            let m = TextMatcher::new(&g, case).unwrap();
            let mut got = Vec::new();
            m.scan(&text, |s, ix| got.push((s, ix)));
            got.sort_by_key(|&(s, ix)| (s.start, s.end, ix));
            assert_eq!(got, naive(&g, case, &text), "round {round} {case:?} {text:?} {g:?}");
        }
    }
}
