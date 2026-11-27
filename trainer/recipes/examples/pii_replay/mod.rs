use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Seek, SeekFrom};

use serde::{Deserialize, Serialize};
use turbo::head::tagger::{TokenTagger, Work};
use turbo_trainer_recipes::corpus::Sentence;
use turbo_trainer_recipes::eval::{Span, spans};

pub const HIDDEN: usize = 768;
pub const TARGETS: [&str; 6] =
    ["city", "first_name", "al_surname", "al_neutral", "company_partial", "company_negative"];
pub const MAX: usize = 24;
pub const MIN: usize = 8;
pub const REPEATS: usize = 4;

pub struct Reader<R> {
    input: R,
    offsets: Vec<u64>,
    at: u64,
    buffer: Vec<u8>,
    hidden: usize,
}

impl<R: Read + Seek> Reader<R> {
    pub fn new(input: R, widths: impl Iterator<Item = usize>, hidden: usize) -> Self {
        let mut offsets = vec![0];
        let mut off = 0;
        for n in widths {
            off += (n * hidden * 4) as u64;
            offsets.push(off);
        }
        Self { input, offsets, at: 0, buffer: Vec::new(), hidden }
    }

    pub fn input_bytes(&self) -> u64 {
        *self.offsets.last().unwrap()
    }

    pub fn read(&mut self, i: usize, words: usize) -> Vec<f32> {
        let (a, b) = (self.offsets[i], self.offsets[i + 1]);
        assert_eq!(b - a, (words * self.hidden * 4) as u64);
        if self.at != a {
            assert_eq!(self.input.seek(SeekFrom::Start(a)).unwrap(), a);
        }
        self.buffer.resize((b - a) as usize, 0);
        self.input.read_exact(&mut self.buffer).unwrap();
        self.at = b;
        self.buffer.as_chunks::<4>().0.iter().map(|k| f32::from_le_bytes(*k)).collect()
    }
}

pub fn predict(tagger: &TokenTagger, row: &Sentence, floats: &[f32]) -> Vec<Span> {
    let ids: Vec<_> = (0..row.words.len() as u32).collect();
    let forms: Vec<_> = row.words.iter().map(String::as_str).collect();
    let (mut scores, mut tags, mut work) = (Vec::new(), Vec::new(), Work::default());
    tagger.scores_forms(floats, &ids, Some(&forms), &mut scores, &mut work).unwrap();
    tagger.decode(&scores, row.words.len(), &mut tags).unwrap();
    tagger
        .spans(&tags)
        .unwrap()
        .into_iter()
        .map(|s| (s.start as usize, s.end as usize, tagger.types()[s.kind as usize].clone()))
        .collect()
}

pub fn signature(row: &Sentence) -> (usize, usize) {
    (row.words.len(), row.labels.iter().filter(|l| *l != "O").count())
}

pub fn persona(i: usize) -> usize {
    i % 45_000
}

#[derive(Clone, Copy, Deserialize, Serialize)]
pub struct Pair {
    pub target: usize,
    pub control: usize,
    pub case: usize,
}

fn overlaps(a: &Span, b: &Span) -> bool {
    a.0 < b.1 && b.0 < a.1
}

pub fn flags(row: &Sentence, gold: &[Span], shared: &BTreeSet<String>) -> [bool; 6] {
    let mut f = [false; 6];
    for (a, b, k) in gold {
        if b - a == 1 && shared.contains(&row.words[*a].to_lowercase()) {
            if k == "city" {
                f[0] = true;
            }
            if k == "first_name" {
                f[1] = true;
            }
        }
        if k == "last_name" && b - a > 1 && row.words[*a].eq_ignore_ascii_case("al") {
            f[2] = true;
        }
        if k == "company_name" && b - a > 1 {
            f[4] = true;
        }
        if k == "first_name" {
            f[5] = true;
        }
    }
    f[3] = row.words.iter().zip(&row.labels).any(|(w, l)| w.eq_ignore_ascii_case("al") && l == "O");
    f
}

pub fn plan<R: Read + Seek>(rows: &[Sentence], reader: &mut Reader<R>, tagger: &TokenTagger) -> Vec<Pair> {
    let gold: Vec<_> = rows.iter().map(|r| spans(&r.labels)).collect();
    let mut lex: BTreeMap<String, [BTreeSet<usize>; 2]> = BTreeMap::new();
    for (i, row) in rows.iter().enumerate() {
        for (a, b, k) in &gold[i] {
            if b - a == 1 && matches!(k.as_str(), "city" | "first_name") {
                lex.entry(row.words[*a].to_lowercase()).or_default()[usize::from(k == "first_name")].insert(persona(i));
            }
        }
    }
    let shared: BTreeSet<_> =
        lex.into_iter().filter_map(|(w, [a, b])| (a.len() >= 2 && b.len() >= 2).then_some(w)).collect();
    let supports: Vec<_> = rows.iter().zip(&gold).map(|(r, g)| flags(r, g, &shared)).collect();
    let mut controls: BTreeMap<(usize, usize, usize), Vec<usize>> = BTreeMap::new();
    let mut cases: [Vec<usize>; 6] = Default::default();
    for (i, row) in rows.iter().enumerate() {
        let f = supports[i];
        let (w, p) = signature(row);
        for (target, &in_role) in f.iter().enumerate() {
            let well_matched = match target {
                0 => gold[i].iter().any(|g| g.2 == "city"),
                1 => gold[i].iter().any(|g| g.2 == "first_name"),
                3 => w > p,
                _ => in_role,
            };
            if well_matched {
                controls.entry((w, p, target)).or_default().push(i);
            }
        }
        // All rare attested al and shared city/person rows. Use a fixed
        // persona sixth for the broader company cases.
        let choose = persona(i) % 6 == 3;
        if f[3] {
            cases[3].push(i);
        }
        if !(f[2] || f[0] || f[1] || (choose && (f[4] || f[5]))) {
            continue;
        }
        let floats = reader.read(i, w);
        let native = predict(tagger, row, &floats);
        for g in gold[i].iter().filter(|g| !native.contains(g)) {
            if g.1 - g.0 == 1 && shared.contains(&row.words[g.0].to_lowercase()) {
                if g.2 == "city" {
                    cases[0].push(i);
                }
                if g.2 == "first_name" {
                    cases[1].push(i);
                }
            }
            if g.2 == "last_name" && g.1 - g.0 > 1 && row.words[g.0].eq_ignore_ascii_case("al") {
                cases[2].push(i);
            }
            if choose && g.2 == "company_name" && g.1 - g.0 > 1 && native.iter().any(|p| p.2 == g.2 && overlaps(p, g)) {
                cases[4].push(i);
            }
        }
        if choose
            && f[5]
            && native
                .iter()
                .any(|n| n.2 == "company_name" && !gold[i].iter().any(|g| g.2 == "company_name" && overlaps(g, n)))
        {
            cases[5].push(i);
        }
    }
    let mut used = BTreeSet::new();
    let mut pairs = Vec::new();
    for target in [3, 0, 1, 5, 2, 4] {
        cases[target].dedup();
        let case_count = cases[target].len();
        let before = pairs.len();
        for &case in &cases[target] {
            if used.contains(&persona(case)) {
                continue;
            }
            let (w, p) = signature(&rows[case]);
            if let Some(&control) = controls.get(&(w, p, target)).and_then(|pool| {
                pool.iter().find(|&&j| {
                    persona(j) != persona(case)
                        && !used.contains(&persona(j))
                        && cases[target].binary_search(&j).is_err()
                })
            }) {
                used.insert(persona(case));
                used.insert(persona(control));
                pairs.push(Pair { target, control, case });
                if pairs.len() - before == MAX {
                    break;
                }
            }
        }
        eprintln!(
            "{}: {case_count} attested candidates, {} exact-matched pairs",
            TARGETS[target],
            pairs.len() - before
        );
    }
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    #[test]
    fn streams_word_rows_out_of_order_with_exact_alignment() {
        let bytes: Vec<_> = (0..12).flat_map(|v| (v as f32).to_le_bytes()).collect();
        let mut reader = Reader::new(Cursor::new(bytes), [2, 1, 3].into_iter(), 2);
        assert_eq!(reader.input_bytes(), 48);
        assert_eq!(reader.read(2, 3), [6., 7., 8., 9., 10., 11.]);
        assert_eq!(reader.read(0, 2), [0., 1., 2., 3.]);
        assert_eq!(reader.read(1, 1), [4., 5.]);
    }
}
