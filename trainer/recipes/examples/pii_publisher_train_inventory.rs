//! Attested publisher TRAIN support, with per-surface persona/context matching.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};
use turbo_trainer_recipes::{eval::spans, nemotron_pii::read_pinned_train_dev};
#[derive(Default)]
struct Support {
    n: usize,
    ids: BTreeSet<usize>,
    ctx: BTreeSet<String>,
    edges: BTreeMap<usize, BTreeSet<String>>,
}
impl Support {
    fn add(&mut self, i: usize, c: String) {
        self.n += 1;
        self.ids.insert(i % 45000);
        self.ctx.insert(c.clone());
        self.edges.entry(i % 45000).or_default().insert(c);
    }
    fn show(&self) -> [usize; 3] {
        [self.n, self.ids.len(), self.ctx.len()]
    }
}
fn matching(a: &Support, b: Option<&Support>) -> usize {
    let mut graph = a.edges.clone();
    if let Some(b) = b {
        for (i, cs) in &b.edges {
            graph.entry(*i).or_default().extend(cs.iter().cloned());
        }
    }
    fn augment<'a>(
        i: usize,
        g: &'a BTreeMap<usize, BTreeSet<String>>,
        seen: &mut BTreeSet<&'a str>,
        matches: &mut BTreeMap<&'a str, usize>,
    ) -> bool {
        for c in &g[&i] {
            if seen.insert(c)
                && matches.get(c.as_str()).copied().is_none_or(|previous| augment(previous, g, seen, matches))
            {
                matches.insert(c, i);
                return true;
            }
        }
        false
    }
    let mut matches = BTreeMap::new();
    for &i in graph.keys() {
        augment(i, &graph, &mut BTreeSet::new(), &mut matches);
    }
    matches.len()
}
fn main() {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    assert_eq!(args.len(), 1, "pii_publisher_train_inventory <publisher corpus>");
    let (rows, _) = read_pinned_train_dev(Path::new(&args[0])).unwrap();
    assert_eq!(rows.len(), 90000);
    let mut roles = BTreeMap::<String, [Support; 3]>::new();
    let mut parts = BTreeMap::<String, [Support; 2]>::new();
    let mut role_tot = [Support::default(), Support::default(), Support::default()];
    let mut surnames = Support::default();
    let particles = [
        "al", "el", "van", "de", "del", "von", "da", "dos", "du", "di", "la", "le", "bin", "st", "san", "ter", "ten",
        "della", "der", "den", "das", "do", "des", "af", "zu",
    ];
    for (i, row) in rows.iter().enumerate() {
        let norm = |w: &str| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
        let context = |a: usize, b: usize| {
            (a.saturating_sub(5)..a)
                .chain(b..(b + 5).min(row.words.len()))
                .enumerate()
                .map(|(j, k)| {
                    let w = norm(&row.words[k]);
                    let w = if row.labels[k] != "O" || w.chars().any(|c| c.is_numeric()) { "*".into() } else { w };
                    if j == a.min(5) { format!("{} {}", "@", w) } else { w }
                })
                .collect::<Vec<_>>()
                .join(" ")
        };
        for (a, b, k) in spans(&row.labels) {
            let w = norm(&row.words[a]);
            if b - a == 1
                && !w.is_empty()
                && w.chars().all(char::is_alphabetic)
                && let Some(r) = ["city", "first_name", "last_name"].iter().position(|&t| k == t)
            {
                let c = context(a, b);
                role_tot[r].add(i, c.clone());
                roles.entry(w.clone()).or_default()[r].add(i, c);
            }
            if b - a > 1 && k == "last_name" {
                let c = context(a, b);
                surnames.add(i, c.clone());
                if particles.contains(&w.as_str()) {
                    parts.entry(w).or_default()[0].add(i, c);
                }
            }
        }
        for (a, w) in row.words.iter().enumerate() {
            let w = norm(w);
            if row.labels[a] == "O" && particles.contains(&w.as_str()) {
                parts.entry(w).or_default()[1].add(i, context(a, a + 1));
            }
        }
    }
    println!(
        "columns occurrences/personas/contexts; role totals {:?}, multi-surname {:?}",
        role_tot.map(|s| s.show()),
        surnames.show()
    );
    for p in [0, 1, 2] {
        let limits: Vec<_> = roles
            .values()
            .map(|v| {
                matching(&v[0], None).min(match p {
                    0 => matching(&v[1], None),
                    1 => matching(&v[2], None),
                    _ => matching(&v[1], Some(&v[2])),
                })
            })
            .collect();
        for threshold in [1, 2, 5] {
            let eligible: Vec<_> = limits.iter().filter(|&&n| n >= threshold).collect();
            println!(
                "role {} threshold {} surfaces {}, balanced pairs per role {}",
                p,
                threshold,
                eligible.len(),
                eligible.iter().map(|&&n| n).sum::<usize>()
            );
        }
    }
    for (k, [a, b]) in parts {
        println!("particle {k} {:?} {:?}", a.show(), b.show());
    }
}
