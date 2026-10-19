//! Release timings of sketching, duplicate grouping and boilerplate on
//! synthetic Markdown pages.
use std::hint::black_box;
use std::time::Instant;

use turbo_analysis::*;
use turbo_analysis_layers::{Boilerplate, Sentences, Structure};
use turbo_group::*;
use turbo_structure::{Syntax, detect};

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

fn paragraph(r: &mut Lcg, words: usize) -> String {
    (0..words).map(|_| format!("w{}", (r.next() % 4000) * (r.next() % 3 + 1))).collect::<Vec<_>>().join(" ") + "."
}

fn page(r: &mut Lcg, n: usize, shared: &[String]) -> String {
    let mut parts = vec!["ACME intranet home".to_owned(), "- Home page\n- About us\n- Contact us".to_owned()];
    for k in 0..40 {
        match r.next() % 20 {
            0 => parts.push(shared[(r.next() as usize) % shared.len()].clone()),
            1 => parts.push(shared[(r.next() as usize) % shared.len()].replacen("w", "edited w", 1)),
            _ => parts.push(paragraph(r, 40 + (k % 40))),
        }
    }
    parts.push(format!("Page {n} of 200, ACME Corp 2024"));
    parts.join("\n\n")
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() {
    let mut r = Lcg(7);
    let shared: Vec<String> = (0..50).map(|_| paragraph(&mut r, 60)).collect();
    let texts: Vec<String> = (0..200).map(|n| page(&mut r, n, &shared)).collect();
    let bytes: usize = texts.iter().map(String::len).sum();
    let docs: Vec<Document> = texts
        .iter()
        .map(|t| {
            let mut d = Document::new(t.clone());
            d.add(&Key::DEFAULT, detect(t, Syntax::Markdown, ViewId::SOURCE), Provenance::by("t")).unwrap();
            d
        })
        .collect();
    let blocks: usize = docs.iter().map(|d| d.get::<Structure>().unwrap().len()).sum();
    println!("{} docs, {:.2} MB, {} blocks", docs.len(), bytes as f64 / 1e6, blocks);
    let sketcher = Sketcher::<Structure>::default();
    let near = NearDuplicates::<Structure>::default();
    let detector = BoilerplateDetector::<Structure>::default();
    let projector = BoilerplateProjector::<Sentences>::default();
    let (mut ts, mut tn, mut tb, mut tc, mut tp) = (vec![], vec![], vec![], vec![], vec![]);
    for _ in 0..15 {
        let t = Instant::now();
        let sk: Vec<Sketches<Structure>> = docs
            .iter()
            .map(|d| sketcher.annotate((d.get::<View>().unwrap(), d.get::<Structure>().unwrap())).unwrap().0)
            .collect();
        ts.push(t.elapsed().as_secs_f64());
        let t = Instant::now();
        for (d, s) in docs.iter().zip(&sk) {
            black_box(near.annotate((d.get::<Structure>().unwrap(), s)).unwrap());
        }
        tn.push(t.elapsed().as_secs_f64());
        let t = Instant::now();
        let flags: Vec<Boilerplate<Structure>> = detector
            .annotate_batch(docs.iter().zip(&sk).map(|(d, s)| (d.get::<Structure>().unwrap(), s)).collect())
            .into_iter()
            .map(|x| x.unwrap().0)
            .collect();
        tb.push(t.elapsed().as_secs_f64());
        let t = Instant::now();
        let batch: Vec<_> = docs.iter().zip(&sk).map(|(d, s)| (d.get::<Structure>().unwrap(), s)).collect();
        let cross = cross_duplicates(&batch, &Thresholds::default()).unwrap();
        tc.push(t.elapsed().as_secs_f64());
        // Sentences: one per paragraph block, to project onto.
        let t = Instant::now();
        for (d, f) in docs.iter().zip(&flags) {
            let st = d.get::<Structure>().unwrap();
            let mut spans = Spans::new(ViewId::SOURCE);
            for (_, s) in st.spans.iter().filter(|(i, _)| st.kind[i.get()] == turbo_analysis_layers::Block::Paragraph) {
                spans.push(s);
            }
            let sentences = Sentences { spans };
            black_box(projector.annotate((st, f, &sentences)).unwrap());
        }
        tp.push(t.elapsed().as_secs_f64());
        black_box(&cross);
        if ts.len() == 1 {
            let flagged: usize = flags.iter().map(|f| f.flagged().count()).sum();
            println!(
                "cross links {}, cross groups {}, flagged blocks {}",
                cross.links.len(),
                cross.groups.len(),
                flagged
            );
            for k in [DuplicateKind::Exact, DuplicateKind::NearDuplicate, DuplicateKind::VersionEdit] {
                let sc: Vec<f64> = cross.links.iter().filter(|l| l.kind == k).map(|l| l.score).collect();
                let lo = sc.iter().copied().fold(1.0, f64::min);
                println!("  {:?}: {} links, least score {lo}", k, sc.len());
            }
        }
    }
    let mb = bytes as f64 / 1e6;
    let (s, n, b, c, p) = (median(ts), median(tn), median(tb), median(tc), median(tp));
    println!("sketch      {:8.3} ms  {:7.1} MB/s  {:6.0} ns/block", s * 1e3, mb / s, s * 1e9 / blocks as f64);
    println!("near-dups   {:8.3} ms  {:7.1} MB/s  {:6.0} ns/block", n * 1e3, mb / n, n * 1e9 / blocks as f64);
    println!("boilerplate {:8.3} ms  {:7.1} MB/s  {:6.0} ns/block", b * 1e3, mb / b, b * 1e9 / blocks as f64);
    println!("cross-dups  {:8.3} ms  {:7.1} MB/s  {:6.0} ns/block", c * 1e3, mb / c, c * 1e9 / blocks as f64);
    println!("project     {:8.3} ms  {:7.1} MB/s", p * 1e3, mb / p);
}
