//! Times the Places pack over prose with no state after a comma and no
//! city label, the text the bare city reading (#2097) adds work to.
//!
//! `cargo run --release -p turbo-pii --example places_bench -- KB ROUNDS [plain]`:
//! KB kilobytes of sentences drawn from a fixed list with a fixed LCG,
//! their capitalized words a mix of sentence starts, given names, common
//! words and city names (with `plain`, only the sentences that name no
//! city), scanned ROUNDS times. Defaults are `256 20`. It prints the best
//! round's time and throughput and the city count.

use std::time::Instant;

use turbo_analysis::ViewId;
use turbo_pii::{Pack, PiiDetector, PiiKind};

/// The sentences, each with whether it names a city.
const SENTENCES: [(bool, &str); 12] = [
    (false, "The committee met on Tuesday and agreed to review the budget again next month. "),
    (true, "In Rouen the weather turned before lunch, so they stayed inside. "),
    (false, "Alice Waters said the menu would change with the season. "),
    (false, "Reading the report took longer than anyone expected. "),
    (true, "Her sister moved to the Boston area after the wedding. "),
    (false, "Price is per person and does not include the port fees. "),
    (false, "The San Francisco Chronicle ran the story on its front page. "),
    (false, "Most of the staff worked from home during the winter. "),
    (false, "He said Harry Denton would bring the wine. "),
    (false, "Mobile devices must be enrolled before they reach the network. "),
    (true, "They flew between Istanbul and Athens twice that year. "),
    (false, "Nothing in this paragraph names a place at all, which is common. "),
];

fn main() {
    let mut args = std::env::args().skip(1);
    let kb: usize = args.next().map_or(256, |a| a.parse().expect("KB"));
    let rounds: usize = args.next().map_or(20, |a| a.parse().expect("ROUNDS"));
    let plain = args.next().is_some_and(|a| a == "plain");
    let mut text = String::with_capacity(kb * 1024 + 128);
    let mut seed: u64 = 0x2097;
    while text.len() < kb * 1024 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let i = (seed >> 33) as usize % SENTENCES.len();
        let (names_city, sentence) = SENTENCES[i];
        if !plain || !names_city {
            text.push_str(sentence);
        }
    }
    let detector = PiiDetector::pack(Pack::Places);
    let mut cities = detector.layer(&text, ViewId::SOURCE).mentions().filter(|m| m.kind == PiiKind::City).count();
    let mut best = f64::INFINITY;
    for _ in 0..rounds {
        let t = Instant::now();
        let layer = detector.layer(&text, ViewId::SOURCE);
        best = best.min(t.elapsed().as_secs_f64());
        cities = layer.mentions().filter(|m| m.kind == PiiKind::City).count();
    }
    println!(
        "{} bytes, best of {rounds}: {:.3} ms, {:.1} MB/s, {cities} cities",
        text.len(),
        best * 1e3,
        text.len() as f64 / best / 1e6
    );
}
