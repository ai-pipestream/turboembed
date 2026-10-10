//! The timing protocol every harness here shares: one warm pass, then
//! three runs of at least two seconds each, embedding the texts in order
//! batch by batch, the best of the three in texts a second.
use std::io::Write;
use std::time::{Duration, Instant};

pub fn args() -> (String, Vec<String>, Vec<usize>, Option<String>) {
    let a: Vec<String> = std::env::args().collect();
    let texts: Vec<String> = serde_json::from_slice(&std::fs::read(&a[2]).unwrap()).unwrap();
    // "-" times nothing: the run only writes the vectors.
    let batches = if a[3] == "-" { Vec::new() } else { a[3].split(',').map(|b| b.trim().parse().unwrap()).collect() };
    (a[1].clone(), texts, batches, a.get(4).cloned())
}

pub fn time(name: &str, b: usize, n: usize, mut pass: impl FnMut()) {
    pass();
    let best = (0..3)
        .map(|_| {
            let (start, mut done) = (Instant::now(), 0);
            while start.elapsed() < Duration::from_secs(2) {
                pass();
                done += n;
            }
            done as f64 / start.elapsed().as_secs_f64()
        })
        .fold(0f64, f64::max);
    println!("{name} batch {b}: {best:.0} texts/s");
}

pub fn write(path: &str, rows: &[Vec<f32>]) {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    for r in rows {
        for v in r {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
    }
}
