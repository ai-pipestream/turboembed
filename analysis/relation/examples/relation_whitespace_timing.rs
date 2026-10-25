//! Release preprocessing timing probe; results are elapsed milliseconds.
use std::{hint::black_box, time::Instant};
fn main() {
    let text =
        "Steve Jobs founded Apple in 1976. Café München\u{1f}résumé @user https://example.org/a_b state-of-the-art!\n"
            .repeat(32);
    for mode in [0, 1] {
        let start = Instant::now();
        for _ in 0..20_000 {
            if mode == 0 {
                black_box(turbo_relation::glirel::words::split(black_box(&text)));
            } else {
                black_box(turbo_relation::gliner2::words::split(black_box(&text)));
            }
        }
        println!("relation{mode} {:.3}", start.elapsed().as_secs_f64() * 1e3);
    }
}
