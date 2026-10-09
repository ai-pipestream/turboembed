//! Texts a second for a static bundle on the CPU backend, for comparing
//! with other programs on the same machine and the same texts: each batch
//! size, at PRECISION_MODEL and FASTEST, with TURBO_TRUNCATE_MODEL and
//! L2 normalization, as Model2Vec's encode runs by default.
//!
//! The bundle and the texts are not in the repository: the test runs when
//! TURBO_SPEED_BUNDLE names the bundle and TURBO_SPEED_TEXTS a JSON array
//! of strings, and passes with a note otherwise. TURBO_SPEED_BATCHES lists
//! the batch sizes (default 1,32,256,1024), each up to the bundle's
//! max_batch. Each cell embeds the texts in
//! order, batch by batch, for at least two seconds after one warm pass,
//! three times, and prints the best. A release build is the one to time.

mod common;

use std::fs;
use std::time::{Duration, Instant};

use common::*;
use turbo::*;

#[test]
fn texts_a_second() {
    let (Some(bundle), Some(texts)) = (std::env::var_os("TURBO_SPEED_BUNDLE"), std::env::var_os("TURBO_SPEED_TEXTS"))
    else {
        println!("TURBO_SPEED_BUNDLE and TURBO_SPEED_TEXTS are not both set: nothing to time");
        return;
    };
    let texts: Vec<String> = serde_json::from_slice(&fs::read(texts).unwrap()).unwrap();
    let texts: Vec<&str> = texts.iter().map(String::as_str).collect();
    let batches: Vec<usize> = std::env::var("TURBO_SPEED_BATCHES")
        .unwrap_or_else(|_| "1,32,256,1024".into())
        .split(',')
        .map(|b| b.trim().parse().unwrap())
        .collect();

    let start = Instant::now();
    let l = Loaded::load(std::path::Path::new(&bundle)).unwrap_or_else(|e| panic!("{e:?}"));
    println!("{}: loaded in {:.1} ms, {} texts", bundle.display(), ms(start.elapsed()), texts.len());
    let max_batch = l.info().max_batch as usize;
    let (batches, over): (Vec<usize>, Vec<usize>) = batches.into_iter().partition(|&b| b <= max_batch);
    if !over.is_empty() {
        println!("  batches {over:?} are over the bundle's max_batch {max_batch}: not timed");
    }
    let mut o = embed_options();
    o.truncate = TURBO_TRUNCATE_MODEL;
    o.normalize = TURBO_NORMALIZE_L2;

    for (name, precision) in [("model", TURBO_PRECISION_MODEL), ("fastest", TURBO_PRECISION_FASTEST)] {
        for &b in &batches {
            let start = Instant::now();
            let s =
                Session::create(l.m, Some(&session_desc(b as u32, 0, precision))).unwrap_or_else(|e| panic!("{e:?}"));
            let made = start.elapsed();
            let pass = || {
                for chunk in texts.chunks(b) {
                    s.embed(chunk, Some(&o)).unwrap_or_else(|e| panic!("{e:?}"));
                }
            };
            pass();
            let best = (0..3)
                .map(|_| {
                    let (start, mut done) = (Instant::now(), 0);
                    while start.elapsed() < Duration::from_secs(2) {
                        pass();
                        done += texts.len();
                    }
                    done as f64 / start.elapsed().as_secs_f64()
                })
                .fold(0f64, f64::max);
            println!(
                "  turbo {name} ({}): batch {b}: {best:.0} texts/s (session made in {:.1} ms)",
                dtype_name(s.info().compute_dtype),
                ms(made)
            );
        }
    }
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn dtype_name(d: u32) -> &'static str {
    match d {
        TURBO_DTYPE_I8 => "I8",
        TURBO_DTYPE_F32 => "F32",
        _ => "other",
    }
}
