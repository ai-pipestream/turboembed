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
//! three times, and prints the best. Each batch's vectors are read into
//! one buffer kept for the cell, as a caller embedding a stream of texts
//! would read them, and the best run's time per batch in write_text, run
//! and the read follows. A release build is the one to time.

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
            let mut vectors = Vec::new();
            // Time in write_text, run and the read, and batches embedded.
            let mut phases = ([Duration::ZERO; 3], 0u32);
            let mut pass = |phases: &mut ([Duration; 3], u32)| {
                for chunk in texts.chunks(b) {
                    let t0 = Instant::now();
                    s.write_text(chunk, Some(&o)).unwrap_or_else(|e| panic!("{e:?}"));
                    let t1 = Instant::now();
                    let r = s.run().unwrap_or_else(|e| panic!("{e:?}"));
                    let t2 = Instant::now();
                    r.read_into(&mut vectors);
                    let t3 = Instant::now();
                    phases.0[0] += t1 - t0;
                    phases.0[1] += t2 - t1;
                    phases.0[2] += t3 - t2;
                    phases.1 += 1;
                }
            };
            pass(&mut phases);
            let (best, at) = (0..3)
                .map(|_| {
                    let mut ph = ([Duration::ZERO; 3], 0u32);
                    let (start, mut done) = (Instant::now(), 0);
                    while start.elapsed() < Duration::from_secs(2) {
                        pass(&mut ph);
                        done += texts.len();
                    }
                    (done as f64 / start.elapsed().as_secs_f64(), ph)
                })
                .fold((0f64, phases), |a, c| if c.0 > a.0 { c } else { a });
            let us = |d: Duration| d.as_secs_f64() * 1e6 / at.1 as f64;
            println!(
                "  turbo {name} ({}): batch {b}: {best:.0} texts/s (session made in {:.1} ms; per batch {:.1} us write, {:.1} run, {:.1} read)",
                dtype_name(s.info().compute_dtype),
                ms(made),
                us(at.0[0]),
                us(at.0[1]),
                us(at.0[2])
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
