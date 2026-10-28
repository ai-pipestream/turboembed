//! Local CoNLL-2003 evaluation: see docs/namefinder-evaluation.md.
use std::{
    env, fs,
    io::{self, Write},
};
use turbo_train_recipes::namefind_eval::{Record, evaluate};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    if !(3..=4).contains(&args.len()) {
        return Err("usage: namefind_eval MODEL.bin CORPUS.conll all|per|org|loc|misc [JVM.json]".into());
    }
    let record = evaluate(&fs::read(&args[0])?, &fs::read_to_string(&args[1])?, &args[2])?;
    eprintln!("turbo {}", serde_json::to_string(&record.summary()?)?);
    if let Some(path) = args.get(3) {
        let reference: Record = serde_json::from_slice(&fs::read(path)?)?;
        let max = record.compare(&reference, 1e-9)?;
        eprintln!("reference {}", serde_json::to_string(&reference.summary()?)?);
        eprintln!("prediction parity PASS; max_probability_error={max}");
    }
    let mut out = io::BufWriter::new(io::stdout().lock());
    serde_json::to_writer_pretty(&mut out, &record)?;
    writeln!(out)?;
    Ok(())
}
