//! The measurement behind `turbo-bench grammar`, built with the
//! `grammar` feature: the English number, unit and time grammars
//! over the goldens number inputs, as the `grammar_bench`
//! example times them.

use std::hint::black_box;
use std::path::Path;

use turbo_analysis::{Layer, ViewId};
use turbo_analysis_layers::DateTime;
use turbo_grammar::engine::Lexed;
use turbo_grammar::{ENGLISH, ENGLISH_UNITS, Grammar, GrammarAnnotator, Parser, TimeAnnotator};

use super::measure::{Options, goldens_text, median_pass, now, out_dir, record, run, same_tree};
use super::{Subject, write};
use crate::cpus::{Cpus, SYSFS_CPUS, pin};
use crate::{BUILD_CHANGES, BUILD_COMMIT, Result, check_build, git};

const USAGE: &str = "turbo-bench grammar [--rounds <n>] [--cpus <list>] [--repo <dir>] [--out <dir>]";

/// The text floor: the example's 100 KiB, in bytes.
const FLOOR: usize = 100 * 1024;

pub fn cmd(args: &[String]) -> Result<()> {
    let mut o = Options::parse(args, &["--rounds", "--cpus", "--repo", "--out"], &[], USAGE)?;
    if !o.positional.is_empty() {
        return Err(format!("{}: not an option\n{USAGE}", o.positional[0]));
    }
    let rounds = o.rounds()?;
    let cpus = o.take("--cpus").map(|l| Cpus::parse(&l, Path::new(SYSFS_CPUS))).transpose()?;
    let repo = o.repo();
    let out = o.out();
    // Refused before anything is measured, and checked again after.
    let before = git::provenance(&repo)?;
    check_build(&before.commit, BUILD_COMMIT, BUILD_CHANGES)?;
    if let Some(c) = &cpus {
        pin(c)?;
    }
    let recorded_at = now();
    let top = Path::new(&before.top).to_owned();
    let (text, source) = goldens_text(&top, &["num", "quantity", "money", "temporal"], FLOOR)?;
    eprintln!("english grammar: {} bytes from inputs/{{num,quantity,money,temporal}}", text.len());

    let annotator = GrammarAnnotator::english();
    let (find_units, find_secs) = median_pass(rounds, || {
        let found = black_box(annotator.find(&text, ViewId::SOURCE));
        Ok((found.numbers.len()
            + found.percents.len()
            + found.ordinals.len()
            + found.quantities.len()
            + found.money.len()) as u64)
    })?;

    let mut lexed = Lexed::default();
    let (lex_units, lex_secs) = median_pass(rounds, || {
        lexed.lex(&text, &ENGLISH.style());
        Ok(lexed.len() as u64)
    })?;

    let mut parser = Parser::new(&ENGLISH);
    let (parse_units, parse_secs) = median_pass(rounds, || {
        parser.parse(&text);
        Ok(parser.chart().nodes.len() as u64)
    })?;

    let mut units_parser = Parser::new(&ENGLISH_UNITS);
    let (units_units, units_secs) = median_pass(rounds, || {
        units_parser.parse(&text);
        Ok(units_parser.chart().nodes.len() as u64)
    })?;

    let times = TimeAnnotator::english();
    let at = DateTime::parse("2016-11-07T00:00:00");
    let (time_units, time_secs) = median_pass(rounds, || {
        let found = black_box(times.find(&text, ViewId::SOURCE, at));
        Ok(found.len() as u64)
    })?;

    let runs = vec![
        run("find", find_units, find_secs, &text),
        run("lex", lex_units, lex_secs, &text),
        run("parse", parse_units, parse_secs, &text),
        run("parse-units", units_units, units_secs, &text),
        run("times", time_units, time_secs, &text),
    ];
    for r in &runs {
        eprintln!("{}: {} units in {:.4} s (median)", r.phase, r.units, r.seconds);
    }

    let after = git::provenance(&repo)?;
    same_tree(&before, &after)?;
    let r = record(
        "grammar",
        Subject { name: "english".into(), sha256: String::new(), load_seconds: None },
        source,
        &text,
        match &cpus {
            Some(c) => format!("the tool ran pinned to CPUs {}, the grammar on one thread", c.list),
            None => "the tool ran unpinned, the grammar on one thread".into(),
        },
        format!(
            "turbo-bench grammar{} --rounds {rounds} --repo <repo> --out <dir>",
            cpus.as_ref().map_or(String::new(), |c| format!(" --cpus {}", c.list))
        ),
        rounds,
        runs,
        &after,
        recorded_at,
    )?;
    let path = write(&r, &out_dir(out, &after))?;
    println!("{}", path.display());
    Ok(())
}
