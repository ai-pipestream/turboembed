//! Dump exact selected values, scores and application counts for the checked-in corpus.
//! Compare dumps from two revisions to check parity, including cases outside the pass lists.
use std::{error::Error, fmt::Write as _, fs, path::Path};
use turbo_grammar::{
    DUTCH, DUTCH_TIMES, ENGLISH, ENGLISH_TIMES, ENGLISH_UNITS, Grammar, Output, Parser, SPANISH, SPANISH_TIMES,
};

fn snapshot(
    root: &Path,
    language: &str,
    family: &str,
    model: &str,
    grammar: &'static dyn Grammar,
    output: Output,
    dump: &mut String,
) -> Result<usize, Box<dyn Error>> {
    let file = root.join(language).join(family).join(format!("{model}.json"));
    let cases: Vec<serde_json::Value> = serde_json::from_str(fs::read_to_string(file)?.trim_start_matches('\u{feff}'))?;
    let mut parser = Parser::new(grammar);
    let mut mentions = Vec::new();
    for (i, case) in cases.iter().enumerate() {
        let text = case["Input"].as_str().ok_or("missing corpus input")?;
        writeln!(dump, "{language}/{family}/{model}:{i} {text:?}")?;
        parser.parse(text);
        mentions.clear();
        parser.select(text, output, &mut mentions);
        for m in &mentions {
            let node = &parser.chart().nodes[m.node as usize];
            write!(dump, "{}..{} score={} apps={} ", m.start, m.end, node.score, node.apps)?;
            if let Some(time) = parser.times().get(&m.value) {
                writeln!(dump, "{time:?}")?;
            } else {
                writeln!(dump, "{:?}", m.value)?;
            }
        }
    }
    Ok(cases.len())
}

fn main() -> Result<(), Box<dyn Error>> {
    let output = std::env::args_os().nth(1).ok_or("usage: corpus_snapshot OUTPUT_FILE")?;
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/recognizers-text");
    let mut dump = String::new();
    let mut count = 0;
    for (language, grammar, times) in [
        ("English", &ENGLISH as &dyn Grammar, &ENGLISH_TIMES as &dyn Grammar),
        ("Spanish", &SPANISH, &SPANISH_TIMES),
        ("Dutch", &DUTCH, &DUTCH_TIMES),
    ] {
        for (model, kind) in
            [("NumberModel", Output::Number), ("OrdinalModel", Output::Ordinal), ("PercentModel", Output::Percent)]
        {
            count += snapshot(&root, language, "Number", model, grammar, kind, &mut dump)?;
        }
        count += snapshot(&root, language, "DateTime", "DateTimeModel", times, Output::Time, &mut dump)?;
    }
    for model in ["AgeModel", "CurrencyModel", "DimensionModel", "TemperatureModel"] {
        count += snapshot(&root, "English", "NumberWithUnit", model, &ENGLISH_UNITS, Output::Quantity, &mut dump)?;
    }
    fs::write(output, dump)?;
    println!("{count} corpus cases recorded");
    Ok(())
}
