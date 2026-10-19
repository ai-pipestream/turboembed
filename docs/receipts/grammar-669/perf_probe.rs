use std::{hint::black_box, time::Instant, path::Path, fs};
use turbo_grammar::{Grammar,Parser,Output,ENGLISH,ENGLISH_UNITS,ENGLISH_TIMES,SPANISH,SPANISH_TIMES,DUTCH};
#[path = "../tests/common/sample.rs"] mod sample;
fn corpus(root: &Path, lang: &str, family: &str, model: &str) -> Vec<String> {
    let raw = fs::read_to_string(root.join(lang).join(family).join(format!("{model}.json"))).unwrap();
    let cases: Vec<serde_json::Value> = serde_json::from_str(raw.trim_start_matches('\u{feff}')).unwrap();
    cases.iter().map(|v|v["Input"].as_str().unwrap().to_owned()).collect()
}
fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/recognizers-text");
    let sample = vec![sample::sample(100 * 1024)];
    let en = corpus(&root,"English","Number","NumberModel");
    let ent = corpus(&root,"English","DateTime","DateTimeModel");
    let unit = corpus(&root,"English","NumberWithUnit","DimensionModel");
    let es = corpus(&root,"Spanish","Number","NumberModel");
    let est = corpus(&root,"Spanish","DateTime","DateTimeModel");
    let nl = corpus(&root,"Dutch","Number","NumberModel");
    let nums = &[Output::Number,Output::Ordinal,Output::Percent][..];
    let configs: [(&str,&dyn Grammar,&Vec<String>,&[Output]);9] = [
        ("sample_numbers",&ENGLISH,&sample,nums), ("sample_units",&ENGLISH_UNITS,&sample,&[Output::Quantity]),
        ("sample_times",&ENGLISH_TIMES,&sample,&[Output::Time]), ("en_numbers",&ENGLISH,&en,nums),
        ("en_datetime",&ENGLISH_TIMES,&ent,&[Output::Time]), ("en_dimension",&ENGLISH_UNITS,&unit,&[Output::Quantity]),
        ("es_numbers",&SPANISH,&es,nums), ("es_datetime",&SPANISH_TIMES,&est,&[Output::Time]),
        ("nl_numbers",&DUTCH,&nl,nums)];
    for (name,grammar,inputs,outputs) in configs {
        let mut p=Parser::new(grammar); let mut out=Vec::new(); let mut hash=14695981039346656037u64;
        for t in inputs {
            p.parse(t); out.clear(); for &o in outputs {p.select(t,o,&mut out);}
            let dump = format!("{} {:?}",p.chart().nodes.len(),out.iter().map(|m|(m.start,m.end,p.chart().nodes[m.node as usize].score,p.chart().nodes[m.node as usize].apps,format!("{:?}",m.value))).collect::<Vec<_>>());
            for b in dump.bytes() {hash=(hash ^ u64::from(b)).wrapping_mul(1099511628211);}
        }
        let now=Instant::now(); let runs=20;
        for _ in 0..runs {for t in inputs {p.parse(black_box(t));out.clear();for &o in outputs {p.select(t,o,&mut out);} black_box(&out);}}
        println!("{name}\t{:.3}\t{hash:016x}",now.elapsed().as_secs_f64()*1e6/f64::from(runs));
    }
}
