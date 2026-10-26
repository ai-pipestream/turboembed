//! Native dependency model operations over pretagged CoNLL-U.
#[path = "support/evaluate.rs"]
mod evaluate;
#[path = "support/train.rs"]
mod train;
use std::{error::Error, fs, path::Path};
use turbo_analysis::Ix;
use turbo_analysis_layers::Words;
use turbo_depparse::{
    DependencyModel, DependencyParser, DependencyParserME, FeedforwardDependencyModel, FeedforwardDependencyParser,
};
use turbo_formats::{conllu::words::ConlluWordReader, source::Field};
use turbo_train::{Manifest, ModelPackage};

const USAGE: &str = "usage: turbo-depparse <inspect|convert|parse> <linear|feedforward> MODEL [OUTPUT | INPUT.conllu OUTPUT.conllu]\n       turbo-depparse train <linear|feedforward> LANGUAGE TRAIN.conllu PARAMS OUTPUT [--projective-only]\n       turbo-depparse evaluate <linear|feedforward> MODEL GOLD.conllu REPORT.tsv MIN_UAS MIN_LAS";
type Result<T> = std::result::Result<T, Box<dyn Error>>;

enum Model {
    Linear(DependencyModel),
    Feedforward(FeedforwardDependencyModel),
}
impl Model {
    fn open(kind: &str, bytes: &[u8]) -> Result<Self> {
        Ok(match kind {
            "linear" => Self::Linear(if bytes.starts_with(b"PK") {
                DependencyModel::open(bytes.to_vec())?
            } else {
                let mut package = ModelPackage::new(Manifest::new("DependencyParserME", "und", 0));
                package.add_entry("depparse.model", bytes.to_vec());
                DependencyModel::open(package.to_bytes())?
            }),
            "feedforward" => Self::Feedforward(FeedforwardDependencyModel::from_bytes(bytes)?),
            _ => return Err(USAGE.into()),
        })
    }
    fn bytes(&self) -> Result<Vec<u8>> {
        Ok(match self {
            Self::Linear(m) => m.bytes().ok_or("linear package bytes unavailable")?.to_vec(),
            Self::Feedforward(m) => m.to_bytes()?,
        })
    }
    fn parse(self, input: &str) -> Result<String> {
        match self {
            Self::Linear(m) => parse(&DependencyParserME::new(m), input),
            Self::Feedforward(m) => {
                let beam = m.default_beam_size();
                parse(&FeedforwardDependencyParser::new(m, beam)?, input)
            }
        }
    }
}

fn parse(parser: &impl DependencyParser<Words>, input: &str) -> Result<String> {
    let mut output = String::new();
    for sentence in ConlluWordReader::new(input)? {
        let mut sentence = sentence?;
        if sentence.words.iter().enumerate().any(|(i, word)| word.id.as_str().parse::<usize>().ok() != Some(i + 1)) {
            return Err("parse requires contiguous syntactic word IDs starting at 1".into());
        }
        let words: Vec<_> = sentence.words.iter().map(|w| w.form.as_str()).collect();
        let tags: Vec<_> = sentence.words.iter().map(|w| w.upos.as_str()).collect();
        if tags.iter().any(|tag| tag.is_empty() || *tag == "_") {
            return Err("parse requires UPOS for every syntactic word".into());
        }
        // The gold HEAD/DEPREL fields are never inputs to inference.
        let graph = parser.parse(&words, &tags)?;
        for (i, word) in sentence.words.iter_mut().enumerate() {
            let arc = graph.arc(Ix::new(i as u32))?;
            word.head = Field::made(arc.head.map_or(0, |head| head.get() + 1).to_string(), word.head.span);
            word.deprel = Field::made(arc.relation.to_owned(), word.deprel.span);
        }
        sentence.write_conllu(&mut output);
    }
    Ok(output)
}

fn write_new(path: &str, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    // Conversion or parsing never silently overwrites a model or source corpus.
    let mut file = fs::OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    Ok(())
}

fn run(args: &[String]) -> Result<()> {
    if args.first().is_some_and(|arg| arg == "train") {
        if args.len() != 6 && !(args.len() == 7 && args[6] == "--projective-only") {
            return Err(USAGE.into());
        }
        return train::run(args);
    }
    let valid = match args.first().map(String::as_str) {
        Some("inspect") => args.len() == 3,
        Some("convert") => args.len() == 4,
        Some("parse") => args.len() == 5,
        Some("evaluate") => args.len() == 7,
        _ => false,
    };
    if !valid {
        return Err(USAGE.into());
    }
    let bytes = fs::read(Path::new(&args[2]))?;
    let model = Model::open(&args[1], &bytes)?;
    match args[0].as_str() {
        "evaluate" => evaluate::run(model, args, &turbo_maxent::sha256(&bytes).to_hex())?,
        "inspect" => {
            println!("kind={} source_sha256={}", args[1], turbo_maxent::sha256(&bytes).to_hex());
            if let Model::Feedforward(m) = model {
                println!(
                    "layout={:?} activation={:?} default_beam={} vocabulary={} transitions={}",
                    m.layout(),
                    m.activation(),
                    m.default_beam_size(),
                    m.vocabulary().len(),
                    m.transitions().len()
                );
            }
        }
        "convert" => {
            let converted = model.bytes()?;
            write_new(&args[3], &converted)?;
            println!(
                "source_sha256={} output_sha256={}",
                turbo_maxent::sha256(&bytes).to_hex(),
                turbo_maxent::sha256(&converted).to_hex()
            );
        }
        "parse" => {
            let input = fs::read_to_string(&args[3])?;
            write_new(&args[4], model.parse(&input)?.as_bytes())?;
        }
        _ => unreachable!(),
    }
    Ok(())
}
fn main() {
    if let Err(error) = run(&std::env::args().skip(1).collect::<Vec<_>>()) {
        eprintln!("{error}");
        std::process::exit(2);
    }
}
