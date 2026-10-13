//! `turbo-bpe-trainer`: trains a BPE model from a UTF-8 corpus, one text
//! per line, and writes it (docs/bpe-trainer.md).
//!
//! ```text
//! turbo-bpe-trainer byte-level <corpus> <tokenizer.json> --vocab-size <n>
//!     [--min-frequency <n>] [--special <token>]... [--lstrip <token>]...
//!     [--add-prefix-space] [--roberta <cls> <sep>]
//! turbo-bpe-trainer sentencepiece <corpus> <model> --vocab-size <n>
//!     [--character-coverage <c>]
//! turbo-bpe-trainer opennlp <corpus> <merges.txt> --merges <n>
//!     [--ties jvm|made-first]
//! ```

use std::process::ExitCode;

use turbo_bpe_trainer::{
    ByteLevelBpe, ByteLevelConfig, OpenNlpBpe, PostProcessor, SentencePieceBpe, SentencePieceConfig, SpecialToken,
    TieOrder,
};

const USAGE: &str = "usage:
  turbo-bpe-trainer byte-level <corpus> <tokenizer.json> --vocab-size <n> [--min-frequency <n>]
      [--special <token>]... [--lstrip <token>]... [--add-prefix-space] [--roberta <cls> <sep>]
  turbo-bpe-trainer sentencepiece <corpus> <model> --vocab-size <n> [--character-coverage <c>]
  turbo-bpe-trainer opennlp <corpus> <merges.txt> --merges <n> [--ties jvm|made-first]";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("turbo-bpe-trainer: {e}");
            ExitCode::FAILURE
        }
    }
}

/// The options after the command and its two paths.
struct Options<'a> {
    rest: &'a [String],
}

impl<'a> Options<'a> {
    /// The value after `--name`, or None when it is not given.
    fn value(&self, name: &str) -> Result<Option<&'a str>, String> {
        let mut found = None;
        for (i, a) in self.rest.iter().enumerate() {
            if a == name {
                let v = self.rest.get(i + 1).ok_or_else(|| format!("{name} needs a value"))?;
                found = Some(v.as_str());
            }
        }
        Ok(found)
    }

    /// Every value given after `--name`.
    fn values(&self, name: &str) -> Vec<&'a str> {
        self.rest.windows(2).filter(|w| w[0] == name).map(|w| w[1].as_str()).collect()
    }

    fn flag(&self, name: &str) -> bool {
        self.rest.iter().any(|a| a == name)
    }

    fn number<T: std::str::FromStr>(&self, name: &str) -> Result<Option<T>, String> {
        match self.value(name)? {
            None => Ok(None),
            Some(v) => v.parse().map(Some).map_err(|_| format!("{name}: {v:?} is not a number")),
        }
    }

    fn required<T: std::str::FromStr>(&self, name: &str) -> Result<T, String> {
        self.number(name)?.ok_or_else(|| format!("{name} is required\n{USAGE}"))
    }
}

fn run(args: &[String]) -> Result<(), String> {
    let [command, corpus, out, rest @ ..] = args else { return Err(USAGE.into()) };
    let o = Options { rest };
    let text = std::fs::read_to_string(corpus).map_err(|e| format!("{corpus}: {e}"))?;
    // Split at line feeds only, as the sentencepiece library reads a
    // file: a carriage return stays in the text. An empty line adds
    // nothing.
    let lines = text.split('\n');
    let bytes = match command.as_str() {
        "byte-level" => {
            let lstrip = o.values("--lstrip");
            let special_tokens = o
                .values("--special")
                .into_iter()
                .map(|t| SpecialToken { content: t.into(), lstrip: lstrip.contains(&t) })
                .collect();
            let post_processor = match rest.iter().position(|a| a == "--roberta") {
                None => PostProcessor::None,
                Some(i) => match (rest.get(i + 1), rest.get(i + 2)) {
                    (Some(cls), Some(sep)) => PostProcessor::Roberta { cls: cls.clone(), sep: sep.clone() },
                    _ => return Err("--roberta needs the cls and sep tokens".into()),
                },
            };
            let config = ByteLevelConfig {
                vocab_size: o.required("--vocab-size")?,
                min_frequency: o.number("--min-frequency")?.unwrap_or(0),
                special_tokens,
                add_prefix_space: o.flag("--add-prefix-space"),
                post_processor,
            };
            let m = ByteLevelBpe::train(lines, config).map_err(|e| e.to_string())?;
            eprintln!("{} pieces, {} merges", m.vocab().len(), m.merges().len());
            m.to_tokenizer_json().into_bytes()
        }
        "sentencepiece" => {
            let config = SentencePieceConfig {
                vocab_size: o.required("--vocab-size")?,
                character_coverage: o.number("--character-coverage")?.unwrap_or(0.9995),
            };
            let m = SentencePieceBpe::train(lines, config).map_err(|e| e.to_string())?;
            eprintln!("{} pieces", m.pieces().len());
            m.to_model_proto()
        }
        "opennlp" => {
            let ties = match o.value("--ties")? {
                None | Some("jvm") => TieOrder::Jvm,
                Some("made-first") => TieOrder::MadeFirst,
                Some(other) => return Err(format!("--ties {other}: expected jvm or made-first")),
            };
            let m = OpenNlpBpe::train_with(lines, o.required("--merges")?, ties).map_err(|e| e.to_string())?;
            eprintln!("{} merges", m.merges().len());
            if ties == TieOrder::Jvm && !m.is_jvm_exact() {
                eprintln!(
                    "the merges may differ from the JVM's: the corpus holds a character outside the Basic \
                     Multilingual Plane, or a bin of the JVM's maps became a tree (docs/bpe-trainer.md)"
                );
            }
            m.to_merges_text().into_bytes()
        }
        _ => return Err(USAGE.into()),
    };
    std::fs::write(out, bytes).map_err(|e| format!("{out}: {e}"))
}
