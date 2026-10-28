//! Optional local corpus evaluation. See ../EVALUATION.md.

use std::path::Path;
use std::process::ExitCode;

use turbo::bundle::{Bundle, sha256_hex};
use turbo_analysis_embed::{EncoderOptions, Precision};
use turbo_analysis_layers::Tag;
use turbo_namefind_dl::{Decoding, NameFinderDl, NameFinderDlOptions};

#[path = "conll_eval/support.rs"]
mod support;
use support::{Counts, Kind};

const USAGE: &str = "usage: conll_eval <bundle> <eng.testb> <upstream.conll> <upstream-revision> <word|piece>\n\
Inputs must already be obtained under their terms. This command downloads nothing.\n\
The upstream file is four-column CoNLL-2003 with predictions in column four.";

fn main() -> ExitCode {
    match run(&std::env::args().skip(1).collect::<Vec<_>>()) {
        Ok(receipt) => {
            print!("{receipt}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("conll_eval: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &[String]) -> Result<String, String> {
    let [bundle, gold_path, peer_path, revision, mode] = args else { return Err(USAGE.into()) };
    let decoding = match mode.as_str() {
        "word" => Decoding::Word,
        "piece" => Decoding::Piece,
        _ => return Err("decoding must be word or piece".into()),
    };
    if revision.len() != 40 || !revision.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("upstream revision must be a full 40-character commit".into());
    }
    let read = |path: &str, role| std::fs::read_to_string(path).map_err(|_| format!("cannot read {role} as UTF-8"));
    let gold_text = read(gold_path, "gold corpus")?;
    let peer_text = read(peer_path, "upstream predictions")?;
    let gold = support::read(&gold_text).map_err(|e| format!("gold: {e}"))?;
    let peer = support::read(&peer_text).map_err(|e| format!("upstream: {e}"))?;
    support::aligned(&gold, &peer)?;
    let bundle = Path::new(bundle);
    let info = Bundle::open(bundle).map_err(|error| format!("cannot open native bundle: {error}"))?;
    if !info.manifest.model.source.commit.eq_ignore_ascii_case(revision) {
        return Err("native bundle and upstream predictions must name the same source commit".into());
    }
    let options = NameFinderDlOptions {
        decoding,
        encoder: EncoderOptions { precision: Precision::Exact, cpu_threads: 2, ..Default::default() },
        ..Default::default()
    };
    let finder = NameFinderDl::open(bundle, &options)
        .map_err(|error| format!("cannot load native token-classification model: {error}"))?;
    let kinds = (0..finder.kinds().len())
        .map(|i| Kind::parse(finder.kinds().label(Tag(i as u16))))
        .collect::<Result<Vec<_>, _>>()?;
    let (mut native, mut upstream) = ([Counts::default(); 5], [Counts::default(); 5]);
    let mut differing_samples = 0;
    let mut agreement = Counts::default();
    for (gold_batch, peer_batch) in gold.chunks(16).zip(peer.chunks(16)) {
        let texts: Vec<_> = gold_batch.iter().map(|s| support::joined(&s.words)).collect();
        let inputs: Vec<_> = texts.iter().map(|(t, _)| (t.as_str(), vec![(0, t.len())])).collect();
        let found = finder.find(&inputs).map_err(|_| "native inference failed")?;
        if found.len() != gold_batch.len() {
            return Err("native inference returned an incorrect sample count".into());
        }
        for (((g, p), (text, spans)), candidates) in gold_batch.iter().zip(peer_batch).zip(&texts).zip(&found) {
            let predictions = support::project(text, spans, candidates, &kinds)?;
            differing_samples += usize::from(predictions != p.entities);
            agreement.update(&p.entities, &predictions, None);
            for (i, kind) in std::iter::once(None).chain(Kind::ALL.map(Some)).enumerate() {
                native[i].update(&g.entities, &predictions, kind);
                upstream[i].update(&g.entities, &p.entities, kind);
            }
        }
    }
    // Aggregate-only output: no input paths, text, tokens or per-document predictions.
    let mut receipt = format!(
        "protocol=conll2003-token-span-v1\nnative_manifest_sha256={}\ngold_sha256={}\nupstream_predictions_sha256={}\nsource_commit={revision}\ndevice=cpu precision=exact cpu_threads=2 decoding={mode}\nsamples={} tokens={}\n",
        info.manifest_sha256,
        sha256_hex(gold_text.as_bytes()),
        sha256_hex(peer_text.as_bytes()),
        gold.len(),
        gold.iter().map(|s| s.words.len()).sum::<usize>()
    );
    for (i, label) in std::iter::once("all").chain(Kind::ALL.map(Kind::name)).enumerate() {
        receipt.push_str(&counts_line("native", label, native[i]));
        receipt.push_str(&counts_line("upstream", label, upstream[i]));
    }
    receipt.push_str(&format!(
        "f1_delta={:.9}\ndiffering_samples={differing_samples}\n",
        native[0].f1() - upstream[0].f1()
    ));
    receipt.push_str(&counts_line("agreement", "all", agreement));
    receipt.push_str(if differing_samples == 0 {
        "parity=exact_entity_agreement\n"
    } else {
        "parity=different_entities\n"
    });
    Ok(receipt)
}

fn counts_line(system: &str, kind: &str, c: Counts) -> String {
    format!(
        "{system}.{kind}: gold={} predicted={} correct={} precision={:.9} recall={:.9} f1={:.9}\n",
        c.gold,
        c.predicted,
        c.correct,
        c.precision(),
        c.recall(),
        c.f1()
    )
}

#[cfg(test)]
#[path = "conll_eval/tests.rs"]
mod tests;
