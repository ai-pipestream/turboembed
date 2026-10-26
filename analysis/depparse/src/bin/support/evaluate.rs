use super::{Model, Result, write_new};
use std::{fs, path::Path};
use turbo_analysis::ViewId;
use turbo_analysis_layers::Words;
use turbo_depparse::{
    DependencyParser, DependencyParserME, DependencySample, FeedforwardDependencyParser, HeldOutEvaluation,
};
use turbo_formats::conllu::{ConlluDependencySample, Tagset, words::ConlluWordReader};
use turbo_unicode::WhitespaceMode;

pub(super) fn run(model: Model, args: &[String], model_hash: &str) -> Result<()> {
    let input = fs::read_to_string(&args[3])?;
    if Path::new(&args[4]).exists() {
        return Err("output already exists".into());
    }
    let uas = args[5].parse::<f64>()?;
    let las = args[6].parse::<f64>()?;
    HeldOutEvaluation::default().passes(uas, las)?;
    let report = match model {
        Model::Linear(m) => evaluate(&DependencyParserME::new(m), &input)?,
        Model::Feedforward(m) => {
            let beam = m.default_beam_size();
            evaluate(&FeedforwardDependencyParser::new(m, beam)?, &input)?
        }
    };
    let passed = report.passes(uas, las)?;
    let provenance = format!(
        "# corpus_sha256={} model_sha256={} minimum_uas={uas} minimum_las={las} passed={passed}\n# inputs=FORM,UPOS relations=exact punctuation=UPOS:PUNCT\n",
        turbo_maxent::sha256(input.as_bytes()).to_hex(),
        model_hash
    );
    write_new(&args[4], format!("{provenance}{}", report.write_tsv()).as_bytes())?;
    if !passed {
        return Err("held-out UAS/LAS gate failed; report retains all sentences and failures".into());
    }
    Ok(())
}
fn evaluate(parser: &impl DependencyParser<Words>, input: &str) -> Result<HeldOutEvaluation> {
    let mut report = HeldOutEvaluation::default();
    for (index, sentence) in ConlluWordReader::new(input)?.enumerate() {
        let sentence = sentence?;
        if sentence.words.iter().any(|word| word.upos.as_str() == "_") {
            return Err(format!("sentence {} requires UPOS", index + 1).into());
        }
        let span = sentence.source;
        let sample = DependencySample::<Words>::from_conllu(ConlluDependencySample::read_sentence(
            &input[span.start as usize..span.end as usize],
            Tagset::U,
            WhitespaceMode::Unicode,
            ViewId::SOURCE,
        )?)?;
        let forms: Vec<_> = sample.tokens().iter().map(String::as_str).collect();
        let tags: Vec<_> = sample.tags().iter().map(String::as_str).collect();
        // Gold HEAD and DEPREL are used only after prediction, for scoring.
        let predicted = parser.parse(&forms, &tags);
        if let Err(error) = &predicted {
            eprintln!("sentence {} failed: {error}", index + 1);
        }
        report.record(&sample, predicted.as_ref().ok())?;
    }
    Ok(report)
}
