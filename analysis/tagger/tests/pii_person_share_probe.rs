//! Probe for #2398 (not for merge): what the trained head reads on names
//! a person and a city share, standing alone.

use std::path::Path;

use turbo_analysis::{Analyzer, Document};
use turbo_analysis_embed::EncoderOptions;
use turbo_analysis_layers::{Subwords, Tokens};
use turbo_pii::nemotron_custom_kinds;
use turbo_tagger::{NEMOTRON_PII_TEMPERATURE, PiiTagger};
use turbo_tokenize::SimpleTokenizer;

const PEOPLE: &[&str] = &[
    "Paris won the race on Saturday.",
    "Columbus drove home after the shift.",
    "They called Salem twice before Dayton answered.",
    "Austin finished the report early.",
    "Florence baked a cake for the office.",
    "Sofia missed the bus this morning.",
    "Charlotte forgot her keys again.",
    "Victoria was promoted in March.",
    "Sydney plays the cello.",
    "Orlando is on leave until Friday.",
    "I had lunch with Austin yesterday.",
    "Dallas and Houston both passed the exam.",
    "Denver is late again.",
    "Phoenix called in sick.",
    "Savannah wants the day off.",
    "Please send the form to Madison by noon.",
    "Dayton answered the phone.",
    "Milan lost his badge.",
    "Valencia will cover the late shift.",
    "My manager is Santiago.",
];

const CITIES: &[&str] = &[
    "Paris is lovely in the spring.",
    "Columbus hosts the fair every August.",
    "Salem voted on the measure last week.",
    "Austin has grown quickly over the past decade.",
    "Charlotte is a banking hub.",
    "Sofia was founded thousands of years ago.",
    "We visited Florence last summer.",
    "Victoria lies on the southern tip of the island.",
    "Sydney hosted the games in 2000.",
    "Orlando draws millions of tourists.",
    "Dallas and Houston are four hours apart.",
    "Denver sits a mile above sea level.",
    "Dayton built the first airplanes.",
    "They left Salem at dawn.",
    "Madison passed a new zoning rule.",
    "Phoenix reached 115 degrees.",
    "Salem is where the old office was.",
    "We saw Columbus from the plane.",
    "The Dayton office still holds the files.",
    "Milan has two football clubs.",
    "Valencia floods most winters.",
    "The flight to Santiago was delayed.",
];

#[test]
fn probe() {
    let Some(bundle_path) = std::env::var_os("TURBO_NEMOTRON_PII_BUNDLE") else {
        return;
    };
    let bundle_dir = turbo_pins::require(Path::new(&bundle_path));
    let tagger = PiiTagger::open(&bundle_dir, None, &EncoderOptions::default())
        .unwrap()
        .with_temperature(NEMOTRON_PII_TEMPERATURE)
        .unwrap();
    let kinds = nemotron_custom_kinds();
    let analyzer = Analyzer::builder().add(SimpleTokenizer::default()).build().unwrap();
    let memo = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../pii/tests/data/human-eval/places-person-names.txt"),
    )
    .unwrap();
    let mut texts: Vec<(&str, String)> = vec![("memo", memo)];
    texts.extend(PEOPLE.iter().map(|t| ("person", (*t).to_owned())));
    texts.extend(CITIES.iter().map(|t| ("city", (*t).to_owned())));
    for (gold, text) in &texts {
        let mut docs = vec![Document::new(text.as_str())];
        for r in analyzer.run(&mut docs) {
            r.unwrap();
        }
        let doc = &docs[0];
        let view = doc.get::<turbo_analysis::View>().unwrap();
        let tokens = doc.get::<Tokens>().unwrap();
        let subwords = doc.get::<Subwords>();
        let got = tagger.model_candidates_person(view, tokens, subwords).unwrap();
        if *gold != "memo" {
            println!("[{gold}] {text}");
        }
        for (mc, person) in got {
            let name = mc.kind.builtin_name().or_else(|| kinds.name(mc.kind));
            let covered = &text[mc.span.range()];
            let before = &text[..mc.span.start as usize];
            let ctx: String = before.chars().rev().take(18).collect::<String>().chars().rev().collect();
            let after: String = text[mc.span.end as usize..].chars().take(18).collect();
            println!(
                "  {gold:<6} {:>4} {covered:<10} kind {:<12} p {:.3} person {:.3}  | {}[{covered}]{}",
                mc.span.start,
                name.unwrap_or_default(),
                mc.probability,
                person,
                ctx.replace('\n', " "),
                after.replace('\n', " ")
            );
        }
    }
}
