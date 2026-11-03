//! Trains the native language model on the checked Tatoeba and FineWeb plans.

use std::error::Error;
use std::path::Path;

use turbo_analysis::ModelRef;
use turbo_train_recipes::fineweb::{self, Avoid, Overlap, Store};
use turbo_train_recipes::langdetect::{EXPORT, LANGDETECT, check_export};
use turbo_train_recipes::langdetect_web::WebInput;
use turbo_train_recipes::langid::{train_planned, write_langid};
use turbo_train_recipes::tatoeba::Export;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let [detailed, cc0, web_dir, out] = args.as_slice() else {
        return Err("usage: turbo-langid-train DETAILED CC0 WEB_RANGES_DIR OUT_DIR".into());
    };
    let export = Export::from_bytes(std::fs::read(detailed)?, std::fs::read(cc0)?)?;
    check_export(&export, EXPORT, false)?;
    let plan = LANGDETECT.plan(LANGDETECT.sentences(&export)?)?;

    let files = fineweb::parse_files(fineweb::FILES_NAME, fineweb::FILES)?;
    let ranges = fineweb::parse_ranges(fineweb::RANGES_NAME, fineweb::RANGES)?;
    let store = Store::new(Path::new(web_dir), Some(&ranges));
    let input = WebInput::checked(fineweb::WEB, &files, &ranges, true, &store)?;
    let avoid = Avoid {
        training: Overlap::new(
            fineweb::WEB.languages.len(),
            fineweb::WEB.split.overlap_min_chars,
            plan.held_out.iter().map(|s| (s.language, s.text.as_ref())),
        ),
        held_out: Overlap::new(
            fineweb::WEB.languages.len(),
            fineweb::WEB.split.overlap_min_chars,
            plan.training.iter().map(|s| (s.language, s.text.as_ref())),
        ),
    };
    let web = fineweb::plan_web(input.files, &fineweb::WEB, input.store, &avoid)?;
    let lists_sha256 = fineweb::lists_digest(input.files, input.ranges);

    let model_ref =
        ModelRef { id: "turbo-mul-tatoeba-langid-1".into(), revision: export.digest(), sha256: "unsealed".into() };
    let outcome = train_planned(&plan, Some(&web), model_ref)?;
    std::fs::create_dir(out)?;
    let model_path = write_langid(&outcome.model, Path::new(out))?;
    let model_sha256 = turbo_sha256::sha256(&std::fs::read(model_path)?).to_hex();
    let receipt = serde_json::json!({
        "model": "turbo-mul-tatoeba-langid-1",
        "model_sha256": model_sha256,
        "tatoeba_export_sha256": export.digest(),
        "web_lists_sha256": lists_sha256,
        "training_texts": outcome.training,
        "held_out_texts": outcome.held_out,
        "held_out_correct": outcome.correct,
        "held_out_accuracy": outcome.correct as f64 / outcome.held_out as f64,
    });
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}
