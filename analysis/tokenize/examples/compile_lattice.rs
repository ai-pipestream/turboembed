//! Compile a UTF-8 MeCab source directory into a prebuilt dictionary.
//! cargo run --release -p turbo-tokenize --example compile_lattice -- SOURCE OUTPUT [legacy]
use turbo_tokenize::{MecabDictionary, WhitespaceMode};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if !(2..=3).contains(&args.len()) || args.get(2).is_some_and(|s| s != "legacy") {
        return Err("expected SOURCE_DIRECTORY OUTPUT_FILE [legacy]".into());
    }
    let mode = if args.len() == 3 { WhitespaceMode::Legacy } else { WhitespaceMode::Unicode };
    let dictionary = MecabDictionary::load(&args[0], mode)?;
    dictionary.write_prebuilt(&args[1])?;
    println!(
        "source sha256={} entries={} surfaces={}",
        dictionary.sha256().to_hex(),
        dictionary.lexicon_len(),
        dictionary.surfaces()
    );
    Ok(())
}
