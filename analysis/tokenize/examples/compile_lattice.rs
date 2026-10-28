//! Compile a UTF-8 MeCab source directory into a prebuilt dictionary.
//! cargo run --release -p turbo-tokenize --example compile_lattice -- SOURCE OUTPUT
use turbo_tokenize::MecabDictionary;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 2 {
        return Err("expected SOURCE_DIRECTORY OUTPUT_FILE".into());
    }
    let dictionary = MecabDictionary::load(&args[0])?;
    dictionary.write_prebuilt(&args[1])?;
    println!(
        "source sha256={} entries={} surfaces={}",
        dictionary.sha256().to_hex(),
        dictionary.lexicon_len(),
        dictionary.surfaces()
    );
    Ok(())
}
