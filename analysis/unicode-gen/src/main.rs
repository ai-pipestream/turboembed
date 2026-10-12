//! Generates the tables in `analysis/unicode/src/tables/` from the Unicode
//! Character Database files in `analysis/unicode/ucd/`.

mod java;
mod rust_enum;
mod trie;
mod ucd;
mod word_break;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: turbo-unicode-gen <ucd directory> <output directory>");
        std::process::exit(2);
    }
    let (ucd_dir, out_dir) = (std::path::Path::new(&args[1]), std::path::Path::new(&args[2]));
    if let Err(error) = java::generate(ucd_dir, out_dir).and_then(|()| word_break::generate(ucd_dir, out_dir)) {
        eprintln!("turbo-unicode-gen: {error}");
        std::process::exit(1);
    }
}
