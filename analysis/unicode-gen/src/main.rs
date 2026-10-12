//! Generates the tables in `analysis/unicode/src/tables/` from the Unicode
//! Character Database files in `analysis/unicode/ucd/`.

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: turbo-unicode-gen <ucd directory> <output directory>");
        std::process::exit(2);
    }
}
