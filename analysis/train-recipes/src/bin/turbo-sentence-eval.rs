//! Offline gold-label sentence evaluation. Never downloads or trains anything.
#[path = "../sentence_eval/mod.rs"]
mod sentence_eval;

fn main() -> std::process::ExitCode {
    match sentence_eval::run(std::env::args_os().skip(1).collect()) {
        Ok(pass) => std::process::ExitCode::from(if pass { 0 } else { 2 }),
        Err(error) => {
            eprintln!("{error}");
            std::process::ExitCode::FAILURE
        }
    }
}
