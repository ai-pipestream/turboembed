//! Dataset download helpers. Classical model commands are disabled under #1511.

mod cli;
mod commands;

use cli::{Args, USAGE, need, only, parse};
use commands::{fetch_ud_command, fineweb_ranges_command};
use std::process::ExitCode;
use turbo_train_recipes::masc::{self, RELEASE};

fn user() -> Result<turbo_zoo::UserId, String> {
    std::env::var("USER")
        .or_else(|_| std::env::var("USERNAME"))
        .map_err(|_| "USER is not set".to_owned())?
        .parse()
        .map_err(|e| format!("USER: {e}"))
}

fn fetch_command(a: Args) -> Result<bool, String> {
    only(&a, "fetch-masc", &["--home"])?;
    let home = need(a.home, "--home")?;
    let path = masc::fetch_release(RELEASE, &home, &user()?).map_err(|e| e.to_string())?;
    println!("{}", path.display());
    Ok(true)
}

fn main() -> ExitCode {
    let result = parse().and_then(|a| match a.command.as_str() {
        "fetch-masc" => fetch_command(a),
        "fetch-ud" => fetch_ud_command(a),
        "fineweb-ranges" => fineweb_ranges_command(a),
        _ => Err(USAGE.to_owned()),
    });
    match result {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("{e}");
            ExitCode::from(2)
        }
    }
}
