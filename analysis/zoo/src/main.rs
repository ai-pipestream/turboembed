//! `turbo-zoo`: list, accept licences for, and fetch the zoo's artifacts.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{SystemTime, UNIX_EPOCH};

use turbo_zoo::{
    Acceptances, Http, Index, Licence, LicenceClass, LocalOnlyReason, Policy, Requirement, Sha256, Sha512, UserId, Zoo,
    ZooError,
};

const USAGE: &str = "usage: turbo-zoo [--index <zoo.json>] <command>
  list                 the artifacts the zoo may fetch, with class and licences
  show <id>            one artifact: licences, training data, files
  accept <licence> [<terms>] [--yes]
                       show the text of a licence (SPDX id, as `show` prints it), and
                       for local-only terms what they allow, and, with --yes after you
                       have read it, record that you accept it;
                       LicenseRef-RestrictedPublic also takes the host's terms address
  revoke <licence> [<terms>]
                       withdraw an acceptance
  accepted             the licences you have accepted
  fetch <id>           fetch and verify an artifact; prints its local files
  hash <file>...       size, SHA-256 and SHA-512 of files, for an index entry

The zoo keeps its cache and acceptance file in $TURBO_ZOO_HOME, else
$XDG_DATA_HOME/turbo-zoo, else ~/.local/share/turbo-zoo. The user is $USER
(or $USERNAME).";

fn main() -> ExitCode {
    match run(std::env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("turbo-zoo: {e}");
            ExitCode::FAILURE
        }
    }
}

fn home() -> Result<PathBuf, String> {
    let var = |k| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    if let Some(h) = var("TURBO_ZOO_HOME") {
        return Ok(h);
    }
    if let Some(x) = var("XDG_DATA_HOME") {
        return Ok(x.join("turbo-zoo"));
    }
    var("HOME").map(|h| h.join(".local/share/turbo-zoo")).ok_or_else(|| "set TURBO_ZOO_HOME or HOME".into())
}

fn user() -> Result<UserId, String> {
    let name = std::env::var("USER").or_else(|_| std::env::var("USERNAME")).map_err(|_| "set USER".to_string())?;
    name.parse()
}

/// The terms `accept` and `revoke` name: a licence with its own text, or
/// one without (LicenseRef-RestrictedPublic) and the terms address `show`
/// prints for it.
fn requirement(licence: &str, terms: Option<&str>) -> Result<Requirement, String> {
    let licence: Licence = licence.parse()?;
    match (Requirement::of(licence), terms) {
        (Some(r), None) => Ok(r),
        (Some(_), Some(_)) => Err(format!("{licence} has its own text; give no terms address")),
        (None, Some(t)) => Ok(Requirement { licence, text: t.to_owned() }),
        (None, None) => Err(format!("{licence} needs the terms address `show` prints for the artifact")),
    }
}

fn run(mut args: Vec<String>) -> Result<(), String> {
    let index = if args.first().map(String::as_str) == Some("--index") {
        if args.len() < 2 {
            return Err(USAGE.into());
        }
        let path = args.remove(1);
        args.remove(0);
        let bytes = std::fs::read(&path).map_err(|e| format!("{path}: {e}"))?;
        Index::parse(&bytes).map_err(|e| e.to_string())?
    } else {
        Index::builtin()
    };
    let e = |e: ZooError| e.to_string();
    let home = home()?;
    let accept_path = home.join("acceptances.json");
    match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["list"] => {
            for a in index.list() {
                let licences: Vec<String> = a.licences().map(|l| l.to_string()).collect();
                println!("{}\t{}\t{}", a.id, a.class, licences.join(" + "));
            }
            Ok(())
        }
        ["show", id] => {
            let a = index.get(id).map_err(e)?;
            println!("id: {}\nkind: {}\nformat: {}\nversion: {}", a.id, a.kind, a.format, a.version);
            if let Some(c) = a.component {
                println!("component: {c}");
            }
            if let Some(l) = &a.language {
                println!("language: {l}");
            }
            println!("class: {}", a.class);
            println!("license: {} ({})", a.license, a.license_terms);
            for t in &a.trained_on {
                println!("trained on: {} under {} ({})", t.name, t.license, t.license_terms);
            }
            for r in a.licences_to_accept(Policy::default()) {
                println!("accept first: {} {}", r.licence, r.text);
            }
            if let Some(o) = &a.origin {
                println!("origin: {o} (fetched from there only)");
            }
            let mut reasons: Vec<LocalOnlyReason> = Vec::new();
            for reason in a.licences().filter_map(Licence::local_only) {
                if !reasons.contains(&reason) {
                    println!("local only: {}", reason.notice());
                    reasons.push(reason);
                }
            }
            for f in &a.files {
                println!("file: {} {} bytes sha256 {} from {}", f.path, f.size, f.sha256, f.url);
            }
            Ok(())
        }
        ["accept", licence, rest @ ..] => {
            let yes = rest.contains(&"--yes");
            let terms: Vec<&str> = rest.iter().copied().filter(|a| *a != "--yes").collect();
            if terms.len() > 1 {
                return Err(USAGE.into());
            }
            let r = requirement(licence, terms.first().copied())?;
            if let Some(reason) = r.licence.refusal() {
                return Err(format!("{}: refused ({reason}); the zoo fetches nothing under it", r.licence));
            }
            if r.licence.class() == LicenceClass::Permissive {
                return Err(format!("{} is {}: there is nothing to accept", r.licence, LicenceClass::Permissive));
            }
            println!("{}: read the text at {}", r.licence, r.text);
            if let Some(reason) = r.licence.local_only() {
                println!("{}", reason.notice());
            }
            if !yes {
                return Err("nothing recorded: once you have read the text and accept it, run the same command \
                            again with --yes"
                    .into());
            }
            let user = user()?;
            let mut acc = Acceptances::load(&accept_path).map_err(e)?;
            let now = SystemTime::now().duration_since(UNIX_EPOCH).map_err(|e| e.to_string())?.as_secs();
            acc.accept_terms(&user, &r, now).map_err(e)?;
            acc.save(&accept_path).map_err(e)?;
            println!("{user} accepts {}", r.licence);
            Ok(())
        }
        ["revoke", licence, rest @ ..] if rest.len() <= 1 => {
            let r = requirement(licence, rest.first().copied())?;
            let user = user()?;
            let mut acc = Acceptances::load(&accept_path).map_err(e)?;
            if acc.revoke(&user, &r) {
                acc.save(&accept_path).map_err(e)?;
                println!("{user} no longer accepts {r}");
            } else {
                println!("{user} had not accepted {r}");
            }
            Ok(())
        }
        ["accepted"] => {
            let user = user()?;
            for a in Acceptances::load(&accept_path).map_err(e)?.iter().filter(|a| a.user == user) {
                println!("{}\t{}\t{}", a.license, a.accepted_at, a.license_text);
            }
            Ok(())
        }
        ["fetch", id] => {
            let user = user()?;
            let zoo = Zoo::new(index, home.join("artifacts"), Policy::default(), &Http);
            // The acceptance file is read only for an artifact that needs it.
            let fetched = zoo.fetch_with(id, &user, || Acceptances::load(&accept_path)).map_err(e)?;
            for f in &fetched.files {
                println!("{}", f.display());
            }
            Ok(())
        }
        ["hash", files @ ..] if !files.is_empty() => {
            for f in files {
                let bytes = std::fs::read(f).map_err(|e| format!("{f}: {e}"))?;
                println!("{}\t{}\t{}\t{f}", bytes.len(), Sha256::of(&bytes), Sha512::of(&bytes));
            }
            Ok(())
        }
        _ => Err(USAGE.into()),
    }
}
