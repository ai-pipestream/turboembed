//! `turbo-zoo import-bundle`: a sealed bundle directory written as an
//! index entry, printed or added to an index file.

use std::path::Path;

use turbo_zoo::import::{BundleImport, add_to, bundle};
use turbo_zoo::{Index, TrainingData};

pub const USAGE: &str =
    "usage: turbo-zoo import-bundle <bundle-dir> --id <id> --base <https url> --license-terms <https url>
    [--trained-on <name> <licence> <https url>]... [--attribution <text>] [--origin <https url>]
    [--component <COMPONENT_*>] [--language <tag>] [--into <zoo.json>]";

/// Runs `import-bundle` with the arguments after the command's name.
pub fn run(args: &[&str]) -> Result<(), String> {
    let [dir, flags @ ..] = args else { return Err(USAGE.into()) };
    let (mut id, mut base, mut terms, mut into) = (None, None, None, None);
    let (mut attribution, mut origin, mut component, mut language) = (None, None, None, None);
    let mut trained_on = Vec::new();
    let mut rest = flags.iter().copied();
    while let Some(flag) = rest.next() {
        let mut value = || rest.next().ok_or_else(|| format!("{flag} needs a value\n{USAGE}"));
        let slot = match flag {
            "--id" => &mut id,
            "--base" => &mut base,
            "--license-terms" => &mut terms,
            "--attribution" => &mut attribution,
            "--origin" => &mut origin,
            "--component" => &mut component,
            "--language" => &mut language,
            "--into" => &mut into,
            "--trained-on" => {
                let (name, license, license_terms) = (value()?, value()?, value()?);
                trained_on.push(TrainingData {
                    name: name.to_owned(),
                    license: license.parse()?,
                    license_terms: license_terms.parse()?,
                });
                continue;
            }
            _ => return Err(format!("{flag}: not an import-bundle flag\n{USAGE}")),
        };
        if slot.replace(value()?).is_some() {
            return Err(format!("{flag} is given twice"));
        }
    }
    fn need<'a>(flag: &str, v: Option<&'a str>) -> Result<&'a str, String> {
        v.ok_or_else(|| format!("{flag} is required\n{USAGE}"))
    }
    let spec = BundleImport {
        id: need("--id", id)?.parse()?,
        base: need("--base", base)?.parse()?,
        license_terms: need("--license-terms", terms)?.parse()?,
        trained_on,
        attribution: attribution.map(str::to_owned),
        origin: origin.map(str::parse).transpose()?,
        component: component.map(str::parse).transpose()?,
        language: language.map(str::parse).transpose()?,
    };
    let entry = bundle(Path::new(dir), spec).map_err(|e| e.to_string())?;
    let Some(into) = into else {
        print!("{}", Index::to_canonical_json(std::slice::from_ref(&entry)));
        return Ok(());
    };
    let held = match std::fs::read(into) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Index::to_canonical_json(&[]).into_bytes(),
        Err(e) => return Err(format!("{into}: {e}")),
    };
    let (id, class, files) = (entry.id.clone(), entry.class, entry.files.len());
    let written = add_to(&held, entry).map_err(|e| format!("{into}: {e}"))?;
    turbo_zoo::import::write(Path::new(into), &written).map_err(|e| e.to_string())?;
    println!("{id}\t{class}\t{files} files\tin {into}");
    Ok(())
}
