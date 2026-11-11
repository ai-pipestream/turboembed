//! `turbo-zoo import-bundle`, `import-hf` and `import-oci`: a model in another layout
//! written as an index entry, printed or added to an index file.

use std::path::Path;

use turbo_zoo::import::{
    BundleImport, Described, HUGGING_FACE, OciImport, RepoImport, add_to, bundle, hugging_face, oci,
};
use turbo_zoo::{Artifact, Format, Http, Index, Kind, OciRegistry, TrainingData};

pub const USAGE: &str =
    "usage: turbo-zoo import-bundle <bundle-dir> --id <id> --base <https url> --license-terms <https url> [...]
       turbo-zoo import-hf <owner/name> <commit> --id <id> --license <SPDX id> --license-terms <https url>
           (--bundle | --kind <KIND_*> --format <FORMAT_*>) [--file <path>]... [--version <text>] [...]
       turbo-zoo import-oci <registry>/<name>@sha256:<digest> --id <id> --license <SPDX id> --license-terms <https url>
           (--bundle | --kind <KIND_*> --format <FORMAT_*>) [--version <text>] [...]
  each takes:
    [--trained-on <name> <licence> <https url>]... [--attribution <text>] [--origin <https url>]
    [--component <COMPONENT_*>] [--language <tag>] [--into <zoo.json>]";

/// Each flag and how many values follow it.
const FLAGS: &[(&str, usize)] = &[
    ("--id", 1),
    ("--base", 1),
    ("--license", 1),
    ("--license-terms", 1),
    ("--trained-on", 3),
    ("--attribution", 1),
    ("--origin", 1),
    ("--component", 1),
    ("--language", 1),
    ("--into", 1),
    ("--kind", 1),
    ("--format", 1),
    ("--version", 1),
    ("--file", 1),
    ("--bundle", 0),
];

/// The flags of one command line, each with its values, in order.
struct Flags<'a>(Vec<(&'a str, &'a [&'a str])>);

impl<'a> Flags<'a> {
    /// `args` read as flags; `allowed` says which of [`FLAGS`] the command takes.
    fn parse(args: &'a [&'a str], allowed: impl Fn(&str) -> bool) -> Result<Flags<'a>, String> {
        let (mut out, mut at) = (Vec::new(), 0);
        while at < args.len() {
            let flag = args[at];
            let Some((_, n)) = FLAGS.iter().find(|(f, _)| *f == flag && allowed(flag)) else {
                return Err(format!("{flag}: not a flag of this command\n{USAGE}"));
            };
            let values = args.get(at + 1..at + 1 + n).ok_or_else(|| format!("{flag} needs {n} value(s)\n{USAGE}"))?;
            out.push((flag, values));
            at += 1 + n;
        }
        Ok(Flags(out))
    }

    fn all(&self, flag: &str) -> impl Iterator<Item = &'a [&'a str]> + '_ {
        let flag = flag.to_owned();
        self.0.iter().filter(move |(f, _)| *f == flag).map(|(_, v)| *v)
    }

    fn has(&self, flag: &str) -> bool {
        self.all(flag).next().is_some()
    }

    /// The one value of `flag`, if it is given; twice is an error.
    fn one(&self, flag: &str) -> Result<Option<&'a str>, String> {
        let mut given = self.all(flag);
        match (given.next(), given.next()) {
            (_, Some(_)) => Err(format!("{flag} is given twice")),
            (first, None) => Ok(first.map(|v| v[0])),
        }
    }

    fn need(&self, flag: &str) -> Result<&'a str, String> {
        self.one(flag)?.ok_or_else(|| format!("{flag} is required\n{USAGE}"))
    }

    /// The fields every importer takes.
    fn described(&self) -> Result<Described, String> {
        let trained_on = self
            .all("--trained-on")
            .map(|v| Ok(TrainingData { name: v[0].to_owned(), license: v[1].parse()?, license_terms: v[2].parse()? }))
            .collect::<Result<_, String>>()?;
        Ok(Described {
            license_terms: self.need("--license-terms")?.parse()?,
            trained_on,
            attribution: self.one("--attribution")?.map(str::to_owned),
            origin: self.one("--origin")?.map(str::parse).transpose()?,
            component: self.one("--component")?.map(str::parse).transpose()?,
            language: self.one("--language")?.map(str::parse).transpose()?,
        })
    }
}

/// Prints `entry` as an index of its own, or adds it to the index file
/// `into` names: last, or in the place of the entry of the same id.
fn deliver(entry: Artifact, into: Option<&str>) -> Result<(), String> {
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

/// Runs `import-bundle` with the arguments after the command's name.
pub fn bundle_dir(args: &[&str]) -> Result<(), String> {
    let [dir, flags @ ..] = args else { return Err(USAGE.into()) };
    let only_hf = ["--license", "--kind", "--format", "--version", "--file", "--bundle"];
    let flags = Flags::parse(flags, |f| !only_hf.contains(&f))?;
    let spec = BundleImport {
        id: flags.need("--id")?.parse()?,
        base: flags.need("--base")?.parse()?,
        described: flags.described()?,
    };
    let entry = bundle(Path::new(dir), spec).map_err(|e| e.to_string())?;
    deliver(entry, flags.one("--into")?)
}

/// Whether the entry is a bundle, or else the kind and format it names.
fn shape(flags: &Flags) -> Result<(bool, Kind, Format), String> {
    let as_bundle = flags.has("--bundle");
    if as_bundle && (flags.has("--kind") || flags.has("--format") || flags.has("--version")) {
        return Err("--bundle takes the kind, format and version from the manifest".into());
    }
    if as_bundle {
        return Ok((true, Kind::Model, Format::Bundle));
    }
    Ok((false, flags.need("--kind")?.parse()?, flags.need("--format")?.parse()?))
}

/// Runs `import-oci` with the arguments after the command's name.
pub fn artifact(args: &[&str]) -> Result<(), String> {
    let [reference, flags @ ..] = args else { return Err(USAGE.into()) };
    let flags = Flags::parse(flags, |f| f != "--base" && f != "--file")?;
    let (as_bundle, kind, format) = shape(&flags)?;
    let spec = OciImport {
        id: flags.need("--id")?.parse()?,
        reference: (*reference).to_owned(),
        bundle: as_bundle,
        kind,
        format,
        version: flags.one("--version")?.map(str::to_owned),
        license: flags.need("--license")?.parse()?,
        described: flags.described()?,
    };
    let into = flags.one("--into")?;
    deliver(oci(&OciRegistry::new(), spec).map_err(|e| e.to_string())?, into)
}

/// Runs `import-hf` with the arguments after the command's name.
pub fn repo(args: &[&str]) -> Result<(), String> {
    let [repo, commit, flags @ ..] = args else { return Err(USAGE.into()) };
    let flags = Flags::parse(flags, |f| f != "--base")?;
    let (as_bundle, kind, format) = shape(&flags)?;
    let spec = RepoImport {
        id: flags.need("--id")?.parse()?,
        repo: (*repo).to_owned(),
        commit: (*commit).to_owned(),
        files: flags.all("--file").map(|v| v[0].to_owned()).collect(),
        bundle: as_bundle,
        kind,
        format,
        version: flags.one("--version")?.map(str::to_owned),
        license: flags.need("--license")?.parse()?,
        described: flags.described()?,
    };
    let into = flags.one("--into")?;
    let entry = hugging_face(&Http::new(), HUGGING_FACE, spec).map_err(|e| e.to_string())?;
    deliver(entry, into)
}
