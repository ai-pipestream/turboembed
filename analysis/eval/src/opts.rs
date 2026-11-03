//! turbo-eval's `--name value` and `--flag` options.

use std::collections::BTreeMap;

use turbo_eval::Result;

use super::USAGE;

/// `--name value` pairs and bare `--flag`s.
pub(crate) struct Opts(pub(crate) BTreeMap<String, Option<String>>);

const FLAGS: [&str; 2] = ["--drift", "--clauses"];

impl Opts {
    pub(crate) fn parse(args: &[String]) -> Result<Opts> {
        let mut m = BTreeMap::new();
        let mut it = args.iter();
        while let Some(a) = it.next() {
            if !a.starts_with("--") {
                return Err(format!("{a}: not an option\n{USAGE}"));
            }
            let v = if FLAGS.contains(&a.as_str()) {
                None
            } else {
                Some(it.next().ok_or_else(|| format!("{a} needs a value"))?.clone())
            };
            if m.insert(a.clone(), v).is_some() {
                return Err(format!("{a} is given twice"));
            }
        }
        Ok(Opts(m))
    }

    pub(crate) fn take(&mut self, name: &str) -> Option<String> {
        self.0.remove(name).flatten()
    }

    pub(crate) fn need(&mut self, name: &str) -> Result<String> {
        self.take(name).ok_or_else(|| format!("{name} is required\n{USAGE}"))
    }

    pub(crate) fn flag(&mut self, name: &str) -> bool {
        self.0.remove(name).is_some()
    }

    pub(crate) fn parsed<T: std::str::FromStr>(&mut self, name: &str) -> Result<Option<T>> {
        self.take(name).map(|v| v.parse().map_err(|_| format!("{name} {v}: not a valid value"))).transpose()
    }

    pub(crate) fn required<T: std::str::FromStr>(&mut self, name: &str) -> Result<T> {
        self.parsed(name)?.ok_or_else(|| format!("{name} is required\n{USAGE}"))
    }
}
