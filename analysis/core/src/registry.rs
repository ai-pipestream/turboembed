//! How add-ons offer their parts to pipelines built from configuration.
//! The shape is OpenNLP's `ComponentProvider` (#1290): a name, a priority,
//! a cheap availability check, a claim on a spec, and a constructor. Two
//! things differ. The contract is the registry's type parameter, so a
//! `Registry<dyn Step>` can only hand out steps and the "exact class or
//! assignable" question does not arise. And registration is explicit: an
//! add-on crate exports its providers and the program lists the add-ons it
//! links, so nothing is found by scanning, reflection or link order.

use std::path::PathBuf;

use crate::analyzer::{Analyzer, Builder};
use crate::annotate::Step;
use crate::error::Error;

/// Where a part's model is and how it is set: what a configuration line
/// says about one step.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Spec {
    /// A model file or directory, when the part needs one.
    pub location: Option<PathBuf>,
    /// Options in the order they were written.
    pub options: Vec<(String, String)>,
}

impl Spec {
    pub fn option<'a>(&'a self, key: &str, default: &'a str) -> &'a str {
        self.options.iter().find(|(k, _)| k == key).map_or(default, |(_, v)| v)
    }

    pub fn with(mut self, key: &str, value: &str) -> Spec {
        self.options.push((key.into(), value.into()));
        self
    }
}

/// One add-on's offer of a part of type `T`: a pipeline step, a stemmer, a
/// gazetteer.
pub trait Provider<T: ?Sized>: Send + Sync {
    /// The name a configuration asks for: "sentences", "pii", "chunker".
    fn name(&self) -> &'static str;
    /// Among providers of one name, the larger wins: a CUDA tagger over the
    /// CPU one.
    fn priority(&self) -> i32 {
        0
    }
    /// Cheap: no model read, no device brought up. A CUDA tagger on a
    /// machine without CUDA says no here.
    fn is_available(&self) -> bool {
        true
    }
    /// Whether it can serve this spec: a model it knows, options it takes.
    fn supports(&self, spec: &Spec) -> bool {
        let _ = spec;
        true
    }
    fn create(&self, spec: &Spec) -> Result<Box<T>, Error>;
}

/// The providers of one kind of part.
pub struct Registry<T: ?Sized> {
    providers: Vec<Box<dyn Provider<T>>>,
}

impl<T: ?Sized> Default for Registry<T> {
    fn default() -> Self {
        Registry { providers: Vec::new() }
    }
}

impl<T: ?Sized> Registry<T> {
    pub fn register(&mut self, p: impl Provider<T> + 'static) -> &mut Self {
        self.providers.push(Box::new(p));
        self
    }

    /// The best available provider of `name` that supports `spec`. Two
    /// with the same top priority is an error naming the tie, never a
    /// silent pick.
    pub fn select(&self, name: &str, spec: &Spec) -> Result<&dyn Provider<T>, Error> {
        let mut fit: Vec<&dyn Provider<T>> = self
            .providers
            .iter()
            .map(|p| p.as_ref())
            .filter(|p| p.name() == name && p.is_available() && p.supports(spec))
            .collect();
        fit.sort_by_key(|p| std::cmp::Reverse(p.priority()));
        match fit.as_slice() {
            [] => Err(Error::Pipeline(format!("no available provider of {name} supports {spec:?}"))),
            [a, b, ..] if a.priority() == b.priority() => {
                Err(Error::Pipeline(format!("two providers of {name} tie at priority {}", a.priority())))
            }
            [best, ..] => Ok(*best),
        }
    }

    pub fn create(&self, name: &str, spec: &Spec) -> Result<Box<T>, Error> {
        self.select(name, spec)?.create(spec)
    }
}

impl Registry<dyn Step> {
    /// A pipeline from configuration: one (name, spec) per step, in any
    /// order. The analyzer orders it and checks it as it would one built
    /// in code.
    pub fn pipeline(&self, steps: &[(&str, Spec)], given: impl FnOnce(Builder) -> Builder) -> Result<Analyzer, Error> {
        let mut b = given(Analyzer::builder());
        for (name, spec) in steps {
            b = b.boxed(self.create(name, spec)?);
        }
        b.build()
    }
}
