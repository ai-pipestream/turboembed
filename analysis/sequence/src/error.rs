use std::fmt;

/// Why a feature generator descriptor, one of its resources, a tag
/// dictionary or a model's manifest could not be read.
#[derive(Debug)]
pub enum Error {
    /// Reading the model file, its XML or a dictionary in it failed.
    Model(turbo_maxent::Error),
    /// The descriptor is well-formed XML but not a valid descriptor; the
    /// message is what OpenNLP reports, or says where it would throw.
    Descriptor(String),
    /// A `generator` element names a class OpenNLP does not have, or one
    /// of the caller's own (custom factories cannot be loaded here).
    UnknownGenerator(Box<str>),
    /// A `generator` element names an OpenNLP class this crate does not
    /// port (see the [`featuregen`](crate::featuregen) module).
    Unsupported(Box<str>),
    /// The descriptor names a resource the model does not hold.
    MissingResource(Box<str>),
    /// A resource could not be parsed: a Brown cluster line whose
    /// frequency is not an integer, or a dictionary that is not valid.
    Resource { name: Box<str>, message: String },
    /// A tag dictionary entry is malformed.
    TagDictionary(String),
    /// A manifest property holds a value OpenNLP cannot parse.
    Manifest { property: Box<str>, value: String },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Model(e) => write!(f, "{e}"),
            Error::Descriptor(message) => write!(f, "feature generator descriptor: {message}"),
            Error::UnknownGenerator(class) => write!(f, "unknown feature generator factory {class:?}"),
            Error::Unsupported(class) => write!(f, "feature generator factory {class:?} is not supported"),
            Error::MissingResource(name) => write!(f, "the model has no resource {name:?}"),
            Error::Resource { name, message } => write!(f, "resource {name:?}: {message}"),
            Error::TagDictionary(message) => write!(f, "tag dictionary: {message}"),
            Error::Manifest { property, value } => write!(f, "manifest property {property}: invalid value {value:?}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Model(e) => Some(e),
            _ => None,
        }
    }
}

impl From<turbo_maxent::Error> for Error {
    fn from(e: turbo_maxent::Error) -> Error {
        Error::Model(e)
    }
}

/// The crate's result type.
pub type Result<T> = std::result::Result<T, Error>;
