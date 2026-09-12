use crate::catalog::CatalogError;

/// Errors from the turboembed engine and C ABI.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Catalog(#[from] CatalogError),
    #[error("{0}")]
    Unavailable(String),
    #[error("{0}")]
    Invalid(String),
    #[error("{0}")]
    Internal(String),
}

impl Error {
    pub fn unavailable(msg: impl Into<String>) -> Self {
        Self::Unavailable(msg.into())
    }

    pub fn invalid(msg: impl Into<String>) -> Self {
        Self::Invalid(msg.into())
    }

    pub fn internal(msg: impl Into<String>) -> Self {
        Self::Internal(msg.into())
    }
}
