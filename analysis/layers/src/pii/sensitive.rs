//! A wrapper for values that must not leave through diagnostics.

use std::fmt;

/// A value that is personal or secret: its `Debug` prints a placeholder
/// and it has no `Display`, so a `{:?}` of a struct holding it, a log line
/// or an error message never carries the contents. Reading the contents
/// is spelled out with [`Sensitive::expose`], so every place a value is
/// copied out can be found by name.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Sensitive<T>(T);

/// What the `Debug` of a [`Sensitive`] prints in place of its contents.
pub const WITHHELD: &str = "<withheld>";

impl<T> Sensitive<T> {
    /// Wraps `value`.
    pub const fn new(value: T) -> Sensitive<T> {
        Sensitive(value)
    }

    /// The contents. Callers that copy them into an output decide that
    /// the output may carry them.
    pub const fn expose(&self) -> &T {
        &self.0
    }

    /// The contents, to change in place.
    pub fn expose_mut(&mut self) -> &mut T {
        &mut self.0
    }

    /// The contents, unwrapped.
    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T> From<T> for Sensitive<T> {
    fn from(value: T) -> Sensitive<T> {
        Sensitive(value)
    }
}

impl<T> fmt::Debug for Sensitive<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Sensitive(")?;
        f.write_str(WITHHELD)?;
        f.write_str(")")
    }
}
