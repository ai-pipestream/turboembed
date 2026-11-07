//! Caller-defined kinds: a table of validated names owned by the layer,
//! detector or policy that uses them. [`super::PiiKind::Custom`] holds an
//! index into such a table, so two documents with different custom kinds
//! share no state.

use std::fmt;

use super::kind::{CUSTOM_CODE_BASE, PiiKind};

/// An index into a [`CustomKinds`] table.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct CustomKindId(u16);

impl CustomKindId {
    /// The id of entry `index`. Only meaningful against the table that
    /// holds it.
    pub const fn new(index: u16) -> CustomKindId {
        CustomKindId(index)
    }

    pub const fn index(self) -> u16 {
        self.0
    }
}

/// Why a kind, a custom kind name or a confidence id was refused. No
/// variant carries a mention's value.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum KindError {
    /// A custom kind name is empty.
    EmptyName,
    /// A custom kind name is longer than [`CustomKinds::MAX_NAME_BYTES`]
    /// bytes; the length is given.
    NameTooLong(usize),
    /// A custom kind name is a built-in kind's name.
    Reserved(PiiKind),
    /// A table repeats a name at this index.
    Duplicate(u16),
    /// A table would exceed [`CustomKinds::MAX`] entries.
    Full,
    /// A kind code is neither a built-in id nor an entry of the table.
    UnknownCode(u32),
    /// A custom kind indexes past the end of its table.
    UnknownCustom(u16),
    /// A confidence id is not one of the four tiers.
    UnknownConfidence(u32),
}

impl fmt::Display for KindError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KindError::EmptyName => f.write_str("a custom pii kind name must not be empty"),
            KindError::NameTooLong(n) => {
                write!(f, "a custom pii kind name is {n} bytes (max {})", CustomKinds::MAX_NAME_BYTES)
            }
            KindError::Reserved(k) => write!(f, "a custom pii kind uses the built-in kind name {k}"),
            KindError::Duplicate(i) => write!(f, "custom pii kind {i} repeats an earlier name"),
            KindError::Full => write!(f, "more than {} custom pii kinds", CustomKinds::MAX),
            KindError::UnknownCode(c) => write!(f, "an unknown pii kind code {c}"),
            KindError::UnknownCustom(i) => write!(f, "custom pii kind {i} is not in the kind table"),
            KindError::UnknownConfidence(c) => write!(f, "an unknown pii confidence id {c}"),
        }
    }
}

impl std::error::Error for KindError {}

/// A table of caller-defined kind names, each 1 to
/// [`CustomKinds::MAX_NAME_BYTES`] UTF-8 bytes, distinct and not a built-in
/// kind's name, at most [`CustomKinds::MAX`] of them. Entry `i` is
/// `PiiKind::Custom(CustomKindId::new(i))`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct CustomKinds {
    names: Vec<Box<str>>,
}

impl CustomKinds {
    /// Maximum UTF-8 byte length of a custom kind name.
    pub const MAX_NAME_BYTES: usize = 64;

    /// Maximum number of custom kinds in one table.
    pub const MAX: usize = 256;

    /// An empty table.
    pub const fn new() -> CustomKinds {
        CustomKinds { names: Vec::new() }
    }

    /// A table of `names` in order, refusing the first invalid or repeated
    /// one.
    pub fn from_names<I, S>(names: I) -> Result<CustomKinds, KindError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut t = CustomKinds::new();
        for (i, name) in names.into_iter().enumerate() {
            let name = name.as_ref();
            if t.find(name).is_some() {
                return Err(KindError::Duplicate(i as u16));
            }
            t.add(name)?;
        }
        Ok(t)
    }

    fn validate(name: &str) -> Result<(), KindError> {
        if name.is_empty() {
            return Err(KindError::EmptyName);
        }
        if name.len() > Self::MAX_NAME_BYTES {
            return Err(KindError::NameTooLong(name.len()));
        }
        if let Some(k) = PiiKind::parse(name) {
            return Err(KindError::Reserved(k));
        }
        Ok(())
    }

    /// The custom kind named `name`, added at the end when new.
    pub fn add(&mut self, name: &str) -> Result<PiiKind, KindError> {
        Self::validate(name)?;
        if let Some(k) = self.find(name) {
            return Ok(k);
        }
        if self.names.len() >= Self::MAX {
            return Err(KindError::Full);
        }
        self.names.push(name.into());
        Ok(PiiKind::Custom(CustomKindId((self.names.len() - 1) as u16)))
    }

    /// The custom kind named `name` in this table.
    pub fn find(&self, name: &str) -> Option<PiiKind> {
        self.names.iter().position(|n| &**n == name).map(|i| PiiKind::Custom(CustomKindId(i as u16)))
    }

    /// The built-in kind named `name`, or the custom kind of this table.
    pub fn resolve(&self, name: &str) -> Option<PiiKind> {
        PiiKind::parse(name).or_else(|| self.find(name))
    }

    /// The name of `kind`: the built-in name, or this table's entry.
    pub fn name(&self, kind: PiiKind) -> Option<&str> {
        match kind {
            PiiKind::Custom(c) => self.names.get(usize::from(c.0)).map(|n| &**n),
            builtin => builtin.builtin_name(),
        }
    }

    /// Whether `kind` is built-in or an entry of this table.
    pub fn check(&self, kind: PiiKind) -> Result<(), KindError> {
        match kind {
            PiiKind::Custom(c) if usize::from(c.0) >= self.names.len() => Err(KindError::UnknownCustom(c.0)),
            _ => Ok(()),
        }
    }

    /// The kind with codec code `code` (see [`PiiKind::code`]).
    pub fn from_code(&self, code: u32) -> Result<PiiKind, KindError> {
        if code >= CUSTOM_CODE_BASE {
            let i = code - CUSTOM_CODE_BASE;
            return match u16::try_from(i) {
                Ok(i) if usize::from(i) < self.names.len() => Ok(PiiKind::Custom(CustomKindId(i))),
                _ => Err(KindError::UnknownCode(code)),
            };
        }
        u16::try_from(code).ok().and_then(PiiKind::from_id).ok_or(KindError::UnknownCode(code))
    }

    /// `kind` of `from` as the same-named kind of this table: built-in
    /// kinds map to themselves, custom kinds by name.
    pub fn translate(&self, from: &CustomKinds, kind: PiiKind) -> Option<PiiKind> {
        match kind {
            PiiKind::Custom(_) => self.find(from.name(kind)?),
            builtin => Some(builtin),
        }
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// The names in table order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.names.iter().map(|n| &**n)
    }

    /// Every custom kind of the table, in order.
    pub fn kinds(&self) -> impl Iterator<Item = PiiKind> {
        (0..self.names.len() as u16).map(|i| PiiKind::Custom(CustomKindId(i)))
    }
}
