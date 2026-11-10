//! What a calling service may receive: the forms a value can leave in, the
//! kinds each form is granted for, and the one intersection that decides
//! a release (#1909). A grant is a set per kind, never a ranking: being
//! granted clear says nothing about token, and the reverse.

use std::collections::BTreeMap;
use std::fmt;

use turbo_analysis_layers::{PiiKind, PiiKinds};

use crate::proto::audit::PiiAccessForm;
use crate::proto::result::PiiValueForm;

/// One form a value can leave in.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Form {
    Token,
    Ciphertext,
    Clear,
}

impl Form {
    pub const ALL: [Form; 3] = [Form::Token, Form::Ciphertext, Form::Clear];

    /// The flag spelling: `token`, `ciphertext` or `clear`.
    pub const fn name(self) -> &'static str {
        match self {
            Form::Token => "token",
            Form::Ciphertext => "ciphertext",
            Form::Clear => "clear",
        }
    }

    pub fn parse(s: &str) -> Option<Form> {
        Form::ALL.into_iter().find(|f| f.name() == s)
    }

    /// The form a request asks for, or `None` for withheld.
    pub fn of_request(form: PiiValueForm) -> Option<Form> {
        match form {
            PiiValueForm::Unspecified | PiiValueForm::Withheld => None,
            PiiValueForm::Token => Some(Form::Token),
            PiiValueForm::Ciphertext => Some(Form::Ciphertext),
            PiiValueForm::Clear => Some(Form::Clear),
        }
    }

    pub const fn audit(self) -> PiiAccessForm {
        match self {
            Form::Token => PiiAccessForm::Token,
            Form::Ciphertext => PiiAccessForm::Ciphertext,
            Form::Clear => PiiAccessForm::Clear,
        }
    }

    const fn bit(self) -> u8 {
        match self {
            Form::Token => 1,
            Form::Ciphertext => 2,
            Form::Clear => 4,
        }
    }
}

/// A set of forms.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Forms(u8);

impl Forms {
    pub const NONE: Forms = Forms(0);
    /// Every form: what an unset release ceiling allows.
    pub const ALL: Forms = Forms(Form::Token.bit() | Form::Ciphertext.bit() | Form::Clear.bit());

    pub const fn of(form: Form) -> Forms {
        Forms(form.bit())
    }

    pub const fn with(self, form: Form) -> Forms {
        Forms(self.0 | form.bit())
    }

    pub const fn contains(self, form: Form) -> bool {
        self.0 & form.bit() != 0
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub const fn union(self, other: Forms) -> Forms {
        Forms(self.0 | other.0)
    }

    pub const fn intersect(self, other: Forms) -> Forms {
        Forms(self.0 & other.0)
    }

    /// The members in [`Form::ALL`] order.
    pub fn iter(self) -> impl Iterator<Item = Form> {
        Form::ALL.into_iter().filter(move |f| self.contains(*f))
    }
}

/// The forms in effect for one mention: what the request asked for, what
/// the server's policy allows, what the request's redaction policy's
/// release ceiling allows for the mention's kind ([`Forms::ALL`] when the
/// kind has none, #1943) and what the calling service is granted for the
/// kind. Every release decision goes through here.
pub const fn effective(requested: Forms, policy: Forms, ceiling: Forms, granted: Forms) -> Forms {
    requested.intersect(policy).intersect(ceiling).intersect(granted)
}

/// The kinds one form is granted for: built-in kinds by id, and whether
/// every caller-defined kind is included (only `*` grants those).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct KindSet {
    pub builtin: PiiKinds,
    pub custom: bool,
}

impl KindSet {
    pub fn contains(&self, kind: PiiKind) -> bool {
        if kind.is_custom() { self.custom } else { self.builtin.contains(kind) }
    }

    fn union(self, other: KindSet) -> KindSet {
        KindSet { builtin: self.builtin.union(other.builtin), custom: self.custom || other.custom }
    }

    /// Every built-in kind and every caller-defined kind.
    fn all() -> KindSet {
        let all: Vec<PiiKind> = PiiKind::builtins().collect();
        KindSet { builtin: PiiKinds::of(&all), custom: true }
    }
}

/// What one calling service is granted: the kinds each form is granted
/// for.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Grant {
    pub token: KindSet,
    pub clear: KindSet,
}

impl Grant {
    /// The forms this grant allows for `kind`.
    pub fn forms(&self, kind: PiiKind) -> Forms {
        let mut f = Forms::NONE;
        if self.token.contains(kind) {
            f = f.with(Form::Token);
        }
        if self.clear.contains(kind) {
            f = f.with(Form::Clear);
        }
        f
    }

    /// Whether this grant allows `form` for any kind at all.
    pub fn allows_any(&self, form: Form) -> bool {
        let set = |s: &KindSet| s.custom || !s.builtin.is_empty();
        match form {
            Form::Token => set(&self.token),
            Form::Clear => set(&self.clear),
            Form::Ciphertext => false,
        }
    }

    /// The rules as one canonical line, for the policy digest.
    pub fn canonical(&self) -> String {
        let set = |s: &KindSet| format!("{:032x}{}", s.builtin.bits(), if s.custom { "+custom" } else { "" });
        format!("token={};clear={}", set(&self.token), set(&self.clear))
    }
}

/// The SHA-256 of a calling service's verified leaf certificate (its DER
/// bytes): the only identity a grant is keyed on.
#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Fingerprint(pub [u8; 32]);

impl Fingerprint {
    /// 64 hexadecimal digits, either case. An error describes the input by
    /// length only, never repeating it.
    pub fn parse(s: &str) -> Result<Fingerprint, String> {
        let b = s.as_bytes();
        if b.len() != 64 {
            return Err(format!("the fingerprint ({} bytes, not shown) is not 64 hexadecimal digits", b.len()));
        }
        let mut out = [0u8; 32];
        for (i, pair) in b.chunks(2).enumerate() {
            let hi = hex(pair[0]).ok_or("the fingerprint (64 bytes, not shown) is not hexadecimal")?;
            let lo = hex(pair[1]).ok_or("the fingerprint (64 bytes, not shown) is not hexadecimal")?;
            out[i] = hi << 4 | lo;
        }
        Ok(Fingerprint(out))
    }
}

impl fmt::Debug for Fingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Fingerprint({self})")
    }
}

impl fmt::Display for Fingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.iter().try_for_each(|b| write!(f, "{b:02x}"))
    }
}

fn hex(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

/// Every configured grant, by service fingerprint.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Grants(pub BTreeMap<Fingerprint, Grant>);

impl Grants {
    /// The grant of `service`; nothing for an unknown or absent service.
    pub fn of(&self, service: Option<&Fingerprint>) -> Grant {
        service.and_then(|s| self.0.get(s)).cloned().unwrap_or_default()
    }

    /// Whether any grant includes `form`.
    pub fn any(&self, form: Form) -> bool {
        self.0.values().any(|g| {
            let s = match form {
                Form::Token => &g.token,
                Form::Clear => &g.clear,
                Form::Ciphertext => return false,
            };
            s.custom || !s.builtin.is_empty()
        })
    }

    /// Adds one `--pii-grant` value, `FINGERPRINT:FORMS:KINDS`: FORMS is
    /// `+`-separated `token` and `clear`, KINDS is a comma-separated list
    /// of built-in kind names, or `*` for every kind, custom kinds
    /// included. Grants of one fingerprint add up.
    pub fn add(&mut self, spec: &str) -> Result<(), String> {
        let mut parts = spec.splitn(3, ':');
        let (Some(fp), Some(forms), Some(kinds)) = (parts.next(), parts.next(), parts.next()) else {
            return Err(super::keys::shape("FINGERPRINT:FORMS:KINDS", spec));
        };
        let fp = Fingerprint::parse(fp)?;
        let kinds = if kinds == "*" {
            KindSet::all()
        } else {
            let mut list = Vec::new();
            for name in kinds.split(',') {
                let k = PiiKind::parse(name)
                    .ok_or_else(|| format!("the kind ({} bytes, not shown) is not a built-in PII kind", name.len()))?;
                if list.contains(&k) {
                    return Err(format!("kind `{name}` given twice"));
                }
                list.push(k);
            }
            KindSet { builtin: PiiKinds::of(&list), custom: false }
        };
        let mut add = Grant::default();
        for name in forms.split('+') {
            let slot = match Form::parse(name) {
                Some(Form::Token) => &mut add.token,
                Some(Form::Clear) => &mut add.clear,
                Some(Form::Ciphertext) => return Err("ciphertext release waits for #1885".into()),
                None => return Err(format!("the form ({} bytes, not shown) is not token or clear", name.len())),
            };
            if *slot != KindSet::default() {
                return Err(format!("form `{name}` given twice"));
            }
            *slot = kinds;
        }
        let g = self.0.entry(fp).or_default();
        g.token = g.token.union(add.token);
        g.clear = g.clear.union(add.clear);
        Ok(())
    }
}
