//! The typed facts a detector records for a mention. Evidence is primary:
//! the [`super::PiiConfidence`] tier is a summary derived from it, and
//! policy reads the facts, never the tier.

use turbo_analysis::{Fnv1a, Span};

/// The check a [`PiiEvidence::Checksum`] fact ran. [`ChecksumAlgorithm::id`]
/// is the stable number shared by the codec and the proto enum.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ChecksumAlgorithm {
    /// Luhn (cards, SINs, IMEIs, SIREN, SIRET).
    Luhn,
    /// ISO 7064 mod 97-10 (IBAN) and the INSEE NIR key.
    Mod97,
    /// Weighted modulus 11 (NHS, CPF, CNPJ).
    Mod11,
    /// The Spanish modulus-23 control letter (DNI, NIE).
    Mod23,
    /// The Spanish CIF control character.
    EsCif,
    /// ISO 7064 mod 11,10 (German tax id).
    Iso7064Mod11_10,
    /// ICAO Doc 9303 7-3-1 check digit.
    Icao9303,
    /// Base58check (legacy Bitcoin addresses).
    Base58Check,
    /// Bech32 and bech32m (segwit addresses).
    Bech32,
    /// EIP-55 mixed-case checksum (Ethereum addresses).
    Eip55,
    /// The ABA routing weights 3, 7, 1.
    AbaWeights,
    /// A check the detector does not name here.
    Other,
    /// The INSEE sum-of-digits modulus 5 rule for La Poste SIRET numbers
    /// (SIREN 356000000), which do not pass Luhn.
    LaPosteMod5,
}

impl ChecksumAlgorithm {
    /// Every algorithm, in id order.
    pub const ALL: [ChecksumAlgorithm; 13] = [
        ChecksumAlgorithm::Luhn,
        ChecksumAlgorithm::Mod97,
        ChecksumAlgorithm::Mod11,
        ChecksumAlgorithm::Mod23,
        ChecksumAlgorithm::EsCif,
        ChecksumAlgorithm::Iso7064Mod11_10,
        ChecksumAlgorithm::Icao9303,
        ChecksumAlgorithm::Base58Check,
        ChecksumAlgorithm::Bech32,
        ChecksumAlgorithm::Eip55,
        ChecksumAlgorithm::AbaWeights,
        ChecksumAlgorithm::Other,
        ChecksumAlgorithm::LaPosteMod5,
    ];

    /// The stable id, 1 to 13. 0 is never an algorithm.
    pub const fn id(self) -> u8 {
        match self {
            ChecksumAlgorithm::Luhn => 1,
            ChecksumAlgorithm::Mod97 => 2,
            ChecksumAlgorithm::Mod11 => 3,
            ChecksumAlgorithm::Mod23 => 4,
            ChecksumAlgorithm::EsCif => 5,
            ChecksumAlgorithm::Iso7064Mod11_10 => 6,
            ChecksumAlgorithm::Icao9303 => 7,
            ChecksumAlgorithm::Base58Check => 8,
            ChecksumAlgorithm::Bech32 => 9,
            ChecksumAlgorithm::Eip55 => 10,
            ChecksumAlgorithm::AbaWeights => 11,
            ChecksumAlgorithm::Other => 12,
            ChecksumAlgorithm::LaPosteMod5 => 13,
        }
    }

    /// The algorithm with stable id `id`.
    pub const fn from_id(id: u32) -> Option<ChecksumAlgorithm> {
        if id == 0 || id > ChecksumAlgorithm::ALL.len() as u32 {
            return None;
        }
        Some(ChecksumAlgorithm::ALL[id as usize - 1])
    }

    pub const fn name(self) -> &'static str {
        match self {
            ChecksumAlgorithm::Luhn => "luhn",
            ChecksumAlgorithm::Mod97 => "mod97",
            ChecksumAlgorithm::Mod11 => "mod11",
            ChecksumAlgorithm::Mod23 => "mod23",
            ChecksumAlgorithm::EsCif => "es-cif",
            ChecksumAlgorithm::Iso7064Mod11_10 => "iso7064-mod11-10",
            ChecksumAlgorithm::Icao9303 => "icao9303",
            ChecksumAlgorithm::Base58Check => "base58check",
            ChecksumAlgorithm::Bech32 => "bech32",
            ChecksumAlgorithm::Eip55 => "eip55",
            ChecksumAlgorithm::AbaWeights => "aba-weights",
            ChecksumAlgorithm::Other => "other",
            ChecksumAlgorithm::LaPosteMod5 => "la-poste-mod5",
        }
    }
}

/// One fact about a mention. Every fact is typed; an unknown fact kind is
/// refused on import. Facts never carry the value.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum PiiEvidence {
    /// The shape alone matched the kind's format: the syntax of an
    /// address, a key or a token, a registered domain or calling code,
    /// allocated number ranges.
    Shape,
    /// A checksum or check digit over the value was computed, and whether
    /// it passed.
    Checksum { algorithm: ChecksumAlgorithm, passed: bool },
    /// A context label for the kind was found in the detector's word
    /// window: where it stands in the same view as the mention, and which
    /// phrase of the kind's label list (the defaults, then the caller's
    /// extras, in order) it is.
    Label { span: Span, phrase: u32 },
    /// A deny list term matched: the list's position in the detector's
    /// composition order and the term's index in that list.
    DenyHit { list: u32, term: u32 },
    /// A rule or dictionary of the detector matched, by the detector's own
    /// rule number.
    Rule { rule: u32 },
}

impl PiiEvidence {
    /// The stable kind tag: shape 1, checksum 2, label 3, deny hit 4,
    /// rule 5. 0 is never a fact.
    pub const fn tag(self) -> u8 {
        match self {
            PiiEvidence::Shape => 1,
            PiiEvidence::Checksum { .. } => 2,
            PiiEvidence::Label { .. } => 3,
            PiiEvidence::DenyHit { .. } => 4,
            PiiEvidence::Rule { .. } => 5,
        }
    }

    pub const fn kind_name(self) -> &'static str {
        match self {
            PiiEvidence::Shape => "shape",
            PiiEvidence::Checksum { .. } => "checksum",
            PiiEvidence::Label { .. } => "label",
            PiiEvidence::DenyHit { .. } => "deny-hit",
            PiiEvidence::Rule { .. } => "rule",
        }
    }

    /// Whether this is a passed checksum.
    pub const fn is_checksum_passed(self) -> bool {
        matches!(self, PiiEvidence::Checksum { passed: true, .. })
    }

    /// Whether this is a context label.
    pub const fn is_label(self) -> bool {
        matches!(self, PiiEvidence::Label { .. })
    }

    /// A fact that holds on its own: a label span is non-empty.
    pub fn check(self) -> Result<(), String> {
        match self {
            PiiEvidence::Label { span, .. } if span.start >= span.end => {
                Err(format!("label evidence span {}..{} is empty", span.start, span.end))
            }
            _ => Ok(()),
        }
    }

    pub fn fold_hash(self, h: &mut Fnv1a) {
        h.u8(self.tag());
        match self {
            PiiEvidence::Shape => {}
            PiiEvidence::Checksum { algorithm, passed } => {
                h.u8(algorithm.id()).bool(passed);
            }
            PiiEvidence::Label { span, phrase } => {
                h.u32(span.start).u32(span.end).u32(phrase);
            }
            PiiEvidence::DenyHit { list, term } => {
                h.u32(list).u32(term);
            }
            PiiEvidence::Rule { rule } => {
                h.u32(rule);
            }
        }
    }
}

/// The evidence column of a layer: one non-empty run of facts per item,
/// in a flat list.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EvidenceList {
    facts: Vec<PiiEvidence>,
    end: Vec<u32>,
}

impl EvidenceList {
    /// Appends an item's facts.
    pub fn push(&mut self, facts: &[PiiEvidence]) {
        self.facts.extend_from_slice(facts);
        self.end.push(self.facts.len() as u32);
    }

    /// Appends an item's facts from an iterator.
    pub fn push_iter(&mut self, facts: impl IntoIterator<Item = PiiEvidence>) {
        self.facts.extend(facts);
        self.end.push(self.facts.len() as u32);
    }

    /// Item `i`'s facts.
    pub fn get(&self, i: usize) -> &[PiiEvidence] {
        let start = if i == 0 { 0 } else { self.end[i - 1] as usize };
        &self.facts[start..self.end[i] as usize]
    }

    /// How many items hold facts.
    pub fn len(&self) -> usize {
        self.end.len()
    }

    pub fn is_empty(&self) -> bool {
        self.end.is_empty()
    }

    /// Every fact of every item, flat.
    pub fn facts(&self) -> &[PiiEvidence] {
        &self.facts
    }

    /// The end offsets of each item's run in [`EvidenceList::facts`].
    pub fn ends(&self) -> &[u32] {
        &self.end
    }

    /// A list of `len` items whose runs end at `ends` in `facts`.
    pub fn from_parts(facts: Vec<PiiEvidence>, ends: Vec<u32>) -> Result<EvidenceList, String> {
        let list = EvidenceList { facts, end: ends };
        list.check()?;
        Ok(list)
    }

    /// The items at `kept`, in that order.
    pub fn retain(&self, kept: impl IntoIterator<Item = usize>) -> EvidenceList {
        let mut out = EvidenceList::default();
        for i in kept {
            out.push(self.get(i));
        }
        out
    }

    /// This list with every label span mapped by `map`, as a layer
    /// projected onto another view maps its spans.
    pub fn map_label_spans<E>(&self, mut map: impl FnMut(Span) -> Result<Span, E>) -> Result<EvidenceList, E> {
        let mut facts = Vec::with_capacity(self.facts.len());
        for &f in &self.facts {
            facts.push(match f {
                PiiEvidence::Label { span, phrase } => PiiEvidence::Label { span: map(span)?, phrase },
                other => other,
            });
        }
        Ok(EvidenceList { facts, end: self.end.clone() })
    }

    /// Every label span of item `i` is a range of `text` on character
    /// boundaries; `what` names the item in an error.
    pub fn check_text(&self, text: &str, what: &str) -> Result<(), String> {
        for (i, facts) in (0..self.len()).map(|i| (i, self.get(i))) {
            for fact in facts {
                if let PiiEvidence::Label { span, .. } = *fact
                    && text.get(span.start as usize..span.end as usize).is_none()
                {
                    return Err(format!(
                        "{what} {i} label evidence span {}..{} is outside the view text or not on UTF-8 boundaries",
                        span.start, span.end
                    ));
                }
            }
        }
        Ok(())
    }

    /// Every item holds at least one fact, runs are contiguous, and every
    /// fact holds on its own.
    pub fn check(&self) -> Result<(), String> {
        let mut start = 0u32;
        for (i, &end) in self.end.iter().enumerate() {
            if end <= start {
                return Err(format!("pii item {i} has no evidence"));
            }
            if end as usize > self.facts.len() {
                return Err(format!("pii item {i} evidence runs past the facts"));
            }
            start = end;
        }
        if start as usize != self.facts.len() {
            return Err("pii evidence facts past the last item".into());
        }
        for (i, f) in self.facts.iter().enumerate() {
            f.check().map_err(|e| format!("pii evidence fact {i}: {e}"))?;
        }
        Ok(())
    }

    pub fn fold_hash(&self, h: &mut Fnv1a) {
        h.usize(self.facts.len()).usize(self.end.len());
        for f in &self.facts {
            f.fold_hash(h);
        }
        for &e in &self.end {
            h.u32(e);
        }
    }
}
