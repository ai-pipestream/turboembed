//! The typed agreement features of a mention, packed into one byte.

/// Grammatical number (`Mention.Number`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Number {
    Singular,
    Plural,
    Unknown,
}

/// Gender, `Neutral` for things and organizations (`Mention.Gender`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Gender {
    Male,
    Female,
    Neutral,
    Unknown,
}

/// Animacy (`Mention.Animacy`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Animacy {
    Animate,
    Inanimate,
    Unknown,
}

/// Grammatical person, known for pronouns only (`Mention.Person`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Person {
    First,
    Second,
    Third,
    Unknown,
}

/// What a mention is, by its form: a proper name, a common noun phrase,
/// a personal pronoun or a demonstrative standing alone as a pronoun.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum MentionKind {
    Proper,
    Nominal,
    Pronominal,
    Demonstrative,
}

/// Where a mention came from: OpenNLP's `CorefMention` kind
/// (`entity`, `nominal`, `pronoun`, `gold`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Origin {
    /// Lifted from the entity layer (`CorefMention.KIND_ENTITY`).
    Entity,
    /// A noun phrase no entity heads (`CorefMention.KIND_NOMINAL`).
    NounPhrase,
    /// A pronoun token (`CorefMention.KIND_PRONOUN`).
    Pronoun,
    /// Read from a corpus annotation (`CorefMention.KIND_GOLD`).
    Gold,
}

impl Origin {
    /// The `CorefMention` kind string.
    pub fn as_str(self) -> &'static str {
        match self {
            Origin::Entity => "entity",
            Origin::NounPhrase => "nominal",
            Origin::Pronoun => "pronoun",
            Origin::Gold => "gold",
        }
    }

    pub(crate) fn from_byte(b: u8) -> Option<Origin> {
        [Origin::Entity, Origin::NounPhrase, Origin::Pronoun, Origin::Gold].get(b as usize).copied()
    }

    pub(crate) fn byte(self) -> u8 {
        self as u8
    }
}

impl MentionKind {
    pub(crate) fn from_byte(b: u8) -> Option<MentionKind> {
        [MentionKind::Proper, MentionKind::Nominal, MentionKind::Pronominal, MentionKind::Demonstrative]
            .get(b as usize)
            .copied()
    }

    pub(crate) fn byte(self) -> u8 {
        self as u8
    }
}

macro_rules! java_names {
    ($t:ident { $($v:ident => $s:literal),+ $(,)? }) => {
        impl $t {
            /// The Java enum constant's name, as OpenNLP's features print it.
            pub fn java_name(self) -> &'static str {
                match self { $($t::$v => $s),+ }
            }
        }
    };
}

java_names!(Number { Singular => "SINGULAR", Plural => "PLURAL", Unknown => "UNKNOWN" });
java_names!(Gender { Male => "MALE", Female => "FEMALE", Neutral => "NEUTRAL", Unknown => "UNKNOWN" });
java_names!(Animacy { Animate => "ANIMATE", Inanimate => "INANIMATE", Unknown => "UNKNOWN" });
java_names!(Person { First => "FIRST", Second => "SECOND", Third => "THIRD", Unknown => "UNKNOWN" });

const NUMBERS: [Number; 3] = [Number::Singular, Number::Plural, Number::Unknown];
const GENDERS: [Gender; 4] = [Gender::Male, Gender::Female, Gender::Neutral, Gender::Unknown];
const ANIMACIES: [Animacy; 3] = [Animacy::Animate, Animacy::Inanimate, Animacy::Unknown];
const PERSONS: [Person; 4] = [Person::First, Person::Second, Person::Third, Person::Unknown];

/// Number, gender, animacy and person in one byte: two bits each, in
/// that order from the low bits. A byte whose number or animacy field
/// holds 3 is not an agreement and is refused by [`Agreement::from_bits`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Agreement(u8);

impl Agreement {
    pub fn new(number: Number, gender: Gender, animacy: Animacy, person: Person) -> Agreement {
        Agreement(number as u8 | (gender as u8) << 2 | (animacy as u8) << 4 | (person as u8) << 6)
    }

    /// The agreement a packed byte holds, `None` for a byte no agreement packs to.
    pub fn from_bits(bits: u8) -> Option<Agreement> {
        if bits & 0b11 == 3 || (bits >> 4) & 0b11 == 3 { None } else { Some(Agreement(bits)) }
    }

    pub fn bits(self) -> u8 {
        self.0
    }

    pub fn number(self) -> Number {
        NUMBERS[(self.0 & 0b11) as usize]
    }

    pub fn gender(self) -> Gender {
        GENDERS[((self.0 >> 2) & 0b11) as usize]
    }

    pub fn animacy(self) -> Animacy {
        ANIMACIES[((self.0 >> 4) & 0b11) as usize]
    }

    pub fn person(self) -> Person {
        PERSONS[(self.0 >> 6) as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_agreement_round_trips_through_its_byte() {
        for n in NUMBERS {
            for g in GENDERS {
                for a in ANIMACIES {
                    for p in PERSONS {
                        let packed = Agreement::new(n, g, a, p);
                        let back = Agreement::from_bits(packed.bits()).unwrap();
                        assert_eq!((back.number(), back.gender(), back.animacy(), back.person()), (n, g, a, p));
                    }
                }
            }
        }
    }

    #[test]
    fn bytes_outside_the_enums_are_refused() {
        assert!(Agreement::from_bits(0b0000_0011).is_none());
        assert!(Agreement::from_bits(0b0011_0000).is_none());
        assert!(Agreement::from_bits(0b1110_1110).is_some());
    }
}
