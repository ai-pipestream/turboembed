//! A whole-tag matcher for the Java regular expression subset that
//! tag patterns in rule files use ([`TagPattern`]).

/// Maximum number of atoms in a compiled [`TagPattern`]: one bit
/// each in a `u64`, and one more for the end of the pattern.
pub const MAX_PATTERN_ATOMS: usize = 63;

/// A Java regular expression from the subset the AnCora head rules use, compiled to a bit-parallel
/// matcher that tests a whole tag (`Pattern.matcher(tag).matches()`) without backtracking.
///
/// It reads a sequence of atoms, each a literal character, `.` (any character but a line
/// terminator), a class (`[MAS]`, `[a-z0-9]`, `[^...]`) or `\` before a character that is not a
/// letter or digit (that character), each optionally followed by `*`, `+` or `?`. `^` at the start
/// and `$` at the end are accepted and change nothing, since the whole tag must match. Groups,
/// alternation, bounded repeats, lazy and possessive quantifiers, escapes such as `\d`, nested
/// classes and class intersections are refused; OpenNLP's rule files use none of them.
///
/// OpenNLP's `es-head-rules` file doubles its backslashes (`GRUP\\.VERB`). In a Java regular
/// expression that is a backslash followed by any character, so those tags never match a type
/// such as `GRUP.VERB`; this reads them the same way.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TagPattern {
    atoms: Vec<Atom>,
    /// For each ASCII character, the atoms that match it.
    ascii: [u64; 128],
    /// The atoms that repeat (`*`).
    repeat: u64,
    /// The atoms that may be passed over (`*` and `?`).
    skip: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Atom {
    Char(char),
    Any,
    Class { negated: bool, ranges: Vec<(char, char)> },
}

impl Atom {
    fn matches(&self, c: char) -> bool {
        match self {
            Atom::Char(x) => *x == c,
            Atom::Any => !matches!(c, '\n' | '\r' | '\u{85}' | '\u{2028}' | '\u{2029}'),
            Atom::Class { negated, ranges } => ranges.iter().any(|&(lo, hi)| lo <= c && c <= hi) != *negated,
        }
    }
}

impl TagPattern {
    /// Compiles a Java regular expression from the subset described on [`TagPattern`].
    pub fn compile(regex: &str) -> Result<Self, String> {
        let mut atoms: Vec<Atom> = Vec::new();
        let mut repeat = 0u64;
        let mut optional = 0u64;
        let mut chars = regex.char_indices().peekable();
        let mut quantifiable = false;
        while let Some((i, c)) = chars.next() {
            match c {
                '*' | '+' | '?' => {
                    if !quantifiable {
                        return Err(format!("dangling {c}"));
                    }
                    if matches!(chars.peek(), Some((_, '*' | '+' | '?' | '{'))) {
                        return Err("lazy, possessive and stacked quantifiers are not supported".into());
                    }
                    let last = atoms.len() - 1;
                    match c {
                        '*' => repeat |= 1 << last,
                        '?' => optional |= 1 << last,
                        _ => {
                            atoms.push(atoms[last].clone());
                            repeat |= 1 << (last + 1);
                        }
                    }
                    quantifiable = false;
                    continue;
                }
                '^' if i == 0 => {
                    quantifiable = false;
                    continue;
                }
                '$' if regex[i..].bytes().all(|b| b == b'$') => break,
                '(' | ')' | '|' | '{' | '}' | '^' | '$' => {
                    return Err(format!("{c} is not supported here"));
                }
                '.' => atoms.push(Atom::Any),
                '\\' => atoms.push(Atom::Char(escaped(chars.next().map(|(_, c)| c))?)),
                '[' => atoms.push(class(&mut chars)?),
                _ => atoms.push(Atom::Char(c)),
            }
            quantifiable = true;
            if atoms.len() > MAX_PATTERN_ATOMS {
                return Err(format!("more than {MAX_PATTERN_ATOMS} atoms"));
            }
        }
        if atoms.len() > MAX_PATTERN_ATOMS {
            return Err(format!("more than {MAX_PATTERN_ATOMS} atoms"));
        }
        let mut ascii = [0u64; 128];
        for (i, atom) in atoms.iter().enumerate() {
            for (b, slot) in ascii.iter_mut().enumerate() {
                if atom.matches(b as u8 as char) {
                    *slot |= 1 << i;
                }
            }
        }
        Ok(Self { atoms, ascii, repeat, skip: repeat | optional })
    }

    /// Whether the pattern matches all of `tag` (`Pattern.matcher(tag).matches()`).
    #[inline]
    pub fn matches(&self, tag: &str) -> bool {
        let mut states = self.closure(1);
        for c in tag.chars() {
            let matching = if c.is_ascii() {
                self.ascii[c as usize]
            } else {
                self.atoms.iter().enumerate().filter(|(_, a)| a.matches(c)).fold(0u64, |m, (i, _)| m | 1 << i)
            };
            let hit = states & matching;
            states = self.closure(((hit & !self.repeat) << 1) | (hit & self.repeat));
            if states == 0 {
                return false;
            }
        }
        states & (1u64 << self.atoms.len()) != 0
    }

    /// Adds the states reached by passing over optional and repeated atoms.
    #[inline]
    fn closure(&self, mut states: u64) -> u64 {
        loop {
            let next = states | ((states & self.skip) << 1);
            if next == states {
                return states;
            }
            states = next;
        }
    }
}

/// The literal character of `\c`: Java reads a backslash before a character that is not an ASCII
/// letter or digit as that character.
fn escaped(c: Option<char>) -> Result<char, String> {
    match c {
        None => Err("trailing backslash".into()),
        Some(c) if c.is_ascii_alphanumeric() => Err(format!("\\{c} is not supported")),
        Some(c) => Ok(c),
    }
}

/// A character class after its `[`, up to and with its `]`.
fn class(chars: &mut std::iter::Peekable<std::str::CharIndices<'_>>) -> Result<Atom, String> {
    let negated = chars.next_if(|&(_, c)| c == '^').is_some();
    let mut ranges = Vec::new();
    loop {
        let lo = match chars.next() {
            None => return Err("unterminated class".into()),
            Some((_, ']')) if !ranges.is_empty() => return Ok(Atom::Class { negated, ranges }),
            Some((_, ']' | '[')) => return Err("empty or nested classes are not supported".into()),
            Some((_, '&')) if chars.peek().is_some_and(|&(_, c)| c == '&') => {
                return Err("class intersections are not supported".into());
            }
            Some((_, '\\')) => escaped(chars.next().map(|(_, c)| c))?,
            Some((_, c)) => c,
        };
        let is_range = chars.peek().is_some_and(|&(_, c)| c == '-') && {
            let mut ahead = chars.clone();
            ahead.next();
            ahead.next().is_some_and(|(_, c)| c != ']')
        };
        if !is_range {
            ranges.push((lo, lo));
            continue;
        }
        chars.next();
        let hi = match chars.next() {
            Some((_, '\\')) => escaped(chars.next().map(|(_, c)| c))?,
            Some((_, '[')) => return Err("nested classes are not supported".into()),
            Some((_, c)) => c,
            None => return Err("unterminated class".into()),
        };
        if hi < lo {
            return Err(format!("illegal range {lo}-{hi}"));
        }
        ranges.push((lo, hi));
    }
}
