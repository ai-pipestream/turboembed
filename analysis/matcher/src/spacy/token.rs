use super::json::Value;
use super::{ImportError, invalid, object, string, strings, unsigned, unsupported};
use crate::{Attr, Cmp, Flag, Iob, SetRelation, Test, Tok, ValueTest};

pub(super) fn read(value: &Value) -> Result<Tok, ImportError> {
    let mut tok = Tok::any();
    for (key, value) in object(value, "token")? {
        let name = key.to_ascii_uppercase();
        if name == "OP" {
            continue;
        }
        let test = if name == "LENGTH" {
            length(value)?
        } else if name == "ENT_IOB" {
            Tok::iob(match string(value, key)? {
                "I" => Iob::Inside,
                "O" => Iob::Outside,
                "B" => Iob::Begin,
                _ => return Err(invalid(key)),
            })
        } else if let Some(flag) = flag(&name) {
            let Value::Bool(on) = value else {
                return Err(invalid(key));
            };
            Tok::flag(flag, *on)
        } else {
            attribute(attr(&name).ok_or_else(|| unsupported(key))?, value)?
        };
        tok = tok.and(test);
    }
    let fields = object(value, "token")?;
    // Case-folded duplicate keys are also ambiguous ("OP" and "op").
    for (i, (key, _)) in fields.iter().enumerate() {
        if fields[..i].iter().any(|(k, _)| k.eq_ignore_ascii_case(key)) {
            return Err(invalid(key));
        }
    }
    for (key, v) in fields {
        if key.eq_ignore_ascii_case("OP") {
            tok = repeat(tok, string(v, key)?)?;
        }
    }
    Ok(tok)
}
fn attr(key: &str) -> Option<Attr> {
    Some(match key {
        "ORTH" | "TEXT" => Attr::Text,
        "LOWER" => Attr::Lower,
        "NORM" => Attr::Norm,
        "LEMMA" => Attr::Lemma,
        "POS" => Attr::Pos,
        "TAG" => Attr::Tag,
        "MORPH" => Attr::Morph,
        "DEP" => Attr::Dep,
        "SHAPE" => Attr::Shape,
        "PREFIX" => Attr::Prefix,
        "SUFFIX" => Attr::Suffix,
        "ENT_TYPE" => Attr::Entity,
        _ => return None,
    })
}
fn flag(key: &str) -> Option<Flag> {
    Some(match key {
        "IS_ALPHA" => Flag::Alpha,
        "IS_ASCII" => Flag::Ascii,
        "IS_DIGIT" => Flag::Digit,
        "IS_LOWER" => Flag::Lower,
        "IS_UPPER" => Flag::Upper,
        "IS_TITLE" => Flag::Title,
        "IS_PUNCT" => Flag::Punct,
        "IS_SPACE" => Flag::Space,
        "IS_BRACKET" => Flag::Bracket,
        "IS_QUOTE" => Flag::Quote,
        "IS_LEFT_PUNCT" => Flag::LeftPunct,
        "IS_RIGHT_PUNCT" => Flag::RightPunct,
        "IS_CURRENCY" => Flag::Currency,
        "LIKE_NUM" => Flag::LikeNum,
        "SPACY" => Flag::SpaceAfter,
        "IS_SENT_START" => Flag::SentStart,
        "IS_STOP" => Flag::Stop,
        _ => return None,
    })
}
fn attribute(attr: Attr, value: &Value) -> Result<Tok, ImportError> {
    if let Value::String(v) = value {
        return Ok(Tok::attr(attr, v));
    }
    let mut tok = Tok::any();
    for (op, v) in object(value, "attribute predicate")? {
        let test = match op.as_str() {
            "IN" | "NOT_IN" => {
                let mut values = strings(v, op)?;
                if attr == Attr::Morph {
                    for v in &mut values {
                        *v = crate::predicates::morph(v).into();
                    }
                }
                if op == "IN" { Test::In(attr, values) } else { Test::NotIn(attr, values) }
            }
            "IS_SUBSET" | "IS_SUPERSET" | "INTERSECTS" if attr == Attr::Morph => {
                let relation = match op.as_str() {
                    "IS_SUBSET" => SetRelation::Subset,
                    "IS_SUPERSET" => SetRelation::Superset,
                    _ => SetRelation::Intersects,
                };
                Test::Value(
                    attr,
                    ValueTest::Morph {
                        values: strings(v, op)?.iter().map(|v| crate::predicates::morph(v).into()).collect(),
                        relation,
                    },
                )
            }
            "FUZZY" | "FUZZY1" | "FUZZY2" | "FUZZY3" | "FUZZY4" | "FUZZY5" | "FUZZY6" | "FUZZY7" | "FUZZY8"
            | "FUZZY9" => {
                let distance = op.strip_prefix("FUZZY").and_then(|s| s.parse().ok());
                let (values, exclude) = if let Value::String(s) = v {
                    (vec![s.as_str().into()].into_boxed_slice(), false)
                } else {
                    let fields = object(v, op)?;
                    let [(set, values)] = fields else {
                        return Err(invalid(op));
                    };
                    if set != "IN" && set != "NOT_IN" {
                        return Err(unsupported(set));
                    }
                    (strings(values, op)?, set == "NOT_IN")
                };
                Test::Value(attr, ValueTest::Fuzzy { values, distance, exclude })
            }
            _ => return Err(unsupported(op)),
        };
        tok = tok.and(Tok::test(test));
    }
    Ok(tok)
}
fn cmp(op: &str) -> Result<Cmp, ImportError> {
    Ok(match op {
        "==" => Cmp::Eq,
        "!=" => Cmp::Ne,
        ">" => Cmp::Gt,
        ">=" => Cmp::Ge,
        "<" => Cmp::Lt,
        "<=" => Cmp::Le,
        _ => return Err(unsupported(op)),
    })
}
fn length(value: &Value) -> Result<Tok, ImportError> {
    if let Value::Number(_) = value {
        return Ok(Tok::length(Cmp::Eq, unsigned(value, "LENGTH")?));
    }
    let mut tok = Tok::any();
    for (op, value) in object(value, "LENGTH")? {
        tok = tok.and(Tok::length(cmp(op)?, unsigned(value, "LENGTH")?));
    }
    Ok(tok)
}
fn repeat(tok: Tok, op: &str) -> Result<Tok, ImportError> {
    Ok(match op {
        "1" => tok,
        "!" => !tok,
        "?" => tok.opt(),
        "*" => tok.star(),
        "+" => tok.plus(),
        _ => {
            let counts = op.strip_prefix('{').and_then(|s| s.strip_suffix('}')).ok_or_else(|| invalid("OP"))?;
            let count = |s: &str| {
                if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
                    return Err(invalid("OP"));
                }
                s.parse::<u32>().map_err(|_| invalid("OP"))
            };
            if let Some((a, b)) = counts.split_once(',') {
                if a.is_empty() && b.is_empty() {
                    return Err(invalid("OP"));
                }
                tok.repeat(if a.is_empty() { 0 } else { count(a)? }, if b.is_empty() { None } else { Some(count(b)?) })
            } else {
                let n = count(counts)?;
                tok.repeat(n, Some(n))
            }
        }
    })
}
