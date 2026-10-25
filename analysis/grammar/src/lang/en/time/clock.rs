//! Times of day: `7:56:30 pm`, `5pm`, `1140 a.m.`, `half past seven`,
//! `20 min past eight`, `noon`, `11ish`, and the parts of the day: `in
//! the morning`, `lunchtime`, `late afternoon`.

use turbo_analysis_layers::{Modifier, Portion};

use crate::engine::{Ctx, Item, Rule, glued, glued_word, node, opt_word, word};
use crate::time::TimeValue;
use crate::time::spec::{Clock, Meridiem, Part, Point, Shown, TimeOfDay};
use crate::value::Value;

use super::super::{FRAC_WORD, INT, NUMERAL};
use super::{AMPM, BARE, CLOCK, DIG, PART, PARTD, PLURAL, clock_of, emit, emit_clock, r, term, term_value, tval};

/// Lexeme `i` as written, when it is all ASCII digits.
pub(crate) fn digits<'a>(c: &Ctx<'a>, i: usize) -> Option<&'a str> {
    let raw = c.raw(i);
    raw.bytes().all(|b| b.is_ascii_digit()).then_some(raw)
}

/// A `DIG` term: the value, and the digit count with 16 added for a
/// leading zero.
pub(crate) fn b_dig(_: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let raw = digits(c, c.start)?;
    if raw.len() > 8 {
        return None;
    }
    let zero = if raw.len() > 1 && raw.starts_with('0') { 16 } else { 0 };
    term_value(raw.parse().ok()?, raw.len() as u8 | zero)
}

pub(crate) fn dig_len(v: &Value) -> u8 {
    term(v).1 & 15
}

pub(crate) fn dig_zero(v: &Value) -> bool {
    term(v).1 & 16 != 0
}

/// The half of the day a written hour leaves open: none above 12 or at 0.
pub(crate) fn open(hour: i64) -> Meridiem {
    if hour > 12 || hour == 0 { Meridiem::Day } else { Meridiem::Unknown }
}

pub(crate) fn hm(c: &Ctx<'_>, h: i64, m: i64, s: i64, shown: Shown) -> Option<Value> {
    if !(0..=24).contains(&h) {
        return None;
    }
    let h = h % 24;
    emit_clock(c, Clock::new(h as u8, u8::try_from(m).ok()?, u8::try_from(s).ok()?, shown, open(h))?)
}

pub(crate) fn b_bare_dig(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if dig_len(&v[0]) > 2 {
        return None;
    }
    hm(c, term(&v[0]).0, 0, 0, Shown::Hour)
}

/// A spelled hour: `seven`, `twelve`.
pub(crate) fn spelled_hour(v: &Value) -> Option<i64> {
    let n = v.num()?;
    (n.has(crate::value::flag::SPELLED) && !n.has(crate::value::flag::DIGITS)).then_some(())?;
    let h = n.value.to_i64()?;
    (1..=12).contains(&h).then_some(h)
}

pub(crate) fn spelled_minutes(v: &Value) -> Option<i64> {
    let n = v.num()?;
    (n.has(crate::value::flag::SPELLED) && !n.has(crate::value::flag::DIGITS)).then_some(())?;
    let m = n.value.to_i64()?;
    (10..60).contains(&m).then_some(m)
}

pub(crate) fn b_bare_word(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    hm(c, spelled_hour(&v[0])?, 0, 0, Shown::Hour)
}

pub(crate) fn b_bare_word_minutes(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    hm(c, spelled_hour(&v[0])?, spelled_minutes(&v[1])?, 0, Shown::Minute)
}

/// `6.45`, `17.30`, `11.00`: an hour and two digits of minutes.
pub(crate) fn b_bare_dot(_: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let (h, m) = c.raw(c.start).split_once('.')?;
    if h.is_empty() || h.len() > 2 || m.len() != 2 || !(h.bytes().chain(m.bytes())).all(|b| b.is_ascii_digit()) {
        return None;
    }
    let (h, m): (i64, i64) = (h.parse().ok()?, m.parse().ok()?);
    if h > 23 || m > 59 {
        return None;
    }
    hm(c, h, m, 0, Shown::Minute)
}

pub(crate) fn b_hm(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if dig_len(&v[0]) > 2 || dig_len(&v[1]) != 2 {
        return None;
    }
    hm(c, term(&v[0]).0, term(&v[1]).0, 0, Shown::Minute)
}

pub(crate) fn b_hms(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if dig_len(&v[0]) > 2 || dig_len(&v[1]) != 2 || dig_len(&v[2]) != 2 {
        return None;
    }
    hm(c, term(&v[0]).0, term(&v[1]).0, term(&v[2]).0, Shown::Second)
}

/// `0730`: four digits with a leading zero.
pub(crate) fn b_hhmm_zero(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if dig_len(&v[0]) != 4 || !dig_zero(&v[0]) {
        return None;
    }
    let n = term(&v[0]).0;
    hm(c, n / 100, n % 100, 0, Shown::Minute)
}

pub(crate) fn meridiem(v: &Value) -> Meridiem {
    match term(v).0 {
        1 => Meridiem::Am,
        2 => Meridiem::Pm,
        _ => Meridiem::Unknown,
    }
}

/// A clock in the half of the day `m` says; a 24-hour time takes none.
pub(crate) fn with_meridiem(c: &Ctx<'_>, k: Clock, m: Meridiem) -> Option<Value> {
    if m == Meridiem::Unknown {
        // `7ampm`: either half, said outright.
        return (k.hour <= 12).then(|| emit_clock(c, Clock { meridiem: Meridiem::Unknown, ..k }))?;
    }
    if k.hour > 12 || k.meridiem == Meridiem::Am || k.meridiem == Meridiem::Pm {
        return None;
    }
    emit_clock(c, Clock::new(k.hour, k.minute, k.second, k.shown, m)?)
}

pub(crate) fn b_clock_ampm(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    with_meridiem(c, clock_of(c, &v[0])?, meridiem(&v[1]))
}

pub(crate) fn b_clock_am(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    with_meridiem(c, clock_of(c, &v[0])?, Meridiem::Am)
}

pub(crate) fn b_clock_pm(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    with_meridiem(c, clock_of(c, &v[0])?, Meridiem::Pm)
}

/// `1140 a.m.`: three or four digits and a half of the day.
pub(crate) fn b_hhmm_ampm(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if !(3..=4).contains(&dig_len(&v[0])) {
        return None;
    }
    let n = term(&v[0]).0;
    let k = Clock::new((n / 100) as u8, u8::try_from(n % 100).ok()?, 0, Shown::Minute, Meridiem::Unknown)?;
    with_meridiem(c, k, meridiem(&v[1]))
}

/// A clock as written, stated outright: `seven o'clock`.
pub(crate) fn b_same_clock(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    clock_of(c, &v[0]).and_then(|k| (k.shown == Shown::Hour).then_some(v[0]))
}

/// `11ish`, `noonish`: the hour as written.
pub(crate) fn b_ish(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let k = clock_of(c, &v[0])?;
    emit_clock(c, Clock { meridiem: Meridiem::Day, ..k })
}

pub(crate) fn minutes_past(c: &Ctx<'_>, m: i64, v: &Value) -> Option<Value> {
    let k = clock_of(c, v)?;
    if k.shown != Shown::Hour || !(1..60).contains(&m) {
        return None;
    }
    emit_clock(c, Clock { minute: m as u8, shown: Shown::Minute, ..k })
}

pub(crate) fn b_half_past(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    minutes_past(c, 30, &v[0])
}

pub(crate) fn b_quarter_past(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    // The quarter word's child comes first.
    (term(&v[0]).0 == 4).then(|| minutes_past(c, 15, &v[1]))?
}

pub(crate) fn b_minutes_past(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let m = v[0].num()?.value.to_i64()?;
    minutes_past(c, m, &v[1])
}

pub(crate) fn b_noon(_: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_clock(c, Clock::new(12, 0, 0, Shown::Hour, Meridiem::Pm)?)
}

pub(crate) fn b_twelve_noon(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    (term(&v[0]).0 == 12).then(|| b_noon(v, c))?
}

pub(crate) fn b_midnight(_: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_clock(c, Clock::new(0, 0, 0, Shown::Hour, Meridiem::Day)?)
}

pub(crate) const PARTS: [Part; 10] = [
    Part::Morning,
    Part::Afternoon,
    Part::Evening,
    Part::Night,
    Part::Daytime,
    Part::NightTime,
    Part::Breakfast,
    Part::Lunch,
    Part::Dinner,
    Part::Dawn,
];

pub(crate) fn part(v: &Value) -> Option<Part> {
    PARTS.get(usize::try_from(term(v).0).ok()?).copied()
}

pub(crate) fn emit_part(c: &Ctx<'_>, p: Part, modifier: Modifier) -> Option<Value> {
    emit(c, TimeValue::Point { point: Point::time(TimeOfDay::Part(p)), modifier, clip: false })
}

/// A part of the day; a plural only after `in the` (`mornings` alone
/// recurs).
pub(crate) fn b_part(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if term(&v[0]).1 & PLURAL != 0 {
        return None;
    }
    emit_part(c, part(&v[0])?, Modifier::NONE)
}

pub(crate) fn b_part_any(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_part(c, part(v.last()?)?, Modifier::NONE)
}

pub(crate) fn b_night_time(_: &[Value], c: &Ctx<'_>) -> Option<Value> {
    emit_part(c, Part::NightTime, Modifier::NONE)
}

pub(crate) fn portion_part(c: &Ctx<'_>, v: &Value, portion: Portion) -> Option<Value> {
    let p = part(v)?;
    if !matches!(p, Part::Dawn | Part::Morning | Part::Afternoon | Part::Evening | Part::Night) {
        return None;
    }
    emit_part(c, p, Modifier::portion(portion))
}

pub(crate) fn b_early_part(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    portion_part(c, v.last()?, Portion::Start)
}

pub(crate) fn b_late_part(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    portion_part(c, v.last()?, Portion::End)
}

/// A clock and a part of the day that sets its half: `8 in the evening`,
/// `in the afternoon at 7`, `one in the morning`.
pub(crate) fn clock_in_part(c: &Ctx<'_>, clock: &Value, part_v: &Value) -> Option<Clock> {
    let k = clock_of(c, clock)?;
    let p = match tval(c, part_v)? {
        TimeValue::Point { point: Point { time: TimeOfDay::Part(p), .. }, .. } => p,
        _ => return None,
    };
    Some(k.with(p.meridiem(k.hour)))
}

/// Whether a part of the day is a meal, which takes no clock time and no
/// date: `dinner seven p.m.` is two mentions.
pub(crate) fn is_meal(p: Part) -> bool {
    matches!(p, Part::Breakfast | Part::Lunch | Part::Dinner)
}

pub(crate) fn part_value(c: &Ctx<'_>, v: &Value) -> Option<Part> {
    match tval(c, v)? {
        TimeValue::Point { point: Point { time: TimeOfDay::Part(p), .. }, .. } => Some(p),
        _ => None,
    }
}

pub(crate) fn b_clock_part(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if is_meal(part_value(c, &v[1])?) {
        return None;
    }
    emit_clock(c, clock_in_part(c, &v[0], &v[1])?)
}

/// An hour alone takes a part of the day said with `in` or `at`: `8 in
/// the evening`, not `2 morning sessions`.
pub(crate) fn b_bare_part(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    let next = c.lexed.len().min(c.end);
    let at = (c.start + 1..next).find(|&i| matches!(c.word(i), "in" | "at"));
    at.is_some().then(|| b_clock_part(v, c))?
}

pub(crate) fn b_part_clock(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if is_meal(part_value(c, &v[0])?) {
        return None;
    }
    emit_clock(c, clock_in_part(c, &v[1], &v[0])?)
}

pub(crate) fn b_am(_: &[Value], _: &Ctx<'_>) -> Option<Value> {
    term_value(1, 0)
}

pub(crate) fn b_pm(_: &[Value], _: &Ctx<'_>) -> Option<Value> {
    term_value(2, 0)
}

pub(crate) fn b_either(_: &[Value], _: &Ctx<'_>) -> Option<Value> {
    term_value(3, 0)
}

const fn rr(name: &'static str, dim: crate::engine::Dim, items: &'static [Item], build: crate::engine::Build) -> Rule {
    r(name, dim, 0, items, build)
}

pub(super) static RULES: &[Rule] = &[
    // Halves of the day.
    rr("ampm_am", AMPM, &[word("am")], b_am),
    rr("ampm_pm", AMPM, &[word("pm")], b_pm),
    rr("ampm_a_m", AMPM, &[word("a"), word("."), word("m"), opt_word(".")], b_am),
    rr("ampm_p_m", AMPM, &[word("p"), word("."), word("m"), opt_word(".")], b_pm),
    rr("ampm_a_m_words", AMPM, &[word("a"), word("m")], b_am),
    rr("ampm_p_m_words", AMPM, &[word("p"), word("m")], b_pm),
    rr("ampm_either", AMPM, &[word("ampm")], b_either),
    // Digits.
    rr("dig", DIG, &[node(NUMERAL)], b_dig),
    // Hours that need context.
    rr("bare_dig", BARE, &[node(DIG)], b_bare_dig),
    rr("bare_word", BARE, &[node(INT)], b_bare_word),
    rr("bare_word_minutes", BARE, &[node(INT), opt_word("-"), node(INT), opt_word("minutes")], b_bare_word_minutes),
    rr("bare_dot", BARE, &[node(NUMERAL)], b_bare_dot),
    // Clock times.
    rr("clock_hm", CLOCK, &[node(DIG), glued_word(":"), glued(DIG)], b_hm),
    rr("clock_hms", CLOCK, &[node(DIG), glued_word(":"), glued(DIG), glued_word(":"), glued(DIG)], b_hms),
    rr("clock_hhmm", CLOCK, &[node(DIG)], b_hhmm_zero),
    rr("clock_bare_ampm", CLOCK, &[node(BARE), node(AMPM)], b_clock_ampm),
    rr("clock_ampm", CLOCK, &[node(CLOCK), node(AMPM)], b_clock_ampm),
    rr("clock_hhmm_ampm", CLOCK, &[node(DIG), node(AMPM)], b_hhmm_ampm),
    rr("clock_bare_a", CLOCK, &[node(BARE), glued_word("a")], b_clock_am),
    rr("clock_bare_p", CLOCK, &[node(BARE), glued_word("p")], b_clock_pm),
    rr("clock_a", CLOCK, &[node(CLOCK), glued_word("a")], b_clock_am),
    rr("clock_p", CLOCK, &[node(CLOCK), glued_word("p")], b_clock_pm),
    rr("clock_dot_ampm", CLOCK, &[node(BARE), glued_word("."), glued(AMPM)], b_clock_ampm),
    rr("clock_oclock", CLOCK, &[node(BARE), word("o"), word("'"), word("clock")], b_same_clock),
    rr("clock_oclock_curly", CLOCK, &[node(BARE), word("o"), word("\u{2019}"), word("clock")], b_same_clock),
    rr("clock_oclock_word", CLOCK, &[node(BARE), word("oclock")], b_same_clock),
    rr("clock_o_clock", CLOCK, &[node(BARE), word("o"), word("clock")], b_same_clock),
    rr("clock_ish", CLOCK, &[node(BARE), glued_word("ish")], b_ish),
    rr("clock_half_past", CLOCK, &[word("half"), word("past"), node(BARE)], b_half_past),
    rr("clock_half_past_c", CLOCK, &[word("half"), word("past"), node(CLOCK)], b_half_past),
    rr("clock_quarter_past", CLOCK, &[node(FRAC_WORD), word("past"), node(BARE)], b_quarter_past),
    rr(
        "clock_minutes_past",
        CLOCK,
        &[
            node(INT),
            opt_word("min"),
            opt_word("mins"),
            opt_word("minute"),
            opt_word("minutes"),
            word("past"),
            node(BARE),
        ],
        b_minutes_past,
    ),
    rr(
        "clock_minutes_past_c",
        CLOCK,
        &[
            node(INT),
            opt_word("min"),
            opt_word("mins"),
            opt_word("minute"),
            opt_word("minutes"),
            word("past"),
            node(CLOCK),
        ],
        b_minutes_past,
    ),
    rr("clock_noon", CLOCK, &[word("noon")], b_noon),
    rr("clock_noonish", CLOCK, &[word("noonish")], b_noon),
    rr("clock_midday", CLOCK, &[word("midday")], b_noon),
    rr("clock_twelve_noon", CLOCK, &[node(DIG), word("noon")], b_twelve_noon),
    rr("clock_midnight", CLOCK, &[word("midnight")], b_midnight),
    rr("clock_mid_night", CLOCK, &[word("mid"), word("night")], b_midnight),
    rr("clock_at_dot", CLOCK, &[word("at"), node(BARE)], b_at_dot),
    rr("clock_at_dot_ampm", CLOCK, &[word("at"), node(BARE), node(AMPM)], b_at_dot_ampm),
    // Parts of the day.
    rr("partd", PARTD, &[node(PART)], b_part),
    rr("partd_in_the", PARTD, &[word("in"), word("the"), node(PART)], b_part_any),
    rr("partd_in", PARTD, &[word("in"), node(PART)], b_part),
    rr("partd_at", PARTD, &[word("at"), node(PART)], b_part),
    rr("partd_night_time", PARTD, &[word("night"), word("-"), word("time")], b_night_time),
    rr(
        "partd_in_the_night_time",
        PARTD,
        &[word("in"), word("the"), word("night"), word("-"), word("time")],
        b_night_time,
    ),
    rr("partd_early", PARTD, &[word("early"), node(PART)], b_early_part),
    rr("partd_early_in_the", PARTD, &[word("early"), word("in"), word("the"), node(PART)], b_early_part),
    rr("partd_late", PARTD, &[word("late"), node(PART)], b_late_part),
    rr("partd_later_in_the", PARTD, &[word("later"), word("in"), word("the"), node(PART)], b_late_part),
    rr("partd_late_in_the", PARTD, &[word("late"), word("in"), word("the"), node(PART)], b_late_part),
    // A clock in a part of the day.
    rr("clock_part", CLOCK, &[node(BARE), node(PARTD)], b_bare_part),
    rr("clock_c_part", CLOCK, &[node(CLOCK), node(PARTD)], b_clock_part),
    rr("clock_part_at", CLOCK, &[node(PARTD), opt_word("at"), node(CLOCK)], b_part_clock),
    rr("clock_part_at_bare", CLOCK, &[node(PARTD), word("at"), node(BARE)], b_part_clock),
];

/// `at 6.45`: a dotted time takes its `at` along.
pub(crate) fn b_at_dot(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    c.raw(c.start + 1).contains('.').then_some(v[0])
}

pub(crate) fn b_at_dot_ampm(v: &[Value], c: &Ctx<'_>) -> Option<Value> {
    if !c.raw(c.start + 1).contains('.') {
        return None;
    }
    b_clock_ampm(v, c)
}
