use turbo_analysis_layers::{Date, DateTime, Modifier, Portion, Relation, TimeGrain, TimeKind};

use super::dates::{Choice, DateTimex, dates};
use super::*;
use crate::time::duration::{Duration, Unit};
use crate::time::spec::{Clock, Edge, Holiday, Meridiem, MonthRef, Part, Point, Range, Recurrence, Shown, TimeOfDay};

fn now(s: &str) -> DateTime {
    DateTime::parse(s).unwrap()
}

/// Each alternative as `timex value` or `timex start..end`.
fn show(r: &Resolved) -> Vec<String> {
    r.alternatives
        .iter()
        .map(|a| match a.value {
            Resolution::Point(i) => format!("{} {i}", a.timex),
            Resolution::Range { start, end } => {
                let f = |x: Option<Instant>| x.map_or(String::new(), |i| i.to_string());
                format!("{} {}..{}", a.timex, f(start), f(end))
            }
            Resolution::Duration { millis } => format!("{} {millis}", a.timex),
            Resolution::NotResolved => format!("{} -", a.timex),
        })
        .collect()
}

fn show_choice(c: &Choice) -> Vec<String> {
    c.readings()
        .map(|(_, d)| {
            let t = match d.timex {
                DateTimex::Date(t) => t.to_string(),
                DateTimex::Range(s, e, l) => format!("({s},{e},{l})"),
            };
            match d.span {
                Some((s, e)) => format!("{t} {s} {e}"),
                None => format!("{t} -"),
            }
        })
        .collect()
}

fn day(n: i16) -> DateSpec {
    DateSpec::Rel { grain: TimeGrain::Day, n }
}

fn clock(hour: u8, minute: u8, shown: Shown, meridiem: Meridiem) -> TimeOfDay {
    TimeOfDay::Clock(Clock::new(hour, minute, 0, shown, meridiem).unwrap())
}

fn cal(year: YearRef, month: u8, day: u8, weekday: u8) -> DateSpec {
    DateSpec::Cal { year, month, day, weekday }
}

fn point(p: Point) -> Resolved {
    resolve(&TimeValue::point(p), now("2016-11-07T16:12:00")).unwrap()
}

#[test]
fn open_years_read_both_ways() {
    let n = now("2016-11-07T00:00:00");
    assert_eq!(
        show_choice(&dates(cal(YearRef::Any, 10, 2, 0), n).unwrap()),
        ["XXXX-10-02 2016-10-02 2016-10-03", "XXXX-10-02 2017-10-02 2017-10-03"]
    );
    let n2 = now("2016-11-02T10:00:00");
    assert_eq!(show_choice(&dates(cal(YearRef::Any, 11, 2, 0), n2).unwrap())[1], "XXXX-11-02 2017-11-02 2017-11-03");
    assert_eq!(show_choice(&dates(cal(YearRef::Any, 0, 0, 5), n).unwrap()).len(), 2);
    assert_eq!(show_choice(&dates(cal(YearRef::Any, 0, 0, 1), n).unwrap())[1], "XXXX-WXX-1 2016-11-07 2016-11-08");
    assert_eq!(show_choice(&dates(cal(YearRef::Any, 12, 0, 0), n).unwrap())[0], "XXXX-12 2015-12-01 2016-01-01");
    assert_eq!(show_choice(&dates(cal(YearRef::Any, 2, 30, 0), n).unwrap()), ["XXXX-02-30 -"]);
    let leap = show_choice(&dates(cal(YearRef::Any, 2, 29, 0), now("2017-03-22T00:00:00")).unwrap());
    assert_eq!(leap, ["XXXX-02-29 2016-02-29 2016-03-01", "XXXX-02-29 2020-02-29 2020-03-01"]);
    let the_4th = show_choice(&dates(cal(YearRef::Any, 0, 4, 0), now("2018-08-07T12:00:00")).unwrap());
    assert_eq!(the_4th, ["XXXX-XX-04 2018-08-04 2018-08-05", "XXXX-XX-04 2018-09-04 2018-09-05"]);
    let short = show_choice(&dates(cal(YearRef::Short(30), 12, 12, 0), n).unwrap());
    assert_eq!(short, ["XX30-12-12 1930-12-12 1930-12-13", "XX30-12-12 2030-12-12 2030-12-13"]);
}

#[test]
fn weekdays_on_days_of_the_month() {
    let r = show_choice(
        &dates(DateSpec::WeekdayDay { weekday: 1, day: 21, this_month: false }, now("2019-01-25T12:00:00")).unwrap(),
    );
    assert_eq!(r, ["XXXX-WXX-1 2019-01-21 2019-01-22", "XXXX-WXX-1 2019-10-21 2019-10-22"]);
    let r = show_choice(
        &dates(DateSpec::WeekdayDay { weekday: 1, day: 21, this_month: false }, now("2019-01-21T12:00:00")).unwrap(),
    );
    assert_eq!(r, ["2019-01-21 2019-01-21 2019-01-22"]);
}

#[test]
fn relative_periods() {
    let n = now("2019-07-04T00:00:00");
    let q = |n: i16| {
        show_choice(&dates(DateSpec::Rel { grain: TimeGrain::Quarter, n }, now("2019-07-04T00:00:00")).unwrap())
    };
    assert_eq!(q(0), ["(2019-07-01,2019-10-01,P3M) 2019-07-01 2019-10-01"]);
    assert_eq!(q(-1)[0].split(' ').next(), Some("(2019-04-01,2019-07-01,P3M)"));
    assert_eq!(
        show_choice(&dates(DateSpec::Rel { grain: TimeGrain::Week, n: 1 }, n).unwrap())[0],
        "2019-W28 2019-07-08 2019-07-15"
    );
    let we =
        show_choice(&dates(DateSpec::Rel { grain: TimeGrain::Weekend, n: 0 }, now("2016-11-11T00:00:00")).unwrap());
    assert_eq!(we, ["2016-W45-WE 2016-11-12 2016-11-14"]);
    let decade =
        show_choice(&dates(DateSpec::Rel { grain: TimeGrain::Decade, n: 1 }, now("2018-07-02T00:00:00")).unwrap());
    assert_eq!(decade, ["(2020-01-01,2030-01-01,P10Y) 2020-01-01 2030-01-01"]);
    let past = show_choice(&dates(DateSpec::WeekdayPast { weekday: 3 }, now("2019-01-31T00:00:00")).unwrap());
    assert_eq!(past, ["2019-01-30 2019-01-30 2019-01-31"]);
    let fortnight = show_choice(&dates(DateSpec::Fortnight { n: 0 }, now("2021-05-20T00:00:00")).unwrap());
    assert_eq!(fortnight, ["(2021-05-17,2021-05-31,P2W) 2021-05-17 2021-05-31"]);
    let same =
        show_choice(&dates(DateSpec::Undefined { grain: TimeGrain::Month }, now("2017-11-08T00:00:00")).unwrap());
    assert_eq!(same, ["XXXX-XX 2017-11-01 2017-12-01"]);
}

#[test]
fn weeks_quarters_decades_and_holidays() {
    let n = now("2016-11-07T00:00:00");
    let lw = DateSpec::MonthWeek { year: YearRef::Any, month: MonthRef::Abs(7), week: 0 };
    assert_eq!(
        show_choice(&dates(lw, n).unwrap()),
        ["XXXX-07-W05 2016-07-25 2016-08-01", "XXXX-07-W05 2017-07-24 2017-07-31"]
    );
    let w1 = DateSpec::Week { year: YearRef::Abs(2015), week: 1 };
    assert_eq!(show_choice(&dates(w1, n).unwrap()), ["2015-W01 2014-12-29 2015-01-05"]);
    let q1 = DateSpec::Quarter { year: YearRef::Any, quarter: 1 };
    let r = show_choice(&dates(q1, now("2019-11-07T00:00:00")).unwrap());
    assert_eq!(r[0], "(XXXX-01-01,XXXX-04-01,P3M) 2019-01-01 2019-04-01");
    assert_eq!(r[1], "(XXXX-01-01,XXXX-04-01,P3M) 2020-01-01 2020-04-01");
    let nineties =
        show_choice(&dates(DateSpec::Decade { year: YearRef::Short(90) }, now("2019-05-23T00:00:00")).unwrap());
    assert_eq!(nineties[0], "(XX90-01-01,XX00-01-01,P10Y) 1990-01-01 2000-01-01");
    assert_eq!(nineties[1], "(XX90-01-01,XX00-01-01,P10Y) 2090-01-01 2100-01-01");
    let c15 = show_choice(&dates(DateSpec::Century { n: 15 }, n).unwrap());
    assert_eq!(c15, ["(1400-01-01,1500-01-01,P100Y) 1400-01-01 1500-01-01"]);
    let easter_monday = DateSpec::Holiday { holiday: Holiday::Easter { days: 1 }, year: YearRef::Any };
    let r = show_choice(&dates(easter_monday, now("2019-08-05T00:00:00")).unwrap());
    assert_eq!(r, ["XXXX-04-22 2019-04-22 2019-04-23", "XXXX-04-22 2020-04-13 2020-04-14"]);
    let next_easter = DateSpec::HolidayNear { holiday: Holiday::Easter { days: 0 }, future: true };
    assert_eq!(
        show_choice(&dates(next_easter, now("2019-06-12T00:00:00")).unwrap())[0],
        "2020-04-12 2020-04-12 2020-04-13"
    );
    let first_monday = DateSpec::NthWeekday { year: YearRef::Any, month: MonthRef::Rel(1), weekday: 1, n: 1 };
    assert_eq!(
        show_choice(&dates(first_monday, now("2018-08-01T12:00:00")).unwrap())[0],
        "XXXX-09-WXX-1-#1 2018-09-03 2018-09-04"
    );
}

#[test]
fn dates_times_and_both() {
    let r = point(Point { date: day(1), time: clock(8, 0, Shown::Minute, Meridiem::Am), ..Point::default() });
    assert_eq!((r.kind, r.grain), (TimeKind::DateTime, TimeGrain::Minute));
    assert_eq!(show(&r), ["2016-11-08T08:00 2016-11-08 08:00:00"]);
    let r = point(Point::time(clock(5, 0, Shown::Hour, Meridiem::Unknown)));
    assert_eq!(show(&r), ["T05 05:00:00", "T17 17:00:00"]);
    assert_eq!(r.preferred, 0, "the first half of the day is preferred");
    let r = point(Point::date(cal(YearRef::Any, 0, 0, 5)));
    assert_eq!(show(&r), ["XXXX-WXX-5 2016-11-04", "XXXX-WXX-5 2016-11-11"]);
    assert_eq!(r.preferred, 1, "the future reading is preferred");
    let r = point(Point { date: day(0), time: TimeOfDay::Part(Part::Evening), ..Point::default() });
    assert_eq!(
        (r.kind, show(&r)),
        (TimeKind::DateTimeRange, vec!["2016-11-07TEV 2016-11-07 16:00:00..2016-11-07 20:00:00".to_string()])
    );
    let r = point(Point::time(TimeOfDay::Now));
    assert_eq!(show(&r), ["PRESENT_REF 2016-11-07 16:12:00"]);
}

#[test]
fn shifts_move_the_reference() {
    let five_minutes = Duration::units(Unit::Minute, 5).unwrap();
    let r = point(Point { time: TimeOfDay::Now, shift: five_minutes, ..Point::default() });
    assert_eq!(show(&r), ["2016-11-07T16:17:00 2016-11-07 16:17:00"]);
    let two_days_ago = Duration::units(Unit::Day, 2).unwrap().negated();
    let r = point(Point { date: day(0), shift: two_days_ago, ..Point::default() });
    assert_eq!((r.kind, show(&r)), (TimeKind::Date, vec!["2016-11-05 2016-11-05".to_string()]));
}

#[test]
fn modifiers_bound_and_cut() {
    let n = now("2018-05-28T00:00:00");
    let year = |y| Point::date(cal(YearRef::Abs(y), 0, 0, 0));
    let m = |relation, portion| Modifier { relation, portion };
    let before = TimeValue::Point { point: year(2018), modifier: Modifier::relation(Relation::Before), clip: false };
    let r = resolve(&before, n).unwrap();
    assert_eq!((r.kind, show(&r)), (TimeKind::DateRange, vec!["2018 ..2018-01-01".to_string()]));
    let after = TimeValue::Point { point: year(2016), modifier: Modifier::relation(Relation::After), clip: false };
    assert_eq!(show(&resolve(&after, n).unwrap()), ["2016 2017-01-01.."]);
    let since_end = TimeValue::Point { point: year(1989), modifier: m(Relation::Since, Portion::End), clip: false };
    let r = resolve(&since_end, n).unwrap();
    assert_eq!((r.modifier.to_string(), show(&r)), ("since-end".to_string(), vec!["1989 1989-09-01..".to_string()]));
    // `later this year`: clipped at the reference, with no modifier shown.
    let this_year = Point::date(DateSpec::Rel { grain: TimeGrain::Year, n: 0 });
    let later = TimeValue::Point { point: this_year, modifier: Modifier::portion(Portion::End), clip: true };
    let r = resolve(&later, n).unwrap();
    assert_eq!((r.modifier, show(&r)), (Modifier::NONE, vec!["2018 2018-07-01..2019-01-01".to_string()]));
    // `end of tomorrow` is its last second, with no modifier shown.
    let end = TimeValue::Point { point: Point::date(day(1)), modifier: Modifier::portion(Portion::End), clip: false };
    let r = resolve(&end, n).unwrap();
    assert_eq!(
        (r.kind, r.modifier, show(&r)),
        (TimeKind::DateTime, Modifier::NONE, vec!["2018-05-29T23:59:59 2018-05-29 23:59:59".to_string()])
    );
    // A day with both a portion and a relation stays DateTimeRange.
    let since_end_day =
        TimeValue::Point { point: Point::date(day(1)), modifier: m(Relation::Since, Portion::End), clip: false };
    let r = resolve(&since_end_day, n).unwrap();
    assert_eq!((r.kind, show(&r)), (TimeKind::DateTimeRange, vec!["2018-05-29 2018-05-29 12:00:00..".to_string()]));
    // `mid 1989` is May to August, as Recognizers-Text gives it.
    let mid = TimeValue::Point { point: year(1989), modifier: Modifier::portion(Portion::Mid), clip: false };
    assert_eq!(show(&resolve(&mid, n).unwrap()), ["1989 1989-05-01..1989-09-01"]);
    // A decade splits as 4, 3 and 3 years.
    let decade = |portion| TimeValue::Point {
        point: Point::date(DateSpec::Decade { year: YearRef::Abs(1980) }),
        modifier: Modifier::portion(portion),
        clip: false,
    };
    let eighties = "(1980-01-01,1990-01-01,P10Y)";
    assert_eq!(show(&resolve(&decade(Portion::Start), n).unwrap()), [format!("{eighties} 1980-01-01..1984-01-01")]);
    assert_eq!(show(&resolve(&decade(Portion::Mid), n).unwrap()), [format!("{eighties} 1984-01-01..1987-01-01")]);
    assert_eq!(show(&resolve(&decade(Portion::End), n).unwrap()), [format!("{eighties} 1987-01-01..1990-01-01")]);
    // A decade without its century prefers the last one by the reference
    // and keeps both readings.
    let late_eighties = TimeValue::Point {
        point: Point::date(DateSpec::Decade { year: YearRef::Short(80) }),
        modifier: Modifier::portion(Portion::End),
        clip: false,
    };
    let r = resolve(&late_eighties, now("2026-03-15T10:00:00")).unwrap();
    let (s, e) =
        ("(XX80-01-01,XX90-01-01,P10Y) 1987-01-01..1990-01-01", "(XX80-01-01,XX90-01-01,P10Y) 2087-01-01..2090-01-01");
    assert_eq!((show(&r), r.preferred), (vec![s.to_string(), e.to_string()], 0));
    let twenties = TimeValue::point(Point::date(DateSpec::Decade { year: YearRef::Short(20) }));
    let r = resolve(&twenties, now("2026-03-15T10:00:00")).unwrap();
    assert_eq!(show(&r)[r.preferred], "(XX20-01-01,XX30-01-01,P10Y) 2020-01-01..2030-01-01");
    // `around 1pm` keeps its point.
    let around = TimeValue::Point {
        point: Point::time(clock(1, 0, Shown::Hour, Meridiem::Pm)),
        modifier: Modifier::relation(Relation::Approx),
        clip: false,
    };
    assert_eq!(show(&resolve(&around, n).unwrap()), ["T13 13:00:00"]);
}

#[test]
fn extreme_dates_and_invalid_inputs_do_not_overflow_or_invert() {
    let min_now = DateTime::midnight(Date::MIN);
    let max_now = DateTime::midnight(Date::MAX);
    assert!(dates(DateSpec::Quarter { year: YearRef::Any, quarter: 0 }, now("2019-11-07T00:00:00")).is_none());
    assert!(dates(DateSpec::Quarter { year: YearRef::Any, quarter: 5 }, now("2019-11-07T00:00:00")).is_none());
    assert!(dates(DateSpec::Half { year: YearRef::Any, half: 0 }, now("2019-11-07T00:00:00")).is_none());
    assert!(dates(DateSpec::Half { year: YearRef::Any, half: 3 }, now("2019-11-07T00:00:00")).is_none());
    assert!(dates(DateSpec::Rel { grain: TimeGrain::Decade, n: i16::MAX }, max_now).is_none());
    assert!(dates(DateSpec::Rel { grain: TimeGrain::Decade, n: i16::MIN }, min_now).is_none());
    assert!(dates(DateSpec::Rel { grain: TimeGrain::Month, n: -1 }, min_now).is_none());
    assert!(dates(DateSpec::Rel { grain: TimeGrain::Year, n: 1 }, max_now).is_none());
    let past_week_clipped = TimeValue::Point {
        point: Point::date(DateSpec::Rel { grain: TimeGrain::Week, n: -2 }),
        modifier: Modifier::portion(Portion::End),
        clip: true,
    };
    assert!(resolve(&past_week_clipped, now("2018-05-28T00:00:00")).is_none());
    let neg_dur = Duration::of(Unit::Day, -1000);
    assert!(resolve(&TimeValue::Duration { length: neg_dur, modifier: Modifier::NONE }, min_now).is_none());
    assert!(resolve(&TimeValue::range(Range::For { start: Point::date(day(0)), length: neg_dur }), min_now).is_none());
    assert!(resolve(&TimeValue::range(Range::Within { length: neg_dur }), min_now).is_none());
}

#[test]
fn ranges_between_points() {
    let n = now("2016-11-07T00:00:00");
    let year = |y| Point::date(cal(YearRef::Abs(y), 0, 0, 0));
    let between =
        |s, e| TimeValue::range(Range::Between { start: s, start_edge: Edge::Start, end: e, end_edge: Edge::Start });
    let r = resolve(&between(year(2014), year(2018)), n).unwrap();
    assert_eq!(
        (r.kind, show(&r)),
        (TimeKind::DateRange, vec!["(2014-01-01,2018-01-01,P4Y) 2014-01-01..2018-01-01".to_string()])
    );
    let to_end = TimeValue::range(Range::Between {
        start: year(2007),
        start_edge: Edge::Start,
        end: year(2008),
        end_edge: Edge::End,
    });
    assert_eq!(show(&resolve(&to_end, n).unwrap()), ["(2007-01-01,2009-01-01,P24M) 2007-01-01..2009-01-01"]);
    let at = |h, m| Point::time(clock(h, 0, Shown::Hour, m));
    let r = resolve(&between(at(5, Meridiem::Unknown), at(6, Meridiem::Pm)), n).unwrap();
    assert_eq!((r.kind, show(&r)), (TimeKind::TimeRange, vec!["(T17,T18,PT1H) 17:00:00..18:00:00".to_string()]));
    let r = resolve(&between(at(3, Meridiem::Unknown), at(8, Meridiem::Unknown)), n).unwrap();
    assert_eq!(r.alternatives.len(), 2, "two open bounds read in both halves of the day");
    let dated = |h| Point { date: day(0), ..at(h, Meridiem::Unknown) };
    let r = resolve(&between(dated(5), dated(7)), now("2016-11-07T16:12:00")).unwrap();
    assert_eq!(r.kind, TimeKind::DateTimeRange);
    assert_eq!(show(&r)[0], "(2016-11-07T05,2016-11-07T07,PT2H) 2016-11-07 05:00:00..2016-11-07 07:00:00");
}

#[test]
fn stretches_from_the_reference() {
    let n = now("2016-11-07T00:00:00");
    let days = Duration::units(Unit::Day, 3).unwrap();
    let next = TimeValue::range(Range::Next { length: days, align: TimeGrain::Day });
    assert_eq!(show(&resolve(&next, n).unwrap()), ["(2016-11-08,2016-11-11,P3D) 2016-11-08..2016-11-11"]);
    let last = TimeValue::range(Range::Next {
        length: Duration::units(Unit::Month, 3).unwrap().negated(),
        align: TimeGrain::Month,
    });
    assert_eq!(show(&resolve(&last, n).unwrap()), ["(2016-08-07,2016-11-07,P3M) 2016-08-07..2016-11-07"]);
    let within = TimeValue::range(Range::Within { length: Duration::units(Unit::Month, 9).unwrap() });
    assert_eq!(show(&resolve(&within, n).unwrap()), ["(2016-11-07,2017-08-07,P9M) 2016-11-07..2017-08-07"]);
    let week = Duration::units(Unit::Week, 1).unwrap();
    let r#for = TimeValue::range(Range::For { start: Point::date(day(1)), length: week });
    assert_eq!(show(&resolve(&r#for, n).unwrap()), ["(2016-11-08,2016-11-15,P1W) 2016-11-08..2016-11-15"]);
    assert!(depends_on_reference(&next));
}

#[test]
fn durations_and_sets() {
    let n = now("2016-11-07T00:00:00");
    let d = TimeValue::Duration { length: Duration::of(Unit::Year, 3500), modifier: Modifier::NONE };
    let r = resolve(&d, n).unwrap();
    assert_eq!(
        (r.kind, r.grain, show(&r)),
        (TimeKind::Duration, TimeGrain::Year, vec!["P3.5Y 110376000000".to_string()])
    );
    let weekly = TimeValue::Set(Recurrence::Every(Duration::units(Unit::Week, 1).unwrap()));
    assert_eq!(show(&resolve(&weekly, n).unwrap()), ["P1W -"]);
    let monday_4pm =
        Point { date: cal(YearRef::Any, 0, 0, 1), time: clock(4, 0, Shown::Hour, Meridiem::Pm), ..Point::default() };
    let r = resolve(&TimeValue::Set(Recurrence::At(monday_4pm)), n).unwrap();
    assert_eq!((r.kind, show(&r)), (TimeKind::Set, vec!["XXXX-WXX-1T16 -".to_string()]));
    assert!(!depends_on_reference(&d) && !depends_on_reference(&weekly));
}

#[test]
fn what_depends_on_the_reference() {
    assert!(depends_on_reference(&TimeValue::point(Point::date(day(1)))));
    assert!(depends_on_reference(&TimeValue::point(Point::date(cal(YearRef::Any, 5, 29, 0)))));
    assert!(!depends_on_reference(&TimeValue::point(Point::date(cal(YearRef::Abs(2016), 11, 7, 0)))));
    assert!(!depends_on_reference(&TimeValue::point(Point::time(clock(5, 0, Shown::Hour, Meridiem::Unknown)))));
    assert!(depends_on_reference(&TimeValue::point(Point::time(TimeOfDay::Now))));
}
