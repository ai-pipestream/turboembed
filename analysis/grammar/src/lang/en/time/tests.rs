use turbo_analysis_layers::DateTime;

use super::*;
use crate::engine::Parser;
use crate::time::resolve;

/// Each mention as `text|kind|timex|modifier`, with its preferred
/// reading, against the reference `at`.
fn found(text: &str, at: &str) -> Vec<String> {
    let now = DateTime::parse(at).unwrap();
    let mut p = Parser::new(&ENGLISH_TIMES);
    p.parse(text);
    let mut ms = Vec::new();
    p.select(text, Output::Time, &mut ms);
    ms.iter()
        .map(|m| {
            let v = p.times().get(&m.value).unwrap();
            let r = resolve(&v, now).unwrap();
            let a = r.alternatives[r.preferred];
            format!("{}|{}|{}|{}", &text[m.start as usize..m.end as usize], r.kind.name(), a.timex, r.modifier)
        })
        .collect()
}

fn one(text: &str) -> String {
    let f = found(text, "2016-11-07T16:12:00");
    assert_eq!(f.len(), 1, "{text}: {f:?}");
    f[0].clone()
}

fn none(text: &str) {
    let f = found(text, "2016-11-07T16:12:00");
    assert!(f.is_empty(), "{text}: {f:?}");
}

#[test]
fn clock_times() {
    assert_eq!(one("back 7:56:30 pm"), "7:56:30 pm|time|T19:56:30|");
    assert_eq!(one("at 5pm"), "5pm|time|T17|");
    assert_eq!(one("1140 a.m. please"), "1140 a.m.|time|T11:40|");
    assert_eq!(one("half past seven"), "half past seven|time|T07:30|");
    assert_eq!(one("at noon"), "noon|time|T12|");
    assert_eq!(one("eight in the evening"), "eight in the evening|time|T20|");
    assert_eq!(one("meet at 6.45"), "at 6.45|time|T06:45|");
    none("Mortgages were at 6.45 percent");
    none("the 5 cars");
}

#[test]
fn parts_of_the_day() {
    assert_eq!(one("in the morning"), "in the morning|timerange|TMO|");
    assert_eq!(one("late afternoon"), "late afternoon|timerange|TAF|end");
    none("Good morning");
    none("We will have lunch together");
}

#[test]
fn dates() {
    assert_eq!(one("back 04th Jan 2019."), "04th Jan 2019|date|2019-01-04|");
    assert_eq!(one("on 2016-11-07"), "2016-11-07|date|2016-11-07|");
    assert_eq!(one("see you 22/04 maybe"), "22/04|date|XXXX-04-22|");
    assert_eq!(one("see you Friday"), "Friday|date|XXXX-WXX-5|");
    assert_eq!(one("next Monday"), "next Monday|date|2016-11-14|");
    assert_eq!(one("tomorrow"), "tomorrow|date|2016-11-08|");
    assert_eq!(one("on the 18th"), "the 18th|date|XXXX-XX-18|");
    none("the 18th of them");
    assert_eq!(one("the first Monday of next month"), "the first Monday of next month|date|XXXX-12-WXX-1-#1|");
    assert_eq!(one("on Christmas"), "Christmas|date|XXXX-12-25|");
}

#[test]
fn ambiguous_words_need_their_cue() {
    assert_eq!(one("in May"), "May|daterange|XXXX-05|");
    none("This may take longer than expected.");
    none("One may ask");
    none("The sun will set");
    none("I paid $12 for it");
    none("up from 21 1/4");
    none("3/4 of a point");
    assert_eq!(found("in April. One vote", "2016-11-07T16:12:00"), ["April|daterange|XXXX-04|"]);
    assert_eq!(one("on Apr. 1st"), "Apr. 1st|date|XXXX-04-01|");
}

#[test]
fn periods() {
    assert_eq!(one("next week"), "next week|daterange|2016-W46|");
    assert_eq!(one("in 2016"), "2016|daterange|2016|");
    assert_eq!(one("Q3 2019"), "Q3 2019|daterange|(2019-07-01,2019-10-01,P3M)|");
    assert_eq!(one("in the 1990s"), "1990s|daterange|(1990-01-01,2000-01-01,P10Y)|");
    assert_eq!(one("in the nineteen eighties"), "nineteen eighties|daterange|(1980-01-01,1990-01-01,P10Y)|");
    assert_eq!(one("this summer"), "this summer|daterange|2016-SU|");
    assert_eq!(one("the same week"), "same week|daterange|XXXX-WXX|ref_undef");
}

#[test]
fn dates_with_times() {
    assert_eq!(one("tomorrow 8:00am"), "tomorrow 8:00am|datetime|2016-11-08T08:00|");
    assert_eq!(one("9am on Tuesday"), "9am on Tuesday|datetime|XXXX-WXX-2T09|");
    assert_eq!(one("this evening"), "this evening|datetimerange|2016-11-07TEV|");
    assert_eq!(one("tonight at 2am"), "tonight at 2am|datetime|2016-11-07T02|");
    assert_eq!(one("now"), "now|datetime|PRESENT_REF|");
    assert_eq!(one("in 5 minutes"), "in 5 minutes|datetime|2016-11-07T16:17:00|");
    assert_eq!(one("2 days ago"), "2 days ago|date|2016-11-05|");
    assert_eq!(one("3 days after January 12th"), "3 days after January 12th|date|2017-01-15|");
}

#[test]
fn durations() {
    assert_eq!(one("for 3.5years"), "3.5years|duration|P3.5Y|");
    assert_eq!(one("for an hour"), "an hour|duration|PT1H|");
    assert_eq!(one("for a few minutes"), "a few minutes|duration|PT3M|");
    assert_eq!(one("one hour and a half"), "one hour and a half|duration|PT1.5H|");
    assert_eq!(one("1 hour and 30 minutes"), "1 hour and 30 minutes|duration|PT1H30M|");
    assert_eq!(one("more than 4 days"), "more than 4 days|duration|P4D|more");
    assert_eq!(one("for 3h"), "3h|duration|PT3H|");
    none("wait a second");
}

#[test]
fn sets() {
    assert_eq!(one("weekly"), "weekly|set|P1W|");
    assert_eq!(one("every two days"), "every two days|set|P2D|");
    assert_eq!(one("once a week"), "once a week|set|P1W|");
    assert_eq!(one("each monday at 4pm"), "each monday at 4pm|set|XXXX-WXX-1T16|");
    assert_eq!(one("3pm each day"), "3pm each day|set|T15|");
    assert_eq!(one("tuesday afternoons"), "tuesday afternoons|set|XXXX-WXX-2TAF|");
    assert_eq!(one("semi-annually"), "semi-annually|set|P0.5Y|");
    assert_eq!(one("every other Friday"), "every other Friday|set|XXXX-WXX-5|");
    assert_eq!(
        one("The board meets every second Thursday at 9 a.m."),
        "every second Thursday at 9 a.m.|set|XXXX-WXX-4T09|"
    );
    assert_eq!(one("every alternate week"), "every alternate week|set|P2W|");
    assert_eq!(one("every second day"), "every second day|set|P2D|");
    assert_eq!(one("it beeps every second"), "every second|set|PT1S|");
}

#[test]
fn ranges() {
    assert_eq!(one("from 2014 to 2018"), "from 2014 to 2018|daterange|(2014-01-01,2018-01-01,P4Y)|");
    assert_eq!(one("from May to Oct 2020"), "from May to Oct 2020|daterange|(2020-05-01,2020-10-01,P5M)|");
    assert_eq!(one("between 3 and 12 of Sept"), "between 3 and 12 of Sept|daterange|(XXXX-09-03,XXXX-09-12,P9D)|");
    assert_eq!(one("5 to 6pm"), "5 to 6pm|timerange|(T17,T18,PT1H)|");
    assert_eq!(one("Monday 8-9am"), "Monday 8-9am|datetimerange|(XXXX-WXX-1T08,XXXX-WXX-1T09,PT1H)|");
    assert_eq!(one("next 3 days"), "next 3 days|daterange|(2016-11-08,2016-11-11,P3D)|");
    assert_eq!(one("within 9 months"), "within 9 months|daterange|(2016-11-07,2017-08-07,P9M)|");
    assert_eq!(one("2 weeks starting May 20th"), "2 weeks starting May 20th|daterange|(2016-05-20,2016-06-03,P2W)|");
    assert_eq!(one("from 9 for 2.5 hrs"), "from 9 for 2.5 hrs|timerange|(T09,T11:30:00,PT2H30M)|");
    assert_eq!(one("the course will last two hours"), "two hours|duration|PT2H|");
}

#[test]
fn modifiers() {
    assert_eq!(one("before 2018"), "before 2018|daterange|2018|before");
    assert_eq!(one("since the end of 1989"), "since the end of 1989|daterange|1989|since-end");
    assert_eq!(one("around 1pm"), "around 1pm|time|T13|approx");
    assert_eq!(one("before around 2pm"), "before around 2pm|timerange|T14|before-approx");
    assert_eq!(one("today before 4pm"), "today before 4pm|datetimerange|2016-11-07T16|before");
    assert_eq!(one("2018 or later"), "2018 or later|daterange|2018|since");
    assert_eq!(one("later this week"), "later this week|daterange|2016-W45|");
    assert_eq!(one("mid-November"), "mid-November|daterange|XXXX-11|mid");
    assert_eq!(one("late 1980s"), "late 1980s|daterange|(1980-01-01,1990-01-01,P10Y)|end");
    assert_eq!(one("in the early seventies"), "early seventies|daterange|(XX70-01-01,XX80-01-01,P10Y)|start");
    assert_eq!(one("by the late eighties"), "by the late eighties|daterange|(XX80-01-01,XX90-01-01,P10Y)|before-end");
    assert_eq!(one("mid-2015"), "mid-2015|daterange|2015|mid");
    assert_eq!(one("end of tomorrow"), "end of tomorrow|datetime|2016-11-08T23:59:59|");
}

#[test]
fn years_from_two_digits() {
    let t = |y, f| year_ref(&Value::Term { value: y, flags: f });
    assert_eq!(t(2016, 0), YearRef::Abs(2016));
    assert_eq!(t(18, SHORT), YearRef::Abs(2018));
    assert_eq!(t(35, SHORT), YearRef::Short(35));
    assert_eq!(t(87, SHORT), YearRef::Abs(1987));
}

#[test]
fn the_grammar_holds_the_number_rules() {
    assert!(ENGLISH_TIMES.rules().len() > ENGLISH.rules().len());
    assert_eq!(ENGLISH_TIMES.name(), "en-times");
    assert_eq!(ENGLISH_TIMES.output(CLOCK), Some(Output::Time));
    assert_eq!(ENGLISH_TIMES.output(MDAY), None);
    let names: std::collections::HashSet<_> = ENGLISH_TIMES.rules().iter().map(|r| r.name).collect();
    assert_eq!(names.len(), ENGLISH_TIMES.rules().len(), "rule names are unique");
}
