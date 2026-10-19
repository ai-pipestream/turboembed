use super::*;

pub(crate) const fn rr(
    name: &'static str,
    dim: crate::engine::Dim,
    items: &'static [Item],
    build: crate::engine::Build,
) -> Rule {
    r(name, dim, 0, items, build)
}

pub(crate) static RULES: &[Rule] = &[
    rr("sep_slash", SEP, &[word("/")], b_sep),
    rr("sep_dash", SEP, &[word("-")], b_sep),
    // Years.
    rr("year_digits", YEAR, &[node(DIG)], b_year_digits),
    rr("year_short", YEAR, &[node(DIG)], b_year_short),
    rr("year_spelled", YEAR, &[node(INT)], b_year_spelled),
    rr("year_pairs", YEAR, &[node(INT), node(INT)], b_year_pairs),
    // Days of the month.
    rr("mday_digits", MDAY, &[node(DIG)], b_mday_digits),
    rr("mday_ordinal", MDAY, &[node(ORD)], b_mday_ordinal),
    rr("mday_suffix", MDAY, &[node(DIG), glued(ORD_SUFFIX)], b_mday_suffix),
    rr("mday_spelled", MDAY, &[node(INT)], b_mday_spelled),
    // A month and a day.
    rr("md_month_day", MD, &[node(MONTH), opt_word("."), opt_word(","), opt_word("the"), node(MDAY)], b_md_month_day),
    rr("md_day_month", MD, &[node(MDAY), opt_word("of"), node(MONTH)], b_md_day_month),
    rr("md_the_day_month", MD, &[word("the"), node(MDAY), opt_word("of"), node(MONTH)], b_md_day_month),
    rr("md_day_of_month", MD, &[node(MDAY), word("day"), word("of"), node(MONTH)], b_md_day_month),
    rr("md_month_sep_day", MD, &[node(MONTH), glued(SEP), glued(DIG)], |v, c| b_md_month_digits(&[v[0], v[2]], c)),
    rr("md_slash", MD, &[node(DIG), glued_word("/"), glued(DIG)], b_md_slash),
    rr("md_paren", MD, &[node(MD), word("("), node(WEEKDAY), word(")")], b_same),
    rr("md_dot", DATE, &[node(NUMERAL)], b_md_dot),
    // Days.
    rr("date_md", DATE, &[node(MD)], b_same),
    rr("date_wd", DATE, &[node(WD)], b_same),
    rr("date_dom", DATE, &[node(DOM)], b_same),
    rr("date_md_year", DATE, &[node(MD), opt_word(","), opt_word("of"), opt_word("in"), node(YEAR)], b_md_year),
    rr("date_mdy_slash", DATE, &[node(DIG), glued_word("/"), glued(DIG), glued_word("/"), glued(DIG)], |v, c| {
        b_mdy(&[v[0], v[1], v[2]], c)
    }),
    rr("date_mdy_dash", DATE, &[node(DIG), glued_word("-"), glued(DIG), glued_word("-"), glued(DIG)], |v, c| {
        b_mdy(&[v[0], v[1], v[2]], c)
    }),
    rr("date_dmy_dot", DATE, &[node(DIG), glued_word("."), glued(DIG), glued_word("."), glued(DIG)], |v, c| {
        b_dmy(&[v[0], v[1], v[2]], c)
    }),
    rr("date_ymd", DATE, &[node(DIG), glued(SEP), glued(DIG), glued(SEP), glued(DIG)], |v, c| {
        b_ymd(&[v[0], v[2], v[4]], c)
    }),
    rr("date_ymd_spaced", DATE, &[node(DIG), node(DIG), node(DIG)], b_ymd_spaced),
    rr("date_yyyymmdd", DATE, &[node(DIG)], b_yyyymmdd),
    rr("date_y_mon_d", DATE, &[node(DIG), glued(SEP), glued(MONTH), glued(SEP), glued(DIG)], b_y_mon_d),
    rr("date_d_mon_y", DATE, &[node(DIG), glued(SEP), glued(MONTH), glued(SEP), glued(DIG)], b_d_mon_y),
    rr("date_mon_d_y", DATE, &[node(MONTH), glued(SEP), glued(DIG), glued(SEP), glued(DIG)], b_mon_d_y),
    rr("date_mon_y_d", DATE, &[node(MONTH), glued(SEP), glued(DIG), glued(SEP), glued(DIG)], b_mon_y_d),
    rr("date_y_d_mon", DATE, &[node(DIG), glued(SEP), glued(DIG), glued(SEP), glued(MONTH)], b_y_d_mon),
    rr("date_d_y_mon", DATE, &[node(DIG), glued(SEP), glued(DIG), glued(SEP), glued(MONTH)], b_d_y_mon),
    rr("date_weekday_md", DATE, &[node(WEEKDAY), opt_word(","), opt_word("-"), node(MD)], b_weekday_date),
    rr("date_weekday_date", DATE, &[node(WEEKDAY), opt_word(","), node(DATE)], b_weekday_date),
    rr("date_weekday_paren", DATE, &[node(WEEKDAY), word("("), node(MD), word(")")], |v, c| {
        b_weekday_date(&[v[0], v[1]], c)
    }),
    rr("date_weekday_dot", DATE, &[node(WEEKDAY), node(NUMERAL)], |v, c| {
        weekday(&v[0]).and_then(|_| b_weekday_dot(v, c))
    }),
    rr("date_dayword", DATE, &[node(DAYWORD)], b_dayword),
    rr("date_the_day", DATE, &[word("the"), word("day")], |_, c| day_rel(c, 0)),
    rr("date_the_day_before", DATE, &[word("the"), word("day"), word("before")], |_, c| day_rel(c, -1)),
    rr("date_the_day_after", DATE, &[word("the"), word("day"), word("after")], |_, c| day_rel(c, 1)),
    rr("date_the_next_day", DATE, &[word("the"), word("next"), word("day")], |_, c| day_rel(c, 1)),
    rr("date_day_after_tomorrow", DATE, &[word("the"), word("day"), word("after"), word("tomorrow")], |_, c| {
        day_rel(c, 2)
    }),
    rr("date_day_after_tomorrow_bare", DATE, &[word("day"), word("after"), word("tomorrow")], |_, c| day_rel(c, 2)),
    rr("date_day_before_yesterday", DATE, &[word("the"), word("day"), word("before"), word("yesterday")], |_, c| {
        day_rel(c, -2)
    }),
    rr("date_day_before_yesterday_bare", DATE, &[word("day"), word("before"), word("yesterday")], |_, c| {
        day_rel(c, -2)
    }),
    rr(
        "date_the_day_of_rel_month",
        DATE,
        &[word("the"), node(MDAY), word("day"), word("of"), node(RELW), word("month")],
        b_rel_month_day,
    ),
    rr(
        "date_day_of_rel_month",
        DATE,
        &[node(MDAY), word("day"), word("of"), node(RELW), word("month")],
        b_rel_month_day,
    ),
    rr(
        "date_the_day_of_the_month",
        DATE,
        &[word("the"), node(MDAY), word("day"), word("of"), word("the"), word("month")],
        |v, c| b_rel_month_day(&v[..1], c),
    ),
    rr("date_day_of_the_month", DATE, &[node(MDAY), word("day"), word("of"), word("the"), word("month")], |v, c| {
        b_rel_month_day(&v[..1], c)
    }),
    rr("date_the_day_this_month", DATE, &[word("the"), node(MDAY), node(RELW), word("month")], b_rel_month_day),
    rr("date_nth_weekday", DATE, &[node(ORD), node(WEEKDAY), word("of"), node(MONTHD)], b_nth_weekday),
    rr("date_nth_weekday_rel", DATE, &[node(ORD), node(WEEKDAY), word("of"), node(PERIOD)], b_nth_weekday),
    rr("date_the_nth_weekday", DATE, &[word("the"), node(ORD), node(WEEKDAY), word("of"), node(MONTHD)], b_nth_weekday),
    rr(
        "date_the_nth_weekday_rel",
        DATE,
        &[word("the"), node(ORD), node(WEEKDAY), word("of"), node(PERIOD)],
        b_nth_weekday,
    ),
    rr("date_last_weekday", DATE, &[word("last"), node(WEEKDAY), word("of"), node(MONTHD)], b_last_weekday),
    rr("date_last_weekday_rel", DATE, &[word("last"), node(WEEKDAY), word("of"), node(PERIOD)], b_last_weekday),
    rr(
        "date_nth_week_weekday",
        DATE,
        &[node(ORD), word("week"), word("of"), node(MONTH), opt_word("."), word("on"), node(WEEKDAY)],
        b_nth_week_weekday,
    ),
    // Weekdays.
    rr("wd", WD, &[node(WEEKDAY)], b_wd),
    rr("wd_rel", WD, &[node(RELW), node(WEEKDAY)], b_wd_rel),
    rr("wd_this_rel", WD, &[word("this"), node(RELW), node(WEEKDAY)], b_wd_this_rel),
    rr("wd_day", WD, &[node(WEEKDAY), node(MDAY)], b_wd_day),
    rr("wd_the_day", WD, &[node(WEEKDAY), word("the"), node(MDAY)], b_wd_the_day),
    rr("wd_week", WD, &[node(RELW), word("week"), opt_word("on"), opt_word("-"), node(WEEKDAY)], b_week_wd),
    rr(
        "wd_on_of_week",
        WD,
        &[word("on"), node(WEEKDAY), word("of"), opt_word("the"), node(RELW), word("week")],
        b_wd_of_week,
    ),
    rr("wd_of_week", WD, &[node(WEEKDAY), word("of"), opt_word("the"), node(RELW), word("week")], b_wd_of_week),
    // Days of an open month.
    rr("dom_the", DOM, &[word("the"), node(MDAY)], b_dom),
    rr("dom_ordinal", DOM, &[node(MDAY)], b_dom_ordinal),
    // Months.
    rr("monthd_month", MONTHD, &[node(MONTH)], b_month),
    rr("monthd_rel", MONTHD, &[node(RELW), node(MONTH)], b_rel_month),
    rr("monthd_year", MONTHD, &[node(MONTH), opt_word(","), opt_word("of"), opt_word("in"), node(YEAR)], b_month_year),
    rr("monthd_sep_year", MONTHD, &[node(MONTH), glued(SEP), glued(YEAR)], b_month_sep_year),
    rr("monthd_year_month", MONTHD, &[node(YEAR), node(MONTH)], b_year_month),
    rr("monthd_ym", MONTHD, &[node(DIG), glued(SEP), glued(DIG)], b_ym_digits),
    rr("monthd_my", MONTHD, &[node(DIG), glued_word("-"), glued(DIG)], b_my_digits),
    rr("monthd_nth", MONTHD, &[node(ORD), word("month"), word("of"), node(PERIOD)], b_nth_month),
    rr("monthd_the_nth", MONTHD, &[word("the"), node(ORD), word("month"), word("of"), node(PERIOD)], b_nth_month),
    // Periods.
    rr("period_year", PERIOD, &[node(YEAR)], b_period_year),
    rr("period_year_spelled", PERIOD, &[node(YEAR)], b_period_year_spelled),
    rr("period_year_word", PERIOD, &[word("year"), node(YEAR)], b_year_word),
    rr("period_eq_year", PERIOD, &[word("="), node(YEAR)], b_year_word),
    rr("period_rel", PERIOD, &[node(RELW), node(UNIT)], b_period_rel),
    rr("period_the_rel", PERIOD, &[word("the"), node(RELW), word("decade")], |v, c| {
        emit_date(c, DateSpec::Rel { grain: TimeGrain::Decade, n: term(&v[0]).0 as i16 })
    }),
    rr("period_the_unit", PERIOD, &[word("the"), node(UNIT)], b_period_the),
    rr("period_weekend", PERIOD, &[node(UNIT)], b_weekend),
    rr("period_week_after_next", PERIOD, &[word("the"), word("week"), word("after"), word("next")], b_week_after_next),
    rr("period_same", PERIOD, &[word("same"), node(UNIT)], b_undefined),
    rr("period_that", PERIOD, &[word("that"), node(UNIT)], b_undefined),
    rr("period_week_n", PERIOD, &[word("week"), node(DIG)], b_week_n),
    rr("period_the_week_n", PERIOD, &[word("the"), word("week"), node(DIG)], b_week_n),
    rr("period_the_year_word", PERIOD, &[word("the"), word("year"), node(YEAR)], b_year_word),
    rr("period_week_n_of", PERIOD, &[word("week"), node(DIG), word("of"), node(PERIOD)], b_week_n),
    rr("period_nth_week", PERIOD, &[node(ORD), word("week"), word("of"), node(PERIOD)], b_nth_week),
    rr("period_nth_week_month", PERIOD, &[node(ORD), word("week"), word("of"), node(MONTHD)], b_nth_week),
    rr("period_the_nth_week", PERIOD, &[word("the"), node(ORD), word("week"), word("of"), node(PERIOD)], b_nth_week),
    rr(
        "period_the_nth_week_month",
        PERIOD,
        &[word("the"), node(ORD), word("week"), word("of"), node(MONTHD)],
        b_nth_week,
    ),
    rr("period_last_week", PERIOD, &[word("last"), word("week"), word("of"), node(PERIOD)], b_last_week),
    rr("period_last_week_month", PERIOD, &[word("last"), word("week"), word("of"), node(MONTHD)], b_last_week),
    rr(
        "period_the_last_week",
        PERIOD,
        &[word("the"), word("last"), word("week"), word("of"), node(PERIOD)],
        b_last_week,
    ),
    rr(
        "period_the_last_week_month",
        PERIOD,
        &[word("the"), word("last"), word("week"), word("of"), node(MONTHD)],
        b_last_week,
    ),
    rr("period_week_of", PERIOD, &[word("the"), word("week"), word("of"), node(DATE)], b_week_of),
    rr("period_week_beginning", PERIOD, &[word("the"), word("week"), word("beginning"), node(DATE)], b_week_of),
    rr(
        "period_week_starting",
        PERIOD,
        &[word("the"), word("week"), word("starting"), opt_word("on"), node(DATE)],
        b_week_of,
    ),
    rr("period_week_commencing", PERIOD, &[word("the"), word("week"), word("commencing"), node(DATE)], b_week_of),
    rr("period_w_c", PERIOD, &[word("w"), glued_word("/"), glued_word("c"), node(DATE)], b_week_of),
    rr("qtr", QTR, &[word("q"), glued(DIG)], b_qtr),
    rr("period_quarter", PERIOD, &[node(QTR)], b_quarter),
    rr("period_year_quarter", PERIOD, &[node(YEAR), node(QTR)], b_year_quarter),
    rr("period_year_sep_quarter", PERIOD, &[node(YEAR), glued_word("-"), glued(QTR)], b_year_quarter),
    rr("period_quarter_year", PERIOD, &[node(QTR), node(YEAR)], b_quarter_year),
    rr("period_quarter_sep_year", PERIOD, &[node(QTR), glued_word("-"), glued(YEAR)], b_quarter_year),
    rr("period_quarter_of_year", PERIOD, &[node(QTR), word("of"), node(YEAR)], b_quarter_year),
    rr("period_nth_quarter", PERIOD, &[node(ORD), node(UNIT), word("of"), node(PERIOD)], b_nth_quarter_of),
    rr(
        "period_the_nth_quarter",
        PERIOD,
        &[word("the"), node(ORD), node(UNIT), word("of"), node(PERIOD)],
        b_nth_quarter_of,
    ),
    rr(
        "period_the_nth_quarter_the_year",
        PERIOD,
        &[word("the"), node(ORD), node(UNIT), word("of"), word("the"), word("year")],
        b_nth_quarter_the_year,
    ),
    rr("period_years_quarter", PERIOD, &[node(PERIOD), word("'"), word("s"), node(ORD), node(UNIT)], b_years_quarter),
    rr(
        "period_years_quarter_curly",
        PERIOD,
        &[node(PERIOD), word("\u{2019}"), word("s"), node(ORD), node(UNIT)],
        b_years_quarter,
    ),
    rr("period_year_half", PERIOD, &[node(YEAR), word("h"), glued(DIG)], b_year_half),
    rr("period_year_sep_half", PERIOD, &[node(YEAR), glued_word("-"), glued_word("h"), glued(DIG)], b_year_half),
    rr("period_decade", PERIOD, &[node(DIG), word("s")], b_decade),
    rr("period_decade_word", PERIOD, &[node(DECADEW)], b_decade_word),
    rr("period_century_decade", PERIOD, &[node(INT), node(DECADEW)], b_century_decade),
    rr("period_century", PERIOD, &[node(ORD), opt_word("-"), word("century")], b_century),
    rr("period_season", PERIOD, &[node(SEASON)], b_season),
    rr("period_rel_season", PERIOD, &[node(RELW), node(SEASON)], b_rel_season),
    rr("period_season_year", PERIOD, &[node(SEASON), opt_word("of"), node(YEAR)], b_season_year),
    rr("period_fiscal_year", PERIOD, &[word("fiscal"), word("year")], b_other_year),
    rr("period_school_year", PERIOD, &[word("school"), word("year")], b_other_year),
    rr("period_fiscal_year_n", PERIOD, &[word("fiscal"), word("year"), node(YEAR)], b_other_year_n),
    rr("period_school_year_n", PERIOD, &[word("school"), word("year"), node(YEAR)], b_other_year_n),
    rr("period_fy", PERIOD, &[word("fy"), node(YEAR)], b_other_year_n),
    rr("period_sy", PERIOD, &[word("sy"), node(YEAR)], b_other_year_n),
    rr("period_fy_glued", PERIOD, &[word("fy"), glued(DIG)], b_other_year_glued),
    rr("period_sy_glued", PERIOD, &[word("sy"), glued(DIG)], b_other_year_glued),
    rr("period_rel_fiscal_year", PERIOD, &[node(RELW), word("fiscal"), word("year")], b_rel_other_year),
    rr("period_rel_school_year", PERIOD, &[node(RELW), word("school"), word("year")], b_rel_other_year),
    rr("period_calendar_year", PERIOD, &[word("calendar"), word("year"), node(YEAR)], b_year_word),
    rr("period_cy", PERIOD, &[word("cy"), node(YEAR)], b_year_word),
    rr("period_cy_glued", PERIOD, &[word("cy"), glued(DIG)], b_calendar_year_glued),
    rr("period_rel_calendar_year", PERIOD, &[node(RELW), word("calendar"), word("year")], b_rel_calendar_year),
    // Holidays.
    rr("hol", HOL, &[node(HOLIDAY)], b_holiday),
    rr("hol_christmas_eve", HOL, &[word("christmas"), word("eve")], |_, c| hol(c, lexicon::CHRISTMAS_EVE)),
    rr("hol_christmas_day", HOL, &[word("christmas"), word("day")], |_, c| hol(c, lexicon::CHRISTMAS)),
    rr("hol_new_year", HOL, &[word("new"), word("year")], |_, c| hol(c, lexicon::NEW_YEAR)),
    rr("hol_new_year_day", HOL, &[word("new"), word("year"), word("day")], |_, c| hol(c, lexicon::NEW_YEAR)),
    rr("hol_new_years", HOL, &[word("new"), word("years"), opt_word("day")], |_, c| hol(c, lexicon::NEW_YEAR)),
    rr("hol_new_year_s", HOL, &[word("new"), word("year"), word("'"), word("s"), opt_word("day")], |_, c| {
        hol(c, lexicon::NEW_YEAR)
    }),
    rr("hol_new_year_s_eve", HOL, &[word("new"), word("year"), word("'"), word("s"), word("eve")], |_, c| {
        hol(c, lexicon::NEW_YEARS_EVE)
    }),
    rr("hol_new_years_eve", HOL, &[word("new"), word("years"), word("eve")], |_, c| hol(c, lexicon::NEW_YEARS_EVE)),
    rr("hol_independence", HOL, &[word("independence"), word("day")], |_, c| hol(c, lexicon::INDEPENDENCE)),
    rr("hol_labor", HOL, &[word("labor"), word("day")], |_, c| hol(c, lexicon::LABOR)),
    rr("hol_labour", HOL, &[word("labour"), word("day")], |_, c| hol(c, lexicon::LABOR)),
    rr("hol_black_friday", HOL, &[word("black"), word("friday")], |_, c| hol(c, lexicon::BLACK_FRIDAY)),
    rr("hol_earth", HOL, &[word("earth"), word("day")], |_, c| hol(c, lexicon::EARTH)),
    rr("hol_easter_monday", HOL, &[word("easter"), word("monday")], |_, c| hol(c, lexicon::EASTER_MONDAY)),
    rr("hol_easter_sunday", HOL, &[word("easter"), word("sunday")], |_, c| hol(c, lexicon::EASTER)),
    rr("hol_good_friday", HOL, &[word("good"), word("friday")], |_, c| hol(c, lexicon::GOOD_FRIDAY)),
    rr("hol_saint_patrick", HOL, &[word("saint"), word("patrick")], |_, c| hol(c, lexicon::PATRICK)),
    rr("hol_st_patrick", HOL, &[word("st"), opt_word("."), word("patrick")], |_, c| hol(c, lexicon::PATRICK)),
    rr("hol_patricks_day", HOL, &[word("patrick"), word("'"), word("s"), word("day")], |_, c| hol(c, lexicon::PATRICK)),
    rr("hol_workers", HOL, &[word("international"), word("workers"), opt_word("'"), word("day")], |_, c| {
        hol(c, lexicon::WORKERS)
    }),
    rr("hol_workers_curly", HOL, &[word("international"), word("workers"), word("\u{2019}"), word("day")], |_, c| {
        hol(c, lexicon::WORKERS)
    }),
    rr("hol_freedom", HOL, &[word("freedom"), word("day")], |_, c| hol(c, lexicon::JUNETEENTH)),
    rr("hol_jubilee", HOL, &[word("jubilee"), word("day")], |_, c| hol(c, lexicon::JUNETEENTH)),
    rr("hol_memorial", HOL, &[word("memorial"), word("day")], |_, c| hol(c, lexicon::MEMORIAL)),
    rr("hol_veterans", HOL, &[word("veterans"), word("day")], |_, c| hol(c, lexicon::VETERANS)),
    rr("hol_valentine", HOL, &[word("valentine"), word("'"), word("s"), word("day")], |_, c| {
        hol(c, lexicon::VALENTINE)
    }),
    rr("hol_mothers", HOL, &[word("mother"), word("'"), word("s"), word("day")], |_, c| hol(c, lexicon::MOTHERS)),
    rr("hol_fathers", HOL, &[word("father"), word("'"), word("s"), word("day")], |_, c| hol(c, lexicon::FATHERS)),
    rr("hol_year", HOL, &[node(HOL), opt_word(","), node(YEAR)], b_holiday_year),
    rr("hol_of_year", HOL, &[node(HOL), word("of"), node(PERIOD)], b_holiday_of_year),
    rr("hol_near", HOL, &[node(RELW), node(HOL)], b_holiday_near),
];
