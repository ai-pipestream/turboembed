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
    rr("sep_slash", SEP, &[word("/")], en::b_sep),
    rr("sep_dash", SEP, &[word("-")], en::b_sep),
    rr("unit_fin_de_semana", UNIT, &[word("fin"), word("de"), word("semana")], |_, _| {
        term_value(words::unit(words::WEEKEND, 1), 0)
    }),
    rr("unit_fines_de_semana", UNIT, &[word("fines"), word("de"), word("semana")], |_, _| {
        term_value(words::unit(words::WEEKEND, 1), PLURAL)
    }),
    // Years and days of the month.
    rr("year_digits", YEAR, &[node(DIG)], en::b_year_digits),
    rr("year_short", YEAR, &[node(DIG)], en::b_year_short),
    rr("year_spelled", YEAR, &[node(INT)], en::b_year_spelled),
    rr("mday_digits", MDAY, &[node(DIG)], en::b_mday_digits),
    rr("mday_ordinal", MDAY, &[node(ORD)], en::b_mday_ordinal),
    rr("mday_spelled", MDAY, &[node(INT)], en::b_mday_spelled),
    // A day and a month.
    rr("month_abbr_dot", MONTH, &[node(MONTH), glued_word(".")], b_abbr_dot),
    rr("weekday_abbr_dot", WEEKDAY, &[node(WEEKDAY), glued_word(".")], b_abbr_dot),
    rr("md_day_de_month", MD, &[node(MDAY), node(DE), node(MONTH)], |v, c| en::b_md_day_month(&[v[0], v[2]], c)),
    rr("md_day_month", MD, &[node(MDAY), node(MONTH)], en::b_md_day_month),
    rr("md_day_de_digits", MD, &[node(MDAY), node(DE), node(DIG)], b_day_of_month_digits),
    rr("md_month_day", MD, &[node(MONTH), node(MDAY)], en::b_md_month_day),
    rr("md_slash", MD, &[node(DIG), glued_word("/"), glued(DIG)], b_dm),
    rr("md_dash", MD, &[node(DIG), glued_word("-"), glued(DIG)], b_dm),
    rr("md_d_mon", MD, &[node(DIG), glued(SEP), glued(MONTH)], b_d_mon),
    rr("md_el_dia", MD, &[word("el"), word("dia"), node(MD)], en::b_same),
    rr("md_primer_dia", MD, &[node(ORD), word("dia"), node(DE), node(MONTH)], |v, c| {
        en::b_md_day_month(&[en::b_mday_ordinal(&v[..1], c)?, v[2]], c)
    }),
    // Dates.
    rr("date_md", DATE, &[node(MD)], en::b_same),
    rr("date_wd", DATE, &[node(WD)], en::b_same),
    rr("date_dom", DATE, &[node(DOM)], en::b_same),
    rr("date_md_year", DATE, &[node(MD), opt_word(","), node(YEAR)], en::b_md_year),
    rr("date_md_de_year", DATE, &[node(MD), opt_word(","), node(DE), node(YEAR)], en::b_md_year),
    rr("date_dmy_slash", DATE, &[node(DIG), glued_word("/"), glued(DIG), glued_word("/"), glued(DIG)], |v, c| {
        en::b_dmy(&[v[0], v[1], v[2]], c)
    }),
    rr("date_dmy_dash", DATE, &[node(DIG), glued_word("-"), glued(DIG), glued_word("-"), glued(DIG)], |v, c| {
        en::b_dmy(&[v[0], v[1], v[2]], c)
    }),
    rr("date_dmy_dot", DATE, &[node(DIG), glued_word("."), glued(DIG), glued_word("."), glued(DIG)], |v, c| {
        en::b_dmy(&[v[0], v[1], v[2]], c)
    }),
    rr("date_ymd", DATE, &[node(DIG), glued(SEP), glued(DIG), glued(SEP), glued(DIG)], |v, c| {
        en::b_ymd(&[v[0], v[2], v[4]], c)
    }),
    rr("date_y_mon_d", DATE, &[node(DIG), glued(SEP), glued(MONTH), glued(SEP), glued(DIG)], en::b_y_mon_d),
    rr("date_d_mon_y", DATE, &[node(DIG), glued(SEP), glued(MONTH), glued(SEP), glued(DIG)], b_d_mon_y),
    rr("date_mon_d_y", DATE, &[node(MONTH), glued(SEP), glued(DIG), glued(SEP), glued(DIG)], en::b_mon_d_y),
    rr("date_mon_y_d", DATE, &[node(MONTH), glued(SEP), glued(DIG), glued(SEP), glued(DIG)], en::b_mon_y_d),
    rr("date_y_d_mon", DATE, &[node(DIG), glued(SEP), glued(DIG), glued(SEP), glued(MONTH)], en::b_y_d_mon),
    rr("date_d_y_mon", DATE, &[node(DIG), glued(SEP), glued(DIG), glued(SEP), glued(MONTH)], en::b_d_y_mon),
    rr("date_weekday_md", DATE, &[node(WEEKDAY), opt_word(","), node(MD)], en::b_weekday_date),
    rr("date_weekday_date", DATE, &[node(WEEKDAY), opt_word(","), node(DATE)], en::b_weekday_date),
    rr("date_dayword", DATE, &[node(DAYWORD)], en::b_dayword),
    rr("date_pasado_manana", DATE, &[word("pasado"), word("mañana")], |_, c| en::day_rel(c, 2)),
    rr("date_el_dia", DATE, &[word("el"), node(UNIT)], |v, c| {
        (term(&v[0]) == (words::unit(words::DAY, 1), 0)).then(|| en::day_rel(c, 0))?
    }),
    rr("date_el_dia_rel", DATE, &[word("el"), node(UNIT), node(RELW)], b_day_rel),
    rr(
        "date_day_this_month",
        DATE,
        &[word("el"), word("dia"), node(MDAY), node(DE), node(RELW), word("mes")],
        |v, c| b_day_this_month(&[v[0], v[2]], c),
    ),
    // Weekdays.
    rr("wd", WD, &[node(WEEKDAY)], en::b_wd),
    rr("wd_rel", WD, &[node(RELW), node(WEEKDAY)], en::b_wd_rel),
    rr("wd_rel_after", WD, &[node(WEEKDAY), node(RELW)], b_wd_after),
    rr("wd_que_viene", WD, &[node(WEEKDAY), word("que"), word("viene")], b_wd_next),
    rr("wd_este_rel", WD, &[word("este"), node(RELW), node(WEEKDAY)], b_wd_this_rel),
    rr("wd_este_pasado", WD, &[word("este"), node(WEEKDAY), word("pasado")], b_wd_past),
    rr("wd_day", WD, &[node(WEEKDAY), node(MDAY)], en::b_wd_day),
    rr("wd_el_day", WD, &[node(WEEKDAY), word("el"), opt_word("dia"), node(MDAY)], en::b_wd_the_day),
    rr("wd_de_la_rel_semana", WD, &[node(WEEKDAY), node(DE), node(ART), node(RELW), word("semana")], b_wd_of_week),
    rr("wd_de_la_semana_rel", WD, &[node(WEEKDAY), node(DE), node(ART), word("semana"), node(RELW)], b_wd_of_week),
    rr("wd_de_rel_semana", WD, &[node(WEEKDAY), node(DE), node(RELW), word("semana")], b_wd_of_week),
    rr("wd_de_semana_rel", WD, &[node(WEEKDAY), node(DE), word("semana"), node(RELW)], b_wd_of_week),
    rr("wd_la_rel_semana", WD, &[node(WEEKDAY), node(ART), node(RELW), word("semana")], b_wd_of_week),
    rr("wd_rel_semana", WD, &[node(WEEKDAY), node(RELW), word("semana")], b_wd_of_week),
    rr(
        "wd_de_la_semana_que_viene",
        WD,
        &[node(WEEKDAY), node(DE), node(ART), word("semana"), word("que"), word("viene")],
        b_wd_of_week_next,
    ),
    rr("wd_semana_rel_dash", WD, &[node(ART), word("semana"), node(RELW), word("-"), node(WEEKDAY)], |v, c| {
        b_week_wd(&v[1..], c)
    }),
    rr("dom_el_dia", DOM, &[word("el"), word("dia"), node(MDAY)], en::b_dom),
    rr("dom_ordinal", DOM, &[node(MDAY)], b_dom_ordinal),
    // Months.
    rr("monthd_month", MONTHD, &[node(MONTH)], en::b_month),
    rr("monthd_rel", MONTHD, &[node(RELW), node(MONTH)], en::b_rel_month),
    rr("monthd_year", MONTHD, &[node(MONTH), opt_word(","), node(YEAR)], |v, c| month_year(c, &v[0], &v[1], false)),
    rr("monthd_de_year", MONTHD, &[node(MONTH), node(DE), node(YEAR)], |v, c| month_year(c, &v[0], &v[2], true)),
    rr("monthd_en_year", MONTHD, &[node(MONTH), word("en"), node(YEAR)], |v, c| month_year(c, &v[0], &v[1], true)),
    rr("monthd_sep_year", MONTHD, &[node(MONTH), glued(SEP), glued(YEAR)], en::b_month_sep_year),
    rr("monthd_my", MONTHD, &[node(DIG), glued_word("-"), glued(DIG)], en::b_my_digits),
    rr("monthd_nth", MONTHD, &[node(ORD), word("mes"), node(DE), node(PERIOD)], |v, c| {
        en::b_nth_month(&[v[0], v[2]], c)
    }),
    rr("monthd_el_nth", MONTHD, &[word("el"), node(ORD), word("mes"), node(DE), node(PERIOD)], |v, c| {
        en::b_nth_month(&[v[0], v[2]], c)
    }),
    // Periods.
    rr("period_year", PERIOD, &[node(YEAR)], en::b_period_year),
    rr("period_year_word", PERIOD, &[word("año"), node(YEAR)], en::b_year_word),
    rr("period_del_year_word", PERIOD, &[word("del"), word("año"), node(YEAR)], en::b_year_word),
    rr("period_rel", PERIOD, &[node(RELW), node(UNIT)], en::b_period_rel),
    rr("period_art_rel", PERIOD, &[node(ART), node(RELW), node(UNIT)], |v, c| en::b_period_rel(&v[1..], c)),
    rr("period_rel_after", PERIOD, &[node(UNIT), node(RELW)], b_period_after),
    rr("period_art_rel_after", PERIOD, &[node(ART), node(UNIT), node(RELW)], |v, c| b_period_after(&v[1..], c)),
    rr("period_que_viene", PERIOD, &[node(UNIT), word("que"), word("viene")], b_period_next),
    rr("period_art_que_viene", PERIOD, &[node(ART), node(UNIT), word("que"), word("viene")], |v, c| {
        b_period_next(&v[1..], c)
    }),
    rr("period_the_unit", PERIOD, &[node(UNIT)], b_period_the),
    rr("period_same", PERIOD, &[word("mismo"), node(UNIT)], en::b_undefined),
    rr("period_same_f", PERIOD, &[word("misma"), node(UNIT)], en::b_undefined),
    rr("period_ese", PERIOD, &[word("ese"), node(UNIT)], en::b_undefined),
    rr("period_esa", PERIOD, &[word("esa"), node(UNIT)], en::b_undefined),
    rr(
        "period_week_after_next",
        PERIOD,
        &[word("la"), word("semana"), word("despues"), word("de"), word("la"), word("proxima")],
        en::b_week_after_next,
    ),
    rr("period_week_n", PERIOD, &[word("semana"), node(DIG)], en::b_week_n),
    rr("period_week_n_of", PERIOD, &[word("semana"), node(DIG), node(DE), node(PERIOD)], b_week_n_of),
    rr("period_week_n_of_year", PERIOD, &[word("semana"), node(DIG), node(DE), node(YEAR)], |v, c| {
        let w = en::small(&v[0])?;
        ((1..=53).contains(&w)).then(|| emit_date(c, DateSpec::Week { year: year_ref(v.last()?), week: w as u8 }))?
    }),
    rr("period_nth_week", PERIOD, &[node(ORD), word("semana"), node(DE), node(PERIOD)], b_nth_week),
    rr("period_nth_week_month", PERIOD, &[node(ORD), word("semana"), node(DE), node(MONTHD)], b_nth_week),
    rr("period_la_nth_week", PERIOD, &[word("la"), node(ORD), word("semana"), node(DE), node(PERIOD)], b_nth_week),
    rr(
        "period_la_nth_week_month",
        PERIOD,
        &[word("la"), node(ORD), word("semana"), node(DE), node(MONTHD)],
        b_nth_week,
    ),
    rr("period_last_week", PERIOD, &[word("la"), word("ultima"), word("semana"), node(DE), node(PERIOD)], |v, c| {
        en::b_last_week(&v[1..], c)
    }),
    rr(
        "period_last_week_month",
        PERIOD,
        &[word("la"), word("ultima"), word("semana"), node(DE), node(MONTHD)],
        |v, c| en::b_last_week(&v[1..], c),
    ),
    rr("period_week_of", PERIOD, &[word("la"), word("semana"), node(DE), node(DATE)], b_week_of),
    rr("period_week_of_dom", PERIOD, &[word("la"), word("semana"), node(DE), node(MDAY)], |v, c| {
        emit_date(c, DateSpec::WeekOf { month: 0, day: u8::try_from(term(v.last()?).0).ok()? })
    }),
    rr(
        "period_week_que_inicia",
        PERIOD,
        &[word("la"), word("semana"), word("que"), word("inicia"), word("el"), node(DATE)],
        b_week_of,
    ),
    rr(
        "period_week_que_comienza",
        PERIOD,
        &[word("la"), word("semana"), word("que"), word("comienza"), word("el"), node(DATE)],
        b_week_of,
    ),
    rr(
        "period_week_que_va",
        PERIOD,
        &[word("la"), word("semana"), word("que"), word("va"), node(DE), node(DATE)],
        b_week_of,
    ),
    rr(
        "period_week_a_partir",
        PERIOD,
        &[word("la"), word("semana"), word("a"), word("partir"), node(DE), node(DATE)],
        b_week_of,
    ),
    rr("qtr", QTR, &[word("t"), glued(DIG)], en::b_qtr),
    rr("period_quarter", PERIOD, &[node(QTR)], en::b_quarter),
    rr("period_year_quarter", PERIOD, &[node(YEAR), node(QTR)], en::b_year_quarter),
    rr("period_year_sep_quarter", PERIOD, &[node(YEAR), glued_word("-"), glued(QTR)], en::b_year_quarter),
    rr("period_nth_part", PERIOD, &[node(ORD), node(UNIT), node(DE), node(PERIOD)], b_nth_part_of),
    rr("period_el_nth_part", PERIOD, &[word("el"), node(ORD), node(UNIT), node(DE), node(PERIOD)], b_nth_part_of),
    rr("period_nth_part_year", PERIOD, &[node(ORD), node(UNIT), node(YEAR)], b_nth_part_year),
    rr("period_nth_part_sep_year", PERIOD, &[node(ORD), node(UNIT), glued_word("-"), glued(YEAR)], b_nth_part_year),
    rr(
        "period_nth_part_the_year",
        PERIOD,
        &[word("el"), node(ORD), node(UNIT), word("del"), word("año")],
        b_nth_part_the_year,
    ),
    rr("period_nth_part_alone", PERIOD, &[node(ORD), node(UNIT)], |v, c| {
        (term(&v[1]).0 == words::unit(words::MONTH_U, 6)).then(|| b_nth_part_the_year(v, c))?
    }),
    rr("period_decade_of", PERIOD, &[word("decada"), node(DE), node(DIG)], b_decade_of),
    rr("period_years_tens", PERIOD, &[word("los"), word("años"), node(TENS)], b_years_tens),
    rr("period_years_digits", PERIOD, &[word("los"), word("años"), node(DIG)], b_years_tens),
    rr("period_anos_tens", PERIOD, &[word("los"), word("anos"), node(TENS)], b_years_tens),
    rr("period_anos_digits", PERIOD, &[word("los"), word("anos"), node(DIG)], b_years_tens),
    rr("period_season", PERIOD, &[node(SEASON)], en::b_season),
    rr("period_the_season", PERIOD, &[node(ART), node(SEASON)], b_season_the),
    rr("period_rel_season", PERIOD, &[node(RELW), node(SEASON)], en::b_rel_season),
    rr("period_season_rel", PERIOD, &[node(SEASON), node(RELW)], b_season_after),
    rr("period_season_year", PERIOD, &[node(SEASON), opt_word("de"), node(YEAR)], en::b_season_year),
    rr("period_year_kind", PERIOD, &[word("año"), node(KIND)], b_other_year),
    rr("period_kind_year", PERIOD, &[node(KIND), word("año")], b_other_year),
    rr("period_year_kind_n", PERIOD, &[word("año"), node(KIND), opt_word("de"), node(YEAR)], b_other_year_n),
    rr("period_kind_year_n", PERIOD, &[node(KIND), word("año"), node(YEAR)], b_other_year_n),
    rr("period_rel_year_kind", PERIOD, &[node(RELW), word("año"), node(KIND)], b_rel_other_year),
    rr("period_rel_kind_year", PERIOD, &[node(RELW), node(KIND), word("año")], b_rel_other_year),
    rr("period_el_rel_year_kind", PERIOD, &[word("el"), node(RELW), word("año"), node(KIND)], b_rel_other_year),
    rr("period_el_rel_kind_year", PERIOD, &[word("el"), node(RELW), node(KIND), word("año")], b_rel_other_year),
    // Holidays.
    rr("hol", HOL, &[node(HOLIDAY)], en::b_holiday),
    rr("hol_trabajador", HOL, &[word("dia"), word("del"), word("trabajador")], |_, c| hol(c, WORKERS)),
    rr("hol_trabajo", HOL, &[word("dia"), word("del"), word("trabajo")], |_, c| hol(c, WORKERS)),
    rr(
        "hol_trabajadores",
        HOL,
        &[word("dia"), word("internacional"), word("de"), word("los"), word("trabajadores")],
        |_, c| hol(c, WORKERS),
    ),
    rr("hol_independencia", HOL, &[word("dia"), word("de"), opt_word("la"), word("independencia")], |_, c| {
        hol(c, INDEPENDENCE)
    }),
    rr("hol_viernes_negro", HOL, &[word("viernes"), word("negro")], |_, c| hol(c, BLACK_FRIDAY)),
    rr("hol_tierra", HOL, &[word("dia"), word("de"), word("la"), word("tierra")], |_, c| hol(c, EARTH)),
    rr("hol_san_patricio", HOL, &[word("dia"), word("de"), word("san"), word("patricio")], |_, c| hol(c, PATRICK)),
    rr("hol_san_valentin", HOL, &[word("dia"), word("de"), word("san"), word("valentin")], |_, c| hol(c, VALENTINE)),
    rr("hol_ano_nuevo", HOL, &[word("año"), word("nuevo")], |_, c| hol(c, words::NEW_YEAR)),
    rr("hol_dia_ano_nuevo", HOL, &[word("dia"), word("de"), word("año"), word("nuevo")], |_, c| {
        hol(c, words::NEW_YEAR)
    }),
    rr("hol_viernes_santo", HOL, &[word("viernes"), word("santo")], |_, c| hol(c, words::GOOD_FRIDAY)),
    rr("hol_year", HOL, &[node(HOL), opt_word(","), node(YEAR)], en::b_holiday_year),
    rr("hol_of_year", HOL, &[node(HOL), node(DE), node(PERIOD)], b_holiday_of),
];
