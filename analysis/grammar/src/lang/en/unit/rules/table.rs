use super::*;

pub(crate) const fn r(name: &'static str, dim: Dim, weight: i16, items: &'static [Item], build: Build) -> Rule {
    Rule { name, dim, items, weight, build }
}

/// An optional word glued to what comes before.
pub(crate) const fn opt_glued_word(text: &'static str) -> Item {
    Item::Word { text, glued: true, optional: true, exact_case: false }
}

pub(crate) const OPT_DOT: Item = opt_glued_word(".");

/// Every unit rule, after the number grammar's.
pub static RULES: &[Rule] = &[
    // Units of measure.
    r("unit_word", UNIT, 0, &[node(UWORD)], b_unit_word),
    r("unit_word_dot", UNIT, 0, &[node_if(UWORD, dotted), glued_word(".")], b_unit_word),
    r("unit_prefixed", UNIT, 0, &[node(SIPRE), node_if(UWORD, spelled_out)], b_prefixed),
    r("unit_square", UNIT, 0, &[word("square"), node_if(UNIT, length)], b_square),
    r("unit_sq", UNIT, 0, &[word("sq"), OPT_DOT, node_if(UNIT, length)], b_square),
    r("unit_squared", UNIT, 0, &[node_if(UNIT, length), glued_if(NUMERAL, two)], b_square),
    r("unit_squared_sign", UNIT, 0, &[node_if(UNIT, length), glued_word("²")], b_square),
    r("unit_squared_caret", UNIT, 0, &[node_if(UNIT, length), glued_word("^"), glued_if(NUMERAL, two)], b_square),
    r("unit_cubic", UNIT, 0, &[word("cubic"), node_if(UNIT, length)], b_cubic),
    r("unit_cubed", UNIT, 0, &[node_if(UNIT, length), glued_if(NUMERAL, three)], b_cubic),
    r("unit_cubed_sign", UNIT, 0, &[node_if(UNIT, length), glued_word("³")], b_cubic),
    r("unit_per", UNIT, 0, &[node_if(UNIT, length), word("per"), node_if(TIMEW, per_time)], b_speed),
    r("unit_slash", UNIT, 0, &[node_if(UNIT, length), word("/"), node_if(TIMEW, per_time)], b_speed),
    r("unit_an_hour", UNIT, 0, &[node_if(UNIT, length), word("an"), word("hour")], b_per_hour),
    r("unit_a_hour", UNIT, 0, &[node_if(UNIT, length), word("a"), word("hour")], b_per_hour),
    r("unit_metric_ton", UNIT, 0, &[word("metric"), node_if(UNIT, ton)], b_metric),
    r("unit_long_ton", UNIT, 0, &[word("long"), node_if(UNIT, ton)], b_long_ton),
    r("unit_short_ton", UNIT, 0, &[word("short"), node_if(UNIT, ton)], b_short_ton),
    r("unit_nautical_mile", UNIT, 0, &[word("nautical"), node_if(UNIT, mile)], b_nautical),
    r("unit_imperial", UNIT, 0, &[word("imperial"), node_if(UNIT, customary)], b_imperial),
    r("unit_uk", UNIT, 0, &[word("uk"), node_if(UNIT, customary)], b_imperial),
    r("unit_british", UNIT, 0, &[word("british"), node_if(UNIT, customary)], b_imperial),
    r("unit_us", UNIT, 0, &[word("us"), node_if(UNIT, customary)], b_us),
    r("unit_fluid", UNIT, 0, &[word("fluid"), node_if(UNIT, fluid)], b_fluid),
    r("unit_fl", UNIT, 0, &[word("fl"), OPT_DOT, node_if(UNIT, fluid)], b_fluid),
    r("unit_light_year", UNIT, 0, &[word("light"), node_if(TIMEW, is_year)], b_light_year),
    r("unit_light_year_hyphen", UNIT, 0, &[word("light"), glued_word("-"), glued_if(TIMEW, is_year)], b_light_year),
    r("unit_degree_temperature", UNIT, 0, &[node(DEG)], b_degree_temperature),
    r("unit_degree_angle", UNIT, 0, &[node(DEG)], b_degree_angle),
    r("unit_degree_dot_temperature", UNIT, 0, &[node_if(DEG, dotted), glued_word(".")], b_degree_temperature),
    r("unit_degree_dot_angle", UNIT, 0, &[node_if(DEG, dotted), glued_word(".")], b_degree_angle),
    r("unit_degree_scale", UNIT, 0, &[node(DEG), node(TSCALE)], b_degree_scale),
    r("unit_scale", UNIT, 0, &[node_if(TSCALE, spelled_out)], b_scale),
    r("unit_inch_mark", UNIT, 0, &[node_if(INCHW, quote_mark)], b_inch_mark),
    r("unit_double_prime", UNIT, 0, &[node_if(INCHW, double_prime)], |_, _| {
        Some(Value::Unit(Unit::Length(Length::Inch)))
    }),
    // Currencies.
    r("cur_word", CUR, 0, &[node(CURW)], b_money_word),
    r("cur_sign", CURP, 0, &[node_if(CURW, prefix_money)], b_money_word),
    r("cur_code", CUR, 0, &[node(CODE)], b_code),
    r("cur_code_before", CURP, 0, &[node_if(CODE, prefix_code)], b_code),
    r("cur_code_dollar", CURP, 0, &[node(CODE), glued_word("$")], b_code_dollar),
    r("cur_nation", CUR, 0, &[node(NATION), node(CURW)], b_nation_money),
    r("cur_nation_dollar", CURP, 0, &[node(NATION), word("$")], b_nation_dollar),
    r("cur_before", CUR, 0, &[node(CURP)], |v, _| Some(v[0])),
    r("nation_hong_kong", NATION, 0, &[word("hong"), word("kong")], |_, _| nation_term("hong kong")),
    r("nation_new_zealand", NATION, 0, &[word("new"), word("zealand")], |_, _| nation_term("new zealand")),
    r("nation_new_taiwan", NATION, 0, &[word("new"), word("taiwan")], |_, _| nation_term("new taiwan")),
    r("nation_united_states", NATION, 0, &[word("united"), word("states")], |_, _| nation_term("united states")),
    r("nation_us_dots", NATION, 0, &[word("u"), glued_word("."), glued_word("s"), OPT_DOT], |_, _| nation_term("u.s.")),
    r("nation_south_african", NATION, 0, &[word("south"), word("african")], |_, _| nation_term("south african")),
    r("nation_south_korean", NATION, 0, &[word("south"), word("korean")], |_, _| nation_term("south korean")),
    r("nation_north_korean", NATION, 0, &[word("north"), word("korean")], |_, _| nation_term("north korean")),
    r("nation_sri_lankan", NATION, 0, &[word("sri"), word("lankan")], |_, _| nation_term("sri lankan")),
    r("nation_trinidad", NATION, 0, &[word("trinidad"), word("and"), word("tobago")], |_, _| {
        nation_term("trinidad and tobago")
    }),
    r("nation_cayman", NATION, 0, &[word("cayman"), word("islands")], |_, _| nation_term("cayman islands")),
    r("nation_solomon", NATION, 0, &[word("solomon"), word("islands")], |_, _| nation_term("solomon islands")),
    // Measured quantities.
    r("qty", QTY, 0, &[node(NUM), node(UNIT)], b_amount_unit),
    r("qty_hyphen", QTY, 0, &[node(NUM), glued_word("-"), glued(UNIT)], b_amount_unit),
    r("qty_parens", QTY, 0, &[node(NUM), word("("), glued(UNIT), glued_word(")")], b_amount_unit),
    r("qty_brackets", QTY, 0, &[node(NUM), word("["), glued(UNIT), glued_word("]")], b_amount_unit),
    r("qty_braces", QTY, 0, &[node(NUM), word("{"), glued(UNIT), glued_word("}")], b_amount_unit),
    r("qty_angles", QTY, 0, &[node(NUM), word("<"), glued(UNIT), glued_word(">")], b_amount_unit),
    r(
        "qty_feet_inches",
        QTY,
        1,
        &[node_if(QTY, feet), opt_word("and"), node_if(NUM, non_negative), node(INCHW)],
        b_feet_inches,
    ),
    r(
        "qty_feet_inch_unit",
        QTY,
        1,
        &[node_if(QTY, feet), opt_word("and"), node_if(NUM, non_negative), node_if(UNIT, inch)],
        b_feet_inches,
    ),
    r(
        "qty_feet_inch_marks",
        QTY,
        1,
        &[node_if(NUMERAL, feet_count), glued_word("'"), glued_if(NUMERAL, inch_count), glued_if(INCHW, inch_mark)],
        b_feet_inch_marks,
    ),
    r(
        "qty_feet_inch_primes",
        QTY,
        1,
        &[node_if(NUMERAL, feet_count), glued_word("′"), glued_if(NUMERAL, inch_count), opt_glued_word("″")],
        b_feet_inch_marks,
    ),
    r(
        "qty_feet_apostrophe_inches",
        QTY,
        0,
        &[node_if(NUMERAL, feet_count), glued_word("'"), glued_if(NUMERAL, inch_count)],
        b_feet_inch_marks,
    ),
    r("qty_feet_prime", QTY, 0, &[node(NUM), glued_word("′")], b_feet_mark),
    // Money.
    r("money_after", MONEY, 0, &[node(NUM), node(CUR)], b_amount_unit),
    r("money_hyphen", MONEY, 0, &[node(NUM), glued_word("-"), glued(CUR)], b_amount_unit),
    r("money_before", MONEY, 0, &[node(CURP), node(NUM)], b_unit_amount),
    r("money_multiplier", MONEY, 0, &[node(CURP), node(NUMERAL), glued(MULT)], b_unit_amount_multiplier),
    r("money_multiplier_after", MONEY, 0, &[node(NUMERAL), node(MULT), node(CUR)], b_amount_multiplier_unit),
    r("money_scale_after", MONEY, 0, &[node(NUM), node(CUR), node(SCALE)], b_amount_unit_scale),
    r("money_million_code", MONEY, 0, &[node(NUM), node(MCODE)], b_amount_million_code),
    r("money_million_code_before", MONEY, 0, &[node(MCODE), node(NUM)], b_million_code_amount),
    r(
        "money_major_minor",
        MONEY,
        1,
        &[node_if(MONEY, whole_major), opt_word("and"), node(NUM), node_if(CUR, minor_unit)],
        b_major_minor,
    ),
    r(
        "money_major_hundredths",
        MONEY,
        0,
        &[node_if(MONEY, whole_major), opt_word("and"), node_if(NUM, hundredths)],
        b_major_hundredths,
    ),
    // Ages.
    r("age_old", AGE, 0, &[node_if(NUM, positive), node_if(TIMEW, age_time), word("old")], b_age),
    r("age_of_age", AGE, 0, &[node_if(NUM, positive), node_if(TIMEW, age_time), word("of"), word("age")], b_age),
    r(
        "age_hyphens",
        AGE,
        0,
        &[node_if(NUM, positive), glued_word("-"), glued_if(TIMEW, age_time), glued_word("-"), glued_word("old")],
        b_age,
    ),
    r("age_yo", AGE, 0, &[node_if(NUM, positive), word("yo")], b_age_years),
    r("age_stated", AGE, 0, &[word("age"), node_if(NUM, positive)], b_age_stated),
    r("age_aged", AGE, 0, &[word("aged"), node_if(NUM, positive)], b_age_stated),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rule_names_are_unique() {
        let mut names: Vec<&str> = RULES.iter().map(|r| r.name).collect();
        names.sort();
        let n = names.len();
        names.dedup();
        assert_eq!(names.len(), n);
    }
}
