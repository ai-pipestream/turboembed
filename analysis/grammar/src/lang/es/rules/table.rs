use super::*;

pub(crate) const fn r(
    name: &'static str,
    dim: crate::engine::Dim,
    weight: i16,
    items: &'static [Item],
    build: crate::engine::Build,
) -> Rule {
    Rule { name, dim, items, weight, build }
}

/// Every rule. A weight above zero makes a reading win over others of the
/// same text: a fraction of a large scale over a mixed number, a mixed
/// number over a spoken decimal.
pub static RULES: &[Rule] = &[
    // Integers in words and digits.
    r("int_unit", INT, 0, &[node(UNIT)], b_word_int),
    r("int_teen", INT, 0, &[node(TEEN)], b_word_int),
    r("int_tens", INT, 0, &[node(TENS)], b_word_int),
    r("int_tens_unit", INT, 0, &[node(TENS), word("y"), node_if(UNIT, unit_1_to_9)], b_tens_unit),
    r("int_tens_i_unit", INT, 0, &[node(TENS), glued_word("i"), glued_if(UNIT, unit_1_to_9)], b_tens_unit),
    r("int_hundreds", INT, 0, &[node(HUNDREDS)], b_hundreds),
    r("int_numeral", INT, 0, &[node_if(NUMERAL, integer)], en::b_numeral_int),
    r("int_scale", INT, 0, &[node_if(INT, positive_integer), node(SCALE)], b_scale),
    r("int_bare_scale", INT, 0, &[node_if(SCALE, singular)], b_bare_scale),
    r("int_sum", INT, 0, &[node_if(INT, grain_from_2), node_if(INT, positive_integer)], b_sum),
    r("int_sum_y", INT, 0, &[node_if(INT, grain_from_3), word("y"), node_if(INT, positive_integer)], b_sum),
    r("int_dozen", INT, 0, &[node_if(INT, small_multiplicand), node(DOZEN)], b_dozen),
    // Numbers.
    r("num_int", NUM, 0, &[node(INT)], en::b_same),
    r("num_numeral", NUM, 0, &[node_if(NUMERAL, not_integer)], en::b_same),
    r("num_numeral_scale", NUM, 0, &[node_if(NUMERAL, not_integer), node(SCALE)], en::b_numeral_scale),
    r("num_suffix_k", NUM, 0, &[node(NUMERAL), word("k")], |v, _| b_suffix(v, 3)),
    r("num_suffix_m", NUM, 0, &[node(NUMERAL), crate::engine::exact("M")], |v, _| b_suffix(v, 6)),
    r("num_suffix_mm", NUM, 0, &[node(NUMERAL), crate::engine::exact("MM")], |v, _| b_suffix(v, 6)),
    r("num_suffix_g", NUM, 0, &[node(NUMERAL), crate::engine::exact("G")], |v, _| b_suffix(v, 9)),
    r("num_suffix_t", NUM, 0, &[node(NUMERAL), crate::engine::exact("T")], |v, _| b_suffix(v, 12)),
    r("num_neg_word", NUM, 0, &[node(NEG_WORD), node_if(NUM, non_negative)], b_neg_word),
    r("num_neg_sign", NUM, 0, &[word("-"), node_if(NUM, en_digits_non_negative)], en::b_neg_sign),
    r("num_neg_minus_sign", NUM, 0, &[word("\u{2212}"), node_if(NUM, en_digits_non_negative)], en::b_neg_sign),
    // Fractions.
    r("den_ordinal", DEN, 0, &[node_if(ORD, denominator_ordinal)], b_den_ordinal),
    r("den_word", DEN, 0, &[node_if(FRAC_WORD, denominator_word)], b_den_word),
    r("den_avo", DEN, 0, &[node_if(INT, spelled_positive_integer), glued(AVO)], b_den_avo),
    r("frac", NUM, 0, &[node_if(INT, spelled_positive_integer), node(DEN)], b_fraction),
    r("frac_half_noun", NUM, 0, &[node_if(FRAC_WORD, noun_half)], b_half_noun),
    r(
        "frac_mixed_con",
        NUM,
        1,
        &[node_if(NUM, non_negative_integer), word("con"), node_if(NUM, positive_fraction)],
        en::b_mixed,
    ),
    r(
        "frac_and_half",
        NUM,
        1,
        &[node_if(NUM, non_negative_integer), word("y"), node_if(FRAC_WORD, one_half)],
        b_and_half,
    ),
    r(
        "frac_and_one_part",
        NUM,
        1,
        &[node_if(NUM, non_negative_integer), word("y"), node_if(NUM, proper_fraction)],
        b_and_one_part,
    ),
    r(
        "frac_scale_and_half",
        NUM,
        2,
        &[node_if(INT, grain_from_6), word("y"), node_if(FRAC_WORD, one_half)],
        b_scale_and_half,
    ),
    r(
        "frac_scale_con_fraction",
        NUM,
        2,
        &[node_if(INT, grain_from_6), word("con"), node_if(NUM, proper_fraction)],
        en::b_scale_and_fraction,
    ),
    r(
        "frac_scale_plus_fraction",
        NUM,
        1,
        &[node_if(INT, grain_from_3), word("y"), node_if(NUM, positive_fraction)],
        en::b_scale_plus_fraction,
    ),
    r(
        "frac_of_scale",
        NUM,
        0,
        &[node_if(NUM, spelled_positive_fraction), opt_word("de"), node(SCALE)],
        en::b_fraction_of_scale,
    ),
    r("frac_word_of_scale", NUM, 0, &[node_if(FRAC_WORD, one_half), node_if(SCALE, singular)], b_word_of_scale),
    r("frac_word_of_dozen", NUM, 0, &[node_if(FRAC_WORD, one_half), node_if(DOZEN, singular)], b_word_of_dozen),
    r(
        "frac_digits",
        NUM,
        0,
        &[node_if(NUMERAL, non_negative_integer), glued_word("/"), glued_if(NUMERAL, positive_integer)],
        en::b_digit_fraction,
    ),
    r(
        "frac_digits_mixed",
        NUM,
        0,
        &[node_if(NUMERAL, non_negative_integer), node_if(NUM, digit_proper_fraction)],
        en::b_mixed,
    ),
    r("frac_vulgar", NUM, 0, &[node(VULGAR)], en::b_vulgar),
    r("frac_over", NUM, 0, &[node_if(INT, integer), word("sobre"), node_if(INT, positive_integer)], en::b_over),
    // Spoken decimals.
    r("digits_unit", DIGSEQ, 0, &[node(UNIT)], en::b_digit_word),
    r("digits_more", DIGSEQ, 0, &[node(DIGSEQ), node(UNIT)], en::b_more_digits),
    r("digits_int", DIGSEQ, 0, &[node_if(INT, spelled_positive_integer)], b_int_digits),
    r("num_coma", NUM, 0, &[node_if(INT, spelled_integer), word("coma"), node(DIGSEQ)], en::b_point),
    r("num_con", NUM, 0, &[node_if(INT, spelled_integer), word("con"), node(DIGSEQ)], en::b_point),
    // Exponents.
    r("sign_plus", SIGN, 0, &[word("+")], en::b_plus),
    r("sign_minus", SIGN, 0, &[word("-")], en::b_minus),
    r("sign_more_plus", SIGN, 0, &[node(SIGN), glued_word("+")], en::b_sign_plus),
    r("sign_more_minus", SIGN, 0, &[node(SIGN), glued_word("-")], en::b_sign_minus),
    r("pow_e", NUM, 0, &[node(NUMERAL), glued_word("e"), glued_if(NUMERAL, integer)], en::b_ten_power),
    r(
        "pow_e_signed",
        NUM,
        0,
        &[node(NUMERAL), glued_word("e"), glued(SIGN), glued_if(NUMERAL, integer)],
        en::b_ten_power,
    ),
    r("pow_caret", NUM, 0, &[node(NUMERAL), glued_word("^"), glued_if(NUMERAL, integer)], en::b_caret),
    r(
        "pow_caret_signed",
        NUM,
        0,
        &[node(NUMERAL), glued_word("^"), glued(SIGN), glued_if(NUMERAL, integer)],
        en::b_caret,
    ),
    r(
        "pow_times_ten",
        NUM,
        0,
        &[node(NUMERAL), glued_word("x"), glued_if(NUMERAL, is_ten), glued_word("^"), glued_if(NUMERAL, integer)],
        en::b_times_ten_power,
    ),
    // Ordinals.
    r("ord_word", ORD, 0, &[node(ORD_WORD)], b_ordinal_word),
    r("ord_tens", ORD, 0, &[node(ORD_TENS)], b_ordinal_word),
    r("ord_hund", ORD, 0, &[node(ORD_HUND)], b_ordinal_word),
    r("ord_scale", ORD, 0, &[node_if(INT, spelled_positive_integer), node(ORD_SCALE)], b_ordinal_scale),
    r("ord_scale_bare", ORD, 0, &[node(ORD_SCALE)], b_ordinal_scale_bare),
    r("ord_sum", ORD, 0, &[node_if(INT, spelled_grain_from_3), node_if(ORD, spelled_ordinal)], b_ordinal_sum),
    r("ord_ord", ORD, 0, &[node_if(ORD, spelled_ordinal), node_if(ORD, spelled_ordinal)], b_ordinal_ordinal),
    r("ord_digits", ORD, 0, &[node_if(NUMERAL, positive_integer), glued(ORD_SUFFIX)], b_ordinal_digits),
    r(
        "ord_digits_dot",
        ORD,
        0,
        &[node_if(NUMERAL, positive_integer), glued_word("."), glued(ORD_SUFFIX)],
        b_ordinal_digits_dot,
    ),
    // Relative ordinals.
    r("rel_word", REL, 0, &[node(REL_WORD)], b_relative_word),
    r("rel_antes_del_ultimo", REL, 0, &[word("antes"), word("del"), word("ultimo")], en::b_end_minus_one),
    r("rel_antes_de_la_ultima", REL, 0, &[word("antes"), word("de"), word("la"), word("ultima")], en::b_end_minus_one),
    r("rel_anterior_al_ultimo", REL, 0, &[word("anterior"), word("al"), word("ultimo")], en::b_end_minus_one),
    r(
        "rel_anterior_a_la_ultima",
        REL,
        0,
        &[word("anterior"), word("a"), word("la"), word("ultima")],
        en::b_end_minus_one,
    ),
    // Percentages.
    r("pct_sign", PCT, 0, &[node(NUM), word("%")], en::b_percent_after),
    r("pct_por_ciento", PCT, 0, &[node(NUM), word("por"), word("ciento")], en::b_percent_after),
    r("pct_por_cien", PCT, 0, &[node(NUM), word("por"), word("cien")], en::b_percent_after),
];

pub(crate) fn en_digits_non_negative(v: &Value) -> bool {
    is_num(v, |n| n.has(DIGITS) && !n.has(SPELLED) && !n.value.is_negative())
}

pub(crate) fn spelled_grain_from_3(v: &Value) -> bool {
    spelled_integer(v) && grain_from_3(v)
}
