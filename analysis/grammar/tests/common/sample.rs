//! A deterministic English text of a given size, for timing: news-like
//! sentences, most with numbers, ordinals or percentages in words or
//! digits, and some with none.

const SENTENCES: &[&str] = &[
    "The council approved a budget of 4.2 million dollars for the third year in a row.",
    "Revenue rose 12% to 1,674.5 million, while costs fell by two and a half percent.",
    "About two thirds of the 3,500 respondents said the next quarter would be harder.",
    "The penultimate stage covers one hundred and twenty-five kilometres of mountain road.",
    "She finished second, 0.25 seconds behind the winner, in her twenty-first race.",
    "Nobody expected the committee to meet again before the end of the season.",
    "Inflation was minus 0.3 percent in March and 1.1 percent in April.",
    "Some 40 000 people attended, a crowd of nearly half a million over the week.",
    "The last but one chapter was rewritten three times by the editors.",
    "Shares closed at 23.75, up from 21 1/4 a week earlier and 3/4 of a point today.",
    "He bought three dozen eggs and a dozen rolls for the 15th anniversary breakfast.",
    "The fund holds 2.5bn in bonds, roughly one fifth of its assets.",
    "Weather permitting, the ferry leaves at the usual time from the old harbour.",
    "Unemployment fell to five point two per cent, the lowest in nineteen years.",
    "One in five households owns a car older than 1998.",
    "The 1st and 2nd editions sold 12,000 copies; the 3rd sold 1.3e4.",
    "A previous version of the report put the figure at 6.23x10^-4.",
    "Production rose by one billion and three quarters million units.",
    "The archive was moved to a new building on the north side of the river.",
    "The vote was 52 to 48, with 7 abstentions and nought spoiled ballots.",
];

/// At least `bytes` bytes of text, built the same way every time.
pub fn sample(bytes: usize) -> String {
    let mut out = String::with_capacity(bytes + 128);
    // A linear congruential sequence picks the sentences.
    let mut state: u64 = 0x2545_f491_4f6c_dd1d;
    while out.len() < bytes {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        let s = SENTENCES[(state >> 33) as usize % SENTENCES.len()];
        out.push_str(s);
        out.push(if (state >> 20).is_multiple_of(7) { '\n' } else { ' ' });
    }
    out
}
