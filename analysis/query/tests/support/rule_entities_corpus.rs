// The reviewed date, time, money and percent set (#1731), shared by the
// gold test and the `rule_entities_gate` example, which scores spaCy and
// Stanza on the same text. Mentions are listed in text order.

type GoldMention = (&'static str, &'static str);

struct GoldDoc {
    text: &'static str,
    expected: &'static [GoldMention],
}

const GOLD_CORPUS: &[GoldDoc] = &[
    GoldDoc {
        text: "On 2026-07-14, the board met at 9:30am and adjourned at 4:15pm after Acme Corp closed a $1.2M round and a $350K bridge, up 15% from Q3 2025.",
        expected: &[
            ("2026-07-14", DATE_ENTITY),
            ("9:30am", TIME_ENTITY),
            ("4:15pm", TIME_ENTITY),
            ("$1.2M", MONEY_ENTITY),
            ("$350K", MONEY_ENTITY),
            ("15%", PERCENT_ENTITY),
            ("Q3 2025", DATE_ENTITY),
        ],
    },
    GoldDoc {
        text: "Report dated July 4, 2026: Yesterday the desk opened at 14:45 and settled at 16:00, paying EUR 250 and £75.50 after a 4.5 percent fee and a 2% rebate.",
        expected: &[
            ("July 4, 2026", DATE_ENTITY),
            ("Yesterday", DATE_ENTITY),
            ("14:45", TIME_ENTITY),
            ("16:00", TIME_ENTITY),
            ("EUR 250", MONEY_ENTITY),
            ("£75.50", MONEY_ENTITY),
            ("4.5 percent", PERCENT_ENTITY),
            ("2%", PERCENT_ENTITY),
        ],
    },
    GoldDoc {
        text: "Between 14th March 2026 and July 2026, shipments of 80 kg over 2.5 km for 3 hours departed at 8:00am and cost $4,500 plus $120 in duties at 8% interest and 1.25% insurance before 2026-09-30.",
        expected: &[
            ("14th March 2026", DATE_ENTITY),
            ("July 2026", DATE_ENTITY),
            ("8:00am", TIME_ENTITY),
            ("$4,500", MONEY_ENTITY),
            ("$120", MONEY_ENTITY),
            ("8%", PERCENT_ENTITY),
            ("1.25%", PERCENT_ENTITY),
            ("2026-09-30", DATE_ENTITY),
        ],
    },
    GoldDoc {
        text: "Filing 2026-11-02: Next month and in Q1 2027 the rate drops by 0.75% while the $99.95 rebate and $25 credit expire at 5pm before the December 15, 2026 deadline and the January 2027 review.",
        expected: &[
            ("2026-11-02", DATE_ENTITY),
            ("Next month", DATE_ENTITY),
            ("Q1 2027", DATE_ENTITY),
            ("0.75%", PERCENT_ENTITY),
            ("$99.95", MONEY_ENTITY),
            ("$25", MONEY_ENTITY),
            ("5pm", TIME_ENTITY),
            ("December 15, 2026", DATE_ENTITY),
            ("January 2027", DATE_ENTITY),
        ],
    },
];
