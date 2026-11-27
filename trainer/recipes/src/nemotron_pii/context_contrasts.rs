//! Balanced same-name contrasts for the #2596 experiment. No held-out probes
//! or memo text is read to construct these rows. Dev varies both names and
//! entire template families.

use crate::Result;
use crate::corpus::Sentence;

use super::spans::to_sentence;

pub const TRAIN: usize = 6_000;
pub const DEV: usize = 600;
pub const KINDS: [&str; 4] = ["city", "first_name", "last_name", "company_name"];

pub(super) const TRAIN_NAMES: &[&str] = &[
    "Alton",
    "Arlen",
    "Ashford",
    "Auden",
    "Banks",
    "Barlow",
    "Barton",
    "Baxter",
    "Beckett",
    "Bennett",
    "Bexley",
    "Bramley",
    "Briar",
    "Calder",
    "Calloway",
    "Carlisle",
    "Carter",
    "Chandler",
    "Chesney",
    "Cresswell",
    "Dalton",
    "Darby",
    "Dawson",
    "Ellery",
    "Emerson",
    "Emmett",
    "Everson",
    "Felton",
    "Galen",
    "Garner",
    "Granger",
    "Hadley",
    "Harlow",
    "Hartley",
    "Hayden",
    "Hollis",
    "Kendall",
    "Kersey",
    "Lennox",
    "Lincoln",
    "Linden",
    "Lowell",
    "Marley",
    "Merritt",
    "Nolan",
    "Orwell",
    "Palmer",
    "Parker",
    "Prescott",
    "Radley",
    "Reeve",
    "Ridley",
    "Ripley",
    "Rowan",
    "Sawyer",
    "Sheridan",
    "Sinclair",
    "Somers",
    "Sutton",
    "Tatum",
    "Thayer",
    "Tilden",
    "Warden",
    "Winslow",
];

const DEV_NAMES: &[&str] = &[
    "Afton",
    "Ainsley",
    "Alden",
    "Ambrose",
    "Arden",
    "Ashton",
    "Bellamy",
    "Blythe",
    "Bracken",
    "Bronson",
    "Clifton",
    "Crofton",
    "Dentonville",
    "Elston",
    "Finley",
    "Gresham",
    "Hawley",
    "Kenton",
    "Lawton",
    "Leland",
    "Loxley",
    "Marston",
    "Mason",
    "Porter",
    "Quinby",
    "Ramsay",
    "Shelton",
    "Stanley",
    "Torrance",
    "Whitney",
    "Winton",
    "Wycliff",
];

// Each family uses the same syntactic position for all four readings. A
// company sometimes has a designator and sometimes is a bare trade name.
// A surname includes both words, not merely the shared token.
const TRAIN_FAMILIES: &[[&str; 4]] = &[
    [
        "At the hearing, {} expanded the bus route to serve outlying blocks.",
        "At the hearing, {} folded the receipt and put it into a coat pocket.",
        "At the hearing, Ms. {de } gave testimony about the missing papers.",
        "At the hearing, { Labs} reported its annual revenue.",
    ],
    [
        "The shipment from {} crossed the county line at dusk.",
        "The message from {} asked where the spare glasses were.",
        "The message from Dr. {Van } named the next conference speaker.",
        "The invoice from { Supply} billed us for six repair kits.",
    ],
    [
        "The mail went to {} on a coach scheduled to stop at the east terminal.",
        "The note went to {} because the spare jacket belonged to that person.",
        "The sealed folder went to counsel {St. } before the trial.",
        "The order went to {} because its quote covered replacement parts.",
    ],
    [
        "By noon, {} had reopened the city hall to residents.",
        "By noon, {} had untangled the headphone cable.",
        "By noon, Professor {de } had signed the manuscript receipt.",
        "By noon, { Works} had shipped the first batch of hand tools.",
    ],
    [
        "According to the county clerk, {} sits just beyond the western boundary.",
        "According to the diary, {} had left a birthday card under the cushion.",
        "According to the directory, Justice {Van } holds a visiting appointment.",
        "According to the quarterly report, {} acquired two equipment suppliers.",
    ],
    [
        "On the last day of the survey, {} opened a neighborhood library branch.",
        "On the last day of the survey, {} knitted a small pair of mittens.",
        "On the last day of the survey, Captain {de } mailed a signed correction.",
        "On the last day of the survey, { Studio} renegotiated the printing contract.",
    ],
    [
        "We heard back from {} after its council voted to fix the levee.",
        "We heard back from {} after the last physical therapy appointment.",
        "We heard back from nurse {Van } after the clinic shift ended.",
        "We heard back from {} after its procurement team reviewed the bids.",
    ],
    [
        "We wrote to {} asking its building office for a municipal map.",
        "We wrote to {} asking whether there was a spare blanket.",
        "We wrote to editor {de } about a correction to the book.",
        "We wrote to { Systems} requesting a replacement circuit.",
    ],
];

const DEV_FAMILIES: &[[&str; 4]] = &[
    [
        "The bid from {} outlined a new public swimming pool for local residents.",
        "The letter from {} described how to handle the injured bird.",
        "The recommendation from Dean {del } covered two graduate fellows.",
        "The brochure from { Design} lists the prices of its commercial fixtures.",
    ],
    [
        "The travelers returned to {} when the ferry service resumed.",
        "The caretaker returned the scarf to {} after the rehearsal.",
        "The board handed the register to Officer {von } for safekeeping.",
        "The merchants sent the ledger to {} for its finance department to review.",
    ],
    [
        "As winter approached, {} opened a public shelter near the reservoir.",
        "As winter approached, {} began packing snacks for the afternoon shift.",
        "As winter approached, Sergeant {del } requested a revised duty roster.",
        "As winter approached, { Foundry} signed a new equipment lease.",
    ],
    [
        "The entry mentions {} because the town limits now include the old pier.",
        "The entry mentions {} because the candle burned a hole in the sleeve.",
        "The entry mentions author {von } because the final edition includes a foreword.",
        "The entry mentions {} because the firm replaced its entire product line.",
    ],
];

/// Generates a labelled BIO row for each rotated four-way contrast.
pub fn documents(dev: bool) -> Result<Vec<Sentence>> {
    let (names, families, total) =
        if dev { (DEV_NAMES, DEV_FAMILIES, DEV) } else { (TRAIN_NAMES, TRAIN_FAMILIES, TRAIN) };
    (0..total)
        .map(|row| {
            let name = names[row % names.len()];
            let hash = (row as u64 ^ 0x2596_05ee).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            let family_index = ((hash ^ (hash >> 23)) as usize) % families.len();
            let mut text = String::new();
            let mut gold = Vec::with_capacity(KINDS.len());
            for rot in 0..KINDS.len() {
                let role = (rot + row / 3) % KINDS.len();
                let index =
                    if row % 2 == 0 { family_index } else { ((hash >> (8 * (role + 1))) as usize) % families.len() };
                let (before, template) = families[index][role].split_once('{').expect("contrast slot");
                let (affix, after) = template.split_once('}').expect("contrast slot end");
                if !text.is_empty() {
                    text.push(' ');
                }
                text.push_str(before);
                let start = text.len();
                if affix.ends_with(' ') {
                    text.push_str(affix);
                    text.push_str(name);
                } else {
                    text.push_str(name);
                    text.push_str(affix);
                }
                let end = text.len();
                text.push_str(after);
                gold.push(format!("{{'start': {start}, 'end': {end}, 'text': 'x', 'label': '{}'}}", KINDS[role]));
            }
            assert!(text.is_ascii(), "offsets count characters");
            to_sentence("context-contrast", row, &text, &format!("[{}]", gold.join(", ")))
                .map(|s| s.expect("four clauses"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_and_dev_are_disjoint_and_annotate_the_same_four_roles() {
        assert!(TRAIN_NAMES.iter().all(|n| !DEV_NAMES.contains(n)));
        assert!(TRAIN_FAMILIES.iter().flatten().all(|t| !DEV_FAMILIES.iter().flatten().any(|d| d == t)));
        for (dev, names, count) in [(false, TRAIN_NAMES, TRAIN), (true, DEV_NAMES, DEV)] {
            let rows = documents(dev).unwrap();
            let distinct = rows.iter().map(|r| &r.words).collect::<std::collections::BTreeSet<_>>();
            assert!(distinct.len() * 2 > rows.len(), "{} unique documents out of {}", distinct.len(), rows.len());
            assert_eq!(rows, documents(dev).unwrap());
            assert_eq!(rows.len(), count);
            for (i, row) in rows.iter().enumerate() {
                let spans = crate::eval::spans(&row.labels);
                assert_eq!(spans.len(), 4);
                for kind in KINDS {
                    let (a, b, _) = spans.iter().find(|(_, _, k)| k == kind).unwrap();
                    assert!(row.words[*a..*b].iter().any(|w| w == names[i % names.len()]));
                    if kind == "last_name" {
                        assert!(b - a >= 2);
                    }
                }
            }
        }
    }
    #[test]
    fn the_generator_does_not_reuse_reserved_names_or_clauses() {
        let fixtures = [
            include_str!("../../../../analysis/tagger/tests/pii_shared_name_sentences.rs"),
            include_str!("../../../../analysis/pii/tests/data/human-eval/places-person-names.txt"),
            include_str!("../../../../analysis/pii/tests/data/human-eval/places-person-names-2.txt"),
            include_str!("../../../../analysis/pii/tests/data/human-eval/places-person-names-3.txt"),
        ];
        let normalize = |s: &str| {
            s.split(|c: char| !c.is_alphanumeric())
                .filter(|s| !s.is_empty())
                .map(str::to_ascii_lowercase)
                .collect::<Vec<_>>()
                .join(" ")
        };
        let fixtures = fixtures.map(normalize);
        for name in TRAIN_NAMES.iter().chain(DEV_NAMES) {
            let lower = name.to_ascii_lowercase();
            assert!(fixtures.iter().all(|t| !t.split(' ').any(|w| w == lower)), "reserved name {name}");
        }
        for template in TRAIN_FAMILIES.iter().chain(DEV_FAMILIES).flatten() {
            let (prefix, slot) = template.split_once('{').unwrap();
            let (_, suffix) = slot.split_once('}').unwrap();
            let (left, right) = (normalize(prefix), normalize(suffix));
            assert!(!right.is_empty());
            for text in &fixtures {
                let words: Vec<_> = text.split(' ').collect();
                let (l, r): (Vec<_>, Vec<_>) = (left.split(' ').collect(), right.split(' ').collect());
                // Match normalized wording around any name up to four
                // tokens long, not just our own generated names.
                assert!(
                    !(1..=4).any(|sl| {
                        words.windows(l.len() + sl + r.len()).any(|w| w.starts_with(&l) && w.ends_with(&r))
                    }),
                    "reserved template {template}"
                );
            }
        }
    }
}
