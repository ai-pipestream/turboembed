//! Scenario-grouped CSC partitions.
//!
//! CSC reuses one scenario under several context ids (part 1 and part 2,
//! friend and colleague variants), sometimes reworded. Contexts whose word
//! sets overlap at `SAME_SCENARIO` or more form one scenario group, and the
//! smallest context id in a group picks its partition. A response text that
//! still lands in several partitions is kept only in the first of train,
//! development and test it reaches, and so is any variant of it that the
//! uncased encoder cannot tell apart ([`encoder_key`]), so no input crosses
//! a split.
use std::collections::{BTreeMap, BTreeSet};

use turbo_unicode::java::to_lowercase;
use turbo_unicode::normalize::{Form, normalize};

use super::corpus::Response;

/// Word-set Jaccard at which two contexts count as one scenario. On the
/// pinned CSV, reworded variants score 0.45 or more and distinct scenarios
/// 0.34 or less.
pub const SAME_SCENARIO: f64 = 0.4;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Partition {
    Train,
    Dev,
    Test,
}

pub fn words(text: &str) -> BTreeSet<String> {
    text.split(|c: char| !c.is_ascii_alphanumeric()).filter(|w| !w.is_empty()).map(str::to_ascii_lowercase).collect()
}

pub fn jaccard(a: &BTreeSet<String>, b: &BTreeSet<String>) -> f64 {
    let union = a.union(b).count();
    if union == 0 { 1.0 } else { a.intersection(b).count() as f64 / union as f64 }
}

/// The key the uncased encoder sees for a response text: canonical
/// decomposition with combining marks dropped (the encoder's accent strip),
/// lowercased, and whitespace collapsed. Two texts with the same key produce
/// the same MiniLM token sequence, so they must share a partition.
pub fn encoder_key(text: &str) -> String {
    let mut decomposed = String::new();
    normalize(Form::Nfd, text, &mut decomposed);
    let mut no_marks = String::with_capacity(decomposed.len());
    for c in decomposed.chars() {
        if !turbo_unicode::category::general_category(c).is_mark() {
            no_marks.push(c);
        }
    }
    let mut lowered = String::new();
    to_lowercase(&no_marks, &mut lowered);
    lowered.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Assigns every response a partition and drops cross-partition repeats,
/// where repeats are counted by the encoder's key, not the exact string.
pub fn assign(mut rows: Vec<Response>) -> Vec<Response> {
    let scenario = scenario_keys(&rows);
    for row in &mut rows {
        row.partition = match scenario[&row.context] % 5 {
            0 => Partition::Test,
            1 => Partition::Dev,
            _ => Partition::Train,
        };
    }
    let keys: Vec<String> = rows.iter().map(|row| encoder_key(&row.text)).collect();
    let mut first: BTreeMap<&str, Partition> = BTreeMap::new();
    for (key, row) in keys.iter().zip(&rows) {
        let p = first.entry(key.as_str()).or_insert(row.partition);
        *p = (*p).min(row.partition);
    }
    let mut i = 0;
    rows.retain(|row| {
        let keep = first[keys[i].as_str()] == row.partition;
        i += 1;
        keep
    });
    rows
}

/// Maps each context id to the smallest context id of its scenario group.
fn scenario_keys(rows: &[Response]) -> BTreeMap<u32, u32> {
    let mut contexts: BTreeMap<u32, BTreeSet<String>> = BTreeMap::new();
    for row in rows {
        contexts.entry(row.context).or_insert_with(|| words(&row.context_text));
    }
    let ids: Vec<u32> = contexts.keys().copied().collect();
    let mut parent: BTreeMap<u32, u32> = ids.iter().map(|&id| (id, id)).collect();
    fn root(parent: &BTreeMap<u32, u32>, mut id: u32) -> u32 {
        while parent[&id] != id {
            id = parent[&id];
        }
        id
    }
    for (i, a) in ids.iter().enumerate() {
        for b in &ids[i + 1..] {
            if jaccard(&contexts[a], &contexts[b]) >= SAME_SCENARIO {
                let (ra, rb) = (root(&parent, *a), root(&parent, *b));
                parent.insert(ra.max(rb), ra.min(rb));
            }
        }
    }
    ids.iter().map(|&id| (id, root(&parent, id))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(context: u32, context_text: &str, text: &str) -> Response {
        Response {
            context,
            context_text: context_text.into(),
            text: text.into(),
            sarcastic: false,
            partition: Partition::Train,
        }
    }

    #[test]
    fn reused_and_reworded_scenarios_share_a_partition() {
        let rows = assign(vec![
            row(105, "Steve bought a pair of shoes after work today.", "Nice shoes."),
            row(233, "Steve bought a pair of shoes after work today.", "Love them."),
            row(201, "Steve bought an expensive pair of shoes after work.", "Fancy."),
            row(112, "You walk into the office and the printer is jammed again.", "Great."),
        ]);
        assert_eq!(rows.len(), 4);
        assert!(rows[..3].iter().all(|r| r.partition == Partition::Test));
        assert_eq!(rows[3].partition, Partition::Train);
    }

    #[test]
    fn repeated_response_text_stays_in_one_partition() {
        let rows = assign(vec![
            row(110, "Steve asks to borrow your car for the weekend.", "Thanks!"),
            row(112, "You walk into the office and the printer is jammed again.", "Thanks!"),
            row(112, "You walk into the office and the printer is jammed again.", "Great."),
        ]);
        let texts: Vec<_> = rows.iter().map(|r| (r.text.as_str(), r.partition)).collect();
        assert_eq!(texts, [("Thanks!", Partition::Train), ("Great.", Partition::Train)]);
        let rows = assign(vec![
            row(110, "Steve asks to borrow your car for the weekend.", "Thanks!"),
            row(111, "Steve brings donuts for the whole team this morning.", "Thanks!"),
        ]);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].partition, Partition::Dev);
    }

    #[test]
    fn uncased_and_whitespace_variants_stay_in_one_partition() {
        // The encoder lowercases its input, so codex's found variants produce
        // identical token sequences: Context 103 train "HAHAHAHAH" vs Context
        // 110 test "hahahahah", and Context 119 train "whatever" vs Context
        // 120 test "Whatever". The scenario texts differ, so only the
        // encoder-key dedup can keep them apart from a second partition.
        let rows = assign(vec![
            row(103, "You finally cleaned your room after three weeks.", "HAHAHAHAH"),
            row(110, "Your friend keeps using your mug without asking.", "hahahahah"),
            row(119, "The meeting ran two hours over on a Friday.", "whatever"),
            row(120, "Your coworker reheats fish in the office microwave.", "Whatever"),
        ]);
        let kept: Vec<_> = rows.iter().map(|r| (r.text.as_str(), r.partition)).collect();
        assert_eq!(kept, [("HAHAHAHAH", Partition::Train), ("whatever", Partition::Train)]);

        // Whitespace and case-fold variants of one text dedupe the same way.
        assert_eq!(encoder_key("  Whatever\t"), encoder_key("whatever"));
        assert_eq!(encoder_key("HAHAHAHAH"), encoder_key("hahahahah"));
        assert_eq!(encoder_key("Café"), encoder_key("cafe"));
    }
}
