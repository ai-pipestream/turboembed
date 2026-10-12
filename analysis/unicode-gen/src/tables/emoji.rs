//! `emoji.rs`: the RGI emoji set and the names, groups and subgroups of
//! `emoji-test.txt`.

use std::collections::{BTreeMap, BTreeSet};

use super::Inputs;
use crate::emit::{RustFile, dec, str_lit};
use crate::ucd::{UcdFile, to_string};

/// The groups of `emoji-test.txt`, in the order of the library's
/// `EmojiGroup` enum.
const GROUPS: [&str; 10] = [
    "Smileys & Emotion",
    "People & Body",
    "Component",
    "Animals & Nature",
    "Food & Drink",
    "Travel & Places",
    "Activities",
    "Objects",
    "Symbols",
    "Flags",
];

const PRESENTATION_SELECTOR: char = '\u{FE0F}';

/// A row of `emoji-test.txt` that names an emoji.
struct Named {
    name: String,
    group: u8,
    subgroup: u8,
}

pub fn render(inputs: &Inputs) -> String {
    let rgi = rgi_set(inputs);
    let (named, subgroups) = names(&inputs.file("emoji-test.txt"));
    let mut file = RustFile::new(
        "The RGI emoji set (every sequence of emoji-sequences.txt and emoji-zwj-sequences.txt) as sorted strings, \
         and the CLDR short name, group and subgroup of every fully-qualified and component row of \
         emoji-test.txt, keyed by the sequence without U+FE0F.",
        &["emoji-sequences.txt", "emoji-zwj-sequences.txt", "emoji-test.txt"],
        inputs.version,
    );
    let rgi: Vec<String> = rgi.iter().map(|s| str_lit(s)).collect();
    file.blank();
    file.array("The RGI emoji sequences, sorted by their UTF-8 bytes.", "RGI", "&str", &rgi);
    let keys: Vec<String> = named.keys().map(|key| str_lit(key)).collect();
    let name_list: Vec<String> = named.values().map(|n| str_lit(&n.name)).collect();
    let groups: Vec<String> = named.values().map(|n| dec(n.group)).collect();
    let subgroup_ids: Vec<String> = named.values().map(|n| dec(n.subgroup)).collect();
    file.blank();
    file.array(
        "The named emoji sequences with every U+FE0F removed, sorted by their UTF-8 bytes.",
        "NAME_KEYS",
        "&str",
        &keys,
    );
    file.blank();
    file.array("The CLDR short name of each sequence of NAME_KEYS.", "NAMES", "&str", &name_list);
    file.blank();
    file.array(
        "The group of each sequence of NAME_KEYS, as the position of the group in EmojiGroup.",
        "GROUPS",
        "u8",
        &groups,
    );
    file.blank();
    file.array(
        "The subgroup of each sequence of NAME_KEYS, as a position in SUBGROUPS.",
        "SUBGROUP_IDS",
        "u8",
        &subgroup_ids,
    );
    let subgroups: Vec<String> = subgroups.iter().map(|s| str_lit(s)).collect();
    file.blank();
    file.array("The subgroup names, in the order emoji-test.txt lists them.", "SUBGROUPS", "&str", &subgroups);
    file.finish()
}

/// Every RGI emoji sequence. Ranges in `emoji-sequences.txt` stand for
/// one single-code-point sequence per code point.
fn rgi_set(inputs: &Inputs) -> BTreeSet<String> {
    let mut set = BTreeSet::new();
    for name in ["emoji-sequences.txt", "emoji-zwj-sequences.txt"] {
        let file = inputs.file(name);
        for record in file.records() {
            if record.field_count() != 3 {
                record.fail("expected `code points ; type ; description`");
            }
            if record.field(0).contains("..") {
                for cp in record.range(0) {
                    set.insert(to_string(&[cp]));
                }
            } else {
                set.insert(to_string(&record.code_points(0)));
            }
        }
    }
    set
}

/// The named rows of `emoji-test.txt` keyed by their sequence without
/// U+FE0F, and the subgroup names in file order.
fn names(file: &UcdFile) -> (BTreeMap<String, Named>, Vec<String>) {
    let mut named = BTreeMap::new();
    let mut subgroups: Vec<String> = Vec::new();
    let mut group: Option<u8> = None;
    for (line, text) in file.lines() {
        if let Some(name) = text.strip_prefix("# group: ") {
            let position = GROUPS.iter().position(|&g| g == name);
            group = Some(position.unwrap_or_else(|| file.fail(line, &format!("unknown group {name}"))) as u8);
            continue;
        }
        if let Some(name) = text.strip_prefix("# subgroup: ") {
            if subgroups.iter().any(|s| s == name) {
                file.fail(line, "subgroup listed twice");
            }
            subgroups.push(name.to_owned());
            continue;
        }
        if text.starts_with('#') || text.trim().is_empty() {
            continue;
        }
        let (Some(group), Some(subgroup)) = (group, subgroups.len().checked_sub(1)) else {
            file.fail(line, "emoji row before the first group and subgroup")
        };
        let Some((data, comment)) = text.split_once('#') else { file.fail(line, "row without a comment") };
        let Some((cps, status)) = data.split_once(';') else { file.fail(line, "expected `code points ; status`") };
        match status.trim() {
            "fully-qualified" | "component" => {}
            "minimally-qualified" | "unqualified" => continue,
            other => file.fail(line, &format!("unknown status {other}")),
        }
        let sequence = parse_sequence(file, line, cps);
        let key: String = sequence.chars().filter(|&c| c != PRESENTATION_SELECTOR).collect();
        let name = comment_name(file, line, comment, &sequence);
        let entry = Named { name, group, subgroup: u8::try_from(subgroup).expect("more than 256 subgroups") };
        if named.insert(key, entry).is_some() {
            file.fail(line, "two named rows differ only in U+FE0F");
        }
    }
    (named, subgroups)
}

fn parse_sequence(file: &UcdFile, line: usize, cps: &str) -> String {
    let cps: Vec<u32> = cps
        .split_ascii_whitespace()
        .map(|cp| match u32::from_str_radix(cp, 16) {
            Ok(value) if (4..=6).contains(&cp.len()) && value < 0x11_0000 => value,
            _ => file.fail(line, &format!("`{cp}` is not a code point")),
        })
        .collect();
    if cps.is_empty() {
        file.fail(line, "empty sequence");
    }
    to_string(&cps)
}

/// The name in a row comment of the form `<emoji> E<version> <name>`.
fn comment_name(file: &UcdFile, line: usize, comment: &str, sequence: &str) -> String {
    let Some(rest) = comment.trim_start().strip_prefix(sequence) else {
        file.fail(line, "the comment does not start with the emoji")
    };
    let rest = rest.trim_start();
    let Some((version, name)) = rest.split_once(' ') else { file.fail(line, "no name after the version") };
    let valid_version = version.strip_prefix('E').is_some_and(|v| {
        v.split('.').count() == 2 && v.split('.').all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()))
    });
    if !valid_version || name.trim().is_empty() {
        file.fail(line, "expected `E<major>.<minor> <name>`");
    }
    name.trim().to_owned()
}
