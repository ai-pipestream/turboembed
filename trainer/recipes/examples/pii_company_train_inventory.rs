//! Source support for a future company-boundary contrast, using only
//! publisher TRAIN. No model fitting and no exported source text.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::{BufReader, Read, Seek};
use std::path::Path;

use serde_json::json;
use turbo::bundle::{Bundle, sha256_hex};
use turbo::head::tagger::{TokenTagger, Work};
use turbo_trainer_recipes::cache::CacheRecord;
use turbo_trainer_recipes::corpus::Sentence;
use turbo_trainer_recipes::eval::spans;
use turbo_trainer_recipes::nemotron_pii::read_pinned_train_dev;

#[derive(Default)]
struct Support {
    instances: usize,
    personas: BTreeSet<usize>,
    contexts: BTreeSet<u64>,
}

impl Support {
    fn note(&mut self, persona: usize, pattern: u64) {
        self.instances += 1;
        self.personas.insert(persona);
        self.contexts.insert(pattern);
    }

    fn independent(&self) -> usize {
        self.personas.len().min(self.contexts.len())
    }
}

fn context(row: &Sentence, a: usize, b: usize) -> u64 {
    let mut hash = DefaultHasher::new();
    let word = |i: usize| {
        if row.labels[i] != "O" {
            row.labels[i].split_once('-').unwrap().1.to_owned()
        } else {
            let s = &row.words[i];
            if s.chars().any(|c| c.is_numeric()) { "<number>".to_owned() } else { s.to_lowercase() }
        }
    };
    (a.saturating_sub(5)..a).map(word).collect::<Vec<_>>().hash(&mut hash);
    "<target>".hash(&mut hash);
    (b..(b + 5).min(row.words.len())).map(word).collect::<Vec<_>>().hash(&mut hash);
    hash.finish()
}

fn anchor(s: &str) -> Option<String> {
    s.chars().all(char::is_alphabetic).then(|| s.to_lowercase())
}

fn main() {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    assert_eq!(args.len(), 3, "pii_company_train_inventory <publisher corpus> <publisher cache> <refit bundle>");
    let cache = Path::new(&args[1]);
    let receipt = fs::read(cache.join("cache.json")).unwrap();
    assert_eq!(sha256_hex(&receipt), "aed82a6bcfbb1875a4e40b841983475e8ce13c4fb0b207640b3f6ade5eae76d9");
    let info: CacheRecord = serde_json::from_slice(&receipt).unwrap();
    assert_eq!(info.layers, [22]);
    let bundle = Bundle::open(Path::new(&args[2])).unwrap();
    assert_eq!(bundle.manifest_sha256, "dd73330bc8bef5bc0f1b9c7b6c004cfbf6713fa1dadcc03a6238a159b6ddc3b7");
    let head = TokenTagger::load(&bundle, "pii").unwrap();
    let (rows, _) = read_pinned_train_dev(Path::new(&args[0])).unwrap();
    assert_eq!(rows.len(), 90_000);
    let mut reader = BufReader::new(File::open(cache.join("train.layer22.f32")).unwrap());
    let (mut bytes, mut scores, mut tags, mut work) = (Vec::new(), Vec::new(), Vec::new(), Work::default());
    let mut negatives: BTreeMap<String, Support> = BTreeMap::new();
    let mut misses: BTreeMap<String, Support> = BTreeMap::new();
    let mut missed = Support::default();
    let (mut offset, mut sampled, mut company, mut multi, mut exact, mut partial, mut wrong, mut absent) =
        (0, 0, 0, 0, 0, 0, 0, 0);
    for (i, row) in rows.iter().enumerate() {
        let persona = i % 45_000;
        let bio = spans(&row.labels);
        for (a, b, k) in &bio {
            if k == "first_name"
                && b - a == 1
                && let Some(anchor) = anchor(&row.words[*a])
            {
                negatives.entry(anchor).or_default().note(persona, context(row, *a, *b));
            }
        }
        let len = row.words.len() * info.encoder.hidden * 4;
        offset += len;
        // Personas not used for the earlier chooser fit or calibration.
        if persona % 6 != 3 {
            reader.seek_relative(len as i64).unwrap();
            continue;
        }
        sampled += 1;
        bytes.resize(len, 0);
        reader.read_exact(&mut bytes).unwrap();
        let forms: Vec<_> = row.words.iter().map(String::as_str).collect();
        let ids: Vec<_> = (0..forms.len() as u32).collect();
        let states: Vec<_> = bytes.as_chunks::<4>().0.iter().map(|v| f32::from_le_bytes(*v)).collect();
        head.scores_forms(&states, &ids, Some(&forms), &mut scores, &mut work).unwrap();
        head.decode(&scores, forms.len(), &mut tags).unwrap();
        let pred: Vec<_> = head
            .spans(&tags)
            .unwrap()
            .into_iter()
            .map(|s| (s.start as usize, s.end as usize, head.types()[s.kind as usize].as_str()))
            .collect();
        for (a, b, k) in bio {
            if k != "company_name" {
                continue;
            }
            company += 1;
            if b - a < 2 {
                continue;
            }
            multi += 1;
            if pred.contains(&(a, b, "company_name")) {
                exact += 1;
                continue;
            }
            if pred.iter().any(|(s, e, kind)| *s < b && *e > a && *kind == "company_name") {
                partial += 1;
            } else if pred.iter().any(|(s, e, _)| *s < b && *e > a) {
                wrong += 1;
            } else {
                absent += 1;
            }
            let p = context(row, a, b);
            missed.note(persona, p);
            if let Some(name) = anchor(&row.words[a]) {
                misses.entry(name).or_default().note(persona, p);
            }
        }
    }
    assert_eq!(reader.stream_position().unwrap(), offset as u64);
    assert_eq!(sampled, 15_000);
    assert_eq!(multi, exact + partial + wrong + absent);
    let overlap: Vec<_> = misses.iter().filter_map(|(key, pos)| negatives.get(key).map(|neg| (pos, neg))).collect();
    let enough = |limit| overlap.iter().filter(|(a, b)| a.independent() >= limit && b.independent() >= limit).count();
    let capacity = |limit| {
        overlap
            .iter()
            .filter(|(a, b)| a.independent() >= limit && b.independent() >= limit)
            .map(|(a, b)| a.independent().min(b.independent()))
            .sum::<usize>()
    };
    println!("{}",serde_json::to_string_pretty(&json!({
        "publisher_train_only":true,"sampled_rows":sampled,"company_gold":company,"company_multi_gold":multi,
        "company_multi_exact":exact,"company_multi_same_type_partial":partial,"company_multi_other_overlap":wrong,"company_multi_uncovered":absent,
        "missed_support":{"instances":missed.instances,"distinct_personas":missed.personas.len(),"distinct_masked_contexts":missed.contexts.len()},
        "name_start_collisions":{"surfaces":overlap.len(),"missed_company_instances":overlap.iter().map(|(s,_)|s.instances).sum::<usize>(),
            "first_name_negative_instances":overlap.iter().map(|(_,s)|s.instances).sum::<usize>(),"at_least_two":enough(2),
            "balanced_upper_capacity_two":capacity(2),"at_least_five":enough(5),"balanced_upper_capacity_five":capacity(5)},
    })).unwrap());
}
