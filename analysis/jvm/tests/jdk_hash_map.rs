//! [`JavaHashMap`] and [`HashMapOrder`] against `java.util.HashMap` on a
//! running JDK 21, over thousands of random scenarios
//! (`tests/jdk/HashMapProbe.java` runs them on the JVM).
//!
//! Each scenario picks a key class (`String`, `CharBuffer`, `Integer`, or
//! a class that is not `Comparable`; some `String` scenarios add `null`
//! and `""`), a constructor (`new HashMap<>()` or
//! `new HashMap<>(n)` for a random `n`), and a run of `put`, `merge`,
//! `putAll` and `clear` calls. Most draw their keys so that their spread
//! hashes share the low 6 to 9 bits with one of a few targets, and some add
//! keys with equal hash codes, so that bins grow long: tables double for a
//! long bin below 64 bins, bins become red-black trees at 64 and more, and
//! later resizes split trees and turn short halves back into lists. The
//! entries, values included, must come out in the JVM's order. A scenario
//! the model reports as not exact (keys with equal hash codes that the JVM
//! orders by `System.identityHashCode`) is counted and skipped.
//!
//! Scenarios that insert with only `put` or only `merge` from a new or
//! cleared map are also run through [`HashMapOrder`].
//!
//! The test runs only with `TURBO_TEST_JDK` set and then needs `java` and
//! `javac` on the PATH. `TURBO_TEST_JDK_SEED` picks the seed (1 by default)
//! and `TURBO_TEST_JDK_COUNT` the number of scenarios (4000 by default).

use std::collections::HashSet;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use turbo_jvm::hash_map::{
    CharBuffer, HashMapOrder, Insert, JavaHash, JavaHashMap, Utf16String, char_buffer_hash, spread, utf16_string_hash,
};
use turbo_jvm::random::JavaRandom;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    String,
    CharBuffer,
    Opaque,
    Integer,
}

#[derive(Clone, Debug)]
enum Op {
    Put(usize),
    Merge(usize),
    PutAll(Vec<usize>),
    Clear,
}

#[derive(Clone, Debug)]
struct Scenario {
    kind: Kind,
    /// `new HashMap<>(n)`, or `new HashMap<>()` for `None`.
    capacity: Option<usize>,
    /// The UTF-16 units of each key (`String`, `CharBuffer`).
    units: Vec<Vec<u16>>,
    /// Whether the first `String` key is `null` (and the second `""`, with
    /// the same hash code).
    null_first: bool,
    /// The hash code of each key (`Integer`, not `Comparable`).
    ints: Vec<i32>,
    ops: Vec<Op>,
}

/// A key whose class is not `Comparable`: equal only to itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Opaque {
    hash: i32,
    index: usize,
}

impl JavaHash for Opaque {
    fn java_hash(&self) -> i32 {
        self.hash
    }
}

impl Scenario {
    fn line(&self) -> String {
        let mut s = String::new();
        s.push_str(match self.kind {
            Kind::String => "s",
            Kind::CharBuffer => "c",
            Kind::Opaque => "o",
            Kind::Integer => "i",
        });
        match self.capacity {
            None => s.push_str(" d "),
            Some(n) => write!(s, " n{n} ").unwrap(),
        }
        let keys: Vec<String> = match self.kind {
            Kind::String | Kind::CharBuffer => self
                .units
                .iter()
                .enumerate()
                .map(|(i, u)| {
                    if i == 0 && self.null_first {
                        "~".to_owned()
                    } else if u.is_empty() {
                        "_".to_owned()
                    } else {
                        u.iter().map(|x| format!("{x:x}")).collect::<Vec<_>>().join(".")
                    }
                })
                .collect(),
            Kind::Opaque | Kind::Integer => self.ints.iter().map(i32::to_string).collect(),
        };
        s.push_str(&keys.join(","));
        for op in &self.ops {
            match op {
                Op::Put(k) => write!(s, " p{k}").unwrap(),
                Op::Merge(k) => write!(s, " m{k}").unwrap(),
                Op::PutAll(ks) => {
                    let ks: Vec<String> = ks.iter().map(usize::to_string).collect();
                    write!(s, " a{}", ks.join(".")).unwrap();
                }
                Op::Clear => s.push_str(" x"),
            }
        }
        s
    }

    /// The spread hash of each key.
    fn spreads(&self) -> Vec<u32> {
        match self.kind {
            Kind::String => self.units.iter().map(|u| spread(utf16_string_hash(u))).collect(),
            Kind::CharBuffer => self.units.iter().map(|u| spread(char_buffer_hash(u))).collect(),
            Kind::Opaque | Kind::Integer => self.ints.iter().map(|&h| spread(h)).collect(),
        }
    }
}

/// What the model gives for a scenario: the line the probe prints, whether
/// the model calls it exact, and whether a bin was a tree at some point.
struct Outcome {
    line: String,
    exact: bool,
    tree: bool,
}

fn run_model<K: JavaHash + Eq + Clone>(keys: &[K], s: &Scenario) -> Outcome {
    let mut map: JavaHashMap<K, (usize, i64)> = match s.capacity {
        None => JavaHashMap::new(),
        Some(n) => JavaHashMap::with_capacity(n),
    };
    let mut exact = true;
    let mut tree = false;
    for (i, op) in s.ops.iter().enumerate() {
        // The probe numbers its ops from 3, their place on the line.
        let value = i as i64 + 3;
        match op {
            Op::Put(k) => map.put(keys[*k].clone(), (*k, value)),
            Op::Merge(k) => map.merge(keys[*k].clone(), (*k, value), |acc, v| acc.1 += v.1),
            Op::PutAll(ks) => {
                map.put_all(ks.iter().map(|&k| (keys[k].clone(), (k, value))).collect::<Vec<_>>().into_iter())
            }
            Op::Clear => {
                exact &= map.is_jvm_exact();
                tree |= map.tree_bins() > 0;
                map.clear();
            }
        }
    }
    exact &= map.is_jvm_exact();
    tree |= map.tree_bins() > 0;
    let line = map.iter().map(|(_, &(k, v))| format!("{k}={v}")).collect::<Vec<_>>().join(",");
    Outcome { line, exact, tree }
}

fn run(s: &Scenario) -> Outcome {
    match s.kind {
        Kind::String => {
            let null = |i: usize| i == 0 && s.null_first;
            let keys: Vec<Option<Utf16String>> =
                s.units.iter().enumerate().map(|(i, u)| (!null(i)).then(|| Utf16String(u.clone()))).collect();
            let outcome = run_model(&keys, s);
            // The same through `String` keys, when every key is valid UTF-16.
            let strings: Option<Vec<Option<String>>> = s
                .units
                .iter()
                .enumerate()
                .map(|(i, u)| if null(i) { Some(None) } else { String::from_utf16(u).ok().map(Some) })
                .collect();
            if let Some(strings) = strings {
                for (k, u) in strings.iter().zip(&s.units) {
                    assert_eq!(k.java_hash(), utf16_string_hash(u));
                }
                let again = run_model(&strings, s);
                assert_eq!((&again.line, again.exact), (&outcome.line, outcome.exact), "String keys: {}", s.line());
            }
            outcome
        }
        Kind::CharBuffer => {
            let keys: Vec<CharBuffer<'_>> = s.units.iter().map(|u| CharBuffer(u)).collect();
            run_model(&keys, s)
        }
        Kind::Opaque => {
            let keys: Vec<Opaque> = s.ints.iter().enumerate().map(|(index, &hash)| Opaque { hash, index }).collect();
            run_model(&keys, s)
        }
        Kind::Integer => run_model(&s.ints, s),
    }
}

/// For a scenario that inserts with only `put` or only `merge` between
/// clears into `new HashMap<>()`, the order [`HashMapOrder`] gives the keys
/// after the last clear, with whether it calls it exact.
fn run_order(s: &Scenario) -> Option<(Vec<usize>, bool)> {
    if s.capacity.is_some() {
        return None;
    }
    let spreads = s.spreads();
    let mut model = HashMapOrder::new();
    let mut out = Vec::new();
    let mut exact = true;
    for segment in s.ops.split(|op| matches!(op, Op::Clear)) {
        let merge = match segment.first() {
            None => false,
            Some(Op::Put(_)) => false,
            Some(Op::Merge(_)) => true,
            Some(_) => return None,
        };
        let mut seen = HashSet::new();
        let mut keys = Vec::new();
        let mut again = false;
        for op in segment {
            let k = match (op, merge) {
                (Op::Put(k), false) | (Op::Merge(k), true) => *k,
                _ => return None,
            };
            if seen.insert(k) {
                keys.push(k as u32);
                again = false;
            } else {
                again = true;
            }
        }
        let insert = if merge { Insert::Merge { again } } else { Insert::Put };
        exact = model.order_keys(&spreads, &keys, insert, &mut out);
    }
    Some((out.into_iter().map(|k| k as usize).collect(), exact))
}

/// Draws scenarios from `rng`.
struct Generator {
    rng: JavaRandom,
}

impl Generator {
    fn below(&mut self, n: usize) -> usize {
        self.rng.next_int_bounded(n as i32) as usize
    }

    fn chance(&mut self, percent: usize) -> bool {
        self.below(100) < percent
    }

    /// A random key of 1 to 6 units, mostly ASCII letters, with some other
    /// BMP characters, surrogates, and characters from U+E000 up.
    fn units(&mut self) -> Vec<u16> {
        let len = 1 + self.below(6);
        (0..len)
            .map(|_| match self.below(10) {
                0 => 0x00C0 + self.below(0x300) as u16,
                1 => 0xD800 + self.below(0x800) as u16,
                2 => 0xE000 + self.below(0x2000) as u16,
                _ => b'A' as u16 + self.below(58) as u16,
            })
            .collect()
    }

    /// A key with the same hash code as `units`: two adjacent units changed
    /// so that their weighted sum stays the same.
    fn twin(&mut self, kind: Kind, units: &[u16]) -> Option<Vec<u16>> {
        if units.len() < 2 {
            return None;
        }
        let i = self.below(units.len() - 1);
        let mut twin = units.to_vec();
        // `String.hashCode` weighs a unit 31 times its successor;
        // `CharBuffer.hashCode` weighs it a 31st of its successor.
        let (up, down) = if kind == Kind::String { (i, i + 1) } else { (i + 1, i) };
        let (step_up, step_down) = (1u16, 31u16);
        twin[up] = twin[up].checked_add(step_up)?;
        twin[down] = twin[down].checked_sub(step_down)?;
        Some(twin)
    }

    fn hash(kind: Kind, units: &[u16]) -> i32 {
        if kind == Kind::String { utf16_string_hash(units) } else { char_buffer_hash(units) }
    }

    fn scenario(&mut self) -> Scenario {
        let kind = [Kind::String, Kind::CharBuffer, Kind::Opaque, Kind::Integer][self.below(4)];
        let n = match self.below(3) {
            0 => 1 + self.below(40),
            1 => 1 + self.below(200),
            _ => 1 + self.below(700),
        };
        // Spread hashes whose low `bits` bits are one of `targets`.
        let clustered = self.chance(80);
        let bits = 6 + self.below(4);
        let mut targets: Vec<u32> = (0..1 + self.below(3)).map(|_| self.below(1 << bits) as u32).collect();
        // `null` and `""` both hash to 0.
        let null_first = kind == Kind::String && self.chance(20);
        if null_first {
            targets[0] = 0;
        }
        let twins = if self.chance(50) { 10 + self.below(40) } else { 0 };
        let fits = |s: u32, targets: &[u32]| !clustered || targets.contains(&(s & ((1 << bits) - 1)));

        let mut units: Vec<Vec<u16>> = Vec::new();
        let mut ints: Vec<i32> = Vec::new();
        match kind {
            Kind::String | Kind::CharBuffer => {
                let mut seen = HashSet::new();
                if null_first {
                    units.push(Vec::new());
                    units.push(Vec::new());
                    seen.insert(Vec::new());
                }
                let mut tries = 0;
                while units.len() < n && tries < 2_000_000 {
                    tries += 1;
                    let key = if !units.is_empty() && self.below(100) < twins {
                        let from = units[self.below(units.len())].clone();
                        match self.twin(kind, &from) {
                            Some(t) => t,
                            None => continue,
                        }
                    } else {
                        self.units()
                    };
                    if fits(spread(Self::hash(kind, &key)), &targets) && seen.insert(key.clone()) {
                        units.push(key);
                    }
                }
            }
            Kind::Opaque | Kind::Integer => {
                let mut seen = HashSet::new();
                while ints.len() < n {
                    let h = if kind == Kind::Opaque && !ints.is_empty() && self.below(100) < twins {
                        ints[self.below(ints.len())]
                    } else {
                        let mut s = self.rng.next_int() as u32;
                        if clustered {
                            s = (s & !((1 << bits) - 1)) | targets[self.below(targets.len())];
                        }
                        // `spread` is its own inverse.
                        spread(s as i32) as i32
                    };
                    if kind == Kind::Opaque || seen.insert(h) {
                        ints.push(h);
                    }
                }
            }
        }
        let n = units.len().max(ints.len());

        let mut ops = Vec::new();
        let capacity;
        if self.chance(30) {
            // Only `put` or only `merge` in each part between clears.
            capacity = None;
            for part in 0..1 + self.below(2) {
                if part > 0 {
                    ops.push(Op::Clear);
                }
                let merge = self.chance(50);
                let mut order: Vec<usize> = (0..n).collect();
                self.rng.shuffle(&mut order);
                order.truncate(1 + self.below(n));
                let mut calls: Vec<usize> = Vec::new();
                for &k in &order {
                    calls.push(k);
                    if self.chance(20) {
                        calls.push(calls[self.below(calls.len())]);
                    }
                }
                if self.chance(30) {
                    calls.push(order[self.below(order.len())]);
                }
                ops.extend(calls.into_iter().map(|k| if merge { Op::Merge(k) } else { Op::Put(k) }));
            }
        } else {
            capacity = if self.chance(30) {
                let sizes = [0, 1, 2, 3, 5, 12, 13, 16, 17, 48, 64, 100, 1000];
                Some(if self.chance(50) { sizes[self.below(sizes.len())] } else { self.below(2000) })
            } else {
                None
            };
            let len = n + self.below(n + 1);
            for _ in 0..len {
                let r = self.below(100);
                ops.push(if r < 48 {
                    Op::Put(self.below(n))
                } else if r < 94 {
                    Op::Merge(self.below(n))
                } else if r < 99 {
                    let mut ks: Vec<usize> = (0..n).collect();
                    self.rng.shuffle(&mut ks);
                    ks.truncate(1 + self.below(n.min(80)));
                    Op::PutAll(ks)
                } else {
                    Op::Clear
                });
            }
        }
        Scenario { kind, capacity, units, null_first, ints, ops }
    }
}

fn jdk_requested() -> bool {
    if std::env::var_os("TURBO_TEST_JDK").is_none() {
        eprintln!("skipped: set TURBO_TEST_JDK to compare with a JDK");
        return false;
    }
    true
}

fn command(command: &mut Command) -> String {
    let output = command.output().unwrap_or_else(|e| panic!("{command:?} does not start: {e}"));
    assert!(output.status.success(), "{command:?} failed: {}", String::from_utf8_lossy(&output.stderr));
    String::from_utf8(output.stdout).expect("the probe prints ASCII")
}

fn setting(name: &str, default: u64) -> u64 {
    std::env::var(name).ok().map(|s| s.parse().unwrap_or_else(|_| panic!("{name} is not a number"))).unwrap_or(default)
}

#[test]
fn java_hash_map_matches_the_jdk_on_random_scenarios() {
    if !jdk_requested() {
        return;
    }
    let seed = setting("TURBO_TEST_JDK_SEED", 1);
    let count = setting("TURBO_TEST_JDK_COUNT", 4000) as usize;
    let mut generator = Generator { rng: JavaRandom::new(seed as i64) };
    let scenarios: Vec<Scenario> = (0..count).map(|_| generator.scenario()).collect();

    let dir: PathBuf = std::env::temp_dir().join(format!("turbo-jvm-hash-map-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("scenarios.txt");
    let lines: Vec<String> = scenarios.iter().map(Scenario::line).collect();
    std::fs::write(&input, lines.join("\n") + "\n").unwrap();
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/jdk/HashMapProbe.java");
    command(Command::new("javac").arg("-d").arg(&dir).arg(&source));
    let output = command(Command::new("java").arg("-cp").arg(&dir).arg("HashMapProbe").arg(&input));
    let _ = std::fs::remove_dir_all(&dir);
    let jvm: Vec<&str> = output.lines().collect();
    assert_eq!(jvm.len(), count);

    let (mut trees, mut inexact, mut orders, mut orders_inexact) = (0, 0, 0, 0);
    let mut mismatches = Vec::new();
    for (i, (s, &jvm)) in scenarios.iter().zip(&jvm).enumerate() {
        let ours = run(s);
        trees += usize::from(ours.tree);
        if !ours.exact {
            inexact += 1;
        } else if ours.line != jvm {
            mismatches.push(format!("scenario {i}: {}\n  ours {}\n  jvm  {jvm}", lines[i], ours.line));
        }
        if let Some((order, exact)) = run_order(s) {
            orders += 1;
            if !exact {
                orders_inexact += 1;
                continue;
            }
            let jvm_order: Vec<usize> = if jvm.is_empty() {
                Vec::new()
            } else {
                jvm.split(',').map(|e| e.split_once('=').unwrap().0.parse().unwrap()).collect()
            };
            if order != jvm_order {
                mismatches
                    .push(format!("scenario {i} (HashMapOrder): {}\n  ours {order:?}\n  jvm  {jvm_order:?}", lines[i]));
            }
        }
    }
    eprintln!(
        "{count} scenarios: {trees} with a tree bin, {inexact} not exact; {orders} through HashMapOrder, \
         {orders_inexact} of them not exact"
    );
    let shown: Vec<&String> = mismatches.iter().take(10).collect();
    assert!(
        mismatches.is_empty(),
        "{} scenarios differ (first {}):\n{}",
        mismatches.len(),
        shown.len(),
        shown.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("\n")
    );
    // The scenarios reach the cases they are drawn for.
    assert!(trees * 4 > count, "{trees} of {count} scenarios have a tree bin");
    assert!(inexact * 4 < count, "{inexact} of {count} scenarios are not exact");
    assert!(orders * 5 > count, "{orders} of {count} scenarios run through HashMapOrder");
}
