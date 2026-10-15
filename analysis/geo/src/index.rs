//! The index behind every gazetteer here (OpenNLP's `GazetteerIndex`): the
//! records in an [`EntryStore`], the match key of every name with the
//! records it names, ranked, a table from source and record id to record,
//! and the representative record of each country.
//!
//! Records arrive in [`Fragment`]s: a loader parses and folds a run of rows
//! into one, possibly on several threads at once, and [`IndexBuilder`]
//! takes the fragments in row order, so the checks that depend on earlier
//! rows (a repeated record id) see the rows in the order OpenNLP adds them.

use std::cell::RefCell;

use crate::entry::RegionCode;
use crate::fold::Folder;
use crate::ranking::{Ranker, by_prior, code_ranks};
use crate::store::{CapacityError, Entry, EntryStore, NONE32, Row};
use crate::table::{HashSlots, hash_bytes};

/// What a loader does with an alternate name that folds to the empty key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Alternates {
    /// Refuse the record; for hand written tables their author can fix.
    Reject,
    /// Leave the name out of the index; for downloaded tables.
    Skip,
}

/// The seed of the name table's hash.
const NAME_SEED: u64 = 0x6E61_6D65;
/// The seed of the record id table's hash, mixed with the source code.
const ID_SEED: u64 = 0x6964_7300;

/// The 676 two letter region codes.
pub(crate) const REGIONS: usize = 26 * 26;

/// Rows parsed and folded, in row order, not yet checked against each
/// other.
pub(crate) struct Fragment {
    pub(crate) store: EntryStore,
    /// The match keys of all rows, one after the other.
    keys: String,
    key_bounds: Vec<u32>,
    /// The hash of each key, taken here so the threads that parse share
    /// the work.
    key_hashes: Vec<u64>,
    /// The first key of each row, and the key count at the end.
    row_keys: Vec<u32>,
    /// The rows with a name that folds to nothing, in row order, each with
    /// the name's place among the record's strings (1 for the canonical
    /// name, 2 and up for the alternates).
    unmatchable: Vec<(u32, u32)>,
    /// The one-based line number of each row, for loaders.
    pub(crate) lines: Vec<u32>,
}

impl Fragment {
    pub(crate) fn new() -> Fragment {
        Fragment {
            store: EntryStore::new(),
            keys: String::new(),
            key_bounds: vec![0],
            key_hashes: Vec::new(),
            row_keys: vec![0],
            unmatchable: Vec::new(),
            lines: Vec::new(),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.store.len()
    }

    /// Stores a checked row with the keys of its names.
    pub(crate) fn push(
        &mut self,
        folder: &mut Folder,
        key: &mut String,
        row: &Row<'_>,
        country: Option<RegionCode>,
        alternates: Alternates,
        line: u32,
    ) -> Result<(), CapacityError> {
        let r = self.store.push(row, country)?;
        folder.fold_into(row.name, key);
        if key.is_empty() {
            self.unmatchable.push((r, 1));
        } else {
            self.push_key(key)?;
        }
        for (k, name) in row.alternates.iter().enumerate() {
            folder.fold_into(name, key);
            if !key.is_empty() {
                self.push_key(key)?;
            } else if alternates == Alternates::Reject && self.unmatchable.last().is_none_or(|&(u, _)| u != r) {
                self.unmatchable.push((r, 2 + k as u32));
            }
        }
        self.row_keys.push((self.key_bounds.len() - 1) as u32);
        self.lines.push(line);
        Ok(())
    }

    fn push_key(&mut self, key: &str) -> Result<(), CapacityError> {
        self.key_hashes.push(hash_bytes(NAME_SEED, key.as_bytes()));
        self.keys.push_str(key);
        self.key_bounds.push(u32::try_from(self.keys.len()).map_err(|_| CapacityError::Text)?);
        Ok(())
    }
}

/// The match keys, each stored once, and the table that finds them.
#[derive(Clone, Debug, Default)]
pub(crate) struct KeyTable {
    pub(crate) text: String,
    pub(crate) bounds: Vec<u32>,
    pub(crate) slots: HashSlots,
}

impl KeyTable {
    fn new() -> KeyTable {
        KeyTable { text: String::new(), bounds: vec![0], slots: HashSlots::default() }
    }

    pub(crate) fn len(&self) -> usize {
        self.bounds.len() - 1
    }

    pub(crate) fn key(&self, k: u32) -> &str {
        let k = k as usize;
        &self.text[self.bounds[k] as usize..self.bounds[k + 1] as usize]
    }

    pub(crate) fn find(&self, key: &str) -> Option<u32> {
        self.slots.find(hash_bytes(NAME_SEED, key.as_bytes()), |k| self.key(k) == key)
    }

    /// The id of `key`, whose hash is `hash`, added if it is new. `hashes`
    /// holds the hash of every key so far, so the table grows without
    /// hashing the key text again.
    fn find_or_insert(&mut self, key: &str, hash: u64, hashes: &mut Vec<u64>) -> Result<u32, CapacityError> {
        if let Some(k) = self.slots.find(hash, |k| self.key(k) == key) {
            return Ok(k);
        }
        let k = u32::try_from(self.len()).map_err(|_| CapacityError::Text)?;
        self.text.push_str(key);
        self.bounds.push(u32::try_from(self.text.len()).map_err(|_| CapacityError::Text)?);
        hashes.push(hash);
        self.slots.insert(hash, k, |k| hashes[k as usize]);
        Ok(k)
    }
}

/// The hash of a record id within a source.
pub(crate) fn id_hash(source: u16, record_id: &str) -> u64 {
    hash_bytes(ID_SEED ^ u64::from(source), record_id.as_bytes())
}

/// Why the builder stopped. A row is named by its place in the fragment.
pub(crate) enum BuildError {
    /// The row repeats the source and record id of an earlier one.
    Duplicate {
        row: usize,
        record_id: String,
    },
    /// A name of the row folds to nothing.
    Unmatchable {
        row: usize,
        name: String,
        source: String,
        record_id: String,
    },
    Capacity(CapacityError),
}

impl From<CapacityError> for BuildError {
    fn from(e: CapacityError) -> BuildError {
        BuildError::Capacity(e)
    }
}

/// Collects fragments into a [`GazetteerIndex`].
pub(crate) struct IndexBuilder {
    store: EntryStore,
    names: KeyTable,
    /// The hash of each key of `names`, while the index is built.
    name_hashes: Vec<u64>,
    /// For each name of each record, the key and the record, in record
    /// order.
    pair_key: Vec<u32>,
    pair_record: Vec<u32>,
    /// The last record paired with each key of `names`, so a record whose
    /// names fold to the same key is paired with it once.
    key_last: Vec<u32>,
    ids: HashSlots,
    regions: Vec<u32>,
    sources: Vec<u16>,
    /// Whether each code is in `sources`.
    source_seen: Vec<bool>,
}

impl IndexBuilder {
    pub(crate) fn new() -> IndexBuilder {
        IndexBuilder {
            store: EntryStore::new(),
            names: KeyTable::new(),
            name_hashes: Vec::new(),
            pair_key: Vec::new(),
            pair_record: Vec::new(),
            key_last: Vec::new(),
            ids: HashSlots::default(),
            regions: vec![NONE32; REGIONS],
            sources: Vec::new(),
            source_seen: Vec::new(),
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.store.len() == 0
    }

    /// Makes room in the record id table for `records` records.
    pub(crate) fn reserve(&mut self, records: usize) {
        let IndexBuilder { store, ids, .. } = self;
        ids.reserve(records, |r| id_hash(store.source[r as usize], Entry::new(store, r).record_id()));
    }

    /// Adds the rows of `fragment` in order. A row whose source and record
    /// id an earlier row has, or one with a name that folds to nothing,
    /// stops the build; OpenNLP checks the id first.
    pub(crate) fn add(&mut self, fragment: Fragment) -> Result<(), BuildError> {
        let base = self.store.len() as u32;
        let rows = fragment.len();
        let Fragment { store, keys, key_bounds, key_hashes, row_keys, unmatchable, .. } = fragment;
        let key_of = |k: u32| &keys[key_bounds[k as usize] as usize..key_bounds[k as usize + 1] as usize];
        self.store.append(store)?;
        let mut ranks = code_ranks(&self.store);
        ranks.resize(self.store.codes.len(), 3);
        self.source_seen.resize(self.store.codes.len(), false);
        let mut unmatchable = unmatchable.iter().peekable();
        for r in 0..rows {
            let record = base + r as u32;
            if !self.insert_id(record) {
                let id = Entry::new(&self.store, record).record_id().to_string();
                return Err(BuildError::Duplicate { row: r, record_id: id });
            }
            if let Some(&&(u, which)) = unmatchable.peek()
                && u as usize == r
            {
                let e = Entry::new(&self.store, record);
                let name = self.store.string(self.store.strings[record as usize] + which).to_string();
                return Err(BuildError::Unmatchable {
                    row: r,
                    name,
                    source: e.source().into(),
                    record_id: e.record_id().into(),
                });
            }
            let source = self.store.source[record as usize];
            if !self.source_seen[usize::from(source)] {
                self.source_seen[usize::from(source)] = true;
                self.sources.push(source);
            }
            for k in row_keys[r]..row_keys[r + 1] {
                let key = self.names.find_or_insert(key_of(k), key_hashes[k as usize], &mut self.name_hashes)?;
                if key as usize == self.key_last.len() {
                    self.key_last.push(NONE32);
                }
                // A record whose names fold to the same key is listed once.
                if self.key_last[key as usize] != record {
                    self.key_last[key as usize] = record;
                    self.pair_key.push(key);
                    self.pair_record.push(record);
                }
            }
            if let Some(region) = Entry::new(&self.store, record).region() {
                let best = &mut self.regions[region.index()];
                if *best == NONE32 || by_prior(&self.store, &ranks, record, *best).is_lt() {
                    *best = record;
                }
            }
        }
        Ok(())
    }

    /// Adds the record `record` of the store to the id table, unless a
    /// record with its source and id is there.
    fn insert_id(&mut self, record: u32) -> bool {
        let store = &self.store;
        let entry = Entry::new(store, record);
        let source = store.source[record as usize];
        let hash = id_hash(source, entry.record_id());
        let same = |other: u32| {
            store.source[other as usize] == source && Entry::new(store, other).record_id() == entry.record_id()
        };
        if self.ids.find(hash, same).is_some() {
            return false;
        }
        self.ids.insert(hash, record, |r| id_hash(store.source[r as usize], Entry::new(store, r).record_id()));
        true
    }

    /// Ranks every key's records and freezes the index.
    pub(crate) fn build(self) -> GazetteerIndex {
        let IndexBuilder {
            mut store,
            mut names,
            name_hashes,
            pair_key,
            pair_record,
            key_last,
            ids,
            regions,
            sources,
            ..
        } = self;
        drop((name_hashes, key_last));
        let n = names.len();
        let mut postings_start = vec![0u32; n + 1];
        for &k in &pair_key {
            postings_start[k as usize + 1] += 1;
        }
        for k in 0..n {
            postings_start[k + 1] += postings_start[k];
        }
        let mut fill = postings_start.clone();
        let mut postings = vec![0u32; pair_key.len()];
        for (&k, &r) in pair_key.iter().zip(&pair_record) {
            postings[fill[k as usize] as usize] = r;
            fill[k as usize] += 1;
        }
        drop((pair_key, pair_record, fill));
        rank_postings(&store, &postings_start, &mut postings);
        store.shrink_to_fit();
        names.text.shrink_to_fit();
        names.bounds.shrink_to_fit();
        GazetteerIndex { store, names, postings_start, postings, ids, regions, sources }
    }
}

/// Sorts each key's records into ranking order, on several threads when
/// there are many keys. The threads get runs of whole keys with about the
/// same number of records each, since a few names can hold most of them.
fn rank_postings(store: &EntryStore, starts: &[u32], postings: &mut [u32]) {
    let ranker = Ranker::new(store);
    let keys = starts.len() - 1;
    let sort_keys = |ranker: &Ranker<'_>, run: &mut [u32], k0: usize, k1: usize, base: usize| {
        let mut scratch = Vec::new();
        for k in k0..k1 {
            let list = &mut run[starts[k] as usize - base..starts[k + 1] as usize - base];
            if list.len() > 1 {
                ranker.sort(list, &mut scratch);
            }
        }
    };
    let threads = if keys < 1 << 16 { 1 } else { std::thread::available_parallelism().map_or(1, |n| n.get()) };
    if threads == 1 {
        sort_keys(&ranker, postings, 0, keys, 0);
        return;
    }
    let total = postings.len();
    std::thread::scope(|scope| {
        let mut rest = postings;
        let (mut k0, mut offset) = (0usize, 0usize);
        for t in 1..=threads {
            // The first key boundary at or after this thread's share.
            let target = (total * t / threads) as u32;
            let k1 = if t == threads { keys } else { k0 + starts[k0..=keys].partition_point(|&s| s < target) };
            if k1 <= k0 {
                continue;
            }
            let end = starts[k1] as usize;
            let (run, tail) = rest.split_at_mut(end - offset);
            rest = tail;
            let (from, base) = (k0, offset);
            let (ranker, sort_keys) = (&ranker, &sort_keys);
            scope.spawn(move || sort_keys(ranker, run, from, k1, base));
            (k0, offset) = (k1, end);
        }
    });
}

/// The finished index. See the module documentation.
pub(crate) struct GazetteerIndex {
    pub(crate) store: EntryStore,
    pub(crate) names: KeyTable,
    /// The records of key `k` are `postings[postings_start[k]..postings_start[k + 1]]`.
    pub(crate) postings_start: Vec<u32>,
    pub(crate) postings: Vec<u32>,
    pub(crate) ids: HashSlots,
    /// The representative record of each region code, or `NONE32`.
    pub(crate) regions: Vec<u32>,
    /// The codes of the sources, in the order their first record came.
    pub(crate) sources: Vec<u16>,
}

thread_local! {
    /// The folder and key buffer of the lookups on this thread.
    static FOLD: RefCell<(Folder, String)> = RefCell::new((Folder::new(), String::new()));
}

/// The match key of `name`, passed to `f`, folded with this thread's
/// folder.
pub(crate) fn with_key<R>(name: &str, f: impl FnOnce(&str) -> R) -> R {
    FOLD.with(|cell| match cell.try_borrow_mut() {
        Ok(mut guard) => {
            let (folder, key) = &mut *guard;
            folder.fold_into(name, key);
            f(key)
        }
        // Only a lookup started from inside `f` gets here.
        Err(_) => f(&Folder::new().fold(name)),
    })
}

impl GazetteerIndex {
    /// The number of records.
    pub(crate) fn len(&self) -> usize {
        self.store.len()
    }

    /// The records of match key `key`, in ranking order.
    pub(crate) fn records_of_key(&self, key: &str) -> &[u32] {
        if key.is_empty() {
            return &[];
        }
        match self.names.find(key) {
            Some(k) => {
                &self.postings[self.postings_start[k as usize] as usize..self.postings_start[k as usize + 1] as usize]
            }
            None => &[],
        }
    }

    /// The records a name matches, in ranking order.
    pub(crate) fn lookup(&self, name: &str) -> Vec<Entry<'_>> {
        with_key(name, |key| self.records_of_key(key).iter().map(|&r| Entry::new(&self.store, r)).collect())
    }

    pub(crate) fn by_id(&self, source: &str, record_id: &str) -> Option<Entry<'_>> {
        let source_code = self.sources.iter().copied().find(|&c| self.store.codes.name(c) == source)?;
        let hash = id_hash(source_code, record_id);
        let store = &self.store;
        let found = self
            .ids
            .find(hash, |r| store.source[r as usize] == source_code && Entry::new(store, r).record_id() == record_id)?;
        Some(Entry::new(store, found))
    }

    pub(crate) fn by_region(&self, code: RegionCode) -> Option<Entry<'_>> {
        let r = self.regions[code.index()];
        (r != NONE32).then(|| Entry::new(&self.store, r))
    }

    pub(crate) fn sources(&self) -> Vec<&str> {
        self.sources.iter().map(|&c| self.store.codes.name(c)).collect()
    }

    /// The number of distinct match keys.
    pub(crate) fn key_count(&self) -> usize {
        self.names.len()
    }

    /// The bytes the index holds on the heap.
    pub(crate) fn heap_bytes(&self) -> usize {
        self.store.heap_bytes()
            + self.names.text.capacity()
            + self.names.bounds.capacity() * 4
            + self.names.slots.heap_bytes()
            + self.postings_start.capacity() * 4
            + self.postings.capacity() * 4
            + self.ids.heap_bytes()
            + self.regions.capacity() * 4
            + self.sources.capacity() * 2
    }
}
