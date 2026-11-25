//! The analyzers `AnalyzeQuery` has built, kept by the request fields that
//! shape them, so that a profile seen before is not compiled again.
//!
//! An analyzer is made from a request's default language, reference date,
//! global synonym rules and field profiles, and from nothing else: the
//! query, the offset mode and the targets do not enter it. Compiling one
//! costs far more than analyzing a query with it, and callers send the
//! same profile with every query.
//!
//! The cache is bounded twice: by the number of analyzers, and by the
//! bytes of their keys. A key is as long as the profile it stands for, so
//! the second bound limits the synonym rules the kept analyzers hold. Past
//! either bound the analyzer used longest ago is dropped.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

use prost::Message;
use turbo_query::OnePassQueryAnalyzer;

use crate::api::Result;
use crate::proto::result::AnalyzeQueryRequest;

/// The most analyzers kept.
pub(super) const MAX_ANALYZERS: usize = 64;

/// The most key bytes kept, over all the analyzers. An analyzer whose key
/// alone is longer is built for its request and not kept.
pub(super) const MAX_KEY_BYTES: usize = 1 << 20;

/// The analyzers of one server, by profile key.
#[derive(Default)]
pub(crate) struct AnalyzerCache(Mutex<Kept>);

#[derive(Default)]
struct Kept {
    /// Each analyzer with the tick of its last use.
    by_key: HashMap<Box<[u8]>, (Arc<OnePassQueryAnalyzer>, u64)>,
    key_bytes: usize,
    tick: u64,
}

/// The fields of `req` that shape its analyzer, as bytes. Two requests
/// with equal keys have equal fields, so they compile to equal analyzers.
pub(super) fn profile_key(req: &AnalyzeQueryRequest) -> Vec<u8> {
    let mut key = Vec::new();
    for text in [req.default_language.trim(), req.reference_date_iso.trim()] {
        key.extend_from_slice(&(text.len() as u64).to_le_bytes());
        key.extend_from_slice(text.as_bytes());
    }
    // The count tells the field profiles from the rules that follow them.
    key.extend_from_slice(&(req.field_profiles.len() as u64).to_le_bytes());
    for profile in &req.field_profiles {
        profile.encode_length_delimited(&mut key).expect("a Vec takes any length");
    }
    for rule in &req.global_synonyms {
        rule.encode_length_delimited(&mut key).expect("a Vec takes any length");
    }
    key
}

impl AnalyzerCache {
    /// The analyzer kept under `key`, or the one `build` makes, which is
    /// then kept. A failed build keeps nothing.
    pub(super) fn get_or_build(
        &self,
        key: Vec<u8>,
        build: impl FnOnce() -> Result<OnePassQueryAnalyzer>,
    ) -> Result<Arc<OnePassQueryAnalyzer>> {
        if let Some(kept) = self.lock().get(&key) {
            return Ok(kept);
        }
        // Built outside the lock, so requests with other profiles do not
        // wait for it. Two first requests of one profile may both build;
        // both then use the one that was kept first.
        let built = Arc::new(build()?);
        Ok(self.lock().keep(key, built))
    }

    fn lock(&self) -> MutexGuard<'_, Kept> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.lock().by_key.len()
    }
}

impl Kept {
    fn get(&mut self, key: &[u8]) -> Option<Arc<OnePassQueryAnalyzer>> {
        self.tick += 1;
        let (analyzer, used) = self.by_key.get_mut(key)?;
        *used = self.tick;
        Some(Arc::clone(analyzer))
    }

    fn keep(&mut self, key: Vec<u8>, built: Arc<OnePassQueryAnalyzer>) -> Arc<OnePassQueryAnalyzer> {
        if let Some(kept) = self.get(&key) {
            return kept;
        }
        if key.len() > MAX_KEY_BYTES {
            return built;
        }
        self.key_bytes += key.len();
        self.by_key.insert(key.into_boxed_slice(), (Arc::clone(&built), self.tick));
        while self.by_key.len() > MAX_ANALYZERS || self.key_bytes > MAX_KEY_BYTES {
            let oldest = self.by_key.iter().min_by_key(|(_, (_, used))| *used).map(|(key, _)| key.clone());
            let Some(oldest) = oldest else { break };
            self.by_key.remove(&oldest);
            self.key_bytes -= oldest.len();
        }
        built
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get(cache: &AnalyzerCache, key: &[u8]) -> Arc<OnePassQueryAnalyzer> {
        cache.get_or_build(key.to_vec(), || Ok(OnePassQueryAnalyzer::new())).unwrap()
    }

    fn holds(cache: &AnalyzerCache, key: &[u8]) -> bool {
        cache.lock().by_key.contains_key(key)
    }

    #[test]
    fn a_key_seen_before_gives_the_analyzer_built_for_it() {
        let cache = AnalyzerCache::default();
        let first = get(&cache, b"a");
        let other = get(&cache, b"b");
        assert!(!Arc::ptr_eq(&first, &other));
        let again = cache.get_or_build(b"a".to_vec(), || panic!("the analyzer of a kept key is not built again"));
        assert!(Arc::ptr_eq(&first, &again.unwrap()));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn a_failed_build_keeps_nothing() {
        let cache = AnalyzerCache::default();
        let failed = cache.get_or_build(b"a".to_vec(), || {
            Err(crate::api::Failure::new(crate::api::TURBO_E_INVALID_ARGUMENT, 0, "no"))
        });
        assert!(failed.is_err());
        assert_eq!(cache.len(), 0);
        get(&cache, b"a");
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn past_the_count_limit_the_analyzer_used_longest_ago_is_dropped() {
        let cache = AnalyzerCache::default();
        let key = |i: usize| (i as u32).to_le_bytes();
        let shared = Arc::new(OnePassQueryAnalyzer::new());
        for i in 0..MAX_ANALYZERS {
            cache.lock().keep(key(i).to_vec(), Arc::clone(&shared));
        }
        // Key 0 is used again, so key 1 is now the one used longest ago.
        assert!(cache.lock().get(&key(0)).is_some());
        cache.lock().keep(key(MAX_ANALYZERS).to_vec(), Arc::clone(&shared));
        assert_eq!(cache.len(), MAX_ANALYZERS);
        assert!(holds(&cache, &key(0)) && !holds(&cache, &key(1)) && holds(&cache, &key(MAX_ANALYZERS)));
    }

    #[test]
    fn the_keys_kept_stay_within_the_byte_limit() {
        let cache = AnalyzerCache::default();
        let shared = Arc::new(OnePassQueryAnalyzer::new());
        let long = |first: u8| {
            let mut key = vec![0u8; MAX_KEY_BYTES / 2];
            key[0] = first;
            key
        };
        cache.lock().keep(long(1), Arc::clone(&shared));
        cache.lock().keep(long(2), Arc::clone(&shared));
        assert_eq!((cache.len(), cache.lock().key_bytes), (2, MAX_KEY_BYTES));
        // A third key of that length takes the place of the first.
        cache.lock().keep(long(3), Arc::clone(&shared));
        assert_eq!((cache.len(), cache.lock().key_bytes), (2, MAX_KEY_BYTES));
        assert!(!holds(&cache, &long(1)) && holds(&cache, &long(2)) && holds(&cache, &long(3)));
        // A key over the limit by itself is answered and not kept.
        let kept = cache.lock().keep(vec![9u8; MAX_KEY_BYTES + 1], Arc::clone(&shared));
        assert!(Arc::ptr_eq(&kept, &shared));
        assert_eq!((cache.len(), cache.lock().key_bytes), (2, MAX_KEY_BYTES));
    }
}
