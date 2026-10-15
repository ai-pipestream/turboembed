//! The set descriptions and the checks before a load or a fetch. None of
//! these touch the network: every fetch here is refused before it starts.

mod common;

use std::fs;
use std::path::Path;

use common::{copy_dir, data, scratch};
use turbo_eval::fetch::{cache_root, fetch};
use turbo_eval::read;
use turbo_eval::sets::{ALL, Digest, Source, find, gutenqa, hicbench, legalbench_rag_mini, qasper};

#[test]
fn names_are_unique_and_found() {
    for s in ALL {
        assert_eq!(find(s.name).map(|f| f.name), Ok(s.name));
        assert_eq!(ALL.iter().filter(|t| t.name == s.name).count(), 1, "{}", s.name);
    }
    assert!(find("msmarco").err().unwrap().starts_with("no set msmarco; the sets are scifact, nfcorpus"));
}

#[test]
fn recorded_digests_are_lower_case_hex() {
    for s in ALL {
        if let Source::Files(files) = s.source {
            for f in files {
                if let Digest::Recorded(d) = f.sha256 {
                    assert!(
                        d.len() == 64 && d.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
                        "{d}"
                    );
                }
                assert!(f.url.starts_with("https://"), "{}", f.url);
            }
        }
    }
}

#[test]
fn a_set_without_a_licence_is_refused() {
    let cache = scratch("sets-licence");
    copy_dir(&data("hicbench"), &cache.join("hicbench"));
    assert_eq!(hicbench::SET.load(&cache).unwrap_err(), "set hicbench has no licence recorded, so it is not used");
    assert_eq!(fetch(&hicbench::SET, &cache).unwrap_err(), "set hicbench has no licence recorded, so it is not used");
    let unlicensed: Vec<&str> = ALL.iter().filter(|s| s.check_license().is_err()).map(|s| s.name).collect();
    assert_eq!(unlicensed, ["conteb-covid-qa", "hicbench"]);
}

#[test]
fn a_recorded_digest_is_checked_before_reading() {
    // The sample archive is not the published one, so its digest differs.
    let cache = scratch("sets-digest");
    copy_dir(&data("qasper"), &cache.join("qasper"));
    let e = qasper::SET.load(&cache).unwrap_err();
    assert!(
        e.ends_with(
            "qasper-test-and-evaluator-v0.3.tgz: SHA-256 is \
             db3e59cbd16f9385bbd1763ee0ae52d07da302a35440726e7ffc010e7d514241, set qasper records \
             72a52a41193e2838b8074f80ac074b94f956b84886c36a61c58a7df4171bdd72"
        ),
        "{e}"
    );
}

#[test]
fn a_set_not_fetched_says_so() {
    let cache = scratch("sets-missing");
    let e = qasper::SET.load(&cache).unwrap_err();
    assert!(e.starts_with("set qasper: ") && e.ends_with("does not exist; fetch the set first"), "{e}");
}

#[test]
fn a_manual_set_loads_from_the_cache() {
    let cache = scratch("sets-manual");
    copy_dir(&data("legalbench-rag"), &cache.join("legalbench-rag-mini"));
    let s = legalbench_rag_mini::SET.load(&cache).unwrap();
    assert_eq!(s.name, "legalbench-rag-mini");
    assert_eq!(s.queries.len(), 3);
    let e = fetch(&legalbench_rag_mini::SET, &cache).unwrap_err();
    assert!(e.starts_with("set legalbench-rag-mini cannot be fetched file by file: download https://"), "{e}");
}

#[test]
fn the_cache_root_given_wins() {
    assert_eq!(cache_root(Some(Path::new("/data/eval"))).unwrap(), Path::new("/data/eval"));
}

#[test]
fn verify_names_a_missing_file() {
    let cache = scratch("sets-verify");
    fs::create_dir_all(cache.join("qasper")).unwrap();
    let e = qasper::SET.verify(&cache.join("qasper")).unwrap_err();
    assert!(e.contains("qasper-test-and-evaluator-v0.3.tgz"), "{e}");
}

#[test]
fn gutenqa_reads_the_files_fetch_writes() {
    let Source::Files(files) = gutenqa::SET.source else { panic!("gutenqa is fetched file by file") };
    let paths: Vec<&str> = files.iter().map(|f| f.path).collect();
    assert_eq!(paths, [read::gutenqa::CHUNKS, read::gutenqa::QUESTIONS]);
    assert!(files.iter().all(|f| matches!(f.sha256, Digest::Recorded(_))));
    // The sample is not the published files, so the digest check refuses
    // it before the reader runs.
    let cache = scratch("sets-gutenqa");
    copy_dir(&data("gutenqa"), &cache.join("gutenqa"));
    let e = gutenqa::SET.load(&cache).unwrap_err();
    assert!(
        e.contains("gutenqa_chunks.parquet: SHA-256 is ")
            && e.ends_with("set gutenqa records ee544a9c7e046464c3ff9329b33e848a9d7a29d7936d4cff92ed1b984549397d"),
        "{e}"
    );
}
