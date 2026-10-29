//! MASC's Penn tags on the recorded release, with TURBO_TEST_MASC_DIR set
//! to the extracted `MASC-3.0.0` directory; a directory that is not the
//! release is refused either way.

use std::path::PathBuf;

use turbo_trainer_recipes::corpus::{Counts, Task};
use turbo_trainer_recipes::masc_pos::read;

#[test]
fn a_directory_that_is_not_the_release_is_refused() {
    let dir = std::env::temp_dir().join(format!("turbo-masc-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("README.txt"), "not MASC").unwrap();
    let e = read(&dir, false).unwrap_err();
    assert!(e.0.contains("corpus digest") && e.0.contains("--unrecorded-corpus"), "{e}");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn the_recorded_release_reads_with_the_recipes_split() {
    let Some(dir) = std::env::var_os("TURBO_TEST_MASC_DIR").map(PathBuf::from) else { return };
    let c = read(&dir, false).unwrap();
    assert!(c.record.recorded);
    assert_eq!(c.record.task, Task::Tags);
    let n = |x: Counts| (x.sentences, x.words);
    assert_eq!(n(c.record.train), (21_340, 372_744));
    assert_eq!(n(c.record.dev), (3_618, 58_893));
    assert_eq!(n(c.record.test), (6_384, 85_292));
    assert_eq!(c.record.skipped, 1);
}
