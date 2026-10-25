//! A temporary name left by a killed process whose pid came round again
//! is passed over: the write takes the next name and leaves the old file
//! as it was. Its own test binary, so this process's write counter starts
//! at 0 and the taken names are the first ones it would try.

use std::fs;

use turbo_zoo::accept::Acceptances;

#[test]
fn a_write_passes_over_temporary_names_a_killed_process_left() {
    let dir = std::env::temp_dir().join(format!("turbo-zoo-leftover-partial-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let dest = dir.join("acceptances.json");
    let left: Vec<_> =
        (0..4).map(|n| dir.join(format!("acceptances.json.partial.{}.{n}", std::process::id()))).collect();
    for p in &left {
        fs::write(p, b"left by a killed write").unwrap();
    }
    Acceptances::default().save(&dest).unwrap();
    assert_eq!(Acceptances::load(&dest).unwrap(), Acceptances::default());
    for p in &left {
        assert_eq!(fs::read(p).unwrap(), b"left by a killed write", "{}", p.display());
    }
    fs::remove_dir_all(&dir).unwrap();
}
