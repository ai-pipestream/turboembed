//! Builds libturbo's cdylib for a test that links a C program against it.
//!
//! `cargo test` builds the rlib the tests link, not the cdylib, so a test
//! that needs the shared library builds it with the profile and target
//! directory the test itself was built with. Neither is read from the test
//! executable's parent directories: cargo 1.100 moved test executables
//! from target/<profile>/deps/ to a hashed directory under
//! target/<profile>/build/, so the grandparent's name is no longer the
//! profile. The target directory comes from CARGO_TARGET_TMPDIR (always
//! target/tmp), and the profile directory is the executable's ancestor
//! that sits directly in it.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The target directory and the name of the profile directory in it
/// (`debug`, `release` or a custom profile's name) that the running test
/// was built under.
fn target_and_profile_dir() -> (PathBuf, String) {
    let target = Path::new(env!("CARGO_TARGET_TMPDIR")).parent().unwrap().to_path_buf();
    let canonical = |p: &Path| p.canonicalize().unwrap_or_else(|_| p.to_path_buf());
    let target_canonical = canonical(&target);
    let exe = canonical(&std::env::current_exe().unwrap());
    let profile_dir = exe
        .ancestors()
        .find(|a| a.parent().is_some_and(|p| p == target_canonical))
        .unwrap_or_else(|| panic!("{} is not under the target directory {}", exe.display(), target.display()));
    let name = profile_dir.file_name().unwrap().to_str().unwrap().to_string();
    (target, name)
}

/// The cargo profile whose output goes to the directory `dir`.
fn profile_of(dir: &str) -> &str {
    match dir {
        "debug" => "dev",
        p => p,
    }
}

/// Builds the cdylib as the library is now and returns the directory that
/// holds it (target/<profile>), for `-L` and the run path.
pub fn build() -> PathBuf {
    let (target, dir) = target_and_profile_dir();
    let out = Command::new(env!("CARGO"))
        .args(["build", "--lib", "--profile", profile_of(&dir), "--manifest-path"])
        .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"))
        .arg("--target-dir")
        .arg(&target)
        .output()
        .expect("cargo build --lib");
    assert!(out.status.success(), "cargo build --lib failed:\n{}", String::from_utf8_lossy(&out.stderr));
    let lib_dir = target.join(&dir);
    let lib = format!("{}turbo{}", std::env::consts::DLL_PREFIX, std::env::consts::DLL_SUFFIX);
    assert!(lib_dir.join(&lib).exists(), "{lib} is not in {}", lib_dir.display());
    lib_dir
}

#[test]
fn the_profile_directory_is_found_under_the_target_directory() {
    let (target, dir) = target_and_profile_dir();
    assert!(target.join(&dir).is_dir(), "{} is not a directory", target.join(&dir).display());
    // The test executable's grandparent no longer names the profile under
    // cargo 1.100; the directory found must be one cargo would build into.
    let expected = if cfg!(debug_assertions) { "debug" } else { "release" };
    if dir == "debug" || dir == "release" {
        assert_eq!(dir, expected);
    }
    assert_eq!(profile_of("debug"), "dev");
    assert_eq!(profile_of("release"), "release");
}
