use super::*;

#[test]
fn the_binary_says_what_to_fetch_and_needs_its_inputs() {
    let exe = env!("CARGO_BIN_EXE_turbo-train-recipes");
    let paths = Command::new(exe).args(["fineweb-ranges", "--paths"]).output().unwrap();
    assert!(paths.status.success(), "{paths:?}");
    let text = String::from_utf8(paths.stdout).unwrap();
    assert_eq!(text.lines().count(), 218);
    let first =
        format!("{}\t{}\tdata/afr_Latn/train/000_00000.parquet", fineweb::FINEWEB_2.repo, fineweb::FINEWEB_2.revision);
    assert_eq!(text.lines().next(), Some(first.as_str()));
    assert!(text.contains(&format!(
        "{}\t{}\tsample/10BT/001_00000.parquet\n",
        fineweb::FINEWEB.repo,
        fineweb::FINEWEB.revision
    )));

    // With nothing fetched, each file's last 8 bytes are needed first.
    let dir = scratch("fineweb-binary");
    let (files, _) = served(&WEB);
    let list = dir.join("files.txt");
    fs::write(&list, fineweb::write_files(&files)).unwrap();
    let web = dir.join("web");
    let need = Command::new(exe)
        .args(["fineweb-ranges", "--need", "--files"])
        .arg(&list)
        .arg("--web")
        .arg(&web)
        .output()
        .unwrap();
    assert!(need.status.success(), "{need:?}");
    let need = String::from_utf8(need.stdout).unwrap();
    assert_eq!(need.lines().count(), files.len());
    let f = &files[0];
    let line = format!(
        "{}\t{}\t{}\t{}",
        fineweb::FINEWEB_2.url(&f.path),
        Store::new(&web, None).file(f, Span { start: f.size - 8, end: f.size }).display(),
        f.size - 8,
        f.size
    );
    assert_eq!(need.lines().next(), Some(line.as_str()));
    let record = Command::new(exe)
        .args(["fineweb-ranges", "--record", "--files"])
        .arg(&list)
        .arg("--web")
        .arg(&web)
        .output()
        .unwrap();
    assert_eq!(record.status.code(), Some(2), "{record:?}");
    assert!(String::from_utf8_lossy(&record.stderr).contains("not fetched"), "{record:?}");

    // A flag a command does not read is refused.
    let paths = Command::new(exe).args(["fineweb-ranges", "--paths", "--web", "x"]).output().unwrap();
    assert_eq!(paths.status.code(), Some(2), "{paths:?}");
    assert!(String::from_utf8_lossy(&paths.stderr).starts_with("fineweb-ranges --paths does not read --web"));

    // Model commands stop before consulting the dataset or model paths.
    let run = Command::new(exe).args(["tatoeba-fineweb", "--references", "missing"]).output().unwrap();
    assert_eq!(run.status.code(), Some(2), "{run:?}");
    assert!(String::from_utf8_lossy(&run.stderr).contains("disabled by #1511"));
    assert!(!dir.join("models").exists());
    fs::remove_dir_all(dir).unwrap();
}
