//! The result table's label cut (`short` in the page's script) on emoji,
//! emoji sequences, combining marks and CJK text: it never leaves a lone
//! surrogate, keeps whole characters, and leaves short text alone. The
//! function is read out of the script the server embeds and run under
//! Node, with and without `Intl.Segmenter`; the test is skipped where no
//! `node` is on the path.

use std::process::Command;

const SCRIPT: &str = include_str!("../static/app.js");

/// The source of `function short`, through its closing brace.
fn short_source() -> &'static str {
    let start = SCRIPT.find("function short(").expect("app.js defines short");
    let end = SCRIPT[start..].find("\n}\n").expect("short has a closing brace") + start + 2;
    &SCRIPT[start..end]
}

/// Runs the checks under Node, with `Intl.Segmenter` removed first when
/// `segmenter` is false so the code point fallback is the one tested.
fn run(segmenter: bool) -> Option<String> {
    let checks = r#"
const texts = [
  "Great product 👍 would buy again",
  "x".repeat(17) + "👍👍👍👍👍👍",
  "x".repeat(21) + "😀😀😀😀😀",
  "👍".repeat(40),
  "👨‍👩‍👧 family 👨‍👩‍👧 family 👨‍👩‍👧 family 👨‍👩‍👧",
  "🇯🇵🇺🇸🇫🇷🇩🇪🇬🇧🇮🇹🇪🇸🇨🇦🇧🇷🇲🇽🇰🇷🇨🇳🇮🇳🇦🇺🇳🇿🇿🇦🇳🇬🇪🇬🇸🇪🇳🇴",
  "e\u0301".repeat(30),
  "東京都の天気は晴れのち曇り、明日は雨が降るでしょう。週末は晴れる見込みです。",
  "short",
  "",
];
const fails = [];
for (const t of texts) {
  for (const n of [18, 22, 28]) {
    const got = short(t, n);
    if (!got.isWellFormed()) fails.push(`lone surrogate: ${JSON.stringify(t)} at ${n}`);
    const cut = got.endsWith("…") && got !== t;
    const kept = cut ? got.slice(0, got.length - 1) : got;
    if (!t.startsWith(kept)) fails.push(`not a prefix: ${JSON.stringify(t)} at ${n}`);
    if (Array.from(got).length > Math.max(n, Array.from(t).length)) fails.push(`too long: ${JSON.stringify(t)} at ${n}`);
    if (Array.from(t).length <= n && got !== t) fails.push(`cut short text: ${JSON.stringify(t)} at ${n}`);
  }
}
if (typeof Intl.Segmenter !== "undefined") {
  // With graphemes, a ZWJ family or a flag is kept whole or dropped whole.
  const fam = short("👨‍👩‍👧".repeat(10), 5);
  if (fam !== "👨‍👩‍👧".repeat(4) + "…") fails.push(`family split: ${JSON.stringify(fam)}`);
  const acc = short("e\u0301".repeat(10), 5);
  if (acc !== "e\u0301".repeat(4) + "…") fails.push(`mark split: ${JSON.stringify(acc)}`);
}
console.log(fails.length ? fails.join("\n") : "OK");
"#;
    let prelude = if segmenter { "" } else { "delete Intl.Segmenter;\n" };
    let program = format!("{prelude}{}\n{checks}", short_source());
    let out = Command::new("node").arg("-e").arg(program).output().ok()?;
    assert!(out.status.success(), "node failed: {}", String::from_utf8_lossy(&out.stderr));
    Some(String::from_utf8(out.stdout).unwrap().trim().to_owned())
}

#[test]
fn labels_are_cut_on_whole_characters() {
    for segmenter in [true, false] {
        match run(segmenter) {
            Some(got) => assert_eq!(got, "OK", "Intl.Segmenter {segmenter}"),
            None => eprintln!("no node on the path: the label cut is not checked"),
        }
    }
}
