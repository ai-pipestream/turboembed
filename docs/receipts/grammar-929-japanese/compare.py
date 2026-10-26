"""Bounded alternating comparison. Run from either checkout; outputs stay local."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

baseline, candidate, main_sha, candidate_sha, output = sys.argv[1:]
binaries = {"main": Path(baseline).resolve(), "candidate": Path(candidate).resolve()}
output = Path(output)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load():
    return [float(x) for x in Path("/proc/loadavg").read_text().split()[:3]]

def run(label, proof=None):
    args = [str(binaries[label])] + ([str(proof)] if proof else [])
    return subprocess.check_output(args, text=True)

proof_dir = Path("target/proof")
proof_dir.mkdir(parents=True, exist_ok=True)
proofs = {}
for label in binaries:
    proof = proof_dir / f"{output.stem}-{label}-english-output.txt"
    run(label, proof)
    proofs[label] = sha(proof)
assert proofs["main"] == proofs["candidate"], proofs
rows = []
for pair in range(8):
    order = ["main", "candidate"] if pair % 2 == 0 else ["candidate", "main"]
    for label in order:
        before = load()
        stdout = run(label)
        rows.append(dict(pair=pair + 1, binary=label, load_before=before, load_after=load(), stdout=stdout))
summary = {}
for grammar in ["en", "en-units"]:
    values = {label: [int(line.rsplit(" ns=", 1)[1]) for row in rows if row["binary"] == label
                     for line in row["stdout"].splitlines() if line.startswith(grammar + " ")]
              for label in binaries}
    summary[grammar] = {label: statistics.median(v) for label, v in values.items()}
    summary[grammar]["candidate_main_ratio"] = summary[grammar]["candidate"] / summary[grammar]["main"]
    summary[grammar]["median_paired_ratio"] = statistics.median(c / m for c, m in zip(values["candidate"], values["main"]))
receipt = dict(main_sha=main_sha, candidate_sha=candidate_sha,
               rustc=subprocess.check_output(["rustc", "--version"], text=True).strip(),
               profile="cargo build --release; default release optimization; jobs=2",
               rustflags_override=bool(os.getenv("RUSTFLAGS") or os.getenv("CARGO_ENCODED_RUSTFLAGS")),
               cpu=next(x.split(":", 1)[1].strip() for x in Path("/proc/cpuinfo").read_text().splitlines()
                        if x.startswith("model name")), logical_cpus=os.cpu_count(),
               source_sha256=sha(Path(__file__).with_name("existing_path.rs")),
               sample_sha256=sha(Path("analysis/grammar/tests/common/sample.rs")),
               binary_sha256={k: sha(v) for k, v in binaries.items()}, output_sha256=proofs,
               protocol="eight alternating-order pairs; three warm iterations then twenty parses and all four output selections per grammar; shared host, no isolation claim",
               summary_median_ns=summary, rows=rows)
output.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(summary, indent=2))
