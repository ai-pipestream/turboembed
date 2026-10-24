"""Measure HailoRT on verified bundle inputs, without Python packages.

This is kernel-throughput evidence, not a capability record or a latency
percentile. The library measurement and conformance must be supplied separately.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import struct
import subprocess


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verified(root, manifest, name):
    entry = next(f for f in manifest["files"] if f["path"] == name)
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("bundle path leaves its directory")
    data = path.read_bytes()
    if len(data) != entry["size"] or digest(data) != entry["sha256"]:
        raise ValueError("bundle file failed integrity check: " + name)
    return data


def tensor(data, name, dtype):
    length = struct.unpack_from("<Q", data)[0]
    header = json.loads(data[8:8 + length])
    entry = header[name]
    if entry["dtype"] != dtype:
        raise ValueError("unexpected tensor dtype: " + name)
    start, end = entry["data_offsets"]
    result = memoryview(data)[8 + length + start:8 + length + end]
    if start < 0 or end < start or len(result) != math.prod(entry["shape"]) * 4:
        raise ValueError("invalid tensor shape or offsets: " + name)
    return entry["shape"], result


def prepare(root, target, batch, work):
    manifest_bytes = (root / "manifest.json").read_bytes()
    m = json.loads(manifest_bytes)
    hef = next(a for a in m["artifacts"] if a["format"] == "FORMAT_HEF" and a["target"] == target)
    seq = hef["fixed_seq"]
    if hef["graph_input"] != "INPUT_EMBEDDINGS" or hef["fixed_batch"] != 1:
        raise ValueError("only single-frame embedding-input HEFs are measured")
    host = next(a for a in m["artifacts"] if a["name"] == hef["host_weights"])
    if len(host["files"]) != 1 or len(hef["files"]) != 1:
        raise ValueError("expected one host-weight file and one HEF")
    weights = verified(root, m, host["files"][0])
    shape, words = tensor(weights, host["tensor_names"]["word_embeddings"], "F32")
    vocab, hidden = shape
    heads = m["architecture"]["heads"]
    ref = verified(root, m, m["reference"]["file"])
    dims, ids_data = tensor(ref, "ids", "I32")
    _, lengths_data = tensor(ref, "lengths", "I32")
    lengths = [v[0] for v in struct.iter_unpack("<i", lengths_data)]
    fitting = [i for i, length in enumerate(lengths) if 0 < length <= seq]
    if batch < len(fitting) or not fitting or batch > m["embed"]["max_batch"]:
        raise ValueError("batch must cover every fitting case and fit the bundle")
    pad = next(s["id"] for s in m["tokenizer"]["special_tokens"] if s["role"] == "SPECIAL_PAD")
    ids, masks, cases = [], [], []
    for row in range(batch):
        case = fitting[row % len(fitting)]
        n = lengths[case]
        start = case * dims[1] * 4
        values = [v[0] for v in struct.iter_unpack("<i", ids_data[start:start + n * 4])]
        if len(values) != n or any(i < 0 or i >= vocab for i in values + [pad]):
            raise ValueError("reference token outside vocabulary")
        ids.extend(values + [pad] * (seq - n))
        masks.extend([1] * n + [0] * (seq - n))
        cases.append(case)
    word_path, bias_path = work / "words.f32", work / "bias.f32"
    with word_path.open("wb") as output:
        for token in ids:
            output.write(words[token * hidden * 4:(token + 1) * hidden * 4])
    with bias_path.open("wb") as output:
        for row in range(batch):
            keys = b"".join(struct.pack("<f", 0.0 if v else -100.0) for v in masks[row * seq:(row + 1) * seq])
            output.write(keys * heads * seq)
    row_bytes = b"turbo-bench rows 1\0" + struct.pack("<II", batch, seq)
    row_bytes += b"".join(struct.pack("<i", v) for v in ids + masks + [0] * len(ids))
    hef_data = verified(root, m, hef["files"][0])
    # A verified copy prevents a changed source HEF from being run later.
    hef_path = work / "model.hef"
    hef_path.write_bytes(hef_data)
    receipt = {"manifest_sha256": digest(manifest_bytes), "hef_sha256": digest(hef_data),
               "word_input_sha256": digest(word_path.read_bytes()), "bias_input_sha256": digest(bias_path.read_bytes()),
               "rows_sha256": digest(row_bytes), "batch": batch, "seq": seq,
               "cases": cases, "live_tokens": sum(masks), "computed_tokens_per_pass": batch * seq}
    return receipt, hef_path, word_path, bias_path


def output(command, work):
    result = subprocess.run(command, cwd=work, capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError("vendor command failed with exit " + str(result.returncode) + ": " + result.stderr[-2000:])
    return result.stdout


def fps_from_text(text):
    values = []
    for line in text.splitlines():
        if line.strip().startswith("FPS:"):
            values.append(float(line.split(":", 1)[1].strip()))
    if len(values) != 1 or not math.isfinite(values[0]) or values[0] <= 0:
        raise ValueError("expected one positive FPS result from one network")
    return values[0]


def run(args):
    args.work.mkdir(parents=True, exist_ok=False)
    work = args.work.resolve()
    cli = Path(shutil.which(args.hailortcli) or args.hailortcli).resolve(strict=True)
    version = output([str(cli), "--version"], work).strip()
    if (args.target == "hailo8" and "version 4." not in version) or (args.target == "hailo10h" and "version 5." not in version):
        raise ValueError("unverified CLI major version for target")
    receipt, hef, words, bias = prepare(args.bundle, args.target, args.batch, work)
    info = output([str(cli), "parse-hef", str(hef)], work)
    inputs = [line.split()[1] for line in info.splitlines() if line.strip().startswith("Input ")]
    names = []
    for suffix in ["/input_layer1", "/input_layer2"]:
        found = [name for name in inputs if name.endswith(suffix)]
        if len(found) != 1:
            raise ValueError("HEF does not have the expected input names")
        names.append(found[0])
    if len(inputs) != 2:
        raise ValueError("HEF has additional inputs")
    if args.target == "hailo8":
        command = [str(cli), "run", str(hef), "--time-to-run", str(args.seconds), "--batch-size", str(args.batch),
                   "--quantized", "false", "--user-format-type", "float32", "--input-files",
                   names[0] + "=" + str(words), names[1] + "=" + str(bias)]
    else:
        command = [str(cli), "run2", "--time-to-run", str(args.seconds), "--mode", "full_async", "--json", str(work / "stats.json"),
                   "set-net", str(hef), "--batch-size", str(args.batch),
                   "set-vstream", names[0], "--type", "float32", "--input-file", str(words),
                   "set-vstream", names[1], "--type", "float32", "--input-file", str(bias)]
    # One untimed process followed by independent measured processes.
    rates = []
    for i in range(args.runs + 1):
        text = output(command, work)
        (work / f"run-{i}.txt").write_text(text)
        if args.target == "hailo8":
            fps = fps_from_text(text)
        else:
            report = json.loads((work / "stats.json").read_text())
            groups = report["network_groups"]
            if len(groups) != 1:
                raise ValueError("expected one measured network")
            fps = float(groups[0]["FPS"])
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError("invalid measured FPS")
            (work / f"run-{i}.json").write_text(json.dumps(report, indent=2))
        if i:
            rates.append(fps)
    sanitized = [a.replace(str(cli), "<hailortcli>").replace(str(work), "<inputs>") for a in command]
    receipt.update({"target": args.target, "cli_version": version, "cli_sha256": digest(cli.read_bytes()),
                    "command": sanitized, "commands": [["<hailortcli>", "--version"], ["<hailortcli>", "parse-hef", "<inputs>/model.hef"]] + [sanitized] * (args.runs + 1), "warmup_processes": 1, "seconds_per_process": args.seconds,
                    "fps_samples": rates, "median_fps": statistics.median(rates),
                    "median_pass_ms": statistics.median(args.batch * 1000 / f for f in rates),
                    "procedure": "Kernel throughput with host F32 input conversion; reciprocal FPS times batch. Not request latency or a p99 measurement. Inputs cycle the same mixed cases as turbo-bench. No tokenization, host gather, pooling or normalization is timed.",
                    "capability_record": False})
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--target", choices=["hailo8", "hailo10h"], required=True)
    parser.add_argument("--work", type=Path, required=True, help="new scratch directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hailortcli", default="hailortcli")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seconds", type=int, default=5)
    args = parser.parse_args()
    if min(args.batch, args.runs, args.seconds) < 1:
        parser.error("batch, runs and seconds must be positive")
    run(args)
