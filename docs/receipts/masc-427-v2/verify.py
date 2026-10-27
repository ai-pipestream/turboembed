"""Validate the committed execution records without training or model loading."""
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GRID = [
    {"iterations": 300, "cutoff": 0, "clearing": "document"},
    {"iterations": 300, "cutoff": 0, "clearing": "sentence"},
    {"iterations": 100, "cutoff": 5, "clearing": "document"},
    {"iterations": 100, "cutoff": 5, "clearing": "sentence"},
]


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read(kind, name, manifest):
    file = f"{kind}-{name}.json.gz"
    packed = (ROOT / file).read_bytes()
    entry = manifest[file]
    check(len(packed) == entry["size"], f"{file}: compressed size")
    check(hashlib.sha256(packed).hexdigest() == entry["sha256"], f"{file}: compressed hash")
    raw = gzip.decompress(packed)
    check(len(raw) == entry["original_size"], f"{file}: original size")
    check(hashlib.sha256(raw).hexdigest() == entry["original_sha256"], f"{file}: original hash")
    return json.loads(raw)


def verify(kind, manifest, run, inputs):
    selection = read(kind, "selection", manifest)
    card = read(kind, "card", manifest)
    gate = read(kind, "gate", manifest)
    model_id = f"turbo-en-masc-3.0.0-ner-{kind}-2"
    reference_id = f"opennlp-en-ner-{kind}-1.5"
    check(selection["model_id"] == card["id"] == gate["candidate"]["id"] == model_id, f"{kind}: identity")
    check(card["gate"]["reference"] == gate["reference"]["id"] == reference_id, f"{kind}: reference identity")
    reference_hash = inputs[f"en-ner-{kind}.bin"]["sha256"]
    check(card["gate"]["reference_sha256"] == gate["reference"]["sha256"] == reference_hash, f"{kind}: reference input")
    corpus_hash = "445449f535cac62e74b2712ad975d7b727e72b8276720dd27492266565ae0afd"
    check(card["corpus"]["corpus_sha256"] == gate["set"]["corpus_sha256"] == corpus_hash, f"{kind}: corpus identity")
    check(card["corpus"]["release_recorded"] is True, f"{kind}: corpus pin validation")
    check(card["corpus"]["split"] == {"every": 10, "held_out_at": 9}, f"{kind}: held-out split")
    check(len(card["corpus"]["held_out_documents"]) == 39, f"{kind}: held-out document list")
    check(len(card["corpus"]["excluded_documents"]) == 2, f"{kind}: excluded document count")
    check(all(x["exclusion"] == "twitter_genre" for x in card["corpus"]["excluded_documents"]), f"{kind}: exclusion")
    check(selection["training_documents"] == 351, f"{kind}: training documents")
    check(selection["training_samples"] == 24958, f"{kind}: training samples")
    check(selection["folds"] == 10, f"{kind}: fold count")
    candidates = selection["candidates"]
    check([c["setting"] for c in candidates] == GRID, f"{kind}: registered grid")
    for candidate in candidates:
        folds = candidate["folds"]
        check(len(folds) == 10, f"{kind}: candidate fold count")
        seen = []
        merged = dict.fromkeys(("predicted", "reference", "correct"), 0)
        for index, fold in enumerate(folds):
            train, test = fold["training_document_indices"], fold["test_document_indices"]
            check(fold["index"] == index, f"{kind}: fold index")
            check(len(set(train)) == len(train) and len(set(test)) == len(test), f"{kind}: duplicate document")
            check(not set(train) & set(test), f"{kind}: train/test overlap")
            check(set(train) | set(test) == set(range(351)), f"{kind}: missing document")
            check(test == list(range(index, 351, 10)), f"{kind}: changed partition")
            check(fold["training_samples"] + fold["test_samples"] == 24958, f"{kind}: sample count")
            seen.extend(test)
            for key in merged:
                merged[key] += fold["scores"][key]
        check(sorted(seen) == list(range(351)), f"{kind}: document test coverage")
        check(all(merged[k] == candidate["merged"][k] for k in merged), f"{kind}: merged counts")
        f1 = 2 * merged["correct"] / (merged["predicted"] + merged["reference"])
        check(math.isclose(f1, candidate["merged"]["f_measure"], abs_tol=1e-12), f"{kind}: merged F1")
    selected = max(range(4), key=lambda i: candidates[i]["merged"]["f_measure"])
    check(selection["selected"] == selected, f"{kind}: winning configuration")
    check(card["training"]["parameters"]["iterations"] == GRID[selected]["iterations"], f"{kind}: iterations")
    check(card["training"]["parameters"]["cutoff"] == GRID[selected]["cutoff"], f"{kind}: cutoff")
    check(card["held_out"]["names"]["documents"] == 39, f"{kind}: held-out documents")
    check(card["held_out"]["samples"] == len(gate["units"]) == 6384, f"{kind}: held-out samples")
    check(gate["resamples"] == 10000 and gate["seed"] == 154, f"{kind}: bootstrap configuration")
    check(card["file"]["sha256"] == gate["candidate"]["sha256"], f"{kind}: model identity")
    check(run["files"][card["file"]["name"]]["sha256"] == card["file"]["sha256"], f"{kind}: output identity")
    for side in ("candidate", "reference"):
        sums = {key: sum(u[side][key] for u in gate["units"]) for key in ("true_positives", "selected", "target")}
        f1 = 200 * sums["true_positives"] / (sums["selected"] + sums["target"])
        check(math.isclose(f1, gate["result"][side]["score"], abs_tol=1e-10), f"{kind}: held-out F1")
    passed = gate["result"]["difference"]["low"] >= -0.5
    check(gate["result"]["pass"] == card["gate"]["pass"] == passed, f"{kind}: margin decision")
    check(run["exit_status"] == (0 if passed else 1), f"{kind}: process status")
    print(f"{kind}: records verified; original accuracy gate {'PASS' if passed else 'FAIL'}")


def main():
    manifest = json.loads((ROOT / "checksums.json").read_text())
    execution = json.loads((ROOT / "execution.json").read_text())
    inputs = json.loads((ROOT / "inputs.json").read_text())
    check(execution["source_sha"] == "8812b1db16ff47890be43be96d6835eea12306af", "execution source")
    check(execution["binary_sha256"] == "be818fba6d2cc25b4f045b0ad7f55950c146a544d3fe5efdf2d0c6ba4976c664", "binary identity")
    check(execution["profile"] == "release", "build profile")
    check(inputs["MASC-3.0.0.zip"]["sha256"] == "0b300d96016233c7c79b43559b477a6bb0ed16bb416d5c517eccb108a6228493", "archive identity")
    references = {
        "person": "687a9263d96b37fced707c9f2ac0560f9edaf54658856395555901924f64dbe4",
        "location": "8fe39e48633f4a86c4132d9c54b396a2d8e0460c1d71e3562dacf976984f447b",
        "organization": "0136c12afe1ac357142260c39bb879b7c9d121e41024114db5a6455b4fd5ba00",
        "date": "1207030923852e1c244919d8f15d9e78c217323728fcf909029abd1703967855",
    }
    for kind, digest in references.items():
        check(inputs[f"en-ner-{kind}.bin"]["sha256"] == digest, f"{kind}: pinned reference")
    kinds = ["person", "location", "organization", "date"]
    check(execution["types"] == kinds and len(execution["results"]) == 4, "incomplete execution")
    check(len(manifest) == 12, "missing original record")
    for kind, run in zip(kinds, execution["results"]):
        check(run["model_id"] == f"turbo-en-masc-3.0.0-ner-{kind}-2", "changed type order")
        verify(kind, manifest, run, inputs)
    print("All 160 CV fold records and 4 final gate records verified. No training or gate resampling performed.")


if __name__ == "__main__":
    main()
