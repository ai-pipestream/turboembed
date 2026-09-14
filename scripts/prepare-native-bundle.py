#!/usr/bin/env python3
"""Prepare a verified TurboEmbed native model bundle from local source files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

FILES = ("openvino_model.xml", "openvino_model.bin", "tokenizer.json", "config.json", "MODEL_CARD.md")
INT32_MAX = 2**31 - 1
REVISION = re.compile(r"[0-9a-f]{40}\Z")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def required_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"{label} is not a regular file: {path}")
    return path


def config_contract(path: Path, max_sequence: int, max_batch: int) -> tuple[int, int]:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key in config.json: {key}")
            result[key] = value
        return result

    try:
        config = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid config.json: {error}") from error
    if not isinstance(config, dict) or config.get("model_type") != "bert":
        raise ValueError("config.json model_type must be 'bert'")
    hidden = config.get("hidden_size")
    vocab = config.get("vocab_size")
    model_max = config.get("max_position_embeddings")
    if type(hidden) is not int or hidden != 384:
        raise ValueError("config.json hidden_size must be 384")
    if type(vocab) is not int or not 1 <= vocab <= INT32_MAX:
        raise ValueError("config.json vocab_size must be between 1 and INT32_MAX")
    if type(model_max) is not int or not 2 <= model_max <= INT32_MAX:
        raise ValueError("config.json max_position_embeddings must be between 2 and INT32_MAX")
    if type(config.get("type_vocab_size")) is not int or config["type_vocab_size"] != 2:
        raise ValueError("config.json type_vocab_size must be 2")
    if not 2 <= max_sequence <= 512:
        raise ValueError("max-sequence must be between 2 and 512")
    if max_sequence > model_max:
        raise ValueError("max-sequence exceeds config.json max_position_embeddings")
    if not 1 <= max_batch <= 32:
        raise ValueError("max-batch must be between 1 and 32")
    return vocab, model_max


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-onnx", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path, metavar="TOKENIZER_JSON")
    parser.add_argument("--config", required=True, type=Path, metavar="CONFIG_JSON")
    parser.add_argument("--model-card", required=True, type=Path, metavar="MODEL_CARD_MD")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--license", required=True, dest="license_id", metavar="SPDX_ID")
    parser.add_argument("--exporter", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-sequence", type=positive_int, default=256)
    parser.add_argument("--max-batch", type=positive_int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(os.path.abspath(args.output_dir))
    if os.path.lexists(output):
        raise SystemExit(f"destination already exists: {output}")
    if "\0" in args.model_id or not 0 < len(args.model_id.encode("utf-8")) < 128:
        raise SystemExit("model-id must be non-empty, contain no NUL, and be under 128 UTF-8 bytes")
    if not REVISION.fullmatch(args.revision):
        raise SystemExit("revision must be exactly 40 lowercase hexadecimal characters")
    if not args.license_id:
        raise SystemExit("license must be non-empty")
    try:
        source = required_file(args.source_onnx, "source ONNX")
        tokenizer = required_file(args.tokenizer, "tokenizer.json")
        config = required_file(args.config, "config.json")
        model_card = required_file(args.model_card, "MODEL_CARD.md")
        exporter = required_file(args.exporter, "exporter")
        config_contract(config, args.max_sequence, args.max_batch)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    output.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(source)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        for src, name in ((tokenizer, "tokenizer.json"), (config, "config.json"), (model_card, "MODEL_CARD.md")):
            shutil.copyfile(src, stage / name)
        vocab_size, _ = config_contract(stage / "config.json", args.max_sequence, args.max_batch)
        completed = subprocess.run(
            [str(exporter), str(source), str(stage)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        tool_version = completed.stdout.strip()
        if not tool_version or "\n" in tool_version:
            raise RuntimeError("exporter must print exactly one non-empty version line")
        if sha256(source) != source_hash:
            raise RuntimeError("source ONNX changed while exporter was running")
        vocab_size, _ = config_contract(stage / "config.json", args.max_sequence, args.max_batch)
        for name in FILES:
            if not (stage / name).is_file():
                raise RuntimeError(f"bundle artifact missing after export: {name}")
        manifest = {
            "schema_version": 1,
            "model": {
                "id": args.model_id,
                "revision": args.revision,
                "license": args.license_id,
                "source_onnx_sha256": source_hash,
            },
            "conversion": {
                "tool": "turboembed-export-model",
                "tool_version": tool_version,
                "command": "turboembed-export-model <SOURCE_ONNX> <OUT_DIR>",
            },
            "contract": {
                "pooling": "mean",
                "normalize": True,
                "precision": "f32",
                "dimension": 384,
                "vocab_size": vocab_size,
                "max_sequence_length": args.max_sequence,
                "max_batch_size": args.max_batch,
                "query_prefix": "",
                "document_prefix": "",
            },
            "files": {name: sha256(stage / name) for name in FILES},
        }
        (stage / "bundle.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if os.path.lexists(output):
            raise FileExistsError(f"destination already exists: {output}")
        os.rename(stage, output)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        raise SystemExit(f"bundle preparation failed: {error}") from error
    finally:
        if stage.exists():
            shutil.rmtree(stage)


if __name__ == "__main__":
    main()
