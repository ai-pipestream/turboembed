#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export the host-side MiniLM embedding tables for the Hailo lane.

The public MiniLM HEFs (Hailo Model Zoo, hailo8/hailo8l) run only the
transformer encoder body on the NPU. Tokenization, the embedding lookup,
the embeddings LayerNorm, pooling, and L2 normalization run on the host
(native/turboembed/src/hailo.cpp). This one-off script extracts those
host-side weights from the pinned Hugging Face checkpoint and writes:

  <model-dir>/embedding_tables.bin  — fp32 tables, "TEMB" v1 header
                                      (format documented in hailo.cpp)
  <model-dir>/config.json           — dim/seq_len/pooling/normalize contract

Pure stdlib (safetensors is a trivial header+json+raw format): works on a
stock Raspberry Pi OS python3 with no pip.

Usage:
  python3 scripts/export-minilm-hailo-tables.py [models/hailo/minilm] \
      [--seq-len 128] [--hf-repo sentence-transformers/all-MiniLM-L6-v2 \
       --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41]
"""

import argparse
import hashlib
import json
import struct
import sys
import tempfile
import urllib.request
from pathlib import Path

MAGIC = 0x54454D42  # 'TEMB'
VERSION = 1

REPO = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

KEYS = {
    "word": "embeddings.word_embeddings.weight",
    "position": "embeddings.position_embeddings.weight",
    "token_type": "embeddings.token_type_embeddings.weight",
    "ln_gamma": "embeddings.LayerNorm.weight",
    "ln_beta": "embeddings.LayerNorm.bias",
}


def hf_download(url: str) -> Path:
    tmp = Path(tempfile.mkstemp(prefix="hailo-export-")[1])
    print(f"  downloading {url}")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    return tmp


def load_safetensors_f32(path: Path) -> dict:
    """name -> (shape, raw LE fp32 bytes). Only F32 tensors are supported."""
    data = path.read_bytes()
    (header_len,) = struct.unpack_from("<Q", data, 0)
    header = json.loads(data[8 : 8 + header_len])
    base = 8 + header_len
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        if meta["dtype"] != "F32":
            continue
        start, end = meta["data_offsets"]
        out[name] = (meta["shape"], data[base + start : base + end])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", nargs="?", default="models/hailo/minilm")
    parser.add_argument("--hf-repo", default=REPO)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="sequence length the HEF was compiled for",
    )
    args = parser.parse_args()

    base = f"https://huggingface.co/{args.hf_repo}/resolve/{args.revision}"
    st_path = hf_download(f"{base}/model.safetensors")
    cfg_path = hf_download(f"{base}/config.json")

    tensors = load_safetensors_f32(st_path)
    missing = [name for name, key in KEYS.items() if key not in tensors]
    if missing:
        have = ", ".join(sorted(tensors.keys())[:8])
        print(f"error: missing tensors {missing}; have: {have}...", file=sys.stderr)
        return 1

    word_shape, word = tensors[KEYS["word"]]
    pos_shape, pos = tensors[KEYS["position"]]
    tt_shape, tt = tensors[KEYS["token_type"]]
    gamma_shape, gamma = tensors[KEYS["ln_gamma"]]
    beta_shape, beta = tensors[KEYS["ln_beta"]]

    vocab_rows, dim = word_shape
    max_pos = pos_shape[0]
    expected = [
        len(word) == vocab_rows * dim * 4,
        pos_shape[1] == dim and len(pos) == max_pos * dim * 4,
        tt_shape == [2, dim] and len(tt) == 2 * dim * 4,
        gamma_shape == [dim] and len(gamma) == dim * 4,
        beta_shape == [dim] and len(beta) == dim * 4,
    ]
    if not all(expected):
        print(
            "error: unexpected tensor shapes: "
            f"word{word_shape} pos{pos_shape} tt{tt_shape} "
            f"gamma{gamma_shape} beta{beta_shape}",
            file=sys.stderr,
        )
        return 1

    with open(cfg_path) as f:
        hf_cfg = json.load(f)
    eps = float(hf_cfg.get("layer_norm_eps", 1e-12))

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    tables = model_dir / "embedding_tables.bin"
    with open(tables, "wb") as out:
        out.write(struct.pack("<5If", MAGIC, VERSION, vocab_rows, max_pos, dim, eps))
        for blob in (word, pos, tt, gamma, beta):
            out.write(blob)

    config = {
        "dim": dim,
        "seq_len": args.seq_len,
        "pooling": "mean",
        "normalize": True,
        "layer_norm_eps": eps,
        "front_end": "word",
        "front_end_note": "official Hailo Model Zoo HEFs take the raw word-embedding rows "
        "(vocab gather, PAD id on padding) and an additive [-10000] attention mask; "
        "position/token-type/LayerNorm are inside the HEF",
        "source": f"{args.hf_repo}@{args.revision}",
        "note": "HEF input is the word-embedding grid; see docs/hailo-embed.md",
    }
    cfg_out = model_dir / "config.json"
    cfg_out.write_text(json.dumps(config, indent=2) + "\n")

    for path in (tables, cfg_out):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"  wrote {path}  sha256={digest}  size={path.stat().st_size}")
    print(f"embedding tables exported: vocab={vocab_rows} max_pos={max_pos} dim={dim}")
    print(f"next: scripts/hailo-select-hef.sh {model_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
