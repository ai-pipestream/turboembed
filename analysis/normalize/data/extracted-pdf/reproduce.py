#!/usr/bin/env python3
"""Fetch the licensed PDFs, verify their hashes and reproduce the text fixtures.

Usage: python3 reproduce.py OUTPUT_DIRECTORY
Requires Poppler pdftotext 26.07.0 on PATH. Python standard library only.
Never rewrites labels or checked-in fixtures.
"""

import hashlib
from pathlib import Path
import subprocess
import sys
import urllib.request

SOURCES = {
    "P19-1423": (
        "f9ee341ae788143c48a5d424812ebd40c8f1312b7eaffcbd52c0aa3fdb79f63c",
        "4a4c955ebcaac4f100a3eee0d9755b0febaa72394695e172f422f60715956f1c",
    ),
    "2020.acl-main.1": (
        "bd2680310de89488ca086aa8378ed095ecf5e4b56588812f62b11667de66d70e",
        "61d8ea51cba932c21984731f870eb08efa34e280b91d20c81487577a5d8c86c8",
    ),
}


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def escaped(text):
    return "".join(
        "\\\\" if c == "\\" else f"\\u{ord(c):04x}"
        if ord(c) < 32 or ord(c) == 0x2014 else c
        for c in text
    ) + "\n"


def main():
    require(len(sys.argv) == 2, "Usage: reproduce.py OUTPUT_DIRECTORY")
    version = subprocess.run(["pdftotext", "-v"], capture_output=True, text=True, check=True)
    require("pdftotext version 26.07.0" in version.stderr, "Use Poppler pdftotext 26.07.0")
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    for name, (pdf_hash, text_hash) in SOURCES.items():
        request = urllib.request.Request(
            f"https://aclanthology.org/{name}.pdf", headers={"User-Agent": "turboembed-evaluation"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            pdf = response.read()
        require(hashlib.sha256(pdf).hexdigest() == pdf_hash, f"PDF changed: {name}")
        source, extracted = output / f"{name}.pdf", output / f"{name}.txt"
        source.write_bytes(pdf)
        subprocess.run(
            ["pdftotext", "-layout", "-f", "2", "-l", "3", "-enc", "UTF-8", str(source), str(extracted)],
            check=True,
        )
        raw = extracted.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == text_hash, f"Extraction changed: {name}")
        encoded = escaped(raw.decode("utf-8")).encode("utf-8")
        expected = Path(__file__).with_name(f"{name}.escaped").read_bytes()
        require(encoded == expected, f"Fixture differs: {name}")
        print(f"{name}: PDF, extracted text and escaped fixture match")


if __name__ == "__main__":
    main()
