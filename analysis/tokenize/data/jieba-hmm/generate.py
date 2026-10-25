"""Pack the pinned jieba BMES probabilities; Python standard library only.

Usage: python3 generate.py DIRECTORY_CONTAINING_PROB_FILES
Download prob_start.py, prob_trans.py and prob_emit.py from jieba/finalseg
at fxsjy/jieba revision 67fa2e36e72f69d9134b8a1037b83fbb070b9775.
The Python inputs are parsed as literals, never executed.
"""
import ast
import hashlib
from pathlib import Path
import struct
import sys

EXPECTED = {
    "start": "14c5706ced5cd3b42eb4873d4b88f7f52a7bdf80fbd767bc4423d361e20c5330",
    "trans": "54dfbc252ed71480d4f0cdfdf516ecfbe44efd0f6c3c64b158e7039f2906c91b",
    "emit": "27d46b1c9efe4dd148fde8be042a21be40e3562d0c7f1273f9de7abae12ebb8d",
}

def read(name):
    data = (Path(sys.argv[1]) / f"prob_{name}.py").read_bytes()
    assert hashlib.sha256(data).hexdigest() == EXPECTED[name], name
    assignments = [node for node in ast.parse(data).body if isinstance(node, ast.Assign)]
    assert len(assignments) == 1
    return ast.literal_eval(assignments[0].value)

start, trans, emit = (read(name) for name in EXPECTED)
states = "BEMS"  # Lexical order also decides ties in jieba's Viterbi.
chars = sorted(set().union(*(values.keys() for values in emit.values())))
assert len(chars) < 65535 and max(map(ord, chars)) < 65536
index = [0] * 65536  # Row zero is an unobserved character.
for row, char in enumerate(chars, 1):
    index[ord(char)] = row
weights = [start[state] if state in "BS" else float("-inf") for state in states]
weights += [trans[a].get(b, float("-inf")) for a in states for b in states]
weights += [-3.14e100] * 4
weights += [emit[state].get(char, -3.14e100) for char in chars for state in states]
# Header: four starts, sixteen transitions. Then the BMP u16 row index,
# then four f64 emissions per row. All numbers are little endian.
out = struct.pack("<20d", *weights[:20]) + struct.pack("<65536H", *index)
out += struct.pack(f"<{len(weights)-20}d", *weights[20:])
dest = Path(__file__).with_name("bmes.bin")
dest.write_bytes(out)
print(len(out), hashlib.sha256(out).hexdigest(), dest.name)
