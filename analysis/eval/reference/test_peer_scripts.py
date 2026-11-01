"""Tests for the peer reference scripts (run: python3 -m unittest
analysis.eval.reference.test_peer_scripts from the repo root is not
a package path; run this file directly: python3
analysis/eval/reference/test_peer_scripts.py).

Covers the acceptance gate on the GLiNER checkpoint fetch, the
FLORES short-text cut over awkward whitespace (through the
converter end to end), and the throughput receipt's per-run output
digests and working-tree provenance.
"""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
CONVERT = os.path.join(HERE, "peers", "flores", "convert_gold.py")
FETCH = os.path.join(HERE, "peers", "gliner", "fetch_checkpoint.py")
GLOTLID_FETCH = os.path.join(HERE, "peers", "glotlid", "fetch_model.py")
GLOTLID_PREDICT = os.path.join(HERE, "peers", "glotlid", "lang_predict.py")
RECEIPT = os.path.join(HERE, "throughput_receipt.py")


def load_convert():
    spec = importlib.util.spec_from_file_location("convert_gold", CONVERT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(argv, env=None, cwd=None):
    return subprocess.run(argv, capture_output=True, text=True, env=env, cwd=cwd)


class ShortOf(unittest.TestCase):
    def test_whitespace(self):
        short_of = load_convert().short_of
        self.assertEqual(short_of("alpha beta gamma delta epsilon"), "alpha beta gamma delta")
        # Leading and repeated spaces make no empty words.
        self.assertEqual(short_of("    alpha beta gamma delta epsilon"), "alpha beta gamma delta")
        self.assertEqual(short_of("alpha  beta   gamma    delta epsilon"), "alpha beta gamma delta")
        # Nonbreaking and ideographic spaces are separators too.
        self.assertEqual(short_of("alpha\u00a0beta\u3000gamma delta epsilon"), "alpha beta gamma delta")
        # Fewer than four words: the character fallback, never empty.
        self.assertEqual(short_of("  one two"), "one two")
        self.assertEqual(len(short_of("一二三四五六七八九十一二三四五六七八九十一")), 20)


class ConverterEndToEnd(unittest.TestCase):
    def test_short_lines_are_scorable(self):
        convert = load_convert()
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "devtest")
            os.mkdir(src)
            for name in convert.FILES.values():
                text = "    alpha  beta\u00a0gamma delta epsilon zeta"
                row = {"id": 1, "split": "devtest", "text": text}
                with open(os.path.join(src, name), "w", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out = os.path.join(tmp, "full.tsv")
            short = os.path.join(tmp, "short.tsv")
            done = run([sys.executable, CONVERT, src, out, "--short-out", short])
            self.assertEqual(done.returncode, 0, done.stderr)
            with open(short, encoding="utf-8") as f:
                lines = f.read().splitlines()
            self.assertEqual(len(lines), len(convert.FILES))
            for line in lines:
                code, _, text = line.partition("\t")
                # The scorer refuses a line whose text is empty.
                self.assertTrue(text.strip(), line)
                self.assertEqual(text, "alpha beta gamma delta")


class FetchGate(unittest.TestCase):
    def fetch(self, extra_env):
        env = {k: v for k, v in os.environ.items() if k != "TURBO_ACCEPT_TERMS"}
        env.update(extra_env)
        return run(
            [sys.executable, FETCH, "urchade/gliner_small-v2.1", "rev", "/nonexistent-dir"],
            env=env,
        )

    def test_unaccepted_refuses_before_download(self):
        done = self.fetch({})
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("refusing to download", done.stderr + done.stdout)

    def test_wrong_acceptance_refuses(self):
        done = self.fetch({"TURBO_ACCEPT_TERMS": "yes"})
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("refusing to download", done.stderr + done.stdout)


class GlotLidFetchGate(unittest.TestCase):
    def fetch(self, extra_env, pythonpath=None):
        env = {k: v for k, v in os.environ.items() if k != "TURBO_ACCEPT_TERMS"}
        env.update(extra_env)
        if pythonpath is not None:
            env["PYTHONPATH"] = pythonpath
        return run(
            [sys.executable, GLOTLID_FETCH, "cis-lmu/glotlid", "rev", "0" * 64, "/nonexistent-dir"],
            env=env,
        )

    def test_unaccepted_refuses_before_download(self):
        done = self.fetch({})
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("refusing to download", done.stderr + done.stdout)

    def test_wrong_acceptance_refuses(self):
        done = self.fetch({"TURBO_ACCEPT_TERMS": "yes"})
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("refusing to download", done.stderr + done.stdout)

    def test_wrong_digest_is_refused_and_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            stub = os.path.join(tmp, "huggingface_hub.py")
            with open(stub, "w", encoding="utf-8") as f:
                f.write(
                    "import os\n"
                    "def hf_hub_download(checkpoint, filename, revision, local_dir):\n"
                    "    os.makedirs(local_dir, exist_ok=True)\n"
                    "    path = os.path.join(local_dir, filename)\n"
                    "    open(path, 'wb').write(b'wrong bytes')\n"
                    "    return path\n"
                )
            model_dir = os.path.join(tmp, "model")
            env = {"TURBO_ACCEPT_TERMS": "cis-lmu/glotlid"}
            env["PYTHONPATH"] = tmp
            done = run(
                [sys.executable, GLOTLID_FETCH, "cis-lmu/glotlid", "rev", "0" * 64, model_dir],
                env=env,
            )
            self.assertNotEqual(done.returncode, 0)
            self.assertIn("expected", done.stderr + done.stdout)
            self.assertFalse(os.path.exists(os.path.join(model_dir, "model.bin")))


class GlotLidPredict(unittest.TestCase):
    def test_script_and_macro_mapping_and_no_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            stub = os.path.join(tmp, "fasttext.py")
            with open(stub, "w", encoding="utf-8") as f:
                f.write(
                    "class Model:\n"
                    "    def predict(self, text, k):\n"
                    "        rows = {\n"
                    "            'Chinese': (['__label__cmn_Hans'], [0.75]),\n"
                    "            'Persian': (['__label__pes_Arab'], [0.5]),\n"
                    "            'Empty': ([], []),\n"
                    "        }\n"
                    "        return rows[text]\n"
                    "def load_model(path):\n"
                    "    return Model()\n"
                )
            gold = os.path.join(tmp, "gold.tsv")
            out = os.path.join(tmp, "out.txt")
            with open(gold, "w", encoding="utf-8") as f:
                f.write("zho\tChinese\npes\tPersian\neng\tEmpty\n")
            env = dict(os.environ, PYTHONPATH=tmp)
            done = run([sys.executable, GLOTLID_PREDICT, "unused", gold, out], env=env)
            self.assertEqual(done.returncode, 0, done.stderr)
            with open(out, encoding="utf-8") as f:
                self.assertEqual(f.read(), "zho\t0.750000\nfas\t0.500000\nund\t0\n")


STUB = """import sys
counter_path, out_path = sys.argv[1], sys.argv[2]
try:
    n = int(open(counter_path).read())
except OSError:
    n = 0
n += 1
open(counter_path, "w").write(str(n))
open(out_path, "w").write(f"run {n}\\n")
"""


class ThroughputReceipt(unittest.TestCase):
    def receipt(self, cwd, tmp):
        counter = os.path.join(tmp, "counter")
        produced = os.path.join(tmp, "produced.txt")
        stub = os.path.join(tmp, "stub.py")
        with open(stub, "w", encoding="utf-8") as f:
            f.write(STUB)
        out = os.path.join(tmp, "receipt.json")
        done = run(
            [
                sys.executable, RECEIPT, out, "10",
                "--runs", "3", "--output", produced,
                "--", sys.executable, stub, counter, produced,
            ],
            cwd=cwd,
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        with open(out, encoding="utf-8") as f:
            return json.load(f)

    def test_per_run_output_hashes_show_difference(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt = self.receipt(tmp, tmp)
            hashes = [r["output_sha256"] for r in receipt["runs"]]
            # The stub writes different bytes each run; the receipt
            # must show that instead of one surviving final file.
            self.assertEqual(len(set(hashes)), 3)
            self.assertIsNone(receipt["commit"])
            self.assertFalse(receipt["working_tree"]["dirty"])

    def test_dirty_tree_is_disclosed(self):
        with tempfile.TemporaryDirectory() as tmp:
            run(["git", "init", "-q"], cwd=tmp)
            tracked = os.path.join(tmp, "tracked.txt")
            with open(tracked, "w", encoding="utf-8") as f:
                f.write("base\n")
            run(["git", "add", "tracked.txt"], cwd=tmp)
            env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
                       GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
            subprocess.run(["git", "commit", "-qm", "base"], cwd=tmp, env=env, check=True)
            with open(tracked, "a", encoding="utf-8") as f:
                f.write("uncommitted\n")
            receipt = self.receipt(tmp, tmp)
            head = run(["git", "rev-parse", "HEAD"], cwd=tmp).stdout.strip()
            self.assertEqual(receipt["commit"], head)
            self.assertTrue(receipt["working_tree"]["dirty"])
            self.assertEqual(len(receipt["working_tree"]["diff_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
