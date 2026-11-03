"""Exercise the exporter with real spaCy tokens; no model download needed.

Run with the pinned peer's spaCy 3.8.7: python test_full_predict.py.
"""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import spacy
from spacy.tokens import Doc

import full_predict


class FullPredict(unittest.TestCase):
    def test_root_and_dependent_heads_in_exported_sentences(self):
        nlp = spacy.blank("en")
        # Use real Token wrappers and known parses, independent of model weights.
        docs = [
            Doc(nlp.vocab, words=["Alice", "left"], heads=[1, 1], deps=["nsubj", "ROOT"]),
            Doc(nlp.vocab, words=["Go", "."], heads=[0, 0], deps=["ROOT", "punct"]),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "words.conllu", Path(tmp) / "full.conllu"
            src.write_text(
                "1\tAlice\t_\t_\t_\t_\t_\t_\t_\t_\n"
                "2\tleft\t_\t_\t_\t_\t_\t_\t_\t_\n\n"
                "1\tGo\t_\t_\t_\t_\t_\t_\t_\t_\n"
                "2\t.\t_\t_\t_\t_\t_\t_\t_\t_\n\n",
                encoding="utf-8",
            )

            def pipe(inputs, batch_size):
                self.assertEqual([[t.text for t in d] for d in inputs],
                                 [["Alice", "left"], ["Go", "."]])
                return iter(docs)

            with patch.object(full_predict.spacy, "load", return_value=nlp), \
                    patch.object(nlp, "pipe", side_effect=pipe), \
                    patch.object(full_predict.sys, "argv", ["full_predict.py", str(src), str(out)]), \
                    contextlib.redirect_stderr(io.StringIO()):
                full_predict.main()

            rows = [line.split("\t") for line in out.read_text(encoding="utf-8").splitlines()
                    if line and not line.startswith("#")]
            self.assertEqual([row[6] for row in rows], ["2", "0", "0", "1"])
            self.assertEqual([row[7] for row in rows], ["nsubj", "ROOT", "ROOT", "punct"])
            self.assertEqual([row[0] for row in rows], ["1", "2", "1", "2"])


if __name__ == "__main__":
    unittest.main()
