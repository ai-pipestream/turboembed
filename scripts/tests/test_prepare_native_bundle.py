import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "prepare-native-bundle.py"


class PrepareNativeBundleTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.onnx = self.write("model.onnx", b"local onnx source")
        self.tokenizer = self.write("source-tokenizer.json", b'{"tokenizer":true}\n')
        self.config = self.write(
            "source-config.json",
            json.dumps(
                {
                    "model_type": "bert",
                    "type_vocab_size": 2,
                    "hidden_size": 384,
                    "vocab_size": 30522,
                    "max_position_embeddings": 512,
                }
            ).encode(),
        )
        self.card = self.write("source-card.md", b"# User supplied model card\n")
        self.exporter = self.root / "fake-exporter.py"
        self.exporter.write_text(
            "#!/usr/bin/env python3\n"
            "import pathlib,sys\n"
            "out=pathlib.Path(sys.argv[2])\n"
            "(out/'openvino_model.xml').write_bytes(b'<model/>')\n"
            "(out/'openvino_model.bin').write_bytes(b'weights')\n"
            "print('2026.3.1-fake')\n"
        )
        self.exporter.chmod(0o755)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, content):
        path = self.root / name
        path.write_bytes(content)
        return path

    def command(self, output):
        return [
            sys.executable,
            str(SCRIPT),
            "--source-onnx", str(self.onnx),
            "--tokenizer", str(self.tokenizer),
            "--config", str(self.config),
            "--model-card", str(self.card),
            "--model-id", "sentence-transformers/all-MiniLM-L6-v2",
            "--revision", "a" * 40,
            "--license", "Apache-2.0",
            "--exporter", str(self.exporter),
            "--output-dir", str(output),
            "--max-sequence", "256",
            "--max-batch", "16",
        ]

    def test_stages_hashes_and_publishes_complete_bundle(self):
        output = self.root / "bundle"
        subprocess.run(self.command(output), check=True)
        self.assertEqual(
            sorted(path.name for path in output.iterdir()),
            ["MODEL_CARD.md", "bundle.json", "config.json", "openvino_model.bin", "openvino_model.xml", "tokenizer.json"],
        )
        manifest = json.loads((output / "bundle.json").read_text())
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["conversion"]["tool_version"], "2026.3.1-fake")
        self.assertEqual(manifest["contract"]["max_batch_size"], 16)
        self.assertEqual(manifest["contract"]["vocab_size"], 30522)
        self.assertEqual(manifest["model"]["source_onnx_sha256"], hashlib.sha256(self.onnx.read_bytes()).hexdigest())
        for name, digest in manifest["files"].items():
            self.assertEqual(digest, hashlib.sha256((output / name).read_bytes()).hexdigest())
        self.assertFalse(any(path.name.startswith(".bundle.stage-") for path in self.root.iterdir()))

    def test_refuses_existing_destination_without_running_exporter(self):
        output = self.root / "bundle"
        output.mkdir()
        marker = output / "keep"
        marker.write_text("unchanged")
        result = subprocess.run(self.command(output), capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(marker.read_text(), "unchanged")
        self.assertFalse((output / "bundle.json").exists())

    def test_refuses_dangling_symlink_destination(self):
        output = self.root / "bundle"
        output.symlink_to(self.root / "missing")
        result = subprocess.run(self.command(output), capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(output.is_symlink())

    def test_export_failure_leaves_no_destination_or_stage(self):
        self.exporter.write_text("#!/usr/bin/env python3\nimport sys\nsys.exit(7)\n")
        output = self.root / "bundle"
        result = subprocess.run(self.command(output), capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(output.exists())
        self.assertFalse(any(path.name.startswith(".bundle.stage-") for path in self.root.iterdir()))

    def test_rejects_contract_before_running_exporter(self):
        for name, flag, value in [
            ("short-sequence", "--max-sequence", "1"),
            ("long-sequence", "--max-sequence", "513"),
            ("large-batch", "--max-batch", "33"),
        ]:
            output = self.root / name
            command = self.command(output)
            command[command.index(flag) + 1] = value
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())

    def test_rejects_untrustworthy_identity_and_config(self):
        cases = []
        bad_revision = self.command(self.root / "bad-revision")
        bad_revision[bad_revision.index("--revision") + 1] = "A" * 40
        cases.append(bad_revision)
        long_id = self.command(self.root / "long-id")
        long_id[long_id.index("--model-id") + 1] = "é" * 64
        cases.append(long_id)
        duplicate = self.write(
            "duplicate-config.json",
            b'{"model_type":"bert","hidden_size":384,"hidden_size":384,"vocab_size":1,"max_position_embeddings":512}',
        )
        duplicate_command = self.command(self.root / "duplicate")
        duplicate_command[duplicate_command.index("--config") + 1] = str(duplicate)
        cases.append(duplicate_command)
        large_vocab = self.write(
            "large-vocab.json",
            b'{"model_type":"bert","hidden_size":384,"vocab_size":2147483648,"max_position_embeddings":512}',
        )
        vocab_command = self.command(self.root / "large-vocab")
        vocab_command[vocab_command.index("--config") + 1] = str(large_vocab)
        cases.append(vocab_command)
        for command in cases:
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertFalse(Path(command[command.index("--output-dir") + 1]).exists())

    def test_detects_source_change_during_export(self):
        self.exporter.write_text(
            "#!/usr/bin/env python3\n"
            "import pathlib,sys\n"
            "source=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2])\n"
            "(out/'openvino_model.xml').write_bytes(b'<model/>')\n"
            "(out/'openvino_model.bin').write_bytes(b'weights')\n"
            "source.write_bytes(b'changed')\n"
            "print('2026.3.1-fake')\n"
        )
        output = self.root / "bundle"
        result = subprocess.run(self.command(output), capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(output.exists())
        self.assertFalse(any(path.name.startswith(".bundle.stage-") for path in self.root.iterdir()))

    def test_manifest_uses_staged_config_after_export(self):
        self.exporter.write_text(
            "#!/usr/bin/env python3\n"
            "import json,pathlib,sys\n"
            "out=pathlib.Path(sys.argv[2])\n"
            "config=json.loads((out/'config.json').read_text()); config['vocab_size']=7\n"
            "(out/'config.json').write_text(json.dumps(config))\n"
            "(out/'openvino_model.xml').write_bytes(b'<model/>')\n"
            "(out/'openvino_model.bin').write_bytes(b'weights')\n"
            "print('2026.3.1-fake')\n"
        )
        output = self.root / "bundle"
        subprocess.run(self.command(output), check=True)
        manifest = json.loads((output / "bundle.json").read_text())
        self.assertEqual(manifest["contract"]["vocab_size"], 7)


if __name__ == "__main__":
    unittest.main()
