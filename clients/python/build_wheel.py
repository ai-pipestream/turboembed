#!/usr/bin/env python3
"""Builds the turboembed_proto Python wheel from server/proto/*.proto (issue #1297)."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import pathlib
import shutil
import tempfile
import zipfile

from grpc_tools import protoc

PACKAGE_NAME = "turboembed_proto"
DIST_NAME = "turboembed_proto"


def normalize_pep440_version(raw: str) -> str:
    cleaned = raw.lstrip("v")
    parts = cleaned.split(".")
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        return cleaned
    return "0.1.0"


def rewrite_relative_imports(pkg_dir: pathlib.Path, modules: set[str]) -> None:
    for py_file in sorted(pkg_dir.glob("*_pb2*.py*")):
        lines = py_file.read_text(encoding="utf-8").splitlines(keepends=True)
        out: list[str] = []
        for line in lines:
            replaced = line
            for mod in modules:
                if line.startswith(f"import {mod} as "):
                    replaced = f"from . {line}"
                    break
                if line.startswith(f"from {mod} import "):
                    replaced = f"from .{mod} import {line[len(f'from {mod} import '):]}"
                    break
            out.append(replaced)
        py_file.write_text("".join(out), encoding="utf-8")


def urlsafe_b64_nopad(digest: bytes) -> str:
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def build_wheel(proto_dir: pathlib.Path, out_dir: pathlib.Path, version: str) -> pathlib.Path:
    pep_version = normalize_pep440_version(version)
    proto_files = sorted(proto_dir.glob("*.proto"))
    if not proto_files:
        raise RuntimeError(f"no .proto files found in {proto_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    wheel_name = f"{DIST_NAME}-{pep_version}-py3-none-any.whl"
    wheel_path = out_dir / wheel_name

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        pkg_dir = tmp_path / PACKAGE_NAME
        pkg_dir.mkdir(parents=True, exist_ok=True)

        args = [
            "grpc_tools.protoc",
            f"-I{proto_dir}",
            f"--python_out={pkg_dir}",
            f"--pyi_out={pkg_dir}",
            f"--grpc_python_out={pkg_dir}",
            *[str(p) for p in proto_files],
        ]
        rc = protoc.main(args)
        if rc != 0:
            raise RuntimeError(f"grpc_tools.protoc exited with {rc}")

        pb2_modules = {p.stem for p in pkg_dir.glob("*_pb2.py")}
        rewrite_relative_imports(pkg_dir, pb2_modules)

        init_lines = [
            '"""Generated Protobuf and gRPC Python stubs for turboembed (v1)."""\n',
            f'__version__ = "{pep_version}"\n',
        ]
        (pkg_dir / "__init__.py").write_text("".join(init_lines), encoding="utf-8")

        dist_info = tmp_path / f"{DIST_NAME}-{pep_version}.dist-info"
        dist_info.mkdir(parents=True, exist_ok=True)
        (dist_info / "METADATA").write_text(
            f"Metadata-Version: 2.1\n"
            f"Name: turboembed-proto\n"
            f"Version: {pep_version}\n"
            f"Summary: Generated Protobuf and gRPC Python client stubs for turboembed.\n"
            f"License: Apache-2.0\n"
            f"Requires-Python: >=3.9\n"
            f"Requires-Dist: grpcio>=1.68.0\n"
            f"Requires-Dist: protobuf>=5.28.0\n",
            encoding="utf-8",
        )
        (dist_info / "WHEEL").write_text(
            "Wheel-Version: 1.0\n"
            "Generator: turboembed-build-wheel\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n",
            encoding="utf-8",
        )
        (dist_info / "top_level.txt").write_text(f"{PACKAGE_NAME}\n", encoding="utf-8")

        records: list[tuple[str, str, str]] = []
        all_files = sorted(
            p for p in tmp_path.rglob("*") if p.is_file() and p.name != "RECORD"
        )
        with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in all_files:
                rel = file_path.relative_to(tmp_path).as_posix()
                data = file_path.read_bytes()
                digest = urlsafe_b64_nopad(hashlib.sha256(data).digest())
                records.append((rel, f"sha256={digest}", str(len(data))))
                info = zipfile.ZipInfo(rel, date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                zf.writestr(info, data)

            record_rel = f"{dist_info.name}/RECORD"
            records.append((record_rel, "", ""))
            buf = io.StringIO()
            writer = csv.writer(buf, lineterminator="\n")
            writer.writerows(records)
            info = zipfile.ZipInfo(record_rel, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, buf.getvalue().encode("utf-8"))

    return wheel_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proto-dir", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--version", type=str, default="0.1.0")
    args = parser.parse_args()
    whl = build_wheel(args.proto_dir.resolve(), args.out_dir.resolve(), args.version)
    print(whl)


if __name__ == "__main__":
    main()
