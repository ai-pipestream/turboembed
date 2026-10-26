#!/usr/bin/env python3
"""Builds the turboembed_proto Python wheel from server/proto/*.proto (issue #1297)."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import pathlib
import re
import shutil
import tempfile
import zipfile

from grpc_tools import protoc

PACKAGE_NAME = "turboembed_proto"
DIST_NAME = "turboembed_proto"

_SEMVER_RE = re.compile(
    r"^v?(?P<core>[0-9]+\.[0-9]+\.[0-9]+)"
    r"(?:-(?P<pre_tag>alpha|a|beta|b|rc|pre|dev)[.-]?(?P<pre_num>[0-9]+))?$"
)
_GRPC_GENERATED_RE = re.compile(r"^GRPC_GENERATED_VERSION = '([0-9]+\.[0-9]+\.[0-9]+)'$", re.M)
_PROTOBUF_GENCODE_RE = re.compile(
    r"ValidateProtobufRuntimeVersion\(\s*_runtime_version\.Domain\.PUBLIC,"
    r"\s*([0-9]+),\s*([0-9]+),\s*([0-9]+),"
)
# grpcio 1.85 dropped Python 3.9; the release job smoke-tests on this floor.
REQUIRES_PYTHON = ">=3.10"

_PRE_MAP = {
    "alpha": "a",
    "a": "a",
    "beta": "b",
    "b": "b",
    "rc": "rc",
    "pre": "rc",
    "dev": ".dev",
}


def normalize_pep440_version(raw: str) -> str:
    m = _SEMVER_RE.match(raw.strip())
    if not m:
        raise ValueError(
            f"unsupported version {raw!r}: expected X.Y.Z or X.Y.Z-(alpha|beta|rc|dev).N"
        )
    core = m.group("core")
    pre_tag = m.group("pre_tag")
    if not pre_tag:
        return core
    return f"{core}{_PRE_MAP[pre_tag]}{m.group('pre_num')}"


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


def _single_version(found: set[str], what: str) -> str:
    if len(found) != 1:
        raise RuntimeError(f"expected one {what} across the generated stubs, found {sorted(found)}")
    return found.pop()


def generated_runtime_floors(pkg_dir: pathlib.Path) -> list[str]:
    """Requires-Dist lines equal to the versions the generated code checks at import.

    The *_grpc.py modules refuse a grpcio older than GRPC_GENERATED_VERSION,
    and the *_pb2.py modules refuse a protobuf runtime older than their
    gencode version or of another major version.
    """
    grpc_found: set[str] = set()
    for py_file in pkg_dir.glob("*_pb2_grpc.py"):
        grpc_found.update(_GRPC_GENERATED_RE.findall(py_file.read_text(encoding="utf-8")))
    pb_found: set[str] = set()
    for py_file in pkg_dir.glob("*_pb2.py"):
        for major, minor, patch in _PROTOBUF_GENCODE_RE.findall(py_file.read_text(encoding="utf-8")):
            pb_found.add(f"{major}.{minor}.{patch}")
    grpc_floor = _single_version(grpc_found, "GRPC_GENERATED_VERSION")
    pb_floor = _single_version(pb_found, "protobuf gencode version")
    pb_next_major = int(pb_floor.split(".")[0]) + 1
    return [f"grpcio>={grpc_floor}", f"protobuf>={pb_floor},<{pb_next_major}"]


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
        requires_dist = "".join(f"Requires-Dist: {req}\n" for req in generated_runtime_floors(pkg_dir))

        init_lines = [
            '"""Generated Protobuf and gRPC Python stubs for turboembed (v1)."""\n',
            f'__version__ = "{pep_version}"\n',
        ]
        (pkg_dir / "__init__.py").write_text("".join(init_lines), encoding="utf-8")

        dist_info = tmp_path / f"{DIST_NAME}-{pep_version}.dist-info"
        dist_info.mkdir(parents=True, exist_ok=True)
        licenses_dir = dist_info / "licenses"
        licenses_dir.mkdir(parents=True, exist_ok=True)
        repo_root = pathlib.Path(__file__).resolve().parents[2]
        for lic_name in ("LICENSE", "NOTICE"):
            lic_src = repo_root / lic_name
            if not lic_src.is_file():
                raise RuntimeError(f"missing repository {lic_name} at {lic_src}")
            shutil.copyfile(lic_src, licenses_dir / lic_name)

        (dist_info / "METADATA").write_text(
            f"Metadata-Version: 2.1\n"
            f"Name: turboembed-proto\n"
            f"Version: {pep_version}\n"
            f"Summary: Generated Protobuf and gRPC Python client stubs for turboembed.\n"
            f"License: Apache-2.0\n"
            f"License-File: licenses/LICENSE\n"
            f"License-File: licenses/NOTICE\n"
            f"Requires-Python: {REQUIRES_PYTHON}\n"
            f"{requires_dist}",
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
