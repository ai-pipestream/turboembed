#!/usr/bin/env python3
"""Builds the cp310-abi3 platform wheel for turboembed-native bundling libturbo (#1469)."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import os
import pathlib
import platform
import re
import shutil
import sys
import tempfile
import zipfile

PACKAGE_NAME = "turboembed_native"
DIST_NAME = "turboembed_native"
REQUIRES_PYTHON = ">=3.10"

_SEMVER_RE = re.compile(
    r"^v?(?P<core>[0-9]+\.[0-9]+\.[0-9]+)"
    r"(?:-(?P<pre_tag>alpha|a|beta|b|rc|pre|dev)[.-]?(?P<pre_num>[0-9]+))?$"
)

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


def default_platform_tag() -> str:
    machine = platform.machine().lower()
    if machine in ("amd64", "x86_64"):
        arch = "x86_64"
    elif machine in ("arm64", "aarch64"):
        arch = "aarch64" if sys.platform.startswith("linux") else "arm64"
    else:
        arch = machine
    if sys.platform.startswith("linux"):
        return f"manylinux_2_28_{arch}"
    if sys.platform == "darwin":
        return f"macosx_11_0_{arch}"
    if sys.platform == "win32":
        return "win_amd64"
    return f"{sys.platform}_{arch}"


def urlsafe_b64_nopad(digest: bytes) -> str:
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def locate_libturbo(repo_root: pathlib.Path, explicit_lib: pathlib.Path | None) -> pathlib.Path:
    if explicit_lib is not None:
        resolved = explicit_lib.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"--lib-path does not exist: {resolved}")
        return resolved
    names = ["libturbo.dylib"] if sys.platform == "darwin" else (["turbo.dll", "libturbo.dll"] if sys.platform == "win32" else ["libturbo.so"])
    target_dir = pathlib.Path(os.environ.get("CARGO_TARGET_DIR", str(repo_root / "target")))
    for profile in ("release", "debug"):
        for name in names:
            cand = target_dir / profile / name
            if cand.is_file():
                return cand
    raise FileNotFoundError(
        f"Could not find built libturbo in {target_dir}/{{release,debug}}; build `cargo build -p turbo` first."
    )


def build_wheel(
    out_dir: pathlib.Path,
    version: str = "0.1.0",
    lib_path: pathlib.Path | None = None,
    platform_tag: str | None = None,
) -> pathlib.Path:
    pep_version = normalize_pep440_version(version)
    plat = platform_tag or default_platform_tag()
    tag = f"cp310-abi3-{plat}"

    here = pathlib.Path(__file__).resolve().parent
    repo_root = here.parents[1]
    src_pkg = here / PACKAGE_NAME
    shared_lib = locate_libturbo(repo_root, lib_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    wheel_name = f"{DIST_NAME}-{pep_version}-{tag}.whl"
    wheel_path = out_dir / wheel_name

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        pkg_dir = tmp_path / PACKAGE_NAME
        pkg_dir.mkdir(parents=True, exist_ok=True)

        for py_file in sorted(src_pkg.glob("*.py")):
            text = py_file.read_text(encoding="utf-8")
            if py_file.name == "__init__.py":
                text = re.sub(r'^__version__ = ".*"$', f'__version__ = "{pep_version}"', text, flags=re.M)
            (pkg_dir / py_file.name).write_text(text, encoding="utf-8")

        shutil.copyfile(shared_lib, pkg_dir / shared_lib.name)

        dist_info = tmp_path / f"{DIST_NAME}-{pep_version}.dist-info"
        dist_info.mkdir(parents=True, exist_ok=True)
        licenses_dir = dist_info / "licenses"
        licenses_dir.mkdir(parents=True, exist_ok=True)
        for lic_name in ("LICENSE", "NOTICE"):
            lic_src = repo_root / lic_name
            if not lic_src.is_file():
                raise RuntimeError(f"missing repository {lic_name} at {lic_src}")
            shutil.copyfile(lic_src, licenses_dir / lic_name)

        (dist_info / "METADATA").write_text(
            f"Metadata-Version: 2.1\n"
            f"Name: turboembed-native\n"
            f"Version: {pep_version}\n"
            f"Summary: In-process native Python bindings for TurboEmbed with zero-copy NumPy and Apache Arrow export.\n"
            f"License: Apache-2.0\n"
            f"License-File: licenses/LICENSE\n"
            f"License-File: licenses/NOTICE\n"
            f"Requires-Python: {REQUIRES_PYTHON}\n",
            encoding="utf-8",
        )
        (dist_info / "WHEEL").write_text(
            f"Wheel-Version: 1.0\n"
            f"Generator: turboembed-native-build-wheel\n"
            f"Root-Is-Purelib: false\n"
            f"Tag: {tag}\n",
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
                mode = 0o755 if file_path.suffix in (".so", ".dylib", ".dll") else 0o644
                info.external_attr = mode << 16
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
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--version", type=str, default="0.1.0")
    parser.add_argument("--lib-path", type=pathlib.Path, default=None)
    parser.add_argument("--platform-tag", type=str, default=None)
    args = parser.parse_args()
    whl = build_wheel(
        out_dir=args.out_dir.resolve(),
        version=args.version,
        lib_path=args.lib_path,
        platform_tag=args.platform_tag,
    )
    print(whl)


if __name__ == "__main__":
    main()
