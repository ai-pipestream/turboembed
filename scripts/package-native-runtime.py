#!/usr/bin/env python3
"""Copy a selected OpenVINO runtime into a fresh TurboEmbed SDK install prefix.

Run after cmake --install. Intel GPU drivers, the system OpenCL ICD loader,
glibc and the C++ runtime remain host prerequisites. No downloads occur.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openvino-root", required=True, type=Path)
    parser.add_argument("--prefix", required=True, type=Path)
    args = parser.parse_args()
    root = args.openvino_root.resolve(strict=True)
    prefix = args.prefix.resolve(strict=True)
    lib = prefix / "lib"
    if not (lib / "libturboembed_prepared.so.1").is_file():
        parser.error("prefix must contain a CMake SDK install with CMAKE_INSTALL_LIBDIR=lib")
    receipt = prefix / "share/turboembed/runtime-files.json"
    if receipt.exists():
        parser.error("runtime already packaged; use a fresh SDK prefix")

    patterns = {
        root / "runtime/lib/intel64": [
            "libopenvino.so*", "libopenvino_intel_gpu_plugin.so*",
            "libopenvino_intel_cpu_plugin.so*", "libopenvino_ir_frontend.so*",
            "libopenvino_onnx_frontend.so*",
        ],
        root / "runtime/3rdparty/tbb/lib": ["libtbb.so*", "libtbbmalloc.so*"],
    }
    sources = []
    for directory, globs in patterns.items():
        for pattern in globs:
            matches = sorted(directory.glob(pattern))
            if not matches:
                parser.error(f"required runtime files missing: {directory / pattern}")
            sources.extend(matches)
    licenses = {
        "OpenVINO-Apache_license.txt": root / "docs/licensing/Apache_license.txt",
        "OpenVINO-LICENSE": root / "docs/licensing/LICENSE",
        "TBB-LICENSE": root / "runtime/3rdparty/tbb/TBB-LICENSE",
    }
    for source in [*sources, *licenses.values()]:
        if not source.is_file():
            parser.error(f"required file missing: {source}")
    for source in sources:
        if (lib / source.name).exists() or (lib / source.name).is_symlink():
            parser.error(f"refusing to overwrite installed file: {source.name}")

    records = {}
    # Copy real files once, with relative links for the SONAME and developer name.
    for source in sources:
        destination = lib / source.name
        if source.is_symlink():
            target = source.resolve().name
            if target not in {p.name for p in sources}:
                parser.error(f"runtime symlink target not included: {source}")
            destination.symlink_to(target)
        else:
            shutil.copy2(source, destination)
        records[str(destination.relative_to(prefix))] = hashlib.sha256(source.read_bytes()).hexdigest()
    license_dir = prefix / "share/turboembed/licenses"
    license_dir.mkdir(parents=True, exist_ok=True)
    for name, source in licenses.items():
        destination = license_dir / name
        shutil.copy2(source, destination)
        records[str(destination.relative_to(prefix))] = hashlib.sha256(source.read_bytes()).hexdigest()
    receipt.write_text(json.dumps({"schema_version": 1, "files": records}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
