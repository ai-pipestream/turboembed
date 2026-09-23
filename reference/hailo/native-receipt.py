#!/usr/bin/env python3
"""The direct-native half of the hailo provider's matched benchmark.

HailoRT's own tool, `hailortcli benchmark`, runs the HEF on the device and
reports frames per second in HW-only mode (the device alone) and in
streaming mode (with the host vstream path). A frame of the MiniLM HEF is
one row of its fixed 128-token window, so the raw runtime's rows per
second is that FPS, and this script turns the two figures into a receipt
of kind `native` with one cell per cell of the libturbo receipt it is
paired with (batch b: p50 = b * 1000 / FPS ms), so `turbo-bench compare`
can read it. The figures are hailortcli's, not a per-cell timing; the
receipt says so.

Usage:
  reference/hailo/native-receipt.py --hef model.hailo8.hef --turbo-receipt hailo-pi5-hailo8-embed.json \
      --commit <sha> --out native.json [--time-to-run 15] [--ssh <host>]
"""
import argparse
import datetime
import json
import platform
import re
import subprocess
import sys


def run(cmd, ssh=None, tolerate_failure=False):
    if ssh:
        cmd = ["ssh", ssh, " ".join(cmd)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 and not tolerate_failure:
        sys.exit(f"error: {' '.join(cmd)}: {p.stderr.strip() or p.stdout.strip()}")
    return p.stdout + p.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hef", required=True)
    ap.add_argument("--turbo-receipt", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--time-to-run", type=int, default=15)
    ap.add_argument("--ssh", default=None, help="run hailortcli on this host")
    a = ap.parse_args()
    turbo = json.load(open(a.turbo_receipt))
    # The tool's third phase (latency) fails to configure the vdevice for
    # this HEF on HailoRT 4.23 after the two FPS phases have run; the FPS
    # figures are complete by then, so a non-zero exit is tolerated and
    # the figures are required instead.
    out = run(["hailortcli", "benchmark", a.hef, "--time-to-run", str(a.time_to_run)], a.ssh, tolerate_failure=True)
    # The tool prints a progress line per second and a summary; the last
    # FPS of each mode is the figure.
    modes = {}
    current = None
    for line in out.replace("\r", "\n").splitlines():
        if "HW-only" in line:
            current = "hw_only"
        elif "streaming mode" in line:
            current = "streaming"
        m = re.search(r"FPS:\s*([0-9.]+)", line)
        if m and current:
            modes[current] = float(m.group(1))
    if "hw_only" not in modes or "streaming" not in modes:
        sys.exit(f"error: hailortcli output has no FPS figures:\n{out[-2000:]}")
    version = run(["hailortcli", "--version"], a.ssh).strip()
    fw = run(["hailortcli", "fw-control", "identify"], a.ssh)
    fw_version = next((l.split(":", 1)[1].strip() for l in fw.splitlines() if "Firmware Version" in l), "")
    board = next((l.split(":", 1)[1].strip() for l in fw.splitlines() if "Board Name" in l), "Hailo")
    host = run(["uname", "-n"], a.ssh).strip()
    osv = run(["uname", "-sr"], a.ssh).strip()
    arch = run(["uname", "-m"], a.ssh).strip()
    # The provider's number is end to end (write, read, host pooling), so the
    # streaming figure is the matched one; HW-only is recorded beside it.
    fps = modes["streaming"]
    cells = []
    for c in turbo["embed"]:
        b = c["batch"]
        p50 = b * 1000.0 / fps
        live = c["live_tokens_per_row"]
        lat = {
            "p50_ms": p50, "mean_ms": p50, "min_ms": p50, "max_ms": p50,
            "rows_per_s": fps, "tokens_per_s": fps * live,
            "iters": int(fps * a.time_to_run),
        }
        cells.append({
            "batch": b, "seq": c["seq"], "live_tokens_per_row": live,
            "token_count_source": c.get("token_count_source", "unrecorded"),
            "text_path": lat, "prepared_tokens_path": lat,
            "prepared_tokens_note": "hailortcli benchmark measures frames per second; every cell is derived from that one rate",
            "per_run": {"host_allocs": None, "provider_allocs": None},
        })
    receipt = {
        "receipt_version": 1,
        "kind": "native",
        "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
        "machine": {"hostname": host, "os": osv, "arch": arch},
        "commit": a.commit,
        "provider": {"id": "hailortcli", "version": "reference", "runtime_version": version, "driver_version": f"firmware {fw_version}"},
        "device": {"name": turbo["device"]["name"], "kind": "Npu", "ordinal": turbo["device"]["ordinal"], "caps": "0x0", "memory_total": 0},
        "bundle": turbo["bundle"],
        "embed": cells,
        "native_reference": (
            f"this is the native side: hailortcli benchmark on {a.hef} for {a.time_to_run} s per mode; "
            f"streaming FPS {modes['streaming']:.2f} (the matched figure), HW-only FPS {modes['hw_only']:.2f}; "
            f"cells derived from the streaming rate ({board}); bundle identity copied from {a.turbo_receipt}"
        ),
    }
    json.dump(receipt, open(a.out, "w"), indent=2)
    print(f"streaming {modes['streaming']:.2f} FPS, HW-only {modes['hw_only']:.2f} FPS; receipt written to {a.out}")


if __name__ == "__main__":
    main()
