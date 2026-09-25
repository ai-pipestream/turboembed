# Raspberry Pi 5 AI HAT setup

How to bring up a Raspberry Pi 5 AI HAT board from a fresh Raspberry Pi OS
install, verify it, and use it with this tree's Hailo provider
([`providers/hailo/`](../providers/hailo/README.md)).

Two accelerator lines exist and they are installed from two different,
mutually exclusive apt metapackages. Pick the one that matches the board
in the machine.

| board | chip | PCI id | apt metapackage | HailoRT | kernel module | `Device Architecture` |
|---|---|---|---|---|---|---|
| AI Kit, AI HAT+ 13 TOPS | Hailo-8L | `1e60:2864` | `hailo-all` | 4.23.0 | `hailo_pci` | `HAILO8L` |
| AI HAT+ 26 TOPS | Hailo-8 | `1e60:2864` | `hailo-all` | 4.23.0 | `hailo_pci` | `HAILO8` |
| AI HAT+ 2 | Hailo-10H | `1e60:45c4` | `hailo-h10-all` | 5.1.1 | `hailo1x_pci` | `HAILO10H` |

Hailo-8 and Hailo-8L share one PCI id and one driver; the chip is told
apart by the runtime, not by the bus. The same M.2 modules on a CM5 IO
board or on an x86_64 host take the same packages where the archive is
available.

The two lines conflict at the package level, so one machine can carry one
of them at a time. From the Raspberry Pi archive on 2026-09-22:

```
Package: hailo-all
Version: 5.1.1
Depends: hailort (>= 4.23.0), hailo-tappas-core (>= 5.1.0), hailort-pcie-driver (>= 4.23.0), rpicam-apps-hailo-postprocess (>= 1.10.0), python3-hailort (>= 4.23.0), python3-hailo-tappas (>= 0:5.1.0)
Conflicts: hailo-h10-all, hailo-meta
Replaces: hailo-h10-all

Package: hailo-h10-all
Version: 5.1.1
Depends: h10-hailort (>= 5.1.1), hailo-tappas-core (>= 5.1.0), h10-hailort-pcie-driver (>= 5.1.1), rpicam-apps-hailo-postprocess (>= 1.10.0), python3-h10-hailort (>= 4.23.0), python3-hailo-tappas (>= 0:5.1.0)
Conflicts: hailo-all, hailo-meta
Replaces: hailo-all
```

Both metapackages carry version 5.1.1, which is the Raspberry Pi packaging
version and not the HailoRT version. The runtime version is the one
`hailortcli --version` prints: 4.23.0 for the Hailo-8 line, 5.1.1 for the
Hailo-10H line. The conflict reaches down to every component:
`h10-hailort` conflicts with `hailort`, `h10-hailort-pcie-driver` with
`hailort-pcie-driver`, and `python3-h10-hailort` with `python3-hailort`
(it also `Provides: pyhailort`, so both lines import as `hailo_platform`).

The two kernel modules bind different PCI ids and cannot substitute for
each other. `hailo1x_pci` 5.1.1 declares
`pci:v00001E60d000026A2`, `d000043A2` and `d000045C4`; `hailo_pci` 4.23.0
declares `pci:v00001E60d00002864`. A Hailo-8 board on the Hailo-10H stack
will enumerate on the PCI bus and get no driver.

## The machines in this document

Every command output below was captured on one of these, on
2026-09-22 unless it names another date.

| machine | board | chip | kernel | stack |
|---|---|---|---|---|
| Hailo-8 Pi 5 | Raspberry Pi 5, AI HAT+ 26 TOPS | Hailo-8 | `6.18.39+rpt-rpi-2712` | `hailo-all` 5.1.1, HailoRT 4.23.0 |
| Hailo-10H Pi 5, first board | Raspberry Pi 5, AI HAT+ 2, USB NVMe boot | Hailo-10H | `6.18.50+rpt-rpi-2712` | `hailo-h10-all` 5.1.1, HailoRT 5.1.1 |
| Hailo-10H Pi 5, replacement board | Raspberry Pi 5 Rev 1.1 16 GB, AI HAT+ 2, USB stick boot | Hailo-10H | `6.18.50+rpt-rpi-2712` | `hailo-h10-all` 5.1.1, HailoRT 5.1.1 |

All run Raspberry Pi OS Trixie (`Debian GNU/Linux 13 (trixie)`), aarch64,
with the 2712 kernel flavor that the Pi 5 uses. The first Hailo-10H board
failed later the same day (section 7) and the same HAT moved to the
replacement board, a fresh install on a 128 GB USB stick with the official
27 W supply and no hub; the outputs in sections 5 and 6 from that machine
are the ones that stand.

## 1. PCIe Gen 3

The Pi 5's PCIe connector comes up at Gen 2 by default, and both Hailo
lines want Gen 3. The AI HAT+ and the AI HAT+ 2 switch the link to Gen 3
themselves, so nothing needs adding for them. The older AI Kit does not;
for it, add this to `/boot/firmware/config.txt` and reboot:

```
dtparam=pciex1_gen=3
```

Check the device and the link speed before installing anything:

```sh
lspci | grep -i hailo
cat /sys/bus/pci/devices/0001:01:00.0/current_link_speed
```

On the replacement Hailo-10H board, 2026-09-24, with no `pciex1_gen`
line in `config.txt`:

```
0001:01:00.0 Co-processor: Hailo Technologies Ltd. Hailo-10H AI Processor (rev 01)
8.0 GT/s PCIe
```

The two outputs below are from 2026-09-22, when both boards also carried
the `config.txt` line:

On the Hailo-8 Pi:

```
dtparam=pciex1_gen=3
0001:01:00.0 Co-processor: Hailo Technologies Ltd. Hailo-8 AI Processor (rev 01)
```

On the first Hailo-10H board:

```
dtparam=pciex1_gen=3
0001:01:00.0 Hailo-10H AI Processor [1e60:45c4]
```

If `lspci` shows no Hailo device at all, the stack cannot be installed:
that is a seating, ribbon, or `config.txt` problem, not a software one.

## 2. Install

Bring the OS and the bootloader EEPROM up to date first, as the
Raspberry Pi AI HAT documentation does, then power the board off:

```sh
sudo apt update
sudo apt full-upgrade -y
sudo rpi-eeprom-update -a
sudo poweroff
```

`rpi-eeprom-update -a` only stages the new image (`pieeprom.upd` in
`/boot/firmware/`); the bootloader writes it on the next boot. The
replacement Hailo-10H board was still on the 2025-06-13 bootloader on
2026-09-24 when its Hailo-10H stopped loading firmware (section 7); it
came back after this step and a cold start on the 2026-05-26 bootloader.

Install `dkms` in the same command as the Hailo stack, or before it. Both
driver packages declare only `Depends: build-essential`; neither pulls
`dkms` in. Their `postinst` tries DKMS first and falls back to a plain
build against the running kernel when `dkms` is absent, which produces a
working module now and no module after the next kernel upgrade. This is
what the fallback looks like, from the `hailo-h10-all` install on the first
Hailo-10H board:

```
Setting up h10-hailort-pcie-driver (5.1.1) ...
Failed to install PCIe driver to the DKMS tree. Trying to install PCIe driver without DKMS
```

Hailo-8 and Hailo-8L:

```sh
sudo apt update
sudo apt install -y dkms hailo-all
sudo reboot
```

Hailo-10H:

```sh
sudo apt update
sudo apt install -y dkms hailo-h10-all
sudo reboot
```

Either install pulls in TAPPAS and the OpenCV and GStreamer development
stacks they need, so it is large. The `hailo-h10-all` install on the first
Hailo-10H board on 2026-09-22 fetched 224 packages, 397 MB, of which the
driver package `h10-hailort-pcie-driver` was 21.9 MB (the driver source plus the
Hailo-10H firmware images that the driver pushes to the chip at probe).

To swap a machine from one line to the other, install the other
metapackage and let apt remove the first through the `Conflicts`; do not
try to hold both.

### What the DKMS build produces

With `dkms` installed, the module lands in the DKMS tree and is rebuilt on
every kernel upgrade. On the Hailo-8 Pi:

```
$ sudo dkms status
hailo_pci/4.23.0, 6.18.39+rpt-rpi-2712, aarch64: installed

$ modinfo hailo_pci
filename:       /lib/modules/6.18.39+rpt-rpi-2712/updates/dkms/hailo_pci.ko.xz
version:        4.23.0
license:        GPL v2
description:    Hailo PCIe driver
author:         Hailo Technologies Ltd.
```

Without it, the fallback build installs straight into the running kernel's
module tree and DKMS knows nothing about it. On the first Hailo-10H board:

```
$ modinfo hailo1x_pci
filename:       /lib/modules/6.18.50+rpt-rpi-2712/kernel/drivers/misc/hailo1x_pci.ko.xz
version:        5.1.1
license:        GPL v2
description:    Hailo PCIe driver
author:         Hailo Technologies Ltd.
```

The source the build uses is `/usr/src/hailort-pcie-driver/` (the same
path for both lines). If a DKMS build fails, its log is at
`/var/lib/dkms/<module>/<version>/build/make.log`.

## 3. Verify

Run these after the reboot. `hailortcli` needs no root.

### Hailo-8, on the Hailo-8 Pi

```
$ lsmod | grep hailo
hailo_pci             147456  0

$ ls -l /dev/hailo*
crw-rw-rw- 1 root root 509, 0 Sep 20 22:09 /dev/hailo0

$ hailortcli --version
HailoRT-CLI version 4.23.0

$ hailortcli scan
Hailo Devices:
[-] Device: 0001:01:00.0

$ hailortcli fw-control identify
Executing on device: 0001:01:00.0
Identifying board
Control Protocol Version: 2
Firmware Version: 4.23.0 (release,app,extended context switch buffer)
Logger Version: 0
Board Name: Hailo-8
Device Architecture: HAILO8

$ python3 -c "import hailo_platform; print(hailo_platform.__version__)"
4.23.0
```

### Hailo-10H, on the first Hailo-10H board

```
$ lsmod | grep hailo
hailo1x_pci           147456  0

$ ls -l /dev/hailo*
crw-rw-rw- 1 root root 236, 0 Sep 22 07:50 /dev/hailo0

$ hailortcli --version
HailoRT-CLI version 5.1.1

$ hailortcli scan
Hailo Devices:
[-] Device: 0001:01:00.0

$ hailortcli fw-control identify
Executing on device: 0001:01:00.0
Identifying board
Control Protocol Version: 2
Firmware Version: 5.1.1 (release,app)
Logger Version: 0
Device Architecture: HAILO10H

$ python3 -c "import hailo_platform; print(hailo_platform.__version__)"
5.1.1
```

Two differences from the Hailo-8 output above. `fw-control identify`
prints no `Board Name` line on the Hailo-10H, so `Device Architecture` is
the field to match a HEF against. The character device's major number is
allocated dynamically, so it differs between the two machines and between
boots; only the name matters.

### The Hailo-10H firmware boot

The two chips differ in how firmware reaches them. The Hailo-8 has a
single image that the runtime loads
(`/lib/firmware/hailo/hailo8_fw.bin` on the Hailo-8 Pi). The Hailo-10H is an
SoC and the driver programs a whole boot chain into it over PCIe at probe: a
certificate, the SCU firmware, a signed device tree picked by the board's
SKU, then SPL, TF-A, the kernel image and the root filesystem. `sudo
dmesg | grep -i hailo` shows the sequence, and it is the fastest way to
tell a firmware problem from a bus problem. From the first Hailo-10H board,
2026-09-22:

```
hailo1x 0001:01:00.0: Probing: Device enabled
hailo1x 0001:01:00.0: Probing: mapped bar 0 - 00000000bdec7665 16384
hailo1x 0001:01:00.0: Probing: mapped bar 2 - 00000000de60275d 4096
hailo1x 0001:01:00.0: Probing: mapped bar 4 - 000000001f4dd10f 16384
hailo1x 0001:01:00.0: Probing: Setting max_desc_page_size to 4096, (PAGE_SIZE=16384)
hailo1x 0001:01:00.0: Probing: Enabled 64 bit dma
hailo1x 0001:01:00.0: Disabling ASPM L0s
hailo1x 0001:01:00.0: Successfully disabled ASPM L0s
hailo1x 0001:01:00.0: Writing file hailo/hailo10h/customer_certificate.bin
hailo1x 0001:01:00.0: File hailo/hailo10h/customer_certificate.bin written successfully
hailo1x 0001:01:00.0: Writing file hailo/hailo10h/scu_fw.bin
hailo1x 0001:01:00.0: File hailo/hailo10h/scu_fw.bin written successfully
hailo1x 0001:01:00.0: Board SKU-ID is: 6
hailo1x 0001:01:00.0: Writing file hailo/hailo10h/u-boot-6.dtb.signed
hailo1x 0001:01:00.0: File hailo/hailo10h/u-boot-6.dtb.signed written successfully
hailo1x 0001:01:00.0: Reading firmware file hailo/hailo10h/u-boot-spl.bin
hailo1x 0001:01:00.0: Reading firmware file hailo/hailo10h/u-boot-tfa.itb
hailo1x 0001:01:00.0: Reading firmware file hailo/hailo10h/fitImage
hailo1x 0001:01:00.0: Reading firmware file hailo/hailo10h/image-fs
hailo1x 0001:01:00.0: Firmware file programmed successfully
hailo1x 0001:01:00.0: Firmware file index 0 programmed successfully
hailo1x 0001:01:00.0: Firmware file programmed successfully
hailo1x 0001:01:00.0: Firmware file index 2 programmed successfully
hailo1x 0001:01:00.0: Firmware file programmed successfully
hailo1x 0001:01:00.0: Firmware file index 3 programmed successfully
hailo1x 0001:01:00.0: Firmware batch programming completed for stage 2
hailo1x 0001:01:00.0: vDMA transfer completed, triggering boot
hailo1x 0001:01:00.0: SOC Firmware Batch loaded successfully
hailo1x 0001:01:00.0: Firmware loaded in 2607 ms
hailo1x 0001:01:00.0: Probing: Added board 1e60-45c4, /dev/hailo0
```

The images ship in `h10-hailort-pcie-driver` and land in
`/lib/firmware/hailo/hailo10h/`. `Board SKU-ID is: 6` is read off the
board and picks which `u-boot-<n>.dtb.signed` is written; on the first
Hailo-10H board that directory holds SKUs 0, 1, 3, 4, 5, 6 and a default, so
a board whose SKU has no file fails here and nowhere else. The chain took 2607 ms, which is
why the device node does not appear the instant the module loads.

This ran at probe both times the module was inserted: once when `apt`
installed it, and again on the next boot. The device is usable straight
after the install, with no reboot.

## 4. Where HEFs come from

A HEF is a compiled network, and it is compiled for one chip. A `hailo8`
HEF does not run on a Hailo-10H and the reverse is also true; the runtime
refuses at configure time and names the architecture. There are three
sources.

### The `hailo-models` package

`hailo-models` is in the Raspberry Pi archive and is pulled in by both
metapackages. It installs into `/usr/share/hailo-models/` and carries both
architectures, with the target in the file name (`_h8`, `_h8l`, `_h10`).
On the Hailo-8 Pi and the first Hailo-10H board, `hailo-models` 1.0.0-2:

```
resnet_v1_50_h10.hef       resnet_v1_50_h8l.hef      scrfd_2.5g_h8l.hef
yolov11m_h10.hef           yolov5n_seg_h10.hef       yolov5n_seg_h8.hef
yolov5n_seg_h8l_mz.hef     yolov5s_personface_h8l.hef
yolov6n_h8.hef             yolov6n_h8l.hef           yolov8m_h10.hef
yolov8m_pose_h10.hef       yolov8s_h8.hef            yolov8s_h8l.hef
yolov8s_pose_h10.hef       yolov8s_pose_h8.hef       yolov8s_pose_h8l_pi.hef
yolox_s_leaky_h8l_rpi.hef
```

These are vision models. They are the quickest way to prove a board works
end to end, and that is all this document uses them for.

### The Hailo Model Zoo

The Model Zoo publishes prebuilt HEFs per architecture, under a
per-architecture path (`hailo8`, `hailo8l`, `hailo10h`), so the same model
name exists once per chip. It covers classification, detection,
segmentation, pose, depth, and a small number of language models. Download
the one matching the `Device Architecture` string that `hailortcli
fw-control identify` printed.

### Text embeddings

The embedding HEF this tree's provider uses is not a vision model and is
not in `hailo-models`. See
[`providers/hailo/README.md`](../providers/hailo/README.md) for the bundle
layout (a `hef` artifact plus a `hailo_tables` artifact holding the word,
position, and token-type tables the Dataflow Compiler cannot fit) and for
the `turbo-bundle import` command that builds it.

The Hailo-8 path has receipts: the Model Zoo's `all_minilm_l6_v2` HEF for
`hailo8`, the tables exported by
[`scripts/export-hailo-tables.py`](../scripts/export-hailo-tables.py), and
the measurements in
[`testdata/receipts/turbo/hailo-2026-09-21.json`](../testdata/receipts/turbo/hailo-2026-09-21.json)
and
[`testdata/receipts/turbo/bench/hailo-pi5-hailo8-embed-2026-09-22b.json`](../testdata/receipts/turbo/bench/hailo-pi5-hailo8-embed-2026-09-22b.json).

The Hailo-10H path does not. There is no public `hailo10h` MiniLM HEF, so
one has to be compiled with the Hailo Dataflow Compiler 5 line, which is
the line that emits `hailo10h` HEFs. The DFC is proprietary, runs on an
x86_64 Linux host, and is not part of either Pi package line, so the
compile happens off the Pi and the resulting HEF is copied over.
[`PLAN.md`](../PLAN.md) section 7 tracks that compile as an open item, and
[`docs/providers.md`](providers.md) lists Hailo-10H as open for the same
reason.

## 5. Prove the board runs

The two tools differ between the lines, so this section is written twice.

### Hailo-8, on the Hailo-8 Pi

`hailortcli run` runs one HEF; `hailortcli benchmark` runs it in three
phases (hardware-only FPS, streaming FPS, hardware latency) and prints a
summary. On `/usr/share/hailo-models/yolov8s_h8.hef`, 2026-09-22:

```
$ hailortcli benchmark /usr/share/hailo-models/yolov8s_h8.hef --time-to-run 5
Starting Measurements...
Measuring FPS in HW-only mode
Network yolov8s/yolov8s: 100% | 1551 | FPS: 309.77 | ETA: 00:00:00
Measuring FPS (and Power on supported platforms) in streaming mode
Network yolov8s/yolov8s: 100% | 1549 | FPS: 309.44 | ETA: 00:00:00
Measuring HW Latency
Network yolov8s/yolov8s: 100% | 520 | HW Latency: 6.66 ms | ETA: 00:00:00

=======
Summary
=======
FPS     (hw_only)                 = 309.58
        (streaming)               = 309.441
Latency (hw)                      = 6.66137 ms
```

### Hailo-10H, on the first Hailo-10H board

`hailortcli run` is gone on this device. HailoRT 5.1.1 answers it with:

```
$ hailortcli run ~/models/hailo10h/resnet_v1_50_h10.hef --time-to-run 5
Running streaming inference ($HOME/models/hailo10h/resnet_v1_50_h10.hef):
  Transform data: true
    Type:      auto
    Quantized: true
[HailoRT CLI] [error] CHECK failed - The "run" command is not supported for this device type! Use "run2" instead.
```

`run2` takes the HEF through a `set-net` subcommand and the timing flag
comes before it:

```
$ hailortcli run2 --time-to-run 5 set-net ~/models/hailo10h/resnet_v1_50_h10.hef
[===================>] 100% 00:00:00
resnet_v1_50: fps: 307.46
```

`benchmark` still exists but prints a different summary from the 4.23 one:
one FPS figure rather than a hardware-only and a streaming figure, no
latency phase, a device temperature readout, and a note that this board
exposes no power or current sensor.

```
$ hailortcli benchmark ~/models/hailo10h/resnet_v1_50_h10.hef --time-to-run 5
Measurement power is not supported
Measurement current is not supported

=======
Summary
=======
resnet_v1_50: FPS: 307.48
0001:01:00.0:
  temperature: mean=33.74 min=31.85 max=34.69
```

```
$ hailortcli benchmark ~/models/hailo10h/yolov8s_pose_h10.hef --time-to-run 5
Measurement power is not supported
Measurement current is not supported

=======
Summary
=======
yolov8s_pose: FPS: 156.75
0001:01:00.0:
  temperature: mean=38.02 min=33.57 max=39.59
```

Both HEFs were copied out of `/usr/share/hailo-models/` into
`~/models/hailo10h/`. These are different networks from the `yolov8s_h8`
run above, so the FPS figures of the different chips are not comparable
with each other; each one only shows that the board in front of you runs.

The same commands on the replacement Hailo-10H board, which carries the same
HAT, give the same figures to a tenth of a frame, which is what a healthy
board looks like next to another healthy board:

```
$ hailortcli benchmark ~/models/hailo10h/resnet_v1_50_h10.hef --time-to-run 5
resnet_v1_50: FPS: 307.38
0001:01:00.0:
  temperature: mean=44.64 min=42.47 max=45.66

$ hailortcli benchmark ~/models/hailo10h/yolov8s_pose_h10.hef --time-to-run 5
yolov8s_pose: FPS: 156.92
0001:01:00.0:
  temperature: mean=49.20 min=45.97 max=50.42
```

## 6. This tree's Hailo provider

[`providers/hailo/`](../providers/hailo/README.md) is a C++ plugin,
`libturbo_provider_hailo.so`, that implements
[`include/turbo/turbo_provider.h`](../include/turbo/turbo_provider.h)
directly and serves `EMBED x TEXT`. It is written against the HailoRT 4.x
vstream API.

### Build and run on a Hailo-8 Pi

From the repository root, on a machine with `hailo-all` installed (the
header is `/usr/include/hailo/hailort.h` and the library
`/usr/lib/libhailort.so`):

```sh
cmake -S providers/hailo -B build/hailo -DCMAKE_BUILD_TYPE=Release
cmake --build build/hailo -j
```

Pass `-DHAILORT_INCLUDE_DIR=` and `-DHAILORT_LIBRARY=` for an install
elsewhere. The vtable tests and the Rust live suite need a MiniLM bundle
built per [`providers/hailo/README.md`](../providers/hailo/README.md):

```sh
export TURBO_LIVE_BUNDLE=$HOME/bundles/minilm-hailo8
./build/hailo/turbo_provider_hailo_test
TURBO_LIVE_LIB=$PWD/build/hailo/libturbo_provider_hailo.so TURBO_LIVE_PROVIDER=hailo \
    cargo test -p turbo-conformance --test live_embed -- --test-threads=1 --nocapture
```

`--test-threads=1` is required, not a preference. Each test opens its own
`hailo_vdevice` and HailoRT admits one per process at a time; a second one
fails with `HAILO_DEVICE_IN_USE`. See [`docs/testing.md`](testing.md) for
the rest of the live-test environment.

What this gets on a Hailo-8: 76 rows per second at every batch and
sequence length on the Hailo-8 Pi, 2026-09-22, about 13.2 ms per row, because
the HEF is batch 1 with a fixed 128-token frame
([`testdata/receipts/turbo/bench/hailo-pi5-hailo8-embed-2026-09-22b.json`](../testdata/receipts/turbo/bench/hailo-pi5-hailo8-embed-2026-09-22b.json)).
That is 1.00x of what `hailortcli benchmark` gets from the same HEF
([`testdata/receipts/turbo/bench/compare-hailo-pi5-hailo8-embed-2026-09-22b.json`](../testdata/receipts/turbo/bench/compare-hailo-pi5-hailo8-embed-2026-09-22b.json)).
The HEF is quantized, so the vectors are not the FP32 ones; see
[`providers/hailo/README.md`](../providers/hailo/README.md) for the cosine
and ranking figures the live suite gates on.

### HailoRT 5: the provider builds, links and finds the device

The provider is written against the HailoRT 4.x C vstream API. Every
HailoRT entry point it calls is one of `hailo_create_vdevice`,
`hailo_init_vdevice_params`, `hailo_release_vdevice`,
`hailo_scan_devices`, `hailo_create_device_by_id`, `hailo_release_device`,
`hailo_identify`, `hailo_get_library_version`, `hailo_create_hef_file`,
`hailo_release_hef`, `hailo_init_configure_params_by_vdevice`,
`hailo_configure_vdevice`, `hailo_hef_make_input_vstream_params`,
`hailo_make_output_vstream_params`, `hailo_create_input_vstreams`,
`hailo_create_output_vstreams`, `hailo_release_input_vstreams`,
`hailo_release_output_vstreams`, `hailo_get_input_vstream_frame_size`,
`hailo_get_output_vstream_frame_size`, `hailo_vstream_write_raw_buffer`,
`hailo_vstream_read_raw_buffer`, and `hailo_get_status_message`
([`providers/hailo/src/provider.cpp`](../providers/hailo/src/provider.cpp)).

That set compiles and links against HailoRT 5.1.1 unchanged. On
the replacement Hailo-10H board (2026-09-22), from the repository root:

```sh
cmake -S providers/hailo -B build/hailo -DCMAKE_BUILD_TYPE=Release
cmake --build build/hailo -j
```

configures with `HailoRT: /usr/lib/libhailort.so (/usr/include)` and
builds `build/hailo/libturbo_provider_hailo.so` and the test binary with
no warnings. The 4.x entry points above are all still exported by
`libhailort.so.5.1.1`.

The vtable tests then run against the real chip. Without a bundle, the
device, capability, struct-size and buffer cases pass and the ten bundle
cases fail naming `TURBO_LIVE_BUNDLE`, which is the harness rule for a
task the device offers (`docs/testing.md`): there is no `hailo10h` MiniLM
HEF to point the variable at (section 4), so on this machine those ten are
red until one exists rather than quietly skipped.

```
$ build/hailo/turbo_provider_hailo_test
test devices_are_hailo_npus
provider `hailo` 2.0.0-alpha.0 from $HOME/turbo/build/hailo/libturbo_provider_hailo.so
  device 0 `Hailo-10H (, 0001:01:00.0)` vendor `Hailo` runtime `HailoRT 5.1.1` driver `firmware 5.1.1` caps 0x3f0402
  ok
test capability_states_the_quantized_floor
  EMBED x TEXT: status 2 dtype 6 reference 12 cosine_floor 0.300 notes `INT8 encoder on Hailo-10H; host tokenize/gather/pool; no receipts for this architecture yet`
  ok
test can_run_checks_the_bundle_and_task
  FAIL TURBO_LIVE_BUNDLE is not set; point it at a bundle directory for this case
  FAILED
...
test buffers_are_host_only
  ok
14 tests, 10 failed checks
```

`turbo-bench discover` through the Rust loader agrees:

```
$ target/release/turbo-bench discover --provider-lib build/hailo/libturbo_provider_hailo.so
[0] hailo:0  Hailo-10H (, 0001:01:00.0)  (Npu, vendor Hailo)
     provider hailo 2.0.0-alpha.0; runtime HailoRT 5.1.1; driver firmware 5.1.1
     offers:
       Embed          x Text   Experimental compute I8 cosine floor 0.300 vs F32 deterministic  (INT8 encoder on Hailo-10H; host tokenize/gather/pool; no receipts for this architecture yet)
```

The empty field before the PCI address is the board name, which the
Hailo-10H firmware does not report (section 3).

What HailoRT 5 did move is the CLI surface: `hailortcli run` is refused on
the Hailo-10H in favor of `run2`, and `benchmark` prints a different
summary (section 5). The C API the provider uses did not move.

So the Hailo-10H is one file away from running through this provider: a
`hailo10h` MiniLM HEF from the DFC 5 line (section 4). The capability cell
stays `EXPERIMENTAL` with the note above until that HEF exists and the
precision and matched-native receipts are taken on it.
[`PLAN.md`](../PLAN.md) section 7 tracks both the HailoRT 5 port and the
Hailo-10H generation work through `hailort::genai::LLM`.

## 7. Troubleshooting

| symptom | cause | what to do |
|---|---|---|
| `lspci` shows no Hailo device | HAT not seated, ribbon reversed, or the PCIe connector not enabled | reseat the HAT and the ribbon, confirm `dtparam=pciex1_gen=3` in `/boot/firmware/config.txt`, reboot |
| `Failed to install PCIe driver to the DKMS tree` during `apt install` | `dkms` is not installed; neither driver package depends on it | `sudo apt install -y dkms`, then `sudo apt install --reinstall hailort-pcie-driver` (or `h10-hailort-pcie-driver`), then reboot |
| module loads now, gone after a kernel upgrade | the module was built by the non-DKMS fallback into the old kernel's tree | same fix as above; `dkms status` should name the module and the running kernel |
| no `/dev/hailo0` and nothing in `lsmod` | the module was not built or not loaded | `modinfo hailo_pci` (or `hailo1x_pci`) to see whether a module exists at all; if not, check `/var/lib/dkms/*/*/build/make.log`; if it exists, `sudo modprobe hailo_pci` and read `sudo dmesg \| grep -i hailo` |
| Hailo-10H: `/dev/hailo0` exists, `hailortcli scan` lists the device, `fw-control identify` fails with `HAILO_DRIVER_OPERATION_FAILED(36)`, and the kernel log says `Device disconnected while opening device` | the chip dropped its state under a bound driver: `lspci -vv` shows it in D3hot with `Mem- BusMaster-`. Seen on the replacement Hailo-10H board on 2026-09-24 after a day of uptime; the trigger is not established | a warm reboot and a driver reload do not recover it (next two rows). Update the bootloader (section 2), then `sudo poweroff` and unplug the supply for a few seconds |
| Hailo-10H: after a warm reboot, no `/dev/hailo0`; the kernel log shows stage 2 complete, then `Timeout waiting for firmware file`, `Failed writing SOC firmware on stage 3` and `probe with driver hailo1x failed with error -110` | the chip did not come back up from its previous failure; a reboot does not always remove power from the HAT | same as the row above. The firmware files are not the problem when `dpkg -V h10-hailort-pcie-driver` prints nothing |
| Hailo-10H: `sudo rmmod hailo1x_pci && sudo modprobe hailo1x_pci` logs `Failed reading device BARs, device may be disconnected` | the chip no longer answers memory reads on the bus | only a cold start recovers it; see the two rows above |
| `/boot/firmware/cmdline.txt` or `config.txt` is empty or missing an edit after the board was unplugged | `/boot/firmware` is FAT and the edit was still in the page cache when power was cut. An empty `cmdline.txt` has no `root=` and the next boot fails | run `sync` after editing anything under `/boot/firmware`, and shut down with `sudo poweroff` before unplugging. Keep a copy of both files before editing them |
| `/dev/hailo0` exists, `hailortcli scan` finds nothing | the runtime and the driver are from different lines | `hailortcli --version` and `modinfo <module> \| grep ^version` must agree (4.23.0 with `hailo_pci`, 5.1.1 with `hailo1x_pci`) |
| `hailortcli` reports the wrong architecture for the HEF | the HEF was compiled for another chip | match the HEF to the `Device Architecture` line from `hailortcli fw-control identify` |
| `HAILO_DEVICE_IN_USE` | a second process or a second `hailo_vdevice` in the same process | one vdevice at a time; run the live suite with `--test-threads=1` and stop any `hailortcli` still running |
| the board resets the moment a USB SSD is plugged in, before any boot | the Pi 5 allows USB devices 600 mA until it negotiates a 5 A supply, and an NVMe in a USB enclosure (or an SSD-class stick) draws more than that at power-up | a 5 A supply (the official 27 W one) so the bootloader raises the budget to 1.6 A; `usb_max_current_enable=1` in `config.txt` for the kernel side; `PSU_MAX_CURRENT=5000` in the EEPROM config for a 5 A supply that does not negotiate; or a powered hub whose brick does not backfeed |
| the board boots, runs cleanly for seconds to minutes, then loses power with nothing in the log; a bare microSD on the official supply does the same; the bootloader eventually shows a LED code of 4 long flashes | the board's power management chip is damaged. On the replacement Hailo-10H board (2026-09-22) this followed a USB hub whose brick backfed 5 V into the Pi's USB port; the journal, once made persistent with `SyncIntervalSec=2s`, showed an idle system with `vcgencmd pmic_read_adc EXT5V_V` at 5.10 to 5.14 V and `get_throttled` 0x0 right up to the cut, and the 4-long-flash family (4/4 board type, 4/5 firmware, 4/6 and 4/7 power failure) is the bootloader's fatal class | replace the board; the SSD, HAT and supply survive. Before that, one five-minute check: the Bootloader recovery image from Raspberry Pi Imager on a microSD, which rewrites the EEPROM and clears the 4/4 and 4/5 codes if they were corruption rather than hardware. Use a hub that does not backfeed, or none; the `hailo-h10-all` install itself needs no reboot, so verify the stack before rebooting |
| `hailortcli benchmark` fails in its third phase after printing both FPS figures | HailoRT 4.23 fails to reconfigure the vdevice for the MiniLM HEF after the two FPS phases | the FPS figures are already complete, so treat a non-zero exit after them as the latency phase only; [`reference/hailo/native-receipt.py`](../reference/hailo/native-receipt.py) tolerates it for this reason |
| `TURBO_E_INVALID_STATE` on every run after one failed run | a vstream write or read failed mid-run and left frames in flight | reload the model; the provider marks it unusable on purpose rather than returning wrong data |
| CMake says HailoRT was not found | no `hailo-all`/`hailo-h10-all`, or an install outside `/usr` | install the metapackage, or pass `-DHAILORT_INCLUDE_DIR=` and `-DHAILORT_LIBRARY=` |

Two further changes are published for Hailo-10H vDMA timeouts under load:
`pcie_aspm=off` in `cmdline.txt`, and the 4 KB page kernel
(`kernel=kernel8.img`) with `options hailo1x_pci force_desc_page_size=4096`
in `/etc/modprobe.d/`. Neither kernel change was needed to recover the
replacement Hailo-10H board on 2026-09-24. The module option changes nothing
on the 16 KB page kernel with driver 5.1.1: the driver already uses
4096-byte descriptor pages there, and the option only changes the probe log
line from `Setting max_desc_page_size to 4096` to
`Force setting max_desc_page_size to 4096`.
