# TurboRerank Metal GELU / erf (Machine C) — SOLIDIFY (7) Apple

HF MiniLM `hidden_act=gelu` is the **erf** form
`0.5 * x * (1 + erf(x / sqrt(2)))`, not `gelu_new` (tanh). CUDA uses
`erff`. Metal Shading Language still has no `erf` / `erfc`, so the
kernel keeps a software special. That is not a CPU fallback.

| Field | Value |
|---|---|
| Host | Machine C, Apple M2, Metal |
| Toolchain | Xcode 26.6 / Metal.xctoolchain 17.6 (`metal_stdlib` 32023) |
| Command | `make metal-erf-probe` + `make test-turborerank-apple` |
| Receipt | `testdata/receipts/turborerank/apple-minilm-l6.json` (`pass=true`, max abs `1.91e-6`) |
| Golden | `testdata/reference_rerank/ms_marco_minilm_l6_berlin.json` (`atol` 2e-3) |
| Berlin logits | `8.84585285`, `-4.32007647`, `-11.27389240` (cosine `1.0`) |

## Why MSL still needs an approximation

`metal_math` (this SDK) lists the Metal 1.1 elementary set plus later
`fma` / `sinpi` / `double` overloads. There is **no** `erf`, `erfc`,
`tgamma`, or `lgamma` in `fast::` or `precise::`.

`make metal-erf-probe` compiles two one-line kernels with the same
`MTLMathModeSafe` options as TurboRerank:

```text
x[i] = erf(x[i]);
x[i] = precise::erf(x[i]);
```

Both fail `newLibraryWithSource` (undeclared identifier). If a future
MSL grows a builtin, that probe **fails on purpose** so GELU switches
to `erf()` instead of keeping a software path.

Not used (and not a silent substitute):

- `tanh` / `gelu_new` — wrong HF activation.
- Host `std::erff` over a mapped buffer — CPU fallback theater.
- MPSGraph / Accelerate vForce — extra graph, extra alloc.

## What replaced A&S 7.1.26

Hastings / A&S 7.1.26 is `1.5e-7` class and only needs `exp`. The
live kernel is a Hart–Cheney piecewise rational (near / mid) plus the
complementary `exp(-x^2)` tail (`erf_approx.hpp`, same Horner in
`metal_api.mm`). Still only MSL-available ops: `fabs`, `copysign`,
`exp`, `fma`, `as_type`.

Host grid vs `erff` on `[-8, 8]` step `1e-3` (Machine C libm):

| path | max \|erf − erff\| | max \|GELU − libm\| | mean \|GELU − libm\| |
|---|---|---|---|
| A&S 7.1.26 | `3.9e-7` | `4.8e-7` | `4.7e-8` |
| Hart + tail | `6.0e-8` | `2.4e-7` | `4.3e-10` |

The Metal grid in `metal_erf_probe` repeats that measurement on-device
(SHARED buffers, no host GELU). Hart must beat A&S vs libm or the
probe fails. Host twin and Metal Hart must stay within `2e-7`.

## Berlin

Measured on Machine C after the Hart swap (`make turborerank-tests` +
`make turborerank-apple-receipt`):

| | relevant | mid | irrelevant | max abs err |
|---|---|---|---|---|
| HF golden | `8.84585285` | `-4.32007599` | `-11.27389431` | — |
| Metal A&S (prior receipt) | `8.84585381` | `-4.32007980` | `-11.27389336` | `3.81e-6` |
| Metal Hart (this) | `8.84585285` | `-4.32007647` | `-11.27389240` | `1.91e-6` |
| CUDA `erff` (Machine A) | `8.84585381` | `-4.32007694` | `-11.27389526` | `1e-6` |

Hart **tightens** Berlin: first logit matches the golden exactly, mid
error drops ~8×, max abs ~2× (`4e-6` → `1.91e-6`). Still inside the
`2e-3` band. Cosine `1.0`. `allocs/forward == 0`. nometal create still
fails loud (180/0 without Metal). Leftover ULP vs CUDA is first-party
`linear_nt` vs cuBLASLt, not the GELU special.

## Commands

```bash
make metal-erf-probe           # MSL erf missing + on-device Hart vs libm
make turborerank-tests         # host erf grid + Metal Berlin + allocs/forward==0
make turborerank-tests-nometal # Metal create fails loud
make turborerank-apple-receipt # writes apple-minilm-l6.json
```
