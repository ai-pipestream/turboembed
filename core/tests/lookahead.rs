//! Limited-lookahead attention (architecture.lookahead, issue 887): the
//! CPU encoder of a banded model gives the f64 encoder whose every query
//! sees the keys before it and at most `lookahead` keys after it, over
//! rows of mixed length in one batch; a band as long as the row is the
//! whole-row encoder; a band of 0 is refused at load; and rule 6 skips the
//! artifact, unread, on a backend whose attention takes no band.

mod common;

use std::path::Path;

use common::*;
use serde_json::json;
use turbo::status::*;
use turbo::*;

const TEXTS: [&str; 4] = [
    "The quick brown fox jumps over the lazy dog while the cat sleeps on the warm windowsill all afternoon.",
    "how do I reset a password",
    "a",
    "Sentence boundaries need context: a full stop after Dr or Mr is not the end of one, and this row is long.",
];

/// The backends whose attention takes a band (turbo::model::runs_band).
const BANDED: [&str; 4] = ["cpu", "cuda", "levelzero", "metal"];

/// The small BERT with `lookahead` keys ahead.
fn banded_fixture(name: &str, lookahead: u32) -> Fixture {
    let mut f = Fixture::model(name);
    f.manifest["architecture"]["lookahead"] = json!(lookahead);
    f.write();
    f
}

/// Each backend of BANDED this machine lists, the CPU always, failing
/// where TURBO_TEST_REQUIRE_<BACKEND>=1 names one it does not, with the
/// precisions to run it at and the largest difference allowed at each:
/// the CPU at its default, a device at MODEL (F32) and FASTEST (F16 where
/// the model's widths allow it, docs/levelzero.md).
fn banded_devices() -> Vec<(&'static str, u32, f64)> {
    let mut rt = std::ptr::null_mut();
    assert_eq!(unsafe { turbo_runtime_create(std::ptr::null(), &mut rt, &mut new_error()) }, 0);
    let mut out = vec![("cpu", TURBO_PRECISION_MODEL, 1e-5)];
    for b in BANDED.into_iter().filter(|&b| b != "cpu") {
        let required = std::env::var(format!("TURBO_TEST_REQUIRE_{}", b.to_uppercase())).is_ok_and(|v| v == "1");
        if first_of(rt, b).is_none() {
            assert!(!required, "TURBO_TEST_REQUIRE_{}=1 and the backend lists no device", b.to_uppercase());
            println!("{b}: skipped: no device");
            continue;
        }
        out.extend([(b, TURBO_PRECISION_MODEL, 1e-5), (b, TURBO_PRECISION_FASTEST, 5e-3)]);
    }
    unsafe { turbo_runtime_release(rt) };
    out
}

/// A session on `backend` at `precision` over the fixture.
fn session_on(f: &Fixture, backend: &str, precision: u32) -> (Loaded, Session) {
    let l = f.load_on(|rt| first_of(rt, backend).unwrap()).unwrap_or_else(|e| panic!("{backend}: {e:?}"));
    let s = Session::create(l.m, Some(&session_desc(0, 0, precision))).unwrap_or_else(|e| panic!("{backend}: {e:?}"));
    (l, s)
}

/// The encoder in f64 with the band: query i attends to keys 0..=i+ahead.
struct Banded {
    st: St,
    layers: usize,
    hidden: usize,
    heads: usize,
    eps: f64,
}

impl Banded {
    fn new(f: &Fixture) -> Banded {
        let a = &f.manifest["architecture"];
        let file = f.manifest["artifacts"][0]["files"][0].as_str().unwrap();
        Banded {
            st: St::read(&f.dir.join(file)),
            layers: a["layers"].as_u64().unwrap() as usize,
            hidden: a["hidden"].as_u64().unwrap() as usize,
            heads: a["heads"].as_u64().unwrap() as usize,
            eps: a["layer_norm_eps"].as_f64().unwrap(),
        }
    }

    fn w(&self, name: &str) -> Vec<f64> {
        self.st.f32s(name).iter().map(|&v| v as f64).collect()
    }

    fn norm(&self, x: &mut [Vec<f64>], prefix: &str) {
        let (w, b) = (self.w(&format!("{prefix}.weight")), self.w(&format!("{prefix}.bias")));
        for r in x {
            let n = r.len() as f64;
            let mean = r.iter().sum::<f64>() / n;
            let var = r.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
            for (i, v) in r.iter_mut().enumerate() {
                *v = (*v - mean) / (var + self.eps).sqrt() * w[i] + b[i];
            }
        }
    }

    fn linear(&self, x: &[Vec<f64>], prefix: &str) -> Vec<Vec<f64>> {
        let (w, b) = (self.w(&format!("{prefix}.weight")), self.w(&format!("{prefix}.bias")));
        let n = x[0].len();
        x.iter()
            .map(|r| (0..b.len()).map(|o| b[o] + (0..n).map(|i| r[i] * w[o * n + i]).sum::<f64>()).collect())
            .collect()
    }

    fn add(x: &mut [Vec<f64>], o: &[Vec<f64>]) {
        for (xr, or) in x.iter_mut().zip(o) {
            xr.iter_mut().zip(or).for_each(|(a, b)| *a += b);
        }
    }

    /// One row mean pooled and normalized.
    fn embed(&self, ids: &[i32], ahead: usize) -> Vec<f64> {
        let (h, n) = (self.hidden, ids.len());
        let (word, pos, ty) = (
            self.w("embeddings.word_embeddings.weight"),
            self.w("embeddings.position_embeddings.weight"),
            self.w("embeddings.token_type_embeddings.weight"),
        );
        let mut x: Vec<Vec<f64>> =
            (0..n).map(|p| (0..h).map(|i| word[ids[p] as usize * h + i] + pos[p * h + i] + ty[i]).collect()).collect();
        self.norm(&mut x, "embeddings.LayerNorm");
        let d = h / self.heads;
        for l in 0..self.layers {
            let at = |s: &str| format!("encoder.layer.{l}.{s}");
            let (q, k, v) = (
                self.linear(&x, &at("attention.self.query")),
                self.linear(&x, &at("attention.self.key")),
                self.linear(&x, &at("attention.self.value")),
            );
            let mut ctx = vec![vec![0.0; h]; n];
            for hd in 0..self.heads {
                let c = hd * d..hd * d + d;
                for i in 0..n {
                    let keys = 0..n.min(i.saturating_add(ahead).saturating_add(1));
                    let s: Vec<f64> = keys
                        .clone()
                        .map(|j| {
                            q[i][c.clone()].iter().zip(&k[j][c.clone()]).map(|(a, b)| a * b).sum::<f64>()
                                / (d as f64).sqrt()
                        })
                        .collect();
                    let max = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let e: Vec<f64> = s.iter().map(|v| (v - max).exp()).collect();
                    let sum: f64 = e.iter().sum();
                    for j in keys {
                        for o in c.clone() {
                            ctx[i][o] += e[j] / sum * v[j][o];
                        }
                    }
                }
            }
            Self::add(&mut x, &self.linear(&ctx, &at("attention.output.dense")));
            self.norm(&mut x, &at("attention.output.LayerNorm"));
            let mut f = self.linear(&x, &at("intermediate.dense"));
            f.iter_mut().flatten().for_each(|v| *v = 0.5 * *v * (1.0 + erf(*v / std::f64::consts::SQRT_2)));
            Self::add(&mut x, &self.linear(&f, &at("output.dense")));
            self.norm(&mut x, &at("output.LayerNorm"));
        }
        let mut m: Vec<f64> = (0..h).map(|i| x.iter().map(|r| r[i]).sum::<f64>() / n as f64).collect();
        let norm = m.iter().map(|a| a * a).sum::<f64>().sqrt().max(1e-12);
        m.iter_mut().for_each(|a| *a /= norm);
        m
    }
}

fn close(got: &[f32], want: &[f64]) -> f64 {
    got.iter().zip(want).map(|(a, b)| (*a as f64 - b).abs()).fold(0.0, f64::max)
}

#[test]
fn a_banded_model_matches_the_banded_reference_on_each_device() {
    let devices = banded_devices();
    for ahead in [1u32, 2, 5] {
        let f = banded_fixture(&format!("lookahead-{ahead}"), ahead);
        let tok = Tok::create(&f.dir).unwrap();
        let reference = Banded::new(&f);
        for &(b, precision, most) in &devices {
            let (_l, s) = session_on(&f, b, precision);
            s.write_text(&TEXTS, None).unwrap();
            let got = s.run().unwrap().rows();
            let (mut worst, mut moved) = (0.0f64, 0.0f64);
            for (t, g) in TEXTS.iter().zip(&got) {
                let ids = tok.row(t, None).unwrap();
                let want = reference.embed(&ids, ahead as usize);
                worst = worst.max(close(g, &want));
                assert!(
                    close(g, &want) < most,
                    "{b}, precision {precision}, lookahead {ahead}, {t:?}: {g:?} vs {want:?}"
                );
                moved = moved.max(close(g, &reference.embed(&ids, usize::MAX)));
            }
            println!(
                "{b}, precision {precision}, lookahead {ahead}: max abs diff {worst:e}, {moved:e} from the whole row"
            );
            // The band changes the vectors: the whole-row encoder is not it.
            assert!(moved > 1e-3, "{b}: lookahead {ahead}: the band changed nothing ({moved})");
        }
    }
}

#[test]
fn a_band_as_long_as_the_row_is_the_whole_row_encoder() {
    let f = banded_fixture("lookahead-wide", 64);
    let tok = Tok::create(&f.dir).unwrap();
    let plain = PlainBert::new(&f.dir);
    for (b, precision, most) in banded_devices() {
        let (_l, s) = session_on(&f, b, precision);
        s.write_text(&TEXTS, None).unwrap();
        for (t, g) in TEXTS.iter().zip(&s.run().unwrap().rows()) {
            let ids = tok.row(t, None).unwrap();
            let want = plain.embed(&ids, &vec![1; ids.len()], &vec![0; ids.len()], TURBO_POOLING_MEAN, 8, true);
            assert!(close(g, &want) < most, "{b}, precision {precision}, {t:?}: {g:?} vs {want:?}");
        }
    }
}

#[test]
fn a_band_of_no_keys_is_refused_at_load() {
    let mut f = Fixture::model("lookahead-zero");
    f.manifest["architecture"]["lookahead"] = json!(0);
    let e = f.load().err().unwrap();
    assert!(e.is(BUNDLE_INVALID, "architecture.lookahead: 0 is not a band of 1 key or more"), "{e:?}");
}

/// Every device's backend but the CPU's, CUDA's, Level Zero's and Metal's runs whole
/// rows, so rule 6 skips the banded artifact there before any file of it is read. The weights
/// are left out of the bundle: the CPU, which runs the band, opens them
/// and finds none, and every other backend refuses with the reason.
#[test]
fn a_banded_artifact_is_skipped_unread_where_attention_takes_no_band() {
    let mut f = Fixture::model("lookahead-skipped");
    f.manifest["architecture"]["lookahead"] = json!(4);
    // npu, listed for the artifact, has no band: the choice names it.
    f.manifest["artifacts"][0]["backends"].as_array_mut().unwrap().push(json!("npu"));
    std::fs::remove_file(f.dir.join("weights/model.safetensors")).unwrap();
    let e = f.load().err().unwrap();
    assert!(e.is(BUNDLE_NOT_FOUND, "weights/model.safetensors"), "{e:?}");

    let bundle = turbo::bundle::Bundle::open(f.write()).unwrap();
    let raw = backend::format_bit(backend::TURBO_FORMAT_SAFETENSORS);
    let b = "npu";
    let e = turbo::model::choose(&bundle.manifest, b, raw, "").unwrap_err();
    let why = format!("weights-f32: architecture.lookahead 4: the {b} backend has no limited-lookahead attention");
    assert_eq!(e.code, BUNDLE_NO_ARTIFACT, "{}", e.message);
    assert!(e.message.contains(&why), "{}", e.message);

    // The same through turbo_model_load where this machine has such a device.
    let mut rt = std::ptr::null_mut();
    assert_eq!(unsafe { turbo_runtime_create(std::ptr::null(), &mut rt, &mut new_error()) }, 0);
    let present = first_of(rt, b).is_some();
    unsafe { turbo_runtime_release(rt) };
    if present {
        let e = f.load_on(|rt| first_of(rt, b).unwrap()).err().unwrap();
        assert!(e.is(BUNDLE_NO_ARTIFACT, &format!("the {b} backend has no limited-lookahead attention")), "{e:?}");
    }
}

/// With `TURBO_TEST_BANDED_BUNDLE` naming a real BERT or RoBERTa bundle,
/// that bundle's weights with a band of `ahead` keys, split into heads of
/// 32, 64 and 128 values (each a model the CPU and a device run alike,
/// reaching every head width's attention kernel; CUDA, which has none for
/// heads of 128, is skipped at that width alone), on each device of
/// BANDED at MODEL and FASTEST, against the CPU on rows long enough to
/// fill the staged kernel's group; skipped without it. The bound is
/// docs/conformance.md's cosine for the session's compute dtype.
#[test]
fn a_real_bundle_with_a_band_matches_the_cpu_on_each_device() {
    let test = "a_real_bundle_with_a_band_matches_the_cpu_on_each_device";
    let Ok(dir) = std::env::var("TURBO_TEST_BANDED_BUNDLE") else {
        return println!("{test}: skipped: TURBO_TEST_BANDED_BUNDLE is not set");
    };
    let src = turbo_pins::require(&dir);
    let devices: Vec<_> = banded_devices().into_iter().filter(|d| d.0 != "cpu").collect();
    let long = TEXTS.join(" ").repeat(4);
    let texts = [long.as_str(), TEXTS[0], TEXTS[1], TEXTS[2]];
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(src.join("manifest.json")).unwrap()).unwrap();
    let hidden = manifest["architecture"]["hidden"].as_u64().unwrap();
    let plain = Loaded::load(&src).unwrap_or_else(|e| panic!("{e:?}"));
    let ps = Session::create(plain.m, Some(&session_desc(4, 0, TURBO_PRECISION_EXACT))).unwrap();
    ps.write_text(&texts, None).unwrap();
    let whole = ps.run().unwrap().rows();
    for (head, ahead) in [(64, 3u32), (32, 16), (128, 1)] {
        // Under the workspace's target, so the files can be hard links
        // where the bundle is on the same file system: a bundle refuses
        // a symbolic link that resolves outside it.
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../target");
        let dir = root.join(format!("banded-tests/{}-{head}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        link_tree(&src, &dir);
        let mut m = manifest.clone();
        m["architecture"]["lookahead"] = json!(ahead);
        m["architecture"]["heads"] = json!(hidden / head);
        // The link is removed first: writing through it would change the bundle's own.
        std::fs::remove_file(dir.join("manifest.json")).unwrap();
        std::fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(&m).unwrap()).unwrap();
        let cpu = Loaded::load(&dir).unwrap_or_else(|e| panic!("{e:?}"));
        let cs = Session::create(cpu.m, Some(&session_desc(4, 0, TURBO_PRECISION_EXACT))).unwrap();
        cs.write_text(&texts, None).unwrap();
        let want = cs.run().unwrap().rows();
        // The band moves these vectors further than any device may.
        let moved = want.iter().zip(&whole).map(|(w, x)| cosine(w, x)).fold(1.0, f64::min);
        assert!(moved < 0.999, "heads of {head}, band {ahead}: the band changed little ({moved})");
        for &(b, precision, _) in &devices {
            let l = match Loaded::load_on(&dir, |rt| first_of(rt, b).unwrap()) {
                Ok(l) => l,
                Err(e) if no_kernel_for_width(b, head, &e) => {
                    println!("{b}, heads of {head}: skipped: {}", e.message);
                    continue;
                }
                Err(e) => panic!("{e:?}"),
            };
            let s = Session::create(l.m, Some(&session_desc(4, 0, precision))).unwrap();
            s.write_text(&texts, None).unwrap();
            let got = s.run().unwrap().rows();
            let dtype = s.info().compute_dtype;
            let floor = if dtype == TURBO_DTYPE_F32 { 0.9999 } else { 0.999 };
            let cos = got.iter().zip(&want).map(|(g, w)| cosine(g, w)).fold(1.0, f64::min);
            println!(
                "{b}, precision {precision}, dtype {dtype}, heads of {head}, band {ahead}: 1 - min cosine {:.3e}",
                1.0 - cos
            );
            assert!(cos >= floor, "{b}, precision {precision}, heads of {head}: cosine {cos}");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

/// Whether a refusal at load is the one gap the real-bundle test skips:
/// CUDA has no attention kernel for heads of 128 and says so, naming the
/// width. Any other backend, width or failure fails the test.
fn no_kernel_for_width(backend: &str, head: u64, e: &Failure) -> bool {
    backend == "cuda" && head == 128 && e.is(UNSUPPORTED, "heads 128 wide")
}

#[test]
fn only_cuda_at_heads_of_128_skips_a_width() {
    let refusal = |head: u64| Failure { code: UNSUPPORTED, field: 0, message: format!("heads {head} wide") };
    assert!(no_kernel_for_width("cuda", 128, &refusal(128)));
    for (backend, head) in [("cuda", 32), ("cuda", 64), ("levelzero", 32), ("levelzero", 64), ("levelzero", 128)] {
        assert!(!no_kernel_for_width(backend, head, &refusal(head)), "{backend}, heads of {head}");
    }
    // The width's own refusal alone: another code or message fails.
    let other = Failure { code: INVALID_ARGUMENT, field: 0, message: "heads 128 wide".into() };
    assert!(!no_kernel_for_width("cuda", 128, &other));
    let other = Failure { code: UNSUPPORTED, field: 0, message: "device memory for the weights".into() };
    assert!(!no_kernel_for_width("cuda", 128, &other));
}

/// Every file under `src`, linked or copied into `dst`.
fn link_tree(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).unwrap();
    for e in std::fs::read_dir(src).unwrap() {
        let (e, to) = {
            let e = e.unwrap();
            let to = dst.join(e.file_name());
            (e, to)
        };
        if e.file_type().unwrap().is_dir() {
            link_tree(&e.path(), &to);
        } else {
            std::fs::hard_link(e.path(), &to).or_else(|_| std::fs::copy(e.path(), &to).map(|_| ())).unwrap();
        }
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(&p, &q)| f64::from(p) * f64::from(q)).sum::<f64>();
    dot(a, b) / (dot(a, a) * dot(b, b)).sqrt()
}
