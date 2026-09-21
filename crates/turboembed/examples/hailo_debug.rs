//! Hailo debug probe: raw vector stats, pairwise cosines, and a centered
//! (Pearson) comparison against the FP32 golden. Bring-up diagnostics for
//! docs/hailo-embed.md — not a test.

#[cfg(feature = "hailo")]
mod imp {
    use turboembed::{Device, EmbedOptions, Engine};

    fn stats(v: &[f32]) -> (f64, f32, f32) {
        let norm = v
            .iter()
            .map(|x| (*x as f64) * (*x as f64))
            .sum::<f64>()
            .sqrt();
        let min = v.iter().copied().fold(f32::INFINITY, f32::min);
        let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        (norm, min, max)
    }

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for (x, y) in a.iter().zip(b.iter()) {
            d += (*x as f64) * (*y as f64);
            na += (*x as f64) * (*x as f64);
            nb += (*y as f64) * (*y as f64);
        }
        d / (na.sqrt() * nb.sqrt())
    }

    pub fn run() {
        let engine = Engine::create(Device::Hailo).expect("create");
        engine.load_model("minilm").expect("load");
        let texts = [
            "hello world",
            "hello there",
            "quantum physics research paper",
            "柏林是德国的首都",
        ];
        let mut vecs = Vec::new();
        let mut opts = EmbedOptions::default();
        opts.normalize = Some(false);
        for t in texts {
            let raw = engine.embed_one("minilm", t, &opts).expect("embed");
            let (norm, min, max) = stats(raw.values());
            println!(
                "{t:35} prenorm l2={norm:9.4} min={min:8.4} max={max:8.4} first4={:?}",
                &raw.values()[..4]
            );
            vecs.push(raw.values().to_vec());
        }
        println!("\npairwise cosines (unnormalized):");
        for i in 0..vecs.len() {
            for j in (i + 1)..vecs.len() {
                println!("  [{i}]~[{j}]: {:.4}", cosine(&vecs[i], &vecs[j]));
            }
        }

        let golden_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../testdata/reference_embeddings/ort_cuda_minilm_short.json");
        let raw = std::fs::read_to_string(&golden_path).expect("golden read");
        let golden: serde_json::Value = serde_json::from_str(&raw).expect("golden parse");
        let golden_vec: Vec<f32> = golden["vector"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let ours = &vecs[0];
        let mean_of = |v: &[f32]| v.iter().map(|x| *x as f64).sum::<f64>() / v.len() as f64;
        let (mg, mo) = (mean_of(&golden_vec), mean_of(ours));
        println!("\ngolden mean {mg:.6}, ours mean {mo:.6}");
        println!("vs golden: plain cos {:.4}", cosine(ours, &golden_vec));
    }
}

fn main() {
    #[cfg(feature = "hailo")]
    imp::run();
    #[cfg(not(feature = "hailo"))]
    eprintln!("hailo_debug needs --features hailo (Raspberry Pi AI HAT+)");
}
