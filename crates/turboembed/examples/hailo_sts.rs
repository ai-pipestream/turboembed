//! Hailo STS quality probe: Spearman rank correlation of Hailo-vs-label
//! similarity scores over testdata/corpus/sts-pairs.jsonl.
//!
//! Run on a provisioned Pi:
//!   cargo run -p turboembed --features hailo --release --example hailo_sts

#[cfg(feature = "hailo")]
mod imp {
    use std::path::PathBuf;

    use serde::Deserialize;
    use turboembed::{Device, EmbedOptions, Engine};

    #[derive(Deserialize)]
    struct Pair {
        score: f64,
        text_a: String,
        text_b: String,
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

    pub fn spearman(xs: &[f64], ys: &[f64]) -> f64 {
        fn ranks(v: &[f64]) -> Vec<f64> {
            let mut idx: Vec<usize> = (0..v.len()).collect();
            idx.sort_by(|a, b| v[*a].partial_cmp(&v[*b]).unwrap());
            let mut r = vec![0.0; v.len()];
            let mut i = 0;
            while i < v.len() {
                let mut j = i;
                while j + 1 < v.len() && v[idx[j + 1]] == v[idx[i]] {
                    j += 1;
                }
                let avg = (i + j) as f64 / 2.0 + 1.0;
                for k in i..=j {
                    r[idx[k]] = avg;
                }
                i = j + 1;
            }
            r
        }
        let (rx, ry) = (ranks(xs), ranks(ys));
        let n = xs.len() as f64;
        let mx = rx.iter().sum::<f64>() / n;
        let my = ry.iter().sum::<f64>() / n;
        let (mut cov, mut vx, mut vy) = (0.0, 0.0, 0.0);
        for i in 0..rx.len() {
            let dx = rx[i] - mx;
            let dy = ry[i] - my;
            cov += dx * dy;
            vx += dx * dx;
            vy += dy * dy;
        }
        cov / (vx.sqrt() * vy.sqrt())
    }

    pub fn run() {
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/corpus/sts-pairs.jsonl");
        let raw = std::fs::read_to_string(&path).expect("corpus read");
        let pairs: Vec<Pair> = raw
            .lines()
            .map(|l| serde_json::from_str(l).expect("pair json"))
            .collect();
        println!("loaded {} pairs from {}", pairs.len(), path.display());

        let engine = Engine::create(Device::Hailo).expect("create hailo");
        engine.load_model("minilm").expect("load minilm");
        let opts = EmbedOptions::default();

        let mut labels = Vec::new();
        let mut scores = Vec::new();
        for p in &pairs {
            let a = engine
                .embed_one("minilm", &p.text_a, &opts)
                .expect("embed a");
            let b = engine
                .embed_one("minilm", &p.text_b, &opts)
                .expect("embed b");
            labels.push(p.score);
            scores.push(cosine(a.values(), b.values()));
        }
        let rho = spearman(&labels, &scores);
        println!(
            "hailo minilm STS pairs: n={} spearman={rho:.4}",
            pairs.len()
        );
        println!(
            "cos range: {:.4} .. {:.4}",
            scores.iter().copied().fold(f64::INFINITY, f64::min),
            scores.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        );
    }
}

fn main() {
    #[cfg(feature = "hailo")]
    imp::run();
    #[cfg(not(feature = "hailo"))]
    eprintln!("hailo_sts needs --features hailo (Raspberry Pi AI HAT+)");
}
