use model2vec_rs::model::StaticModel;
use static_peers::*;

fn main() {
    let (dir, texts, batches, out) = args();
    let start = std::time::Instant::now();
    let m = StaticModel::from_pretrained(&dir, None, None, None).unwrap();
    println!("model2vec-rs: loaded in {:.1} ms", start.elapsed().as_secs_f64() * 1e3);
    for &b in &batches {
        time("model2vec-rs", b, texts.len(), || {
            for c in texts.chunks(b) {
                std::hint::black_box(m.encode_with_args(c, Some(512), b));
            }
        });
    }
    if let Some(out) = out {
        write(&out, &m.encode_with_args(&texts, Some(512), 1024));
    }
}
