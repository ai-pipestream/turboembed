use statembed::StaticEmbedding;
use static_peers::*;

fn main() {
    let (dir, texts, batches, out) = args();
    let t: Vec<&str> = texts.iter().map(String::as_str).collect();
    let name = if cfg!(feature = "rayon") { "statembed+rayon" } else { "statembed" };
    let start = std::time::Instant::now();
    let mut m = StaticEmbedding::from_dir(&dir, Some(true), None, None).unwrap();
    m.init().unwrap();
    println!("{name}: loaded in {:.1} ms", start.elapsed().as_secs_f64() * 1e3);
    for &b in &batches {
        time(name, b, t.len(), || {
            if b == 1 {
                for x in &t {
                    std::hint::black_box(m.embed_text(x).unwrap());
                }
            } else {
                for c in t.chunks(b) {
                    std::hint::black_box(m.embed_texts(c, Some(b)).unwrap());
                }
            }
        });
    }
    if let Some(out) = out {
        // A text the library refuses is a row of NaN, which the score counts.
        let dim = m.embed_text("a").unwrap().len();
        let rows: Vec<Vec<f32>> = t.iter().map(|x| m.embed_text(x).unwrap_or_else(|_| vec![f32::NAN; dim])).collect();
        write(&out, &rows);
    }
}
