//! `Encoder::run_layer_ranges` (#1616): the ranges from each of the last
//! layers in one pass, the last block what `run_ranges` gives bit for
//! bit, on the sealed BERT (2 layers) and ModernBERT (3 layers).

use std::path::{Path, PathBuf};

use turbo::turbo_token_range;
use turbo_analysis_embed::*;

fn testdata() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata")
}

#[test]
fn the_last_block_is_the_plain_range_run() {
    for (bundle, layers) in [("tiny-bert-bundle", 2u32), ("tiny-modernbert-bundle", 3), ("tiny-deberta-bundle", 2)] {
        let o = EncoderOptions { max_batch: 2, max_seq: 16, max_ranges: 12, ..EncoderOptions::default() };
        let mut enc = Encoder::open(&testdata().join(bundle), &o).unwrap();
        let mut rows = Rows::new(16, 0);
        let a = rows.push(&[&[5, 9, 12, 7, 30, 2]]);
        let b = rows.push(&[&[11, 4, 8]]);
        let r = |row, start, end| turbo_token_range { row, start, end };
        let ranges = [r(a, 0, 6), r(a, 2, 3), r(b, 0, 3), r(b, 1, 2)];
        let (mut plain, mut tapped, mut via_opts) = (Vec::new(), Vec::new(), Vec::new());
        enc.run_ranges(&rows, &ranges, &mut plain).unwrap();
        enc.run_layer_ranges(&rows, &ranges, layers, &mut tapped).unwrap();
        let mut enc_opts = Encoder::open(&testdata().join(bundle), &EncoderOptions { layers, ..o }).unwrap();
        enc_opts.run_ranges(&rows, &ranges, &mut via_opts).unwrap();
        assert_eq!(
            via_opts.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            tapped.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "{bundle}: EncoderOptions.layers"
        );
        assert_eq!(tapped.len(), layers as usize * plain.len(), "{bundle}");
        let last = &tapped[(layers as usize - 1) * plain.len()..];
        assert_eq!(
            last.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            plain.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "{bundle}"
        );
        assert_ne!(&tapped[..plain.len()], last, "{bundle}: the first block is an earlier layer's");
        // The session holds twelve vectors: one range more than fits.
        let over = vec![ranges[1]; 12 / layers as usize + 1];
        let e = enc.run_layer_ranges(&rows, &over, layers, &mut tapped).unwrap_err();
        assert_eq!(e.code, turbo::status::CAPACITY, "{bundle}: {e:?}");
    }
}
