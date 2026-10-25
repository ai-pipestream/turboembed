//! Values captured from the production methods before shared-hash migration.
use turbo_analysis::Annotator;
use turbo_structure::{KeyValueAnnotator, TableGridAnnotator};
#[test]
fn table_settings_keep_previous_hash_values() {
    let kv: Vec<_> = [0, 1, 6, u16::MAX]
        .into_iter()
        .map(|max_key_tokens| KeyValueAnnotator { max_key_tokens }.config_hash())
        .collect();
    let grid: Vec<_> = [0, 1, 1_000_000, u32::MAX as usize, usize::MAX]
        .into_iter()
        .map(|max_cells| TableGridAnnotator { max_cells }.config_hash())
        .collect();

    assert_eq!(kv, [11061404734286987699, 11061403634775359488, 11061411331356756965, 11133459029818539162]);
    assert_eq!(
        grid,
        [13941525070242382259, 13941523970730754048, 15041036698453382259, 13941522165804301466]
            .into_iter()
            .chain([if usize::BITS == 64 { 4505217903955541146 } else { 13941522165804301466 }])
            .collect::<Vec<u64>>()
    );
}
