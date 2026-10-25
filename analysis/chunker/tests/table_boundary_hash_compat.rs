use turbo_analysis::Annotator;
use turbo_chunker::TableBoundaryAnnotator;

#[test]
fn boundary_settings_preserve_incoming_identities() {
    let actual = [(100_000, 0), (100_000, -100), (1, 0), (0, -1), (i32::MAX, i32::MIN)].map(
        |(inside_cost, row_boundary_cost)| TableBoundaryAnnotator { inside_cost, row_boundary_cost }.config_hash(),
    );
    // Actual public-method values captured before the incoming loop migration.
    assert_eq!(
        actual,
        [2973721608113647949, 343166301830395193, 589727492704079044, 343057468895825504, 94855848640228586]
    );
}
