//! No-terms distillation must complete embedding progress before reduction.

use std::fs;
use std::path::Path;

use turbo_bundle::distill::{self, Progress};
use turbo_bundle::recipe::Recipe;

#[test]
fn base_only_distillation_reports_every_completed_row_before_reduction() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let work = std::env::temp_dir().join(format!("turbo-bundle-progress-{}", std::process::id()));
    fs::create_dir_all(&work).unwrap();
    let bundle = work.join("bundle");
    let mut recipe = Recipe::load(&root.join("testdata/tiny-static-recipe/recipe.json")).unwrap();
    let mut events = Vec::new();
    let staged =
        distill::stage_with(&recipe, &root.join("testdata/tiny-bert-bundle"), &bundle, &[], &mut |p| events.push(p))
            .unwrap();
    assert!(staged.added.is_empty() && staged.refused.is_empty());
    let Progress::Kept { kept, added: 0, .. } = events[0] else { panic!("missing base-only kept count") };
    let embedded: Vec<_> = events
        .iter()
        .enumerate()
        .filter_map(|(i, p)| match p {
            Progress::Embedded { done, total } => Some((i, *done, *total)),
            _ => None,
        })
        .collect();
    eprintln!("base-only kept={kept}, embedding progress={embedded:?}");
    assert_eq!(embedded.last().map(|(_, done, total)| (*done, *total)), Some((kept, kept)));
    assert!(embedded.iter().all(|(_, done, total)| done <= total && *total == kept));
    assert!(embedded.windows(2).all(|w| w[0].1 < w[1].1));
    let reduced = events.iter().position(|p| matches!(p, Progress::Reduced { .. })).unwrap();
    assert!(embedded.iter().all(|(i, _, _)| *i < reduced));
    assert!(matches!(events.last(), Some(Progress::Measured { .. })));
    distill::seal(&mut recipe, &bundle).unwrap();
    fs::remove_dir_all(work).unwrap();
}
