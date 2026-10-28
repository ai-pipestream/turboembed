# turbo-geo

Offline gazetteer entity extraction (`GeoEntityLinker`) and forward/reverse geocoding (`Geocoder`) over a compact memory-mappable binary index (`TRBGEO\x01\x00`), scoring candidate Administrative, Country, and Place entries by fuzzy matching, hierarchical containment, population prior, and proximity clustering without network dependencies.

## Features

- **Gazetteer entity linking (`GeoEntityLinker`)**: Matches named-entity location spans (`analysis/geo/src/linker/mod.rs`) against a memory-mapped or owned `GeoIndex` (`analysis/geo/src/index.rs`), combining normalized exact-key and fuzzy lookup (`analysis/geo/src/fold.rs`, `analysis/geo/src/linker/fuzzy.rs`) with country/region context boosts and spatial clustering (`analysis/geo/src/linker/spatial.rs`).
- **Forward and reverse geocoding (`Geocoder`, `GeocodeAnnotator`)**: Resolves free-form place strings and coordinate pairs (`analysis/geo/src/geocode.rs`, `analysis/geo/src/geocode/annotator.rs`) into structured `GeoPlace` layers on `Document`.
- **Multi-source data compilers**: Ingestion builders for GeoNames (`analysis/geo/src/geonames.rs`), Overture Maps (`analysis/geo/src/overture.rs`), and administrative hierarchy overlays (`analysis/geo/src/overlay.rs`, `analysis/geo/src/containment.rs`), plus an embedded starter gazetteer (`analysis/geo/src/bundled.rs`).

## Testing, Accuracy, and Speed

- **Test suite**: Covered by `analysis/geo/tests/bundled.rs`, `analysis/geo/tests/loaders.rs`, `analysis/geo/tests/mention_admission.rs`, `analysis/geo/tests/names.rs`, `analysis/geo/tests/overlay.rs`, and `analysis/geo/tests/snapshot.rs`.
- **Accuracy and speed**: Memory-mapped `TRBGEO\x01\x00` lookup runs in zero-copy read-only slices with $O(1)$ exact-hash lookup and bounded candidate scoring. See `analysis/geo/data/README.md` for gazetteer provenance and licensing.
