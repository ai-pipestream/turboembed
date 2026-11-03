# turbo-geo

Offline gazetteer lookup (`Gazetteer`, `GazetteerIndex`), gazetteer name finding (`NameFinder`, `NameAnnotator`), mention geocoding (`Geocoder`, `PopulationPriorGeocoder`, `SpatialCoherenceGeocoder`, `GeocodeAnnotator`), and place profile comparison (`PlaceProfiles`) without network dependencies.

## Features

- **Gazetteer index and snapshot persistence (`Gazetteer`, `GazetteerIndex`, `analysis/geo/src/index.rs`, `analysis/geo/src/snapshot/mod.rs`)**: Columnar storage with `Folder` normalized match keys (`analysis/geo/src/fold.rs`), open-addressing key lookup, and binary snapshots starting with the 8-byte `TURBOGEO` magic and `u32` format version followed by length-prefixed little-endian arrays and a trailing `u64` checksum (verified and copied into owned column allocations with offset and code bounds checks on read in `analysis/geo/src/snapshot/read.rs`).
- **Mention geocoding and name finding (`Geocoder`, `GeocodeAnnotator`, `NameFinder`, `NameAnnotator`)**: `NameFinder` and `NameAnnotator` (`analysis/geo/src/names/mod.rs`) match gazetteer names in text over a mapped automaton image (`name_image`) and emit location entity mentions; `Geocoder::resolve` (`analysis/geo/src/geocode.rs`) resolves document text and mention spans (`PopulationPriorGeocoder`, `SpatialCoherenceGeocoder`), and `GeocodeAnnotator` (`analysis/geo/src/geocode/annotator.rs`) consumes entity mentions and writes the `Locations` (`turbo:locations`) layer on `Document`.
- **Multi-source gazetteers and overlays**: Loaders for GeoNames (`GeoNamesGazetteer`, `analysis/geo/src/geonames.rs`), Overture Maps (`OvertureGazetteer`, `analysis/geo/src/overture.rs`), user and in-memory tables (`UserGazetteer`, `InMemoryGazetteer`), overlay and containment rules (`OverlayGazetteer`, `analysis/geo/src/overlay.rs`, `analysis/geo/src/containment.rs`), and the bundled Natural Earth populated-places gazetteer (`BundledGazetteer`, `analysis/geo/src/bundled.rs`).

## Testing, Accuracy, and Speed

- **Test suite**: Covered by `analysis/geo/tests/bundled.rs`, `analysis/geo/tests/loaders.rs`, `analysis/geo/tests/mention_admission.rs`, `analysis/geo/tests/names.rs`, `analysis/geo/tests/overlay.rs`, and `analysis/geo/tests/snapshot.rs`.
- **Accuracy and speed**: Columnar `GazetteerIndex` lookup uses open-addressing tag-byte hash slots with pre-ranked posting lists, while `TURBOGEO` snapshots reload pre-folded columns and tables without reparsing or refolding source text. See `analysis/geo/data/README.md` for gazetteer provenance and licensing.
