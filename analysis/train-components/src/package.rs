//! What `BaseModel` adds to a model file for a tool factory's artifacts,
//! and the order the JVM writes the zip entries in.

use turbo_train::package::OPENNLP_VERSION;
use turbo_train::{Manifest, ModelPackage, TimeZone};

use crate::error::Result;
use crate::java_hash::string_order;

/// `BaseModel.FACTORY_NAME`: the manifest entry naming the factory class.
pub const FACTORY_KEY: &str = "factory";
/// The class `BaseModel.serialize` records for a `Dictionary` artifact,
/// under `serializer-class-<entry name>`.
pub const DICTIONARY_SERIALIZER: &str = "opennlp.tools.util.model.DictionarySerializer";

/// When and by which OpenNLP a model file was written: the instant
/// `BaseModel` records as `Timestamp`, the `OpenNLP-Version`, and the
/// time zone the date line and the zip entries' times are in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stamp<'v> {
    pub timestamp_ms: i64,
    pub version: &'v str,
    pub zone: TimeZone,
}

impl Stamp<'static> {
    /// A model written by this port at `timestamp_ms`, in UTC.
    pub fn at(timestamp_ms: i64) -> Stamp<'static> {
        Stamp { timestamp_ms, version: OPENNLP_VERSION, zone: TimeZone::UTC }
    }
}

impl Stamp<'_> {
    /// The base manifest of a model of `component` for `language`.
    pub fn manifest(&self, component: &str, language: &str) -> Manifest {
        Manifest::new(component, language, self.timestamp_ms).with_version(self.version).with_zone(self.zone)
    }
}

/// The manifest key `serializer-class-<entry>`.
pub fn serializer_class_key(entry: &str) -> String {
    format!("serializer-class-{entry}")
}

/// `BaseModel.serialize` writes the artifacts in the iteration order of
/// its `HashMap<String, Object>`, whose keys were put in `insertion`
/// order (the manifest first, then the factory's artifacts, then the
/// model and its descriptor). This is that order, as indices into
/// `insertion`. The map never holds more than a few entries, so its table
/// stays at 16 buckets.
pub fn artifact_order(insertion: &[&str]) -> Result<Vec<usize>> {
    Ok(string_order(insertion.iter().copied()))
}

/// Adds `artifacts` (name and bytes) to `package` in the JVM's zip order
/// for a map that held the manifest and then these names in this order,
/// and places the manifest where the JVM writes it. For a sentence or
/// token model, with or without a dictionary, `manifest.properties`
/// hashes to an earlier bucket than `sent.model`, `token.model` and
/// `abbreviations.dictionary` and comes first; a part-of-speech model's
/// `tags.tagdict` comes before it.
pub fn add_artifacts(package: &mut ModelPackage, artifacts: Vec<(&str, Vec<u8>)>) -> Result<()> {
    let mut names: Vec<&str> = vec![turbo_train::package::MANIFEST_ENTRY];
    names.extend(artifacts.iter().map(|(n, _)| *n));
    let order = artifact_order(&names)?;
    let mut artifacts: Vec<Option<(&str, Vec<u8>)>> = artifacts.into_iter().map(Some).collect();
    let mut added = 0;
    for i in order {
        if i == 0 {
            package.place_manifest(added);
            continue;
        }
        let (name, bytes) = artifacts[i - 1].take().expect("each index once");
        package.add_entry(name, bytes);
        added += 1;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_manifest_comes_first_and_the_dictionary_where_the_fixtures_have_it() {
        // SentenceModel: the factory's dictionary goes in before sent.model,
        // and the file has manifest, sent.model, abbreviations.dictionary
        // (fixture en-abbrev.bin). TokenizerModel likewise, and the file has
        // manifest, abbreviations.dictionary, token.model (fixture
        // tokenizer-abbreviations.bin).
        let sentence = ["manifest.properties", "abbreviations.dictionary", "sent.model"];
        assert_eq!(artifact_order(&sentence).unwrap(), vec![0, 2, 1]);
        let token = ["manifest.properties", "abbreviations.dictionary", "token.model"];
        assert_eq!(artifact_order(&token).unwrap(), vec![0, 1, 2]);
        assert_eq!(artifact_order(&["manifest.properties", "sent.model"]).unwrap(), vec![0, 1]);
        assert_eq!(artifact_order(&["manifest.properties", "token.model"]).unwrap(), vec![0, 1]);
        // POSModel: the factory's tag dictionary, then the classifier and
        // the descriptor; the file has tags.tagdict, manifest.properties,
        // generator.featuregen, pos.model (fixture pos-fixture.bin), and
        // without a dictionary manifest.properties, generator.featuregen,
        // pos.model (the published models).
        let pos = ["manifest.properties", "tags.tagdict", "pos.model", "generator.featuregen"];
        assert_eq!(artifact_order(&pos).unwrap(), vec![1, 0, 3, 2]);
        assert_eq!(
            artifact_order(&["manifest.properties", "pos.model", "generator.featuregen"]).unwrap(),
            vec![0, 2, 1]
        );
        assert_eq!(artifact_order(&["manifest.properties", "lemmatizer.model"]).unwrap(), vec![0, 1]);
    }

    #[test]
    fn the_manifest_is_placed_where_the_jvm_writes_it() {
        let mut package = ModelPackage::new(Stamp::at(0).manifest("POSTaggerME", "en"));
        add_artifacts(
            &mut package,
            vec![("tags.tagdict", vec![1]), ("pos.model", vec![2]), ("generator.featuregen", vec![3])],
        )
        .unwrap();
        assert_eq!(package.entry_names(), ["tags.tagdict", "manifest.properties", "generator.featuregen", "pos.model"]);
        let mut package = ModelPackage::new(Stamp::at(0).manifest("SentenceDetectorME", "en"));
        add_artifacts(&mut package, vec![("abbreviations.dictionary", vec![1]), ("sent.model", vec![2])]).unwrap();
        assert_eq!(package.entry_names(), ["manifest.properties", "sent.model", "abbreviations.dictionary"]);
    }
}
