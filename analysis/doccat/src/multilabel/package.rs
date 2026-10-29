//! The multi-label model's package: an OpenNLP-style `.bin` zip archive
//! (`turbo_train::ModelPackage`) whose `manifest.properties` names the
//! component and the feature factory and whose model entry holds the
//! [`MultiLabelDoccatModel`] bytes, thresholds included.

use turbo_maxent::ModelFile;
use turbo_train::{Manifest, ModelPackage};

use super::error::MultiLabelError;
use super::model::{MODEL_ENTRY, MultiLabelDoccatModel};
use crate::model::FACTORY_NAME;

/// The `Component-Name` of a multi-label model package.
pub const COMPONENT_NAME: &str = "MultiLabelDoccat";

/// `MultiLabelDoccat` serialized: the manifest with the factory entries
/// ([`crate::DoccatFactory::manifest_entries`]) and the model entry, which
/// carries the labels, heads, thresholds, `max_labels`, and `min_score`.
pub fn package_multilabel_model(
    model: &MultiLabelDoccatModel,
    timestamp_ms: i64,
) -> Result<ModelPackage, MultiLabelError> {
    let entry = model.write()?;
    let mut manifest = Manifest::new(COMPONENT_NAME, model.language(), timestamp_ms);
    manifest.set("factory", FACTORY_NAME);
    for (key, value) in factory_entries(model) {
        manifest.set(&key, &value);
    }
    let mut package = ModelPackage::new(manifest);
    package.add_entry(MODEL_ENTRY, entry);
    Ok(package)
}

/// The manifest property naming the feature generators, from the model's
/// context generator: `doccat.featureGenerators`, the comma-separated
/// generator specifications, as `DoccatFactory.createManifestEntries` builds it.
fn factory_entries(model: &MultiLabelDoccatModel) -> Vec<(String, String)> {
    let builtins = model.context_generator().builtins();
    if builtins.is_empty() {
        return Vec::new();
    }
    let specs: Vec<String> = builtins.iter().map(crate::model::encode_single_generator).collect();
    vec![(crate::model::FEATURE_GENERATORS_KEY.to_owned(), specs.join(","))]
}

/// Opens the `.bin` archive's bytes as a [`MultiLabelDoccatModel`]:
/// validates `Component-Name`, reads the model entry, and opens it as
/// [`MultiLabelDoccatModel::open`]. The thresholds calibrated into the
/// model round-trip here.
pub fn open_package(bytes: Vec<u8>) -> Result<MultiLabelDoccatModel, MultiLabelError> {
    let file = ModelFile::open(bytes).map_err(|e| MultiLabelError::Package(e.to_string()))?;
    if file.component_name() != COMPONENT_NAME {
        return Err(MultiLabelError::Package(format!(
            "expected Component-Name {COMPONENT_NAME}, got {:?}",
            file.component_name()
        )));
    }
    let entry = file.entry(MODEL_ENTRY).map_err(|e| MultiLabelError::Package(e.to_string()))?;
    Ok(MultiLabelDoccatModel::open(&entry)?.with_package_identity(file.opennlp_version(), &file.sha256().to_hex()))
}
