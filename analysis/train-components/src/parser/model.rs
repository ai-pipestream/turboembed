//! The `ParserModel` file: `new ParserModel(...)` serialized, and the copies
//! `ParserModel.update*Model` make. `turbo_parser::ParserModel` reads it.

use turbo_parser::model::{
    ATTACH_MODEL_ENTRY_NAME, BUILD_MODEL_ENTRY_NAME, CHECK_MODEL_ENTRY_NAME, CHUNKER_TAGGER_MODEL_ENTRY_NAME,
    COMPONENT_NAME, HEAD_RULES_MODEL_ENTRY_NAME, HeadRulesArtifact, PARSER_TAGGER_MODEL_ENTRY_NAME, PARSER_TYPE,
    POS_MODEL_SERIALIZER, ParserModel, ParserModelError, ParserType,
};
use turbo_train::ModelPackage;

use crate::error::Result;
use crate::package::{Stamp, add_artifacts, serializer_class_key};

/// Packages and opens a new [`ParserModel`] from serialized sub-model artifacts
/// (`new ParserModel(...)`): the manifest with `manifest_entries`, the parser type and the
/// serializer classes of the head rules and the tagger, and the entries in the JVM's zip order.
/// `attach_bytes` must be given for a `TREEINSERT` model and only for one.
#[allow(clippy::too_many_arguments)]
pub fn from_serialized_parts(
    language: &str,
    build_bytes: Vec<u8>,
    check_bytes: Vec<u8>,
    attach_bytes: Option<Vec<u8>>,
    tagger_bytes: Vec<u8>,
    chunker_bytes: Vec<u8>,
    head_rules: HeadRulesArtifact,
    parser_type: ParserType,
    manifest_entries: &[(&str, &str)],
    stamp: Stamp<'_>,
) -> Result<ParserModel> {
    if parser_type == ParserType::Chunking && attach_bytes.is_some() {
        return Err(ParserModelError::UnexpectedAttachModel.into());
    }
    if parser_type == ParserType::TreeInsert && attach_bytes.is_none() {
        return Err(ParserModelError::MissingArtifact(ATTACH_MODEL_ENTRY_NAME).into());
    }

    let mut manifest = stamp.manifest(COMPONENT_NAME, language);
    for &(k, v) in manifest_entries {
        manifest.set(k, v);
    }
    manifest.set(PARSER_TYPE, parser_type.as_str());
    manifest.set(&serializer_class_key(HEAD_RULES_MODEL_ENTRY_NAME), head_rules.serializer_class());
    manifest.set(&serializer_class_key(PARSER_TAGGER_MODEL_ENTRY_NAME), POS_MODEL_SERIALIZER);

    let mut package = ModelPackage::new(manifest);
    let mut artifacts: Vec<(&str, Vec<u8>)> = Vec::with_capacity(6);
    artifacts.push((BUILD_MODEL_ENTRY_NAME, build_bytes));
    artifacts.push((CHECK_MODEL_ENTRY_NAME, check_bytes));
    if let Some(attach) = attach_bytes {
        artifacts.push((ATTACH_MODEL_ENTRY_NAME, attach));
    }
    artifacts.push((PARSER_TAGGER_MODEL_ENTRY_NAME, tagger_bytes));
    artifacts.push((CHUNKER_TAGGER_MODEL_ENTRY_NAME, chunker_bytes));
    artifacts.push((HEAD_RULES_MODEL_ENTRY_NAME, head_rules.serialize_to_bytes()));
    add_artifacts(&mut package, artifacts)?;
    Ok(ParserModel::from_bytes(package.to_bytes())?)
}

/// A copy of `model` with `build.model` replaced by `build_bytes` (`ParserModel.updateBuildModel`).
pub fn update_build_model(model: &ParserModel, build_bytes: Vec<u8>) -> Result<ParserModel> {
    with_replaced_entry(model, BUILD_MODEL_ENTRY_NAME, build_bytes)
}

/// A copy of `model` with `check.model` replaced by `check_bytes` (`ParserModel.updateCheckModel`).
pub fn update_check_model(model: &ParserModel, check_bytes: Vec<u8>) -> Result<ParserModel> {
    with_replaced_entry(model, CHECK_MODEL_ENTRY_NAME, check_bytes)
}

/// A copy of `model` with `parsertager.postagger` replaced by `tagger_bytes`
/// (`ParserModel.updateTaggerModel`).
pub fn update_tagger_model(model: &ParserModel, tagger_bytes: Vec<u8>) -> Result<ParserModel> {
    with_replaced_entry(model, PARSER_TAGGER_MODEL_ENTRY_NAME, tagger_bytes)
}

/// A copy of `model` with `parserchunker.chunker` replaced by `chunker_bytes`
/// (`ParserModel.updateChunkerModel`).
pub fn update_chunker_model(model: &ParserModel, chunker_bytes: Vec<u8>) -> Result<ParserModel> {
    with_replaced_entry(model, CHUNKER_TAGGER_MODEL_ENTRY_NAME, chunker_bytes)
}

fn with_replaced_entry(model: &ParserModel, target_name: &'static str, new_bytes: Vec<u8>) -> Result<ParserModel> {
    let get = |name: &'static str| -> Vec<u8> {
        if name == target_name {
            new_bytes.clone()
        } else {
            model.file().entry(name).expect("validated entry").into_owned()
        }
    };
    let attach = match model.parser_type() {
        ParserType::Chunking => None,
        ParserType::TreeInsert => Some(get(ATTACH_MODEL_ENTRY_NAME)),
    };
    from_serialized_parts(
        model.language(),
        get(BUILD_MODEL_ENTRY_NAME),
        get(CHECK_MODEL_ENTRY_NAME),
        attach,
        get(PARSER_TAGGER_MODEL_ENTRY_NAME),
        get(CHUNKER_TAGGER_MODEL_ENTRY_NAME),
        model.head_rules().clone(),
        model.parser_type(),
        &[],
        Stamp::at(0),
    )
}

#[cfg(test)]
mod tests {
    use turbo_parser::model::HEAD_RULES_SERIALIZER_KEY;

    use super::*;

    #[test]
    fn the_reader_looks_for_the_head_rules_serializer_where_the_writer_puts_it() {
        assert_eq!(serializer_class_key(HEAD_RULES_MODEL_ENTRY_NAME), HEAD_RULES_SERIALIZER_KEY);
    }
}
