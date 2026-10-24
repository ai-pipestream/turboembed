//! The header's constants as the Rust mirrors give them.

use std::process::Command;

use crate::{include, run, scratch};
use turbo::*;

#[test]
fn mirrored_constants_match_the_header() {
    use turbo::status::*;
    // Every constant the Rust side mirrors belongs here.
    let constants: &[(&str, i64)] = &[
        ("TURBO_ERROR_MESSAGE_LEN", TURBO_ERROR_MESSAGE_LEN as i64),
        ("TURBO_TRUNCATE_MODEL", TURBO_TRUNCATE_MODEL.into()),
        ("TURBO_TRUNCATE_NONE", TURBO_TRUNCATE_NONE.into()),
        ("TURBO_TRUNCATE_RIGHT", TURBO_TRUNCATE_RIGHT.into()),
        ("TURBO_TRUNCATE_LEFT", TURBO_TRUNCATE_LEFT.into()),
        ("TURBO_PROMPT_NONE", TURBO_PROMPT_NONE.into()),
        ("TURBO_PROMPT_QUERY", TURBO_PROMPT_QUERY.into()),
        ("TURBO_PROMPT_DOCUMENT", TURBO_PROMPT_DOCUMENT.into()),
        ("TURBO_NORMALIZE_MODEL", TURBO_NORMALIZE_MODEL.into()),
        ("TURBO_NORMALIZE_NONE", TURBO_NORMALIZE_NONE.into()),
        ("TURBO_NORMALIZE_L2", TURBO_NORMALIZE_L2.into()),
        ("TURBO_POOLING_MODEL", TURBO_POOLING_MODEL.into()),
        ("TURBO_POOLING_MEAN", TURBO_POOLING_MEAN.into()),
        ("TURBO_POOLING_CLS", TURBO_POOLING_CLS.into()),
        ("TURBO_POOLING_LAST", TURBO_POOLING_LAST.into()),
        ("TURBO_STAGE_MAX", TURBO_STAGE_MAX as i64),
        ("TURBO_OUTPUT_DIMS_MAX", TURBO_OUTPUT_DIMS_MAX as i64),
        ("TURBO_EMBED_STAGE_TOKENIZE", TURBO_EMBED_STAGE_TOKENIZE as i64),
        ("TURBO_EMBED_STAGE_UPLOAD", TURBO_EMBED_STAGE_UPLOAD as i64),
        ("TURBO_EMBED_STAGE_LOOKUP", TURBO_EMBED_STAGE_LOOKUP as i64),
        ("TURBO_EMBED_STAGE_ENCODE", TURBO_EMBED_STAGE_ENCODE as i64),
        ("TURBO_EMBED_STAGE_POOL", TURBO_EMBED_STAGE_POOL as i64),
        ("TURBO_EMBED_STAGE_NORMALIZE", TURBO_EMBED_STAGE_NORMALIZE as i64),
        ("TURBO_EMBED_STAGE_DOWNLOAD", TURBO_EMBED_STAGE_DOWNLOAD as i64),
        ("TURBO_EMBED_STAGE_COUNT", TURBO_EMBED_STAGE_COUNT as i64),
        ("TURBO_STAGE_UNUSED", TURBO_STAGE_UNUSED.into()),
        ("TURBO_STAGE_HOST", TURBO_STAGE_HOST.into()),
        ("TURBO_STAGE_DEVICE", TURBO_STAGE_DEVICE.into()),
        ("TURBO_STAGE_FUSED", TURBO_STAGE_FUSED.into()),
        ("TURBO_BERT_EMBEDDING_TENSORS", turbo::backend::TURBO_BERT_EMBEDDING_TENSORS.into()),
        ("TURBO_BERT_LAYER_TENSORS", turbo::backend::TURBO_BERT_LAYER_TENSORS.into()),
        ("TURBO_FAMILY_BERT", turbo::backend::TURBO_FAMILY_BERT.into()),
        ("TURBO_FAMILY_ROBERTA", turbo::backend::TURBO_FAMILY_ROBERTA.into()),
        ("TURBO_FAMILY_STATIC", turbo::backend::TURBO_FAMILY_STATIC.into()),
        ("TURBO_STATIC_EMBEDDINGS", turbo::backend::TURBO_STATIC_EMBEDDINGS.into()),
        ("TURBO_STATIC_WEIGHTS", turbo::backend::TURBO_STATIC_WEIGHTS.into()),
        ("TURBO_STATIC_MAPPING", turbo::backend::TURBO_STATIC_MAPPING.into()),
        ("TURBO_STATIC_TENSORS", turbo::backend::TURBO_STATIC_TENSORS.into()),
        ("TURBO_FORMAT_SAFETENSORS", turbo::backend::TURBO_FORMAT_SAFETENSORS.into()),
        ("TURBO_FORMAT_OPENVINO_IR", turbo::backend::TURBO_FORMAT_OPENVINO_IR.into()),
        ("TURBO_FORMAT_HEF", turbo::backend::TURBO_FORMAT_HEF.into()),
        ("TURBO_FORMAT_GGUF", turbo::backend::TURBO_FORMAT_GGUF.into()),
        ("TURBO_FORMAT_ONNX", turbo::backend::TURBO_FORMAT_ONNX.into()),
        (
            "TURBO_FORMAT_BIT(TURBO_FORMAT_SAFETENSORS)",
            turbo::backend::format_bit(turbo::backend::TURBO_FORMAT_SAFETENSORS).into(),
        ),
        ("TURBO_FORMAT_BIT(TURBO_FORMAT_HEF)", turbo::backend::format_bit(turbo::backend::TURBO_FORMAT_HEF).into()),
        ("TURBO_FORMAT_BIT(TURBO_FORMAT_ONNX)", turbo::backend::format_bit(turbo::backend::TURBO_FORMAT_ONNX).into()),
        ("TURBO_INPUT_TOKEN_IDS", turbo::backend::TURBO_INPUT_TOKEN_IDS.into()),
        ("TURBO_INPUT_EMBEDDINGS", turbo::backend::TURBO_INPUT_EMBEDDINGS.into()),
        ("TURBO_OUTPUT_HIDDEN_STATES", turbo::backend::TURBO_OUTPUT_HIDDEN_STATES.into()),
        ("TURBO_TASK_EMBED", TURBO_TASK_EMBED.into()),
        ("TURBO_DEVICE_CPU", TURBO_DEVICE_CPU.into()),
        ("TURBO_DEVICE_GPU", TURBO_DEVICE_GPU.into()),
        ("TURBO_DEVICE_IGPU", TURBO_DEVICE_IGPU.into()),
        ("TURBO_DEVICE_NPU", TURBO_DEVICE_NPU.into()),
        ("TURBO_DTYPE_I8", TURBO_DTYPE_I8.into()),
        ("TURBO_DTYPE_I32", TURBO_DTYPE_I32.into()),
        ("TURBO_DTYPE_I64", TURBO_DTYPE_I64.into()),
        ("TURBO_DTYPE_F16", TURBO_DTYPE_F16.into()),
        ("TURBO_DTYPE_BF16", TURBO_DTYPE_BF16.into()),
        ("TURBO_DTYPE_F32", TURBO_DTYPE_F32.into()),
        ("TURBO_DTYPE_F64", TURBO_DTYPE_F64.into()),
        ("TURBO_PLACE_HOST", TURBO_PLACE_HOST.into()),
        ("TURBO_PLACE_PINNED", TURBO_PLACE_PINNED.into()),
        ("TURBO_PLACE_DEVICE", TURBO_PLACE_DEVICE.into()),
        ("TURBO_PLACE_SHARED", TURBO_PLACE_SHARED.into()),
        ("TURBO_HANDLE_HOST_PTR", TURBO_HANDLE_HOST_PTR.into()),
        ("TURBO_HANDLE_CUDA_PTR", TURBO_HANDLE_CUDA_PTR.into()),
        ("TURBO_HANDLE_CL_MEM", TURBO_HANDLE_CL_MEM.into()),
        ("TURBO_HANDLE_ZE_USM", TURBO_HANDLE_ZE_USM.into()),
        ("TURBO_HANDLE_MTL_BUFFER", TURBO_HANDLE_MTL_BUFFER.into()),
        ("TURBO_HANDLE_DMABUF_FD", TURBO_HANDLE_DMABUF_FD.into()),
        ("TURBO_PRECISION_MODEL", TURBO_PRECISION_MODEL.into()),
        ("TURBO_PRECISION_FASTEST", TURBO_PRECISION_FASTEST.into()),
        ("TURBO_PRECISION_EXACT", TURBO_PRECISION_EXACT.into()),
        ("TURBO_AUTOTUNE_RUNTIME", TURBO_AUTOTUNE_RUNTIME.into()),
        ("TURBO_AUTOTUNE_OFF", TURBO_AUTOTUNE_OFF.into()),
        ("TURBO_AUTOTUNE_ON", TURBO_AUTOTUNE_ON.into()),
        ("TURBO_AUTOTUNE_RETUNE", TURBO_AUTOTUNE_RETUNE.into()),
        ("TURBO_TUNED_DEFAULT", TURBO_TUNED_DEFAULT.into()),
        ("TURBO_TUNED_FORCED", TURBO_TUNED_FORCED.into()),
        ("TURBO_TUNED_MEASURED", TURBO_TUNED_MEASURED.into()),
        ("TURBO_TUNED_CACHE", TURBO_TUNED_CACHE.into()),
        ("TURBO_CHOICES_LEN", TURBO_CHOICES_LEN as i64),
        ("TURBO_NUMERIC_F32_FMA", turbo::backend::TURBO_NUMERIC_F32_FMA.into()),
        ("TURBO_NUMERIC_TF32", turbo::backend::TURBO_NUMERIC_TF32.into()),
        ("TURBO_NUMERIC_F16_F32ACC", turbo::backend::TURBO_NUMERIC_F16_F32ACC.into()),
        ("TURBO_NUMERIC_F16_CHUNKACC", turbo::backend::TURBO_NUMERIC_F16_CHUNKACC.into()),
        ("TURBO_CAP_UNSUPPORTED", turbo::backend::TURBO_CAP_UNSUPPORTED.into()),
        ("TURBO_CAP_EXPERIMENTAL", turbo::backend::TURBO_CAP_EXPERIMENTAL.into()),
        ("TURBO_CAP_SUPPORTED", turbo::backend::TURBO_CAP_SUPPORTED.into()),
        ("TURBO_OK", OK.into()),
        ("TURBO_E_INVALID_ARGUMENT", INVALID_ARGUMENT.into()),
        ("TURBO_E_INVALID_STRUCT_SIZE", INVALID_STRUCT_SIZE.into()),
        ("TURBO_E_INVALID_UTF8", INVALID_UTF8.into()),
        ("TURBO_E_INVALID_HANDLE", INVALID_HANDLE.into()),
        ("TURBO_E_INVALID_SHAPE", INVALID_SHAPE.into()),
        ("TURBO_E_INVALID_STATE", INVALID_STATE.into()),
        ("TURBO_E_INVALID_ENUM", INVALID_ENUM.into()),
        ("TURBO_E_UNSUPPORTED", UNSUPPORTED.into()),
        ("TURBO_E_UNSUPPORTED_OPTION", UNSUPPORTED_OPTION.into()),
        ("TURBO_E_UNSUPPORTED_TASK", UNSUPPORTED_TASK.into()),
        ("TURBO_E_OUT_OF_MEMORY", OUT_OF_MEMORY.into()),
        ("TURBO_E_BUSY", BUSY.into()),
        ("TURBO_E_CAPACITY", CAPACITY.into()),
        ("TURBO_E_DEVICE_NOT_FOUND", DEVICE_NOT_FOUND.into()),
        ("TURBO_E_DEVICE_UNAVAILABLE", DEVICE_UNAVAILABLE.into()),
        ("TURBO_E_RUNTIME", RUNTIME.into()),
        ("TURBO_E_BUNDLE_NOT_FOUND", BUNDLE_NOT_FOUND.into()),
        ("TURBO_E_BUNDLE_INVALID", BUNDLE_INVALID.into()),
        ("TURBO_E_BUNDLE_INTEGRITY", BUNDLE_INTEGRITY.into()),
        ("TURBO_E_BUNDLE_NO_ARTIFACT", BUNDLE_NO_ARTIFACT.into()),
        ("TURBO_E_INTERNAL", INTERNAL.into()),
        ("TURBO_E_PANIC", PANIC.into()),
    ];
    // Where the core puts each BERT tensor is where the header says.
    let roles =
        turbo::model::BERT_EMBEDDING_ROLES.iter().enumerate().chain(turbo::model::BERT_LAYER_ROLES.iter().enumerate());
    let roles: Vec<(String, i64)> = roles
        .map(|(i, &r)| (format!("TURBO_BERT_{}", turbo::manifest::role_name(r).to_uppercase()), i as i64))
        .collect();
    let constants: Vec<(&str, i64)> =
        constants.iter().copied().chain(roles.iter().map(|(n, i)| (n.as_str(), *i))).collect();
    let mut src = String::from(
        "#include <stdio.h>\n#include <turbo/turbo.h>\n#include <turbo/turbo_backend.h>\nint main(void) {\n",
    );
    for (name, _) in &constants {
        src += &format!("  printf(\"{name} %lld\\n\", (long long)({name}));\n");
    }
    src += "  return 0;\n}\n";
    let d = scratch("constants");
    std::fs::write(d.join("constants.c"), src).unwrap();
    run(Command::new("cc")
        .args(["-std=c11", "-Wall", "-Werror", "-o"])
        .arg(d.join("constants"))
        .arg(d.join("constants.c"))
        .arg("-I")
        .arg(include()));
    let out = run(&mut Command::new(d.join("constants")));
    let want: String = constants.iter().map(|(name, v)| format!("{name} {v}\n")).collect();
    assert_eq!(out, want);
    for (name, v) in constants.iter().filter(|(n, _)| n.starts_with("TURBO_OK") || n.starts_with("TURBO_E_")) {
        assert_eq!(turbo_status_name_str(*v as i32), *name);
    }
    std::fs::remove_dir_all(d).unwrap();
}

fn turbo_status_name_str(code: i32) -> String {
    unsafe { std::ffi::CStr::from_ptr(turbo_status_name(code)) }.to_str().unwrap().to_owned()
}
