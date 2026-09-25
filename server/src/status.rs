//! Turbo status codes as gRPC statuses (docs/kserve.md, Errors).

use tonic::metadata::MetadataMap;
use tonic::{Code, Status};

use crate::api::*;

/// The gRPC status code for a turbo status code.
pub fn grpc_code(code: i32) -> Code {
    match code {
        TURBO_OK => Code::Ok,
        TURBO_E_INVALID_ARGUMENT => Code::InvalidArgument,
        TURBO_E_INVALID_STRUCT_SIZE => Code::Internal,
        TURBO_E_INVALID_UTF8 => Code::InvalidArgument,
        TURBO_E_INVALID_HANDLE => Code::Internal,
        TURBO_E_INVALID_SHAPE => Code::InvalidArgument,
        TURBO_E_INVALID_STATE => Code::FailedPrecondition,
        TURBO_E_INVALID_ENUM => Code::InvalidArgument,
        TURBO_E_UNSUPPORTED => Code::Unimplemented,
        TURBO_E_UNSUPPORTED_OPTION => Code::Unimplemented,
        TURBO_E_UNSUPPORTED_TASK => Code::Unimplemented,
        TURBO_E_OUT_OF_MEMORY => Code::ResourceExhausted,
        TURBO_E_BUSY => Code::Unavailable,
        TURBO_E_CAPACITY => Code::OutOfRange,
        TURBO_E_DEVICE_NOT_FOUND => Code::FailedPrecondition,
        TURBO_E_DEVICE_UNAVAILABLE => Code::Unavailable,
        TURBO_E_RUNTIME => Code::Internal,
        TURBO_E_BUNDLE_NOT_FOUND => Code::NotFound,
        TURBO_E_BUNDLE_INVALID => Code::FailedPrecondition,
        TURBO_E_BUNDLE_INTEGRITY => Code::DataLoss,
        TURBO_E_BUNDLE_NO_ARTIFACT => Code::FailedPrecondition,
        TURBO_E_INTERNAL => Code::Internal,
        TURBO_E_PANIC => Code::Internal,
        _ => Code::Unknown,
    }
}

/// The fields of turbo_embed_options, 1-based: what a write's
/// turbo_error.field names.
pub const EMBED_OPTIONS: [&str; 6] = ["truncate", "max_tokens", "prompt_role", "normalize", "pooling", "output_dim"];

/// The fields of turbo_session_desc, 1-based: what turbo_session_create's
/// turbo_error.field names.
pub const SESSION_DESC: [&str; 3] = ["max_batch", "max_seq", "precision"];

/// `<name>: <message>`, or `<name> field <n> (<field>): <message>` when the
/// failure names a field of `fields`, the struct the call took.
pub fn describe(f: &Failure, fields: &[&str]) -> String {
    let name = status_name(f.code);
    if f.field == 0 {
        return format!("{name}: {}", f.message);
    }
    match fields.get(f.field as usize - 1) {
        Some(field) => format!("{name} field {} ({field}): {}", f.field, f.message),
        None => format!("{name} field {}: {}", f.field, f.message),
    }
}

/// The gRPC status a client receives: the mapped code, the message, and
/// trailing metadata turbo-code and turbo-field.
pub fn to_status(f: &Failure, fields: &[&str]) -> Status {
    let mut md = MetadataMap::new();
    md.insert("turbo-code", f.code.to_string().parse().expect("decimal is ASCII"));
    md.insert("turbo-field", f.field.to_string().parse().expect("decimal is ASCII"));
    Status::with_metadata(grpc_code(f.code), describe(f, fields), md)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every code turbo.h defines, read from the header itself, and the
    /// gRPC code docs/kserve.md gives it.
    #[test]
    fn every_header_code_maps_as_the_table_says() {
        let table: &[(&str, Code)] = &[
            ("TURBO_OK", Code::Ok),
            ("TURBO_E_INVALID_ARGUMENT", Code::InvalidArgument),
            ("TURBO_E_INVALID_STRUCT_SIZE", Code::Internal),
            ("TURBO_E_INVALID_UTF8", Code::InvalidArgument),
            ("TURBO_E_INVALID_HANDLE", Code::Internal),
            ("TURBO_E_INVALID_SHAPE", Code::InvalidArgument),
            ("TURBO_E_INVALID_STATE", Code::FailedPrecondition),
            ("TURBO_E_INVALID_ENUM", Code::InvalidArgument),
            ("TURBO_E_UNSUPPORTED", Code::Unimplemented),
            ("TURBO_E_UNSUPPORTED_OPTION", Code::Unimplemented),
            ("TURBO_E_UNSUPPORTED_TASK", Code::Unimplemented),
            ("TURBO_E_OUT_OF_MEMORY", Code::ResourceExhausted),
            ("TURBO_E_BUSY", Code::Unavailable),
            ("TURBO_E_CAPACITY", Code::OutOfRange),
            ("TURBO_E_DEVICE_NOT_FOUND", Code::FailedPrecondition),
            ("TURBO_E_DEVICE_UNAVAILABLE", Code::Unavailable),
            ("TURBO_E_RUNTIME", Code::Internal),
            ("TURBO_E_BUNDLE_NOT_FOUND", Code::NotFound),
            ("TURBO_E_BUNDLE_INVALID", Code::FailedPrecondition),
            ("TURBO_E_BUNDLE_INTEGRITY", Code::DataLoss),
            ("TURBO_E_BUNDLE_NO_ARTIFACT", Code::FailedPrecondition),
            ("TURBO_E_INTERNAL", Code::Internal),
            ("TURBO_E_PANIC", Code::Internal),
        ];
        let header = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../include/turbo/turbo.h")).unwrap();
        let codes: Vec<(String, i32)> = header
            .lines()
            .filter_map(|l| {
                let mut w = l.split_whitespace();
                (w.next()? == "#define").then_some(())?;
                let name = w.next()?;
                (name == "TURBO_OK" || name.starts_with("TURBO_E_")).then_some(())?;
                Some((name.to_string(), w.next()?.parse().ok()?))
            })
            .collect();
        assert_eq!(codes.len(), table.len(), "the header's codes: {codes:?}");
        for (name, code) in &codes {
            let want = table.iter().find(|(n, _)| n == name).unwrap_or_else(|| panic!("{name} is not in the table")).1;
            assert_eq!(grpc_code(*code), want, "{name}");
            assert_eq!(&status_name(*code), name, "turbo_status_name({code})");
        }
        // A code the table does not have.
        for code in [1, 255, 263, 770, 1284, 1538, -1, i32::MAX] {
            assert_eq!(grpc_code(code), Code::Unknown, "{code}");
            assert_eq!(status_name(code), "TURBO_E_UNKNOWN");
        }
    }

    #[test]
    fn messages_and_trailing_metadata() {
        let s = to_status(&Failure::new(TURBO_E_UNSUPPORTED_OPTION, 6, "no such width"), &EMBED_OPTIONS);
        assert_eq!(s.code(), Code::Unimplemented);
        assert_eq!(s.message(), "TURBO_E_UNSUPPORTED_OPTION field 6 (output_dim): no such width");
        assert_eq!(s.metadata().get("turbo-code").unwrap(), "513");
        assert_eq!(s.metadata().get("turbo-field").unwrap(), "6");
        let s = to_status(&Failure::new(TURBO_E_BUSY, 0, "held"), &EMBED_OPTIONS);
        assert_eq!(s.message(), "TURBO_E_BUSY: held");
        assert_eq!(s.metadata().get("turbo-field").unwrap(), "0");
        let s = to_status(&Failure::new(4242, 0, "odd"), &EMBED_OPTIONS);
        assert_eq!(s.code(), Code::Unknown);
        assert_eq!(s.message(), "TURBO_E_UNKNOWN: odd");
        assert_eq!(s.metadata().get("turbo-code").unwrap(), "4242");
        let f = Failure::new(TURBO_E_UNSUPPORTED_OPTION, 3, "not in F16");
        assert_eq!(describe(&f, &SESSION_DESC), "TURBO_E_UNSUPPORTED_OPTION field 3 (precision): not in F16");
    }
}
