//! cxx bridge to `ov::genai::TextEmbeddingPipeline`.
//!
//! Compiled only with the `genai` feature. The C++ side lives in
//! `cxx/text_embedding.{hpp,cpp}` and is the only translation unit that
//! includes OpenVINO GenAI headers.

#[cxx::bridge(namespace = "inferstream_ov")]
pub(crate) mod ffi {
    unsafe extern "C++" {
        include!("text_embedding.hpp");

        type Pipeline;

        fn load_pipeline(
            models_path: &str,
            device: &str,
            pooling: u8,
            normalize: bool,
            max_length: u32,
            pad_to_max_length: bool,
        ) -> Result<UniquePtr<Pipeline>>;

        fn embed_documents(self: &Pipeline, texts: &Vec<String>) -> Result<Vec<f32>>;
        fn embedding_dim(self: &Pipeline) -> usize;
        fn device(self: &Pipeline) -> String;

        fn available_devices() -> Result<Vec<String>>;
    }
}
