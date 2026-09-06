#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("failed to load tokenizer: {0}")]
	Tokenizer(String),

	#[error("failed to tokenize candidates: {0}")]
	Tokenization(String),

	#[error("ONNX Runtime error: {0}")]
	Ort(#[from] ort::Error),

	#[error("model returned an invalid output")]
	InvalidOutput,
}