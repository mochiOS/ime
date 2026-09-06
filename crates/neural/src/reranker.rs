use std::path::Path;
use engine::Candidate;
use tokenizers::Tokenizer;
use crate::Error;

const DEFAULT_MAX_LENGTH: usize = 96;

pub struct Reranker {
	tokenizer: Tokenizer,
	max_length: usize,
}

impl Reranker {
	pub fn open(
		tokenizer_path: impl AsRef<Path>,
	) -> Result<Self, Error> {
		let tokenizer = Tokenizer::from_file(tokenizer_path)
			.map_err(|error| Error::Tokenizer(error.to_string()))?;

		Ok(Self {
			tokenizer,
			max_length: DEFAULT_MAX_LENGTH,
		})
	}

	pub fn rank(
		&self,
		candidates: Vec<Candidate>,
	) -> Result<Vec<Candidate>, Error> {
		if candidates.len() <= 1 {
			return Ok(candidates);
		}

		let texts = candidates
			.iter()
			.map(|candidate| candidate.text.as_str())
			.collect::<Vec<_>>();

		let encodings = self
			.tokenizer
			.encode_batch(texts, true)
			.map_err(|error| Error::Tokenization(error.to_string()))?;

		let _ = encodings;

		todo!("ONNX model inference");
	}
}