use std::path::Path;
use engine::Candidate;
use tokenizers::Tokenizer;
use crate::Error;

const DEFAULT_MAX_LENGTH: usize = 96;

pub struct Reranker {
	tokenizer: Tokenizer,
	session: Session,
	max_length: usize,
}

impl Reranker {
    pub fn open(
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
    ) -> Result<Self, Error> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|error| Error::Tokenizer(error.to_string()))?;

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .with_inter_threads(1)?
            .commit_from_file(model_path)?;

        Ok(Self {
            tokenizer,
            session,
            max_length: DEFAULT_MAX_LENGTH,
        })
    }

    pub fn rank(
        &mut self,
        mut candidates: Vec<Candidate>,
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

        let batch_size = encodings.len();

        let mut input_ids =
            vec![0_i64; batch_size * self.max_length];

        let mut attention_mask =
            vec![0_i64; batch_size * self.max_length];

        for (batch_index, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();

            let length = ids.len().min(self.max_length);

            let offset = batch_index * self.max_length;

            for index in 0..length {
                input_ids[offset + index] = ids[index] as i64;
                attention_mask[offset + index] = mask[index] as i64;
            }
        }

        let input_ids = Tensor::<i64>::from_array((
            [batch_size, self.max_length],
            input_ids,
        ))?;

        let attention_mask = Tensor::<i64>::from_array((
            [batch_size, self.max_length],
            attention_mask,
        ))?;

        let outputs = self.session.run(ort::inputs! {
            "input_ids" => input_ids,
            "attention_mask" => attention_mask,
        })?;

        let output = outputs
            .get("score")
            .ok_or(Error::InvalidOutput)?;

        let (_, scores) = output.try_extract_tensor::<f32>()?;

        if scores.len() != candidates.len() {
            return Err(Error::InvalidOutput);
        }

        let mut ranked = candidates
            .drain(..)
            .zip(scores.iter().copied())
            .collect::<Vec<_>>();

        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(
            ranked
                .into_iter()
                .map(|(candidate, _)| candidate)
                .collect(),
        )
    }
}