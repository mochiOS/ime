use std::collections::HashMap;

use crate::Error;
use crate::LanguageModelOrder;
use crate::mime::{BIGRAM_SIZE, TRIGRAM_SIZE, UNIGRAM_SIZE, VOCAB_ENTRY_SIZE, read_i32, read_u32};

const BOS_TOKEN: &str = "<s>";
const EOS_TOKEN: &str = "</s>";
const UNKNOWN_TOKEN: &str = "<unk>";
const BACKOFF_PENALTY: i32 = 500;
const UNKNOWN_COST: i32 = 12000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LanguageModelCost {
    pub(crate) cost: i32,
    pub(crate) order: LanguageModelOrder,
    pub(crate) backoff_penalty: i32,
    pub(crate) unknown: bool,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct LanguageModel {
    vocabulary: HashMap<String, u32>,

    unigrams: HashMap<u32, i32>,
    bigrams: HashMap<(u32, u32), i32>,
    trigrams: HashMap<(u32, u32, u32), i32>,

    bos_id: Option<u32>,
    eos_id: Option<u32>,
    unknown_id: Option<u32>,
}

impl LanguageModel {
    pub(crate) fn from_mime_bytes(
        bytes: &[u8],
        string_table: &[u8],
        vocabulary_count: usize,
        unigram_count: usize,
        bigram_count: usize,
        trigram_count: usize,
        vocabulary_offset: usize,
        unigram_offset: usize,
        bigram_offset: usize,
        trigram_offset: usize,
        end_offset: usize,
    ) -> Result<Self, Error> {
        let vocabulary_end = vocabulary_offset
            .checked_add(vocabulary_count.checked_mul(VOCAB_ENTRY_SIZE).ok_or(
                Error::InvalidDictionary("language model vocabulary size overflow"),
            )?)
            .ok_or(Error::InvalidDictionary(
                "language model vocabulary overflow",
            ))?;

        let unigram_end = unigram_offset
            .checked_add(
                unigram_count
                    .checked_mul(UNIGRAM_SIZE)
                    .ok_or(Error::InvalidDictionary("unigram size overflow"))?,
            )
            .ok_or(Error::InvalidDictionary("unigram section overflow"))?;

        let bigram_end = bigram_offset
            .checked_add(
                bigram_count
                    .checked_mul(BIGRAM_SIZE)
                    .ok_or(Error::InvalidDictionary("bigram size overflow"))?,
            )
            .ok_or(Error::InvalidDictionary("bigram section overflow"))?;

        let trigram_end = trigram_offset
            .checked_add(
                trigram_count
                    .checked_mul(TRIGRAM_SIZE)
                    .ok_or(Error::InvalidDictionary("trigram size overflow"))?,
            )
            .ok_or(Error::InvalidDictionary("trigram section overflow"))?;

        if vocabulary_end > unigram_offset
            || unigram_end > bigram_offset
            || bigram_end > trigram_offset
            || trigram_end > end_offset
            || end_offset > bytes.len()
        {
            return Err(Error::InvalidDictionary(
                "language model section offsets are invalid",
            ));
        }

        let mut vocabulary = HashMap::with_capacity(vocabulary_count);

        for word_id in 0..vocabulary_count {
            let offset = vocabulary_offset + word_id * VOCAB_ENTRY_SIZE;

            let string_offset = read_u32(bytes, offset).ok_or(Error::InvalidDictionary(
                "truncated language model vocabulary",
            ))? as usize;

            let string_len = read_u32(bytes, offset + 4).ok_or(Error::InvalidDictionary(
                "truncated language model vocabulary",
            ))? as usize;

            let end = string_offset
                .checked_add(string_len)
                .ok_or(Error::InvalidDictionary(
                    "language model string range overflow",
                ))?;

            let value = string_table
                .get(string_offset..end)
                .ok_or(Error::InvalidDictionary(
                    "language model string is outside string table",
                ))?;

            let value = std::str::from_utf8(value).map_err(|_| Error::InvalidUtf8)?;

            vocabulary.insert(value.to_owned(), word_id as u32);
        }

        let mut unigrams = HashMap::with_capacity(unigram_count);

        for index in 0..unigram_count {
            let offset = unigram_offset + index * UNIGRAM_SIZE;

            let word =
                read_u32(bytes, offset).ok_or(Error::InvalidDictionary("truncated unigram"))?;

            let cost =
                read_i32(bytes, offset + 4).ok_or(Error::InvalidDictionary("truncated unigram"))?;

            unigrams.insert(word, cost);
        }

        let mut bigrams = HashMap::with_capacity(bigram_count);

        for index in 0..bigram_count {
            let offset = bigram_offset + index * BIGRAM_SIZE;

            let previous =
                read_u32(bytes, offset).ok_or(Error::InvalidDictionary("truncated bigram"))?;

            let current =
                read_u32(bytes, offset + 4).ok_or(Error::InvalidDictionary("truncated bigram"))?;

            let cost =
                read_i32(bytes, offset + 8).ok_or(Error::InvalidDictionary("truncated bigram"))?;

            bigrams.insert((previous, current), cost);
        }

        let mut trigrams = HashMap::with_capacity(trigram_count);

        for index in 0..trigram_count {
            let offset = trigram_offset + index * TRIGRAM_SIZE;

            let before_previous =
                read_u32(bytes, offset).ok_or(Error::InvalidDictionary("truncated trigram"))?;

            let previous =
                read_u32(bytes, offset + 4).ok_or(Error::InvalidDictionary("truncated trigram"))?;

            let current =
                read_u32(bytes, offset + 8).ok_or(Error::InvalidDictionary("truncated trigram"))?;

            let cost = read_i32(bytes, offset + 12)
                .ok_or(Error::InvalidDictionary("truncated trigram"))?;

            trigrams.insert((before_previous, previous, current), cost);
        }

        let bos_id = vocabulary.get(BOS_TOKEN).copied();

        let eos_id = vocabulary.get(EOS_TOKEN).copied();

        let unknown_id = vocabulary.get(UNKNOWN_TOKEN).copied();

        Ok(Self {
            vocabulary,
            unigrams,
            bigrams,
            trigrams,
            bos_id,
            eos_id,
            unknown_id,
        })
    }

    pub(crate) fn initial_context(&self) -> (Option<u32>, Option<u32>) {
        (None, self.bos_id)
    }

    pub(crate) fn word_id(&self, surface: &str) -> Option<u32> {
        self.vocabulary.get(surface).copied()
    }

    pub(crate) fn context_word_id(&self, word: Option<u32>) -> Option<u32> {
        word.or(self.unknown_id)
    }

    pub(crate) fn cost(
        &self,
        before_previous: Option<u32>,
        previous: Option<u32>,
        current: Option<u32>,
    ) -> i32 {
        self.cost_details(before_previous, previous, current).cost
    }

    pub(crate) fn cost_details(
        &self,
        before_previous: Option<u32>,
        previous: Option<u32>,
        current: Option<u32>,
    ) -> LanguageModelCost {
        if self.vocabulary.is_empty() {
            return LanguageModelCost {
                cost: 0,
                order: LanguageModelOrder::Disabled,
                backoff_penalty: 0,
                unknown: false,
            };
        }

        let Some(current) = current else {
            return LanguageModelCost {
                cost: UNKNOWN_COST,
                order: LanguageModelOrder::Unknown,
                backoff_penalty: 0,
                unknown: true,
            };
        };

        let mut best = None;

        if let Some(previous) = previous {
            if let Some(cost) = self.bigrams.get(&(previous, current)) {
                best = Some(LanguageModelCost {
                    cost: cost.saturating_add(BACKOFF_PENALTY),
                    order: LanguageModelOrder::Bigram,
                    backoff_penalty: BACKOFF_PENALTY,
                    unknown: false,
                });
            }
        }

        if let (Some(before_previous), Some(previous)) = (before_previous, previous) {
            if let Some(cost) = self.trigrams.get(&(before_previous, previous, current)) {
                best = Some(min_cost(
                    best,
                    LanguageModelCost {
                        cost: *cost,
                        order: LanguageModelOrder::Trigram,
                        backoff_penalty: 0,
                        unknown: false,
                    },
                ));
            }
        }

        if let Some(cost) = self.unigrams.get(&current) {
            return min_cost(
                best,
                LanguageModelCost {
                    cost: cost.saturating_add(BACKOFF_PENALTY * 2),
                    order: LanguageModelOrder::Unigram,
                    backoff_penalty: BACKOFF_PENALTY * 2,
                    unknown: false,
                },
            );
        }

        best.unwrap_or(LanguageModelCost {
            cost: UNKNOWN_COST,
            order: LanguageModelOrder::Unknown,
            backoff_penalty: 0,
            unknown: true,
        })
    }

    pub(crate) fn eos_cost(&self, before_previous: Option<u32>, previous: Option<u32>) -> i32 {
        self.cost(before_previous, previous, self.eos_id)
    }
}

fn min_cost(previous: Option<LanguageModelCost>, current: LanguageModelCost) -> LanguageModelCost {
    match previous {
        Some(previous) if previous.cost <= current.cost => previous,
        _ => current,
    }
}
