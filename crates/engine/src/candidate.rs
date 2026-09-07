#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    pub reading: String,
    pub surface: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    pub text: String,
    pub cost: i32,
    pub segments: Vec<Segment>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentCost {
    pub reading: String,
    pub surface: String,
    pub word_cost: i32,
    pub connection_cost: i32,
    pub surface_penalty: i32,
    pub segment_penalty: i32,
    pub language_cost: i32,
    pub language_order: LanguageModelOrder,
    pub backoff_penalty: i32,
    pub unknown_language_model: bool,
    pub base_cost: i32,
    pub total_cost: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LanguageModelOrder {
    Trigram,
    Bigram,
    Unigram,
    Unknown,
    Disabled,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateCost {
    pub text: String,
    pub cost: i32,
    pub base_cost: i32,
    pub language_cost: i32,
    pub eos_connection_cost: i32,
    pub eos_language_cost: i32,
    pub segments: Vec<SegmentCost>,
}
