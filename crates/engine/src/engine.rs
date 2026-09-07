use crate::{Candidate, CandidateCost, Error, Segment, SegmentCost, dictionary::*};
use std::path::Path;

const UNKNOWN_WORD_COST: i32 = 10_000;
const UNKNOWN_CONTEXT_ID: u16 = 0;
const BOS_EOS_CONTEXT_ID: u16 = 0;
const SAME_SURFACE_BONUS: i32 = -1_000;
const ASCII_SURFACE_PENALTY: i32 = 30_000;
const KATAKANA_SURFACE_PENALTY: i32 = 10_000;
const KATAKANA_NO_PARTICLE_PENALTY: i32 = 5_000;
const UNUSUAL_CHARACTER_PENALTY: i32 = 15_000;
const SEGMENT_PENALTY: i32 = 500;
const SINGLE_CHAR_SEGMENT_PENALTY: i32 = 2_500;
const TWO_CHAR_SEGMENT_PENALTY: i32 = 750;
const MIN_INTERNAL_BEAM: usize = 128;
const MAX_INTERNAL_BEAM: usize = 512;
const INTERNAL_BEAM_MULTIPLIER: usize = 32;

#[derive(Debug, Clone)]
pub struct Engine {
    dictionary: Dictionary,
}

#[derive(Clone)]
struct ConversionPath {
    text: String,
    base_cost: i32,
    language_cost: i32,
    right_id: u16,
    segments: Vec<Segment>,
    segment_costs: Vec<SegmentCost>,
    segment_count: usize,
    before_previous_word: Option<u32>,
    previous_word: Option<u32>,
}

impl Engine {
    pub fn new(dictionary: Dictionary) -> Self {
        Self { dictionary }
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        Ok(Self {
            dictionary: Dictionary::open(path)?,
        })
    }

    pub fn convert(&self, reading: &str) -> Option<Candidate> {
        self.candidates(reading, 1).into_iter().next()
    }

    pub fn candidates(&self, reading: &str, limit: usize) -> Vec<Candidate> {
        self.candidate_costs(reading, limit)
            .into_iter()
            .map(|candidate| Candidate {
                text: candidate.text,
                cost: candidate.cost,
                segments: candidate
                    .segments
                    .into_iter()
                    .map(|segment| Segment {
                        reading: segment.reading,
                        surface: segment.surface,
                    })
                    .collect(),
            })
            .collect()
    }

    pub fn candidate_costs(&self, reading: &str, limit: usize) -> Vec<CandidateCost> {
        let beam = internal_beam_width(limit);
        self.candidate_costs_with_beam(reading, limit, beam)
    }

    pub fn candidates_with_beam(
        &self,
        reading: &str,
        limit: usize,
        internal_beam: usize,
    ) -> Vec<Candidate> {
        self.candidate_costs_with_beam(reading, limit, internal_beam)
            .into_iter()
            .map(|candidate| Candidate {
                text: candidate.text,
                cost: candidate.cost,
                segments: candidate
                    .segments
                    .into_iter()
                    .map(|segment| Segment {
                        reading: segment.reading,
                        surface: segment.surface,
                    })
                    .collect(),
            })
            .collect()
    }

    pub fn candidate_costs_with_beam(
        &self,
        reading: &str,
        limit: usize,
        internal_beam: usize,
    ) -> Vec<CandidateCost> {
        if reading.is_empty() || limit == 0 {
            return Vec::new();
        }

        let internal_beam = internal_beam.max(limit).max(1);
        let boundaries = char_boundaries(reading);

        let mut paths: Vec<Vec<ConversionPath>> = vec![Vec::new(); reading.len() + 1];

        let (before_previous_word, previous_word) =
            self.dictionary.language_model().initial_context();

        paths[0].push(ConversionPath {
            text: String::new(),
            base_cost: 0,
            language_cost: 0,
            right_id: BOS_EOS_CONTEXT_ID,
            segments: Vec::new(),
            segment_costs: Vec::new(),
            segment_count: 0,
            before_previous_word,
            previous_word,
        });

        for &start in &boundaries[..boundaries.len() - 1] {
            if paths[start].is_empty() {
                continue;
            }

            let suffix = &reading[start..];

            let matches = self.dictionary.lookup_prefix(suffix);

            if matches.is_empty() {
                let Some(end) = next_boundary(&boundaries, start) else {
                    continue;
                };

                let unknown = &reading[start..end];

                let previous_paths = paths[start].clone();

                for path in previous_paths {
                    let connection_cost =
                        self.dictionary
                            .matrix()
                            .cost(path.right_id, UNKNOWN_CONTEXT_ID) as i32;

                    let current_word = self.dictionary.language_model().word_id(unknown);

                    let language_model_cost = self.dictionary.language_model().cost_details(
                        path.before_previous_word,
                        path.previous_word,
                        current_word,
                    );
                    let segment_penalty = Self::segment_penalty(unknown);
                    let base_delta = connection_cost + UNKNOWN_WORD_COST + segment_penalty;

                    let mut text = path.text.clone();

                    text.push_str(unknown);

                    let mut segments = path.segments.clone();

                    segments.push(Segment {
                        reading: unknown.to_string(),
                        surface: unknown.to_string(),
                    });
                    let mut segment_costs = path.segment_costs.clone();
                    segment_costs.push(SegmentCost {
                        reading: unknown.to_string(),
                        surface: unknown.to_string(),
                        word_cost: UNKNOWN_WORD_COST,
                        connection_cost,
                        surface_penalty: 0,
                        segment_penalty,
                        language_cost: language_model_cost.cost,
                        language_order: language_model_cost.order,
                        backoff_penalty: language_model_cost.backoff_penalty,
                        unknown_language_model: language_model_cost.unknown,
                        base_cost: path.base_cost + base_delta,
                        total_cost: path.base_cost
                            + base_delta
                            + path.language_cost
                            + language_model_cost.cost,
                    });

                    paths[end].push(ConversionPath {
                        text,

                        base_cost: path.base_cost + base_delta,

                        language_cost: path.language_cost + language_model_cost.cost,

                        right_id: UNKNOWN_CONTEXT_ID,

                        segments,
                        segment_costs,

                        segment_count: path.segment_count + 1,

                        before_previous_word: path.previous_word,

                        previous_word: self
                            .dictionary
                            .language_model()
                            .context_word_id(current_word),
                    });
                }

                Self::prune_paths(&mut paths[end], internal_beam);

                continue;
            }

            let previous_paths = paths[start].clone();

            for entry in matches {
                let end = start + entry.reading.len();

                for path in &previous_paths {
                    let connection_cost =
                        self.dictionary.matrix().cost(path.right_id, entry.left_id) as i32;

                    let current_word = self.dictionary.language_model().word_id(&entry.surface);

                    let language_model_cost = self.dictionary.language_model().cost_details(
                        path.before_previous_word,
                        path.previous_word,
                        current_word,
                    );
                    let surface_penalty = Self::surface_penalty(&entry.reading, &entry.surface);
                    let segment_penalty = Self::segment_penalty(&entry.reading);
                    let word_cost = entry.cost as i32;
                    let base_delta =
                        connection_cost + word_cost + surface_penalty + segment_penalty;

                    let mut text = path.text.clone();

                    text.push_str(&entry.surface);

                    let mut segments = path.segments.clone();

                    segments.push(Segment {
                        reading: entry.reading.clone(),

                        surface: entry.surface.clone(),
                    });
                    let mut segment_costs = path.segment_costs.clone();
                    segment_costs.push(SegmentCost {
                        reading: entry.reading.clone(),
                        surface: entry.surface.clone(),
                        word_cost,
                        connection_cost,
                        surface_penalty,
                        segment_penalty,
                        language_cost: language_model_cost.cost,
                        language_order: language_model_cost.order,
                        backoff_penalty: language_model_cost.backoff_penalty,
                        unknown_language_model: language_model_cost.unknown,
                        base_cost: path.base_cost + base_delta,
                        total_cost: path.base_cost
                            + base_delta
                            + path.language_cost
                            + language_model_cost.cost,
                    });

                    paths[end].push(ConversionPath {
                        text,

                        base_cost: path.base_cost + base_delta,

                        language_cost: path.language_cost + language_model_cost.cost,

                        right_id: entry.right_id,

                        segments,
                        segment_costs,

                        segment_count: path.segment_count + 1,

                        before_previous_word: path.previous_word,

                        previous_word: self
                            .dictionary
                            .language_model()
                            .context_word_id(current_word),
                    });
                }

                Self::prune_paths(&mut paths[end], internal_beam);
            }
        }

        let mut results = paths[reading.len()]
            .drain(..)
            .map(|path| {
                let eos_cost = self
                    .dictionary
                    .matrix()
                    .cost(path.right_id, BOS_EOS_CONTEXT_ID) as i32;

                let language_eos_cost = self
                    .dictionary
                    .language_model()
                    .eos_cost(path.before_previous_word, path.previous_word);
                let base_cost = path.base_cost + eos_cost;
                let language_cost = path.language_cost + language_eos_cost;

                CandidateCost {
                    text: path.text,

                    cost: base_cost + language_cost,

                    base_cost,

                    language_cost,

                    eos_connection_cost: eos_cost,

                    eos_language_cost: language_eos_cost,

                    segments: path.segment_costs,
                }
            })
            .collect::<Vec<_>>();

        results.sort_by_key(|candidate| candidate.cost);

        let mut seen = std::collections::HashSet::new();
        results.retain(|candidate| seen.insert(candidate.text.clone()));

        results.truncate(limit);

        results
    }

    fn prune_paths(paths: &mut Vec<ConversionPath>, limit: usize) {
        paths.sort_by_key(|path| path.base_cost + path.language_cost);
        paths.dedup_by(|a, b| {
            a.text == b.text
                && a.right_id == b.right_id
                && a.previous_word == b.previous_word
                && a.before_previous_word == b.before_previous_word
        });
        paths.truncate(limit);
    }

    fn segment_penalty(reading: &str) -> i32 {
        match reading.chars().count() {
            0 => 0,
            1 => SINGLE_CHAR_SEGMENT_PENALTY,
            2 => TWO_CHAR_SEGMENT_PENALTY,
            _ => SEGMENT_PENALTY,
        }
    }

    fn surface_penalty(reading: &str, surface: &str) -> i32 {
        if reading == surface {
            return SAME_SURFACE_BONUS;
        }

        let reading_has_ascii = reading.chars().any(|ch| ch.is_ascii_alphanumeric());

        let surface_has_ascii = surface.chars().any(|ch| ch.is_ascii_alphanumeric());

        if surface_has_ascii && !reading_has_ascii {
            return ASCII_SURFACE_PENALTY;
        }

        let reading_is_hiragana = reading
            .chars()
            .all(|ch| matches!(ch, '\u{3040}'..='\u{309f}'));

        let surface_is_katakana = surface
            .chars()
            .all(|ch| matches!(ch, '\u{30a0}'..='\u{30ff}'));

        if reading_is_hiragana && surface_is_katakana {
            return KATAKANA_SURFACE_PENALTY;
        }

        let mut penalty = 0;

        if reading_is_hiragana && reading.contains('の') && surface.contains('ノ') {
            penalty += KATAKANA_NO_PARTICLE_PENALTY;
        }

        let unusual_characters = surface
            .chars()
            .filter(|ch| {
                !reading.contains(*ch)
                    && !matches!(
                        ch,
                        '\u{3040}'..='\u{309f}'
                            | '\u{30a0}'..='\u{30ff}'
                            | '\u{3400}'..='\u{4dbf}'
                            | '\u{4e00}'..='\u{9fff}'
                            | '\u{f900}'..='\u{faff}'
                    )
            })
            .count();

        penalty + UNUSUAL_CHARACTER_PENALTY * unusual_characters as i32
    }
}

fn internal_beam_width(limit: usize) -> usize {
    limit
        .saturating_mul(INTERNAL_BEAM_MULTIPLIER)
        .clamp(MIN_INTERNAL_BEAM, MAX_INTERNAL_BEAM)
}

fn char_boundaries(input: &str) -> Vec<usize> {
    let mut boundaries = input
        .char_indices()
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    boundaries.push(input.len());
    boundaries
}

fn next_boundary(boundaries: &[usize], current: usize) -> Option<usize> {
    boundaries
        .iter()
        .copied()
        .find(|&boundary| boundary > current)
}
