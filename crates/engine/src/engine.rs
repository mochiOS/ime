use crate::{Candidate, Error, Segment, dictionary::*};
use std::path::Path;

const UNKNOWN_WORD_COST: i32 = 10_000;
const UNKNOWN_CONTEXT_ID: u16 = 0;
const BOS_EOS_CONTEXT_ID: u16 = 0;
const SAME_SURFACE_BONUS: i32 = -1_000;
const ASCII_SURFACE_PENALTY: i32 = 30_000;
const KATAKANA_SURFACE_PENALTY: i32 = 10_000;
const UNUSUAL_CHARACTER_PENALTY: i32 = 15_000;
const SEGMENT_PENALTY: i32 = 500;
const SINGLE_CHAR_SEGMENT_PENALTY: i32 = 2_500;
const TWO_CHAR_SEGMENT_PENALTY: i32 = 750;

#[derive(Debug, Clone)]
pub struct Engine {
    dictionary: Dictionary,
}

#[derive(Clone)]
struct ConversionPath {
    text: String,
    cost: i32,
    right_id: u16,
    segments: Vec<Segment>,
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
        if reading.is_empty() || limit == 0 {
            return Vec::new();
        }

        let boundaries = char_boundaries(reading);

        let mut paths: Vec<Vec<ConversionPath>> = vec![Vec::new(); reading.len() + 1];

        let (before_previous_word, previous_word) =
            self.dictionary.language_model().initial_context();

        paths[0].push(ConversionPath {
            text: String::new(),
            cost: 0,
            right_id: BOS_EOS_CONTEXT_ID,
            segments: Vec::new(),
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

                    let language_cost = self.dictionary.language_model().cost(
                        path.before_previous_word,
                        path.previous_word,
                        current_word,
                    );

                    let mut text = path.text.clone();

                    text.push_str(unknown);

                    let mut segments = path.segments.clone();

                    segments.push(Segment {
                        reading: unknown.to_string(),
                        surface: unknown.to_string(),
                    });

                    paths[end].push(ConversionPath {
                        text,

                        cost: path.cost
                            + connection_cost
                            + UNKNOWN_WORD_COST
                            + SEGMENT_PENALTY
                            + language_cost,

                        right_id: UNKNOWN_CONTEXT_ID,

                        segments,

                        segment_count: path.segment_count + 1,

                        before_previous_word: path.previous_word,

                        previous_word: current_word,
                    });
                }

                Self::prune_paths(&mut paths[end], limit);

                continue;
            }

            let previous_paths = paths[start].clone();

            for entry in matches {
                let end = start + entry.reading.len();

                for path in &previous_paths {
                    let connection_cost =
                        self.dictionary.matrix().cost(path.right_id, entry.left_id) as i32;

                    let current_word = self.dictionary.language_model().word_id(&entry.surface);

                    let language_cost = self.dictionary.language_model().cost(
                        path.before_previous_word,
                        path.previous_word,
                        current_word,
                    );

                    let mut text = path.text.clone();

                    text.push_str(&entry.surface);

                    let mut segments = path.segments.clone();

                    segments.push(Segment {
                        reading: entry.reading.clone(),

                        surface: entry.surface.clone(),
                    });

                    paths[end].push(ConversionPath {
                        text,

                        cost: path.cost
                            + connection_cost
                            + entry.cost as i32
                            + Self::surface_penalty(&entry.reading, &entry.surface)
                            + SEGMENT_PENALTY
                            + language_cost,

                        right_id: entry.right_id,

                        segments,

                        segment_count: path.segment_count + 1,

                        before_previous_word: path.previous_word,

                        previous_word: current_word,
                    });
                }

                Self::prune_paths(&mut paths[end], limit);
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

                Candidate {
                    text: path.text,

                    cost: path.cost + eos_cost + language_eos_cost,

                    segments: path.segments,
                }
            })
            .collect::<Vec<_>>();

        results.sort_by_key(|candidate| candidate.cost);

        results.dedup_by(|a, b| a.text == b.text);

        results.truncate(limit);

        results
    }

    fn prune_paths(paths: &mut Vec<ConversionPath>, limit: usize) {
        paths.sort_by_key(|path| path.cost);
        paths.dedup_by(|a, b| {
            a.text == b.text
                && a.right_id == b.right_id
                && a.previous_word == b.previous_word
                && a.before_previous_word == b.before_previous_word
        });
        paths.truncate(limit);
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

        UNUSUAL_CHARACTER_PENALTY * unusual_characters as i32
    }
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
