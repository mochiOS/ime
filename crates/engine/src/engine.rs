use crate::{Candidate, CandidateCost, Error, Segment, SegmentCost, dictionary::*};
use std::path::Path;
use std::rc::Rc;

const UNKNOWN_WORD_COST: i32 = 10_000;
const UNKNOWN_CONTEXT_ID: u16 = 0;
const BOS_EOS_CONTEXT_ID: u16 = 0;
const ASCII_WORD_COST: i32 = 2_000;
const SAME_SURFACE_BONUS: i32 = -1_000;
const ASCII_SURFACE_PENALTY: i32 = 30_000;
const KATAKANA_SURFACE_PENALTY: i32 = 10_000;
const KATAKANA_NO_PARTICLE_PENALTY: i32 = 5_000;
const COUNTER_PAIR_BONUS: i32 = -6_000;
const COUNTER_NO_BONUS: i32 = -3_000;
const WEEKDAY_SUFFIX_BONUS: i32 = -6_000;
const UNUSUAL_CHARACTER_PENALTY: i32 = 15_000;
const SEGMENT_PENALTY: i32 = 500;
const SINGLE_CHAR_SEGMENT_PENALTY: i32 = 2_500;
const TWO_CHAR_SEGMENT_PENALTY: i32 = 750;
const MIN_INTERNAL_BEAM: usize = 64;
const MAX_INTERNAL_BEAM: usize = 128;
const INTERNAL_BEAM_MULTIPLIER: usize = 2;

#[derive(Debug, Clone)]
pub struct Engine {
    dictionary: Dictionary,
}

struct ConversionPath {
    base_cost: i32,
    language_cost: i32,
    eos_connection_cost: i32,
    eos_language_cost: i32,
    right_id: u16,
    segment_count: usize,
    before_previous_word: Option<u32>,
    previous_word: Option<u32>,
    parent: Option<Rc<ConversionPath>>,
    segment: Option<Segment>,
    segment_cost: Option<SegmentCost>,
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
        let beam = internal_beam_width(limit);
        self.candidates_internal(reading, limit, beam, false)
            .into_iter()
            .map(path_to_candidate)
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
        self.candidates_internal(reading, limit, internal_beam, false)
            .into_iter()
            .map(path_to_candidate)
            .collect()
    }

    pub fn candidate_costs_with_beam(
        &self,
        reading: &str,
        limit: usize,
        internal_beam: usize,
    ) -> Vec<CandidateCost> {
        self.candidates_internal(reading, limit, internal_beam, true)
            .into_iter()
            .map(path_to_candidate_cost)
            .collect()
    }

    fn candidates_internal(
        &self,
        reading: &str,
        limit: usize,
        internal_beam: usize,
        include_costs: bool,
    ) -> Vec<Rc<ConversionPath>> {
        if reading.is_empty() || limit == 0 {
            return Vec::new();
        }

        let internal_beam = internal_beam.max(limit).max(1);
        let boundaries = char_boundaries(reading);

        let mut paths: Vec<Vec<Rc<ConversionPath>>> = vec![Vec::new(); reading.len() + 1];

        let (before_previous_word, previous_word) =
            self.dictionary.language_model().initial_context();

        paths[0].push(Rc::new(ConversionPath {
            base_cost: 0,
            language_cost: 0,
            eos_connection_cost: 0,
            eos_language_cost: 0,
            right_id: BOS_EOS_CONTEXT_ID,
            segment_count: 0,
            before_previous_word,
            previous_word,
            parent: None,
            segment: None,
            segment_cost: None,
        }));

        for &start in &boundaries[..boundaries.len() - 1] {
            if paths[start].is_empty() {
                continue;
            }

            let suffix = &reading[start..];
            let matches = self.dictionary.lookup_prefix(suffix);
            let previous_paths = paths[start].clone();

            if let Some((ascii, end)) = ascii_run(reading, &boundaries, start) {
                for path in &previous_paths {
                    paths[end].push(self.extend_unknown_with_cost(
                        path,
                        ascii,
                        ASCII_WORD_COST,
                        include_costs,
                    ));
                }

                Self::prune_paths(&mut paths[end], internal_beam);
            }

            if matches.is_empty() {
                let Some(end) = next_boundary(&boundaries, start) else {
                    continue;
                };

                let unknown = &reading[start..end];

                for path in &previous_paths {
                    paths[end].push(self.extend_unknown_with_cost(
                        path,
                        unknown,
                        UNKNOWN_WORD_COST,
                        include_costs,
                    ));
                }

                Self::prune_paths(&mut paths[end], internal_beam);

                continue;
            }

            for entry in matches {
                let end = start + entry.reading.len();

                for path in &previous_paths {
                    paths[end].push(self.extend_dictionary_entry(path, entry, include_costs));
                }

                Self::prune_paths(&mut paths[end], internal_beam);
            }
        }

        let mut results = paths[reading.len()]
            .drain(..)
            .map(|path| self.finish_path(path))
            .collect::<Vec<_>>();

        results.sort_by_key(|path| path.base_cost + path.language_cost);

        let mut seen = std::collections::HashSet::new();
        results.retain(|path| seen.insert(path_text(path)));

        results.truncate(limit);

        results
    }

    fn extend_unknown_with_cost(
        &self,
        path: &Rc<ConversionPath>,
        unknown: &str,
        word_cost: i32,
        include_costs: bool,
    ) -> Rc<ConversionPath> {
        let connection_cost = self
            .dictionary
            .matrix()
            .cost(path.right_id, UNKNOWN_CONTEXT_ID) as i32;
        let current_word = self.dictionary.language_model().word_id(unknown);
        let language_model_cost = self.dictionary.language_model().cost_details(
            path.before_previous_word,
            path.previous_word,
            current_word,
        );
        let segment_penalty = Self::segment_penalty(unknown);
        let base_delta = connection_cost + word_cost + segment_penalty;
        let base_cost = path.base_cost + base_delta;
        let language_cost = path.language_cost + language_model_cost.cost;

        Rc::new(ConversionPath {
            base_cost,
            language_cost,
            eos_connection_cost: 0,
            eos_language_cost: 0,
            right_id: UNKNOWN_CONTEXT_ID,
            segment_count: path.segment_count + 1,
            before_previous_word: path.previous_word,
            previous_word: self
                .dictionary
                .language_model()
                .context_word_id(current_word),
            parent: Some(Rc::clone(path)),
            segment: Some(Segment {
                reading: unknown.to_string(),
                surface: unknown.to_string(),
            }),
            segment_cost: include_costs.then(|| SegmentCost {
                reading: unknown.to_string(),
                surface: unknown.to_string(),
                word_cost,
                connection_cost,
                surface_penalty: 0,
                segment_penalty,
                language_cost: language_model_cost.cost,
                language_order: language_model_cost.order,
                backoff_penalty: language_model_cost.backoff_penalty,
                unknown_language_model: language_model_cost.unknown,
                base_cost,
                total_cost: base_cost + language_cost,
            }),
        })
    }

    fn extend_dictionary_entry(
        &self,
        path: &Rc<ConversionPath>,
        entry: &DictionaryEntry,
        include_costs: bool,
    ) -> Rc<ConversionPath> {
        let connection_cost = self.dictionary.matrix().cost(path.right_id, entry.left_id) as i32;
        let current_word = self.dictionary.language_model().word_id(&entry.surface);
        let language_model_cost = self.dictionary.language_model().cost_details(
            path.before_previous_word,
            path.previous_word,
            current_word,
        );
        let surface_penalty = Self::surface_penalty(&entry.reading, &entry.surface)
            + Self::context_surface_adjustment(path, &entry.reading, &entry.surface);
        let segment_penalty = Self::segment_penalty(&entry.reading);
        let word_cost = entry.cost as i32;
        let base_delta = connection_cost + word_cost + surface_penalty + segment_penalty;
        let base_cost = path.base_cost + base_delta;
        let language_cost = path.language_cost + language_model_cost.cost;

        Rc::new(ConversionPath {
            base_cost,
            language_cost,
            eos_connection_cost: 0,
            eos_language_cost: 0,
            right_id: entry.right_id,
            segment_count: path.segment_count + 1,
            before_previous_word: path.previous_word,
            previous_word: self
                .dictionary
                .language_model()
                .context_word_id(current_word),
            parent: Some(Rc::clone(path)),
            segment: Some(Segment {
                reading: entry.reading.clone(),
                surface: entry.surface.clone(),
            }),
            segment_cost: include_costs.then(|| SegmentCost {
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
                base_cost,
                total_cost: base_cost + language_cost,
            }),
        })
    }

    fn finish_path(&self, path: Rc<ConversionPath>) -> Rc<ConversionPath> {
        let eos_cost = self
            .dictionary
            .matrix()
            .cost(path.right_id, BOS_EOS_CONTEXT_ID) as i32;
        let language_eos_cost = self
            .dictionary
            .language_model()
            .eos_cost(path.before_previous_word, path.previous_word);

        Rc::new(ConversionPath {
            base_cost: path.base_cost + eos_cost,
            language_cost: path.language_cost + language_eos_cost,
            eos_connection_cost: eos_cost,
            eos_language_cost: language_eos_cost,
            right_id: path.right_id,
            segment_count: path.segment_count,
            before_previous_word: path.before_previous_word,
            previous_word: path.previous_word,
            parent: Some(path),
            segment: None,
            segment_cost: None,
        })
    }

    fn prune_paths(paths: &mut Vec<Rc<ConversionPath>>, limit: usize) {
        paths.sort_by_key(|path| path.base_cost + path.language_cost);
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

    fn context_surface_adjustment(
        path: &ConversionPath,
        reading: &str,
        surface: &str,
    ) -> i32 {
        let mut adjustment = 0;

        if reading == "ようび" && surface == "曜日" {
            adjustment += WEEKDAY_SUFFIX_BONUS;
        }

        if let Some(previous) = path.segment.as_ref() {
            if is_numeric_surface(&previous.surface) && is_counter_surface(surface) {
                adjustment += COUNTER_PAIR_BONUS;
            }

            if previous.reading != "の" && is_counter_surface(&previous.surface) && surface == "の"
            {
                adjustment += COUNTER_NO_BONUS;
            }
        }

        adjustment
    }
}

fn path_to_candidate(path: Rc<ConversionPath>) -> Candidate {
    let segments = path_segments(&path);
    let text = segments
        .iter()
        .map(|segment| segment.surface.as_str())
        .collect();

    Candidate {
        text,
        cost: path.base_cost + path.language_cost,
        segments,
    }
}

fn path_to_candidate_cost(path: Rc<ConversionPath>) -> CandidateCost {
    CandidateCost {
        text: path_text(&path),
        cost: path.base_cost + path.language_cost,
        base_cost: path.base_cost,
        language_cost: path.language_cost,
        eos_connection_cost: path.eos_connection_cost,
        eos_language_cost: path.eos_language_cost,
        segments: path_segment_costs(&path),
    }
}

fn path_text(path: &ConversionPath) -> String {
    path_segments(path)
        .iter()
        .map(|segment| segment.surface.as_str())
        .collect()
}

fn path_segments(path: &ConversionPath) -> Vec<Segment> {
    let mut segments = Vec::with_capacity(path.segment_count);
    let mut current = Some(path);

    while let Some(path) = current {
        if let Some(segment) = &path.segment {
            segments.push(segment.clone());
        }

        current = path.parent.as_deref();
    }

    segments.reverse();
    segments
}

fn path_segment_costs(path: &ConversionPath) -> Vec<SegmentCost> {
    let mut segment_costs = Vec::with_capacity(path.segment_count);
    let mut current = Some(path);

    while let Some(path) = current {
        if let Some(segment_cost) = &path.segment_cost {
            segment_costs.push(segment_cost.clone());
        }

        current = path.parent.as_deref();
    }

    segment_costs.reverse();
    segment_costs
}

fn is_numeric_surface(surface: &str) -> bool {
    !surface.is_empty()
        && surface
            .chars()
            .all(|ch| ch.is_ascii_digit() || matches!(ch, '〇' | '一' | '二' | '三' | '四' | '五' | '六' | '七' | '八' | '九' | '十' | '百' | '千' | '万' | '億' | '兆'))
}

fn is_counter_surface(surface: &str) -> bool {
    matches!(
        surface,
        "匹"
            | "頭"
            | "羽"
            | "本"
            | "枚"
            | "個"
            | "粒"
            | "つ"
            | "人"
            | "台"
            | "冊"
            | "歳"
            | "才"
            | "回"
            | "件"
            | "軒"
            | "杯"
            | "着"
            | "足"
            | "階"
            | "番"
            | "円"
    )
}

fn internal_beam_width(limit: usize) -> usize {
    limit
        .saturating_mul(INTERNAL_BEAM_MULTIPLIER)
        .clamp(MIN_INTERNAL_BEAM, MAX_INTERNAL_BEAM)
}

fn ascii_run<'a>(input: &'a str, boundaries: &[usize], start: usize) -> Option<(&'a str, usize)> {
    let first = input[start..].chars().next()?;

    if !first.is_ascii_alphanumeric() {
        return None;
    }

    let end = boundaries
        .iter()
        .copied()
        .skip_while(|&boundary| boundary <= start)
        .take_while(|&boundary| {
            input[start..boundary]
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric())
        })
        .last()?;

    Some((&input[start..end], end))
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
