use crate::Error;
use crate::language_model::LanguageModel;
use crate::mime::{
    ENTRY_SIZE, HEADER_SIZE, MAGIC, VERSION, read_i16, read_u16, read_u32, read_u64,
};
use std::fs;
use std::path::Path;

const MAX_PREFIX_MATCHES_PER_READING: usize = 64;
const MAX_LOOKUP_READING_CHARS: usize = 24;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DictionaryEntry {
    pub reading: String,
    pub surface: String,
    pub left_id: u16,
    pub right_id: u16,
    pub cost: i16,
}

#[derive(Debug, Clone, Default)]
pub struct ConnectionMatrix {
    previous_size: usize,
    next_size: usize,
    costs: Vec<i16>,
}

impl ConnectionMatrix {
    pub fn new(previous_size: usize, next_size: usize, costs: Vec<i16>) -> Result<Self, Error> {
        let expected = previous_size
            .checked_mul(next_size)
            .ok_or(Error::InvalidDictionary(
                "connection matrix dimensions overflow",
            ))?;
        if costs.len() != expected {
            return Err(Error::InvalidDictionary(
                "connection matrix size does not match dimensions",
            ));
        }
        Ok(Self {
            previous_size,
            next_size,
            costs,
        })
    }

    pub fn previous_size(&self) -> usize {
        self.previous_size
    }

    pub fn next_size(&self) -> usize {
        self.next_size
    }

    pub fn cost(&self, previous_right_id: u16, next_left_id: u16) -> i16 {
        let previous = previous_right_id as usize;
        let next = next_left_id as usize;
        if previous >= self.previous_size || next >= self.next_size {
            return 0;
        }
        self.costs[previous * self.next_size + next]
    }
}

#[derive(Debug, Clone, Default)]
pub struct Dictionary {
    entries: Vec<DictionaryEntry>,
    matrix: ConnectionMatrix,
    language_model: LanguageModel,
}

impl Dictionary {
    pub fn new(mut entries: Vec<DictionaryEntry>) -> Self {
        entries
            .sort_unstable_by(|a, b| a.reading.cmp(&b.reading).then_with(|| a.cost.cmp(&b.cost)));
        Self {
            entries,
            matrix: ConnectionMatrix::default(),
            language_model: LanguageModel::default(),
        }
    }

    pub fn with_matrix(mut entries: Vec<DictionaryEntry>, matrix: ConnectionMatrix) -> Self {
        entries
            .sort_unstable_by(|a, b| a.reading.cmp(&b.reading).then_with(|| a.cost.cmp(&b.cost)));
        Self {
            entries,
            matrix,
            language_model: LanguageModel::default(),
        }
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let bytes = fs::read(path)?;
        Self::from_mime_bytes(&bytes)
    }

    pub fn entries(&self) -> &[DictionaryEntry] {
        &self.entries
    }

    pub fn matrix(&self) -> &ConnectionMatrix {
        &self.matrix
    }

    pub(crate) fn lookup_prefix<'a>(&'a self, input: &str) -> Vec<&'a DictionaryEntry> {
        let mut matches = Vec::new();
        for end in input
            .char_indices()
            .take(MAX_LOOKUP_READING_CHARS)
            .map(|(index, ch)| index + ch.len_utf8())
        {
            let reading = &input[..end];
            let start = self
                .entries
                .partition_point(|entry| entry.reading.as_str() < reading);
            let stop = self
                .entries
                .partition_point(|entry| entry.reading.as_str() <= reading);
            matches.extend(
                self.entries[start..stop]
                    .iter()
                    .take(MAX_PREFIX_MATCHES_PER_READING),
            );
        }
        matches
    }

    pub(crate) fn language_model(&self) -> &LanguageModel {
        &self.language_model
    }

    fn from_mime_bytes(bytes: &[u8]) -> Result<Self, Error> {
        if bytes.len() < HEADER_SIZE {
            return Err(Error::InvalidDictionary("file is smaller than the header"));
        }

        if bytes[0..4] != MAGIC {
            return Err(Error::InvalidDictionary("bad magic"));
        }

        let version = read_u16(bytes, 4).ok_or(Error::InvalidDictionary("missing version"))?;

        if version != VERSION {
            return Err(Error::UnsupportedVersion(version));
        }

        let entry_count =
            read_u32(bytes, 8).ok_or(Error::InvalidDictionary("missing entry count"))? as usize;

        let previous_size = read_u32(bytes, 12)
            .ok_or(Error::InvalidDictionary("missing matrix previous size"))?
            as usize;

        let next_size = read_u32(bytes, 16)
            .ok_or(Error::InvalidDictionary("missing matrix next size"))?
            as usize;

        let vocabulary_count = read_u32(bytes, 20).ok_or(Error::InvalidDictionary(
            "missing language model vocabulary count",
        ))? as usize;

        let unigram_count =
            read_u32(bytes, 24).ok_or(Error::InvalidDictionary("missing unigram count"))? as usize;

        let bigram_count =
            read_u32(bytes, 28).ok_or(Error::InvalidDictionary("missing bigram count"))? as usize;

        let trigram_count =
            read_u32(bytes, 32).ok_or(Error::InvalidDictionary("missing trigram count"))? as usize;

        let entry_offset = usize::try_from(
            read_u64(bytes, 40).ok_or(Error::InvalidDictionary("missing entry offset"))?,
        )
        .map_err(|_| Error::InvalidDictionary("entry offset is too large"))?;

        let string_offset = usize::try_from(
            read_u64(bytes, 48).ok_or(Error::InvalidDictionary("missing string offset"))?,
        )
        .map_err(|_| Error::InvalidDictionary("string offset is too large"))?;

        let matrix_offset = usize::try_from(
            read_u64(bytes, 56).ok_or(Error::InvalidDictionary("missing matrix offset"))?,
        )
        .map_err(|_| Error::InvalidDictionary("matrix offset is too large"))?;

        let vocabulary_offset = usize::try_from(
            read_u64(bytes, 64).ok_or(Error::InvalidDictionary("missing vocabulary offset"))?,
        )
        .map_err(|_| Error::InvalidDictionary("vocabulary offset is too large"))?;

        let unigram_offset = usize::try_from(
            read_u64(bytes, 72).ok_or(Error::InvalidDictionary("missing unigram offset"))?,
        )
        .map_err(|_| Error::InvalidDictionary("unigram offset is too large"))?;

        let bigram_offset = usize::try_from(
            read_u64(bytes, 80).ok_or(Error::InvalidDictionary("missing bigram offset"))?,
        )
        .map_err(|_| Error::InvalidDictionary("bigram offset is too large"))?;

        let trigram_offset = usize::try_from(
            read_u64(bytes, 88).ok_or(Error::InvalidDictionary("missing trigram offset"))?,
        )
        .map_err(|_| Error::InvalidDictionary("trigram offset is too large"))?;

        let end_offset = usize::try_from(
            read_u64(bytes, 96).ok_or(Error::InvalidDictionary("missing end offset"))?,
        )
        .map_err(|_| Error::InvalidDictionary("end offset is too large"))?;

        let entry_bytes = entry_count
            .checked_mul(ENTRY_SIZE)
            .and_then(|size| entry_offset.checked_add(size))
            .ok_or(Error::InvalidDictionary("entry table overflow"))?;

        if entry_bytes > bytes.len()
            || string_offset < entry_bytes
            || matrix_offset < string_offset
            || vocabulary_offset < matrix_offset
            || unigram_offset < vocabulary_offset
            || bigram_offset < unigram_offset
            || trigram_offset < bigram_offset
            || end_offset < trigram_offset
            || end_offset > bytes.len()
        {
            return Err(Error::InvalidDictionary("section offsets are invalid"));
        }

        let string_table = &bytes[string_offset..matrix_offset];

        let mut entries = Vec::with_capacity(entry_count);

        for index in 0..entry_count {
            let offset = entry_offset + index * ENTRY_SIZE;

            let reading_offset = read_u32(bytes, offset)
                .ok_or(Error::InvalidDictionary("truncated entry"))?
                as usize;

            let reading_len = read_u32(bytes, offset + 4)
                .ok_or(Error::InvalidDictionary("truncated entry"))?
                as usize;

            let surface_offset = read_u32(bytes, offset + 8)
                .ok_or(Error::InvalidDictionary("truncated entry"))?
                as usize;

            let surface_len = read_u32(bytes, offset + 12)
                .ok_or(Error::InvalidDictionary("truncated entry"))?
                as usize;

            let left_id =
                read_u16(bytes, offset + 16).ok_or(Error::InvalidDictionary("truncated entry"))?;

            let right_id =
                read_u16(bytes, offset + 18).ok_or(Error::InvalidDictionary("truncated entry"))?;

            let cost =
                read_i16(bytes, offset + 20).ok_or(Error::InvalidDictionary("truncated entry"))?;

            let reading = read_string(string_table, reading_offset, reading_len)?;

            let surface = read_string(string_table, surface_offset, surface_len)?;

            entries.push(DictionaryEntry {
                reading,
                surface,
                left_id,
                right_id,
                cost,
            });
        }

        let matrix_len = previous_size
            .checked_mul(next_size)
            .ok_or(Error::InvalidDictionary("matrix dimensions overflow"))?;

        let matrix_bytes = matrix_len
            .checked_mul(2)
            .and_then(|size| matrix_offset.checked_add(size))
            .ok_or(Error::InvalidDictionary("matrix size overflow"))?;

        if matrix_bytes > vocabulary_offset {
            return Err(Error::InvalidDictionary("truncated connection matrix"));
        }

        let mut costs = Vec::with_capacity(matrix_len);

        for index in 0..matrix_len {
            costs.push(
                read_i16(bytes, matrix_offset + index * 2)
                    .ok_or(Error::InvalidDictionary("truncated connection matrix"))?,
            );
        }

        let matrix = ConnectionMatrix::new(previous_size, next_size, costs)?;

        let language_model = LanguageModel::from_mime_bytes(
            bytes,
            string_table,
            vocabulary_count,
            unigram_count,
            bigram_count,
            trigram_count,
            vocabulary_offset,
            unigram_offset,
            bigram_offset,
            trigram_offset,
            end_offset,
        )?;

        entries
            .sort_unstable_by(|a, b| a.reading.cmp(&b.reading).then_with(|| a.cost.cmp(&b.cost)));

        Ok(Self {
            entries,
            matrix,
            language_model,
        })
    }
}

fn read_string(table: &[u8], offset: usize, len: usize) -> Result<String, Error> {
    let end = offset
        .checked_add(len)
        .ok_or(Error::InvalidDictionary("string range overflow"))?;
    let bytes = table.get(offset..end).ok_or(Error::InvalidDictionary(
        "string range is outside string table",
    ))?;
    let text = std::str::from_utf8(bytes).map_err(|_| Error::InvalidUtf8)?;
    Ok(text.to_owned())
}
