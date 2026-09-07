use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

const MAGIC: [u8; 4] = *b"MIME";
const VERSION: u16 = 1;

const HEADER_SIZE: u64 = 104;
const ENTRY_SIZE: u64 = 24;
const VOCAB_ENTRY_SIZE: u64 = 8;
const UNIGRAM_SIZE: u64 = 8;
const BIGRAM_SIZE: u64 = 12;
const TRIGRAM_SIZE: u64 = 16;
const LM_SCALE: f64 = 800.0;
const LM_ALPHA: f64 = 0.1;
const ORTHOGRAPHIC_VARIANT_PENALTY: i32 = 4_000;
const BOS_TOKEN: &str = "<s>";
const EOS_TOKEN: &str = "</s>";
const UNKNOWN_TOKEN: &str = "<unk>";

#[derive(Debug)]
struct SourceEntry {
    reading: String,
    surface: String,
    left_id: u16,
    right_id: u16,
    cost: i16,
}

#[derive(Debug)]
struct Args {
    lexicons: Vec<PathBuf>,
    matrix: PathBuf,
    lm_corpus: PathBuf,
    lm_min_unigram: u64,
    lm_min_bigram: u64,
    lm_min_trigram: u64,
    output: PathBuf,
}

struct LanguageModelSource {
    vocabulary: Vec<String>,
    unigrams: Vec<(u32, i32)>,
    bigrams: Vec<(u32, u32, i32)>,
    trigrams: Vec<(u32, u32, u32, i32)>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("mimec: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    let mut entries = Vec::new();
    for path in &args.lexicons {
        entries.extend(read_sudachi_csv(path)?);
    }
    entries.sort_unstable_by(|a, b| a.reading.cmp(&b.reading).then_with(|| a.cost.cmp(&b.cost)));

    let (previous_size, next_size, matrix) = read_matrix(&args.matrix)?;

    println!("building language model...");

    let language_model = build_language_model(
        &args.lm_corpus,
        args.lm_min_unigram,
        args.lm_min_bigram,
        args.lm_min_trigram,
    )?;

    write_mime(
        &args.output,
        &entries,
        previous_size,
        next_size,
        &matrix,
        &language_model,
    )?;

    println!(
        "wrote {} entries, {} words, {} unigrams, {} bigrams, {} trigrams to {}",
        entries.len(),
        language_model.vocabulary.len(),
        language_model.unigrams.len(),
        language_model.bigrams.len(),
        language_model.trigrams.len(),
        args.output.display(),
    );
    Ok(())
}

fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
    let mut lexicons = Vec::new();
    let mut matrix = None;
    let mut lm_corpus = None;
    let mut output = None;
    let mut lm_min_unigram = 2;
    let mut lm_min_bigram = 2;
    let mut lm_min_trigram = 2;
    let mut args = env::args_os().skip(1);

    while let Some(arg) = args.next() {
        match arg.to_str() {
            Some("--lex") => {
                lexicons.push(PathBuf::from(args.next().ok_or("--lex requires a path")?));
            }

            Some("--matrix") => {
                matrix = Some(PathBuf::from(
                    args.next().ok_or("--matrix requires a path")?,
                ));
            }

            Some("--lm-corpus") => {
                lm_corpus = Some(PathBuf::from(
                    args.next().ok_or("--lm-corpus requires a path")?,
                ));
            }

            Some("--lm-min-unigram") => {
                lm_min_unigram = args
                    .next()
                    .ok_or("--lm-min-unigram requires a value")?
                    .to_string_lossy()
                    .parse()?;
            }

            Some("--lm-min-bigram") => {
                lm_min_bigram = args
                    .next()
                    .ok_or("--lm-min-bigram requires a value")?
                    .to_string_lossy()
                    .parse()?;
            }

            Some("--lm-min-trigram") => {
                lm_min_trigram = args
                    .next()
                    .ok_or("--lm-min-trigram requires a value")?
                    .to_string_lossy()
                    .parse()?;
            }
            Some("-o" | "--output") => {
                output = Some(PathBuf::from(
                    args.next().ok_or("--output requires a path")?,
                ));
            }
            Some("-h" | "--help") => {
                print_usage();
                std::process::exit(0);
            }

            _ => {
                return Err(format!("unknown argument: {}", arg.to_string_lossy(),).into());
            }
        }
    }
    if lexicons.is_empty() {
        return Err("at least one --lex <Sudachi CSV> is required".into());
    }
    Ok(Args {
        lexicons,
        matrix: matrix.ok_or("--matrix <matrix.def> is required")?,
        lm_corpus: lm_corpus.ok_or("--lm-corpus <corpus.txt> is required")?,
        lm_min_unigram,
        lm_min_bigram,
        lm_min_trigram,
        output: output.unwrap_or_else(|| PathBuf::from("ja.mime")),
    })
}

fn print_usage() {
    println!(
        "Usage: mimec \
--lex <lex.csv> [--lex <lex.csv> ...] \
--matrix <matrix.def> \
--lm-corpus <tokenized.txt> \
[--lm-min-unigram <count>] \
[--lm-min-bigram <count>] \
[--lm-min-trigram <count>] \
[-o ja.mime]"
    );
}

fn read_sudachi_csv(path: &Path) -> Result<Vec<SourceEntry>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut text = String::new();
    reader.read_to_string(&mut text)?;
    let rows = parse_csv(&text)?;
    let mut entries = Vec::with_capacity(rows.len());
    for (line, row) in rows.into_iter().enumerate() {
        if row.len() < 12 {
            return Err(format!(
                "{}:{}: expected at least 12 CSV columns, got {}",
                path.display(),
                line + 1,
                row.len()
            )
            .into());
        }
        let left: i32 = row[1]
            .parse()
            .map_err(|_| format!("{}:{}: invalid left ID", path.display(), line + 1))?;
        let right: i32 = row[2]
            .parse()
            .map_err(|_| format!("{}:{}: invalid right ID", path.display(), line + 1))?;
        if left < 0 || right < 0 {
            continue;
        }
        let surface = row[4].clone();
        let mut cost: i32 = row[3]
            .parse()
            .map_err(|_| format!("{}:{}: invalid word cost", path.display(), line + 1))?;
        if let Some(normalized) = row.get(12) {
            if should_penalize_orthographic_variant(&surface, normalized) {
                cost = cost
                    .saturating_add(ORTHOGRAPHIC_VARIANT_PENALTY)
                    .min(i16::MAX as i32);
            }
        }
        if left > u16::MAX as i32
            || right > u16::MAX as i32
            || cost < i16::MIN as i32
            || cost > i16::MAX as i32
        {
            return Err(format!(
                "{}:{}: numeric field is outside .mime range",
                path.display(),
                line + 1
            )
            .into());
        }
        let source_reading = if row[11].is_empty() {
            row[0].as_str()
        } else {
            row[11].as_str()
        };
        let reading = katakana_to_hiragana(source_reading);
        if reading.is_empty() || surface.is_empty() {
            continue;
        }
        entries.push(SourceEntry {
            reading,
            surface,
            left_id: left as u16,
            right_id: right as u16,
            cost: cost as i16,
        });
    }
    Ok(entries)
}

fn read_matrix(path: &Path) -> Result<(usize, usize, Vec<i16>), Box<dyn std::error::Error>> {
    let reader = BufReader::new(File::open(path)?);
    let mut lines = reader.lines();
    let header = lines.next().ok_or("matrix.def is empty")??;
    let mut dims = header.split_whitespace();
    let previous_size: usize = dims
        .next()
        .ok_or("matrix.def header misses previous size")?
        .parse()?;
    let next_size: usize = dims
        .next()
        .ok_or("matrix.def header misses next size")?
        .parse()?;
    let len = previous_size
        .checked_mul(next_size)
        .ok_or("matrix dimensions overflow")?;
    let mut costs = vec![0i16; len];
    for (index, line) in lines.enumerate() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut fields = line.split_whitespace();
        let previous: usize = fields
            .next()
            .ok_or_else(|| format!("matrix.def:{}: missing previous ID", index + 2))?
            .parse()?;
        let next: usize = fields
            .next()
            .ok_or_else(|| format!("matrix.def:{}: missing next ID", index + 2))?
            .parse()?;
        let cost: i32 = fields
            .next()
            .ok_or_else(|| format!("matrix.def:{}: missing cost", index + 2))?
            .parse()?;
        if previous >= previous_size
            || next >= next_size
            || cost < i16::MIN as i32
            || cost > i16::MAX as i32
        {
            return Err(format!("matrix.def:{}: value outside declared range", index + 2).into());
        }
        costs[previous * next_size + next] = cost as i16;
    }
    Ok((previous_size, next_size, costs))
}

fn write_mime(
    path: &Path,
    entries: &[SourceEntry],
    previous_size: usize,
    next_size: usize,
    matrix: &[i16],
    language_model: &LanguageModelSource,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut strings = Vec::<u8>::new();
    let mut interned = HashMap::<String, (u32, u32)>::new();
    let mut encoded_entries = Vec::with_capacity(entries.len());
    for entry in entries {
        let reading = intern_string(&entry.reading, &mut strings, &mut interned)?;
        let surface = intern_string(&entry.surface, &mut strings, &mut interned)?;
        encoded_entries.push((reading, surface, entry.left_id, entry.right_id, entry.cost));
    }

    let mut encoded_vocabulary = Vec::with_capacity(language_model.vocabulary.len());
    for word in &language_model.vocabulary {
        encoded_vocabulary.push(intern_string(word, &mut strings, &mut interned)?);
    }

    let entry_offset = HEADER_SIZE;
    let string_offset = entry_offset + ENTRY_SIZE * entries.len() as u64;
    let matrix_offset = string_offset + strings.len() as u64;
    let vocabulary_offset = matrix_offset + matrix.len() as u64 * 2;
    let unigram_offset = vocabulary_offset + VOCAB_ENTRY_SIZE * encoded_vocabulary.len() as u64;
    let bigram_offset = unigram_offset + UNIGRAM_SIZE * language_model.unigrams.len() as u64;
    let trigram_offset = bigram_offset + BIGRAM_SIZE * language_model.bigrams.len() as u64;
    let end_offset = trigram_offset + TRIGRAM_SIZE * language_model.trigrams.len() as u64;
    let mut writer = BufWriter::new(File::create(path)?);

    writer.write_all(&MAGIC)?;
    writer.write_all(&VERSION.to_le_bytes())?;
    writer.write_all(&0u16.to_le_bytes())?;
    writer.write_all(&(entries.len() as u32).to_le_bytes())?;
    writer.write_all(&(previous_size as u32).to_le_bytes())?;
    writer.write_all(&(next_size as u32).to_le_bytes())?;
    writer.write_all(&(language_model.vocabulary.len() as u32).to_le_bytes())?;
    writer.write_all(&(language_model.unigrams.len() as u32).to_le_bytes())?;
    writer.write_all(&(language_model.bigrams.len() as u32).to_le_bytes())?;
    writer.write_all(&(language_model.trigrams.len() as u32).to_le_bytes())?;
    writer.write_all(&0u32.to_le_bytes())?;
    writer.write_all(&entry_offset.to_le_bytes())?;
    writer.write_all(&string_offset.to_le_bytes())?;
    writer.write_all(&matrix_offset.to_le_bytes())?;
    writer.write_all(&vocabulary_offset.to_le_bytes())?;
    writer.write_all(&unigram_offset.to_le_bytes())?;
    writer.write_all(&bigram_offset.to_le_bytes())?;
    writer.write_all(&trigram_offset.to_le_bytes())?;
    writer.write_all(&end_offset.to_le_bytes())?;

    for (reading, surface, left_id, right_id, cost) in encoded_entries {
        writer.write_all(&reading.0.to_le_bytes())?;
        writer.write_all(&reading.1.to_le_bytes())?;
        writer.write_all(&surface.0.to_le_bytes())?;
        writer.write_all(&surface.1.to_le_bytes())?;
        writer.write_all(&left_id.to_le_bytes())?;
        writer.write_all(&right_id.to_le_bytes())?;
        writer.write_all(&cost.to_le_bytes())?;
        writer.write_all(&0u16.to_le_bytes())?;
    }

    writer.write_all(&strings)?;

    for cost in matrix {
        writer.write_all(&cost.to_le_bytes())?;
    }

    for (offset, len) in encoded_vocabulary {
        writer.write_all(&offset.to_le_bytes())?;
        writer.write_all(&len.to_le_bytes())?;
    }

    for &(word, cost) in &language_model.unigrams {
        writer.write_all(&word.to_le_bytes())?;
        writer.write_all(&cost.to_le_bytes())?;
    }

    for &(previous, current, cost) in &language_model.bigrams {
        writer.write_all(&previous.to_le_bytes())?;
        writer.write_all(&current.to_le_bytes())?;
        writer.write_all(&cost.to_le_bytes())?;
    }

    for &(before_previous, previous, current, cost) in &language_model.trigrams {
        writer.write_all(&before_previous.to_le_bytes())?;
        writer.write_all(&previous.to_le_bytes())?;
        writer.write_all(&current.to_le_bytes())?;
        writer.write_all(&cost.to_le_bytes())?;
    }

    writer.flush()?;

    Ok(())
}

fn intern_string(
    text: &str,
    strings: &mut Vec<u8>,
    interned: &mut HashMap<String, (u32, u32)>,
) -> Result<(u32, u32), Box<dyn std::error::Error>> {
    if let Some(value) = interned.get(text) {
        return Ok(*value);
    }
    let offset = u32::try_from(strings.len()).map_err(|_| "string table exceeds 4 GiB")?;
    let len = u32::try_from(text.len()).map_err(|_| "single string exceeds 4 GiB")?;
    strings.extend_from_slice(text.as_bytes());
    let value = (offset, len);
    interned.insert(text.to_owned(), value);
    Ok(value)
}

fn katakana_to_hiragana(text: &str) -> String {
    text.chars()
        .map(|ch| {
            if ('ァ'..='ヶ').contains(&ch) {
                char::from_u32(ch as u32 - 0x60).unwrap_or(ch)
            } else {
                ch
            }
        })
        .collect()
}

fn should_penalize_orthographic_variant(surface: &str, normalized: &str) -> bool {
    if normalized.is_empty()
        || normalized == "*"
        || normalized == surface
        || surface
            .chars()
            .all(|ch| matches!(ch, '\u{3040}'..='\u{309f}'))
    {
        return false;
    }

    surface
        .chars()
        .any(|ch| matches!(ch, '\u{3400}'..='\u{4dbf}' | '\u{4e00}'..='\u{9fff}' | '\u{f900}'..='\u{faff}'))
}

fn parse_csv(input: &str) -> Result<Vec<Vec<String>>, Box<dyn std::error::Error>> {
    let mut rows = Vec::new();
    let mut row = Vec::new();
    let mut field = String::new();
    let mut chars = input.chars().peekable();
    let mut quoted = false;
    while let Some(ch) = chars.next() {
        if quoted {
            match ch {
                '"' if chars.peek() == Some(&'"') => {
                    chars.next();
                    field.push('"');
                }
                '"' => quoted = false,
                _ => field.push(ch),
            }
            continue;
        }
        match ch {
            '"' if field.is_empty() => quoted = true,
            ',' => row.push(std::mem::take(&mut field)),
            '\n' => {
                if field.ends_with('\r') {
                    field.pop();
                }
                row.push(std::mem::take(&mut field));
                if !row.iter().all(String::is_empty) {
                    rows.push(std::mem::take(&mut row));
                } else {
                    row.clear();
                }
            }
            _ => field.push(ch),
        }
    }
    if quoted {
        return Err("unterminated quoted CSV field".into());
    }
    if !field.is_empty() || !row.is_empty() {
        if field.ends_with('\r') {
            field.pop();
        }
        row.push(field);
        rows.push(row);
    }
    Ok(rows)
}

fn build_language_model(
    path: &Path,
    min_unigram: u64,
    min_bigram: u64,
    min_trigram: u64,
) -> Result<LanguageModelSource, Box<dyn std::error::Error>> {
    let mut raw_unigrams = HashMap::<String, u64>::new();

    {
        let reader = BufReader::new(File::open(path)?);

        for line in reader.lines() {
            let line = line?;

            for word in line.split('\t').filter(|word| !word.is_empty()) {
                *raw_unigrams.entry(word.to_owned()).or_default() += 1;
            }
        }
    }

    let mut vocabulary = raw_unigrams
        .iter()
        .filter_map(|(word, count)| {
            if *count >= min_unigram {
                Some(word.clone())
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    vocabulary.push(BOS_TOKEN.to_owned());

    vocabulary.push(EOS_TOKEN.to_owned());

    vocabulary.push(UNKNOWN_TOKEN.to_owned());

    vocabulary.sort_unstable();
    vocabulary.dedup();

    let word_ids = vocabulary
        .iter()
        .enumerate()
        .map(|(id, word)| (word.clone(), id as u32))
        .collect::<HashMap<_, _>>();

    let bos_id = word_ids[BOS_TOKEN];
    let eos_id = word_ids[EOS_TOKEN];
    let unknown_id = word_ids[UNKNOWN_TOKEN];

    let mut unigrams = HashMap::<u32, u64>::new();

    let mut bigrams = HashMap::<(u32, u32), u64>::new();

    let mut trigrams = HashMap::<(u32, u32, u32), u64>::new();

    let reader = BufReader::new(File::open(path)?);

    for line in reader.lines() {
        let line = line?;

        let mut words = Vec::new();

        words.push(bos_id);

        for word in line.split('\t').filter(|word| !word.is_empty()) {
            words.push(word_ids.get(word).copied().unwrap_or(unknown_id));
        }

        words.push(eos_id);

        for &word in &words {
            *unigrams.entry(word).or_default() += 1;
        }

        for pair in words.windows(2) {
            *bigrams.entry((pair[0], pair[1])).or_default() += 1;
        }

        for triple in words.windows(3) {
            *trigrams
                .entry((triple[0], triple[1], triple[2]))
                .or_default() += 1;
        }
    }

    let total_unigrams = unigrams.values().sum::<u64>();

    let vocabulary_size = vocabulary.len();

    let mut encoded_unigrams = Vec::new();

    for (&word, &count) in &unigrams {
        if count < min_unigram && word != bos_id && word != eos_id && word != unknown_id {
            continue;
        }

        let cost = probability_cost(count, total_unigrams, vocabulary_size);

        encoded_unigrams.push((word, cost));
    }

    let mut encoded_bigrams = Vec::new();

    for (&(previous, current), &count) in &bigrams {
        if count < min_bigram {
            continue;
        }

        let denominator = unigrams.get(&previous).copied().unwrap_or(1);

        let cost = probability_cost(count, denominator, vocabulary_size);

        encoded_bigrams.push((previous, current, cost));
    }

    let mut encoded_trigrams = Vec::new();

    for (&(before_previous, previous, current), &count) in &trigrams {
        if count < min_trigram {
            continue;
        }

        let denominator = bigrams
            .get(&(before_previous, previous))
            .copied()
            .unwrap_or(1);

        let cost = probability_cost(count, denominator, vocabulary_size);

        encoded_trigrams.push((before_previous, previous, current, cost));
    }

    encoded_unigrams.sort_unstable();
    encoded_bigrams.sort_unstable();
    encoded_trigrams.sort_unstable();

    Ok(LanguageModelSource {
        vocabulary,
        unigrams: encoded_unigrams,
        bigrams: encoded_bigrams,
        trigrams: encoded_trigrams,
    })
}

fn probability_cost(count: u64, denominator: u64, vocabulary_size: usize) -> i32 {
    let numerator = count as f64 + LM_ALPHA;

    let denominator = denominator as f64 + LM_ALPHA * vocabulary_size as f64;

    let probability = (numerator / denominator).clamp(f64::MIN_POSITIVE, 1.0);

    (-probability.ln() * LM_SCALE)
        .round()
        .clamp(0.0, i32::MAX as f64) as i32
}
