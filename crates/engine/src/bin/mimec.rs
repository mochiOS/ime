use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

const MAGIC: [u8; 4] = *b"MIME";
const VERSION: u16 = 1;
const HEADER_SIZE: u64 = 48;
const ENTRY_SIZE: u64 = 24;

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
	output: PathBuf,
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
	write_mime(&args.output, &entries, previous_size, next_size, &matrix)?;

	println!("wrote {} entries to {}", entries.len(), args.output.display());
	Ok(())
}

fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
	let mut lexicons = Vec::new();
	let mut matrix = None;
	let mut output = None;
	let mut args = env::args_os().skip(1);
	while let Some(arg) = args.next() {
		match arg.to_str() {
			Some("--lex") => lexicons.push(PathBuf::from(args.next().ok_or("--lex requires a path")?)),
			Some("--matrix") => matrix = Some(PathBuf::from(args.next().ok_or("--matrix requires a path")?)),
			Some("-o" | "--output") => output = Some(PathBuf::from(args.next().ok_or("--output requires a path")?)),
			Some("-h" | "--help") => {
				print_usage();
				std::process::exit(0);
			}
			_ => return Err(format!("unknown argument: {}", arg.to_string_lossy()).into()),
		}
	}
	if lexicons.is_empty() {
		return Err("at least one --lex <Sudachi CSV> is required".into());
	}
	Ok(Args {
		lexicons,
		matrix: matrix.ok_or("--matrix <matrix.def> is required")?,
		output: output.unwrap_or_else(|| PathBuf::from("ja.mime")),
	})
}

fn print_usage() {
	println!("Usage: mimec --lex <lex.csv> [--lex <lex.csv> ...] --matrix <matrix.def> [-o ja.mime]");
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
			return Err(format!("{}:{}: expected at least 12 CSV columns, got {}", path.display(), line + 1, row.len()).into());
		}
		let left: i32 = row[1].parse().map_err(|_| format!("{}:{}: invalid left ID", path.display(), line + 1))?;
		let right: i32 = row[2].parse().map_err(|_| format!("{}:{}: invalid right ID", path.display(), line + 1))?;
		if left < 0 || right < 0 {
			continue;
		}
		let cost: i32 = row[3].parse().map_err(|_| format!("{}:{}: invalid word cost", path.display(), line + 1))?;
		if left > u16::MAX as i32 || right > u16::MAX as i32 || cost < i16::MIN as i32 || cost > i16::MAX as i32 {
			return Err(format!("{}:{}: numeric field is outside .mime range", path.display(), line + 1).into());
		}

		let surface = row[4].clone();
		let source_reading = if row[11].is_empty() { row[0].as_str() } else { row[11].as_str() };
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
	let previous_size: usize = dims.next().ok_or("matrix.def header misses previous size")?.parse()?;
	let next_size: usize = dims.next().ok_or("matrix.def header misses next size")?.parse()?;
	let len = previous_size.checked_mul(next_size).ok_or("matrix dimensions overflow")?;
	let mut costs = vec![0i16; len];
	for (index, line) in lines.enumerate() {
		let line = line?;
		let line = line.trim();
		if line.is_empty() || line.starts_with('#') {
			continue;
		}
		let mut fields = line.split_whitespace();
		let previous: usize = fields.next().ok_or_else(|| format!("matrix.def:{}: missing previous ID", index + 2))?.parse()?;
		let next: usize = fields.next().ok_or_else(|| format!("matrix.def:{}: missing next ID", index + 2))?.parse()?;
		let cost: i32 = fields.next().ok_or_else(|| format!("matrix.def:{}: missing cost", index + 2))?.parse()?;
		if previous >= previous_size || next >= next_size || cost < i16::MIN as i32 || cost > i16::MAX as i32 {
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
) -> Result<(), Box<dyn std::error::Error>> {
	let mut strings = Vec::<u8>::new();
	let mut interned = HashMap::<String, (u32, u32)>::new();
	let mut encoded_entries = Vec::with_capacity(entries.len());
	for entry in entries {
		let reading = intern_string(&entry.reading, &mut strings, &mut interned)?;
		let surface = intern_string(&entry.surface, &mut strings, &mut interned)?;
		encoded_entries.push((reading, surface, entry.left_id, entry.right_id, entry.cost));
	}

	let entry_offset = HEADER_SIZE;
	let string_offset = entry_offset + ENTRY_SIZE * entries.len() as u64;
	let matrix_offset = string_offset + strings.len() as u64;
	let mut writer = BufWriter::new(File::create(path)?);

	writer.write_all(&MAGIC)?;
	writer.write_all(&VERSION.to_le_bytes())?;
	writer.write_all(&0u16.to_le_bytes())?;
	writer.write_all(&(entries.len() as u32).to_le_bytes())?;
	writer.write_all(&(previous_size as u32).to_le_bytes())?;
	writer.write_all(&(next_size as u32).to_le_bytes())?;
	writer.write_all(&0u32.to_le_bytes())?;
	writer.write_all(&entry_offset.to_le_bytes())?;
	writer.write_all(&string_offset.to_le_bytes())?;
	writer.write_all(&matrix_offset.to_le_bytes())?;

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
