use std::env;
use std::io::{self, BufRead, BufWriter, Write};
use engine::Engine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let mut args = env::args().skip(1);

	let mut dictionary = None;
	let mut limit = 16;

	while let Some(arg) = args.next() {
		match arg.as_str() {
			"--dictionary" => {
				dictionary = args.next();
			}

			"--limit" => {
				let value = args
					.next()
					.ok_or("--limit requires a value")?;

				limit = value.parse()?;
			}

			_ => {
				return Err(
					format!("unknown argument: {arg}").into()
				);
			}
		}
	}

	let dictionary =
		dictionary.ok_or("--dictionary is required")?;

	let engine = Engine::open(dictionary)?;

	let stdin = io::stdin();
	let mut stdout = BufWriter::new(io::stdout());

	for line in stdin.lock().lines() {
		let reading = line?;

		if reading.is_empty() {
			writeln!(stdout, "0")?;
			stdout.flush()?;
			continue;
		}

		let candidates = engine.candidates(
			&reading,
			limit,
		);

		writeln!(
			stdout,
			"{}",
			candidates.len()
		)?;

		for candidate in candidates {
			writeln!(
				stdout,
				"{}",
				candidate.text
			)?;
		}

		stdout.flush()?;
	}

	Ok(())
}