use std::env;

use engine::Engine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let mut args = env::args().skip(1);

	let mut dictionary = None;
	let mut text = None;
	let mut limit = 16;

	while let Some(arg) = args.next() {
		match arg.as_str() {
			"--dictionary" => {
				dictionary = args.next();
			}

			"--text" => {
				text = args.next();
			}

			"--limit" => {
				let value = args
					.next()
					.ok_or("--limit requires a value")?;

				limit = value.parse()?;
			}

			_ => {
				return Err(format!(
					"unknown argument: {arg}"
				).into());
			}
		}
	}

	let dictionary =
		dictionary.ok_or("--dictionary is required")?;

	let text =
		text.ok_or("--text is required")?;

	let engine = Engine::open(dictionary)?;

	for candidate in engine.candidates(&text, limit) {
		println!("{}", candidate.text);
	}

	Ok(())
}