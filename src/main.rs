use std::io::{self, Write};
use std::time::Duration;

use crossterm::{
	cursor,
	event::{self, Event, KeyCode, KeyEventKind},
	execute,
	terminal::{
		self,
		Clear,
		ClearType,
		EnterAlternateScreen,
		LeaveAlternateScreen,
	},
};

use engine::Engine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let engine = Engine::open("vendor/ja/ja.mime")?;

	terminal::enable_raw_mode()?;

	let mut stdout = io::stdout();

	execute!(
		stdout,
		EnterAlternateScreen,
		cursor::Hide
	)?;

	let result = run(&engine, &mut stdout);

	execute!(
		stdout,
		cursor::Show,
		LeaveAlternateScreen
	)?;

	terminal::disable_raw_mode()?;

	result
}

fn run(
	engine: &Engine,
	stdout: &mut impl Write,
) -> Result<(), Box<dyn std::error::Error>> {
	let mut input = String::new();

	draw(engine, stdout, &input)?;

	loop {
		let Event::Key(key) = event::read()? else {
			continue;
		};

		if key.kind != KeyEventKind::Press {
			continue;
		}

		let changed = match key.code {
			KeyCode::Esc => break,

			KeyCode::Enter => {
				if input.is_empty() {
					false
				} else {
					input.clear();
					true
				}
			}

			KeyCode::Backspace => {
				input.pop().is_some()
			}

			KeyCode::Char(ch) => {
				input.push(ch);
				true
			}

			_ => false,
		};

		if changed {
			draw(engine, stdout, &input)?;
		}
	}

	Ok(())
}

fn draw(
	engine: &Engine,
	stdout: &mut impl Write,
	input: &str,
) -> Result<(), Box<dyn std::error::Error>> {
	let (_, height) = terminal::size()?;

	execute!(
		stdout,
		cursor::MoveTo(0, 0),
		Clear(ClearType::All)
	)?;

	execute!(
		stdout,
		cursor::MoveTo(0, 0),
		crossterm::style::Print("IME engine test"),
		cursor::MoveTo(0, 1),
		crossterm::style::Print("Esc: exit / Enter: clear"),
		cursor::MoveTo(0, 3),
		crossterm::style::Print(format!("> {input}"))
	)?;

	if input.is_empty() {
		stdout.flush()?;
		return Ok(());
	}

	let candidates = engine.candidates(input, 10);

	let start_y = 5;
	let available_lines = height.saturating_sub(start_y);

	for (index, candidate) in candidates
		.iter()
		.take(available_lines as usize)
		.enumerate()
	{
		let mut segments = String::new();

		for (segment_index, segment) in candidate.segments.iter().enumerate() {
			if segment_index != 0 {
				segments.push_str(" / ");
			}

			segments.push_str(&format!(
				"{} -> {}",
				segment.reading,
				segment.surface
			));
		}

		let line = format!(
			"{:>2}. {}  [{}]  cost={}",
			index + 1,
			candidate.text,
			segments,
			candidate.cost
		);

		execute!(
			stdout,
			cursor::MoveTo(0, start_y + index as u16),
			crossterm::style::Print(line)
		)?;
	}

	stdout.flush()?;

	Ok(())
}