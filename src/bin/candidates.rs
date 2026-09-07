use engine::Engine;
use std::env;
use std::io::{self, BufRead, BufWriter, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);

    let mut dictionary = None;
    let mut limit = 16;
    let mut debug_costs = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--dictionary" => {
                dictionary = args.next();
            }

            "--limit" => {
                let value = args.next().ok_or("--limit requires a value")?;

                limit = value.parse()?;
            }

            "--debug-costs" => {
                debug_costs = true;
            }

            _ => {
                return Err(format!("unknown argument: {arg}").into());
            }
        }
    }

    let dictionary = dictionary.ok_or("--dictionary is required")?;

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

        let candidates = engine.candidate_costs(&reading, limit);

        writeln!(stdout, "{}", candidates.len())?;

        for candidate in candidates {
            writeln!(stdout, "{}", candidate.text)?;

            if debug_costs {
                writeln!(
                    stdout,
                    "  total={} base={} lm={} eos_conn={} eos_lm={}",
                    candidate.cost,
                    candidate.base_cost,
                    candidate.language_cost,
                    candidate.eos_connection_cost,
                    candidate.eos_language_cost
                )?;

                for segment in candidate.segments {
                    writeln!(
                        stdout,
                        "  {} -> {} word={} conn={} surface={} segment={} lm={} order={:?} backoff={} unk_lm={} base={} total={}",
                        segment.reading,
                        segment.surface,
                        segment.word_cost,
                        segment.connection_cost,
                        segment.surface_penalty,
                        segment.segment_penalty,
                        segment.language_cost,
                        segment.language_order,
                        segment.backoff_penalty,
                        segment.unknown_language_model,
                        segment.base_cost,
                        segment.total_cost
                    )?;
                }
            }
        }

        stdout.flush()?;
    }

    Ok(())
}
