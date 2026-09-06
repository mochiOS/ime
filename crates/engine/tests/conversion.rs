use engine::{ConnectionMatrix, Dictionary, DictionaryEntry, Engine};

fn entry(reading: &str, surface: &str, left_id: u16, right_id: u16, cost: i16) -> DictionaryEntry {
	DictionaryEntry {
		reading: reading.to_owned(),
		surface: surface.to_owned(),
		left_id,
		right_id,
		cost,
	}
}

#[test]
fn converts_single_word() {
	let dictionary = Dictionary::new(vec![
		entry("にほんご", "日本語", 0, 0, 100),
		entry("にほん", "日本", 0, 0, 200),
		entry("ご", "語", 0, 0, 200),
	]);
	let engine = Engine::new(dictionary);
	let candidates = engine.convert("にほんご");
	assert_eq!(candidates[0].text, "日本語");
	assert_eq!(candidates[0].cost, 100);
}

#[test]
fn connection_cost_changes_best_path() {
	let matrix = ConnectionMatrix::new(
		2,
		2,
		vec![
			0, 0,
			0, 500,
		],
	).unwrap();
	let dictionary = Dictionary::with_matrix(vec![
		entry("あ", "A", 0, 1, 0),
		entry("い", "B", 1, 0, 0),
		entry("あい", "AB", 0, 0, 100),
	], matrix);
	let engine = Engine::new(dictionary);
	let candidates = engine.convert("あい");
	assert_eq!(candidates[0].text, "AB");
	assert_eq!(candidates[0].cost, 100);
}

#[test]
fn keeps_unknown_text_as_is() {
	let dictionary = Dictionary::new(vec![entry("にほん", "日本", 0, 0, 100)]);
	let engine = Engine::new(dictionary);
	let candidates = engine.convert("にほんabc");
	assert_eq!(candidates[0].text, "日本abc");
}
