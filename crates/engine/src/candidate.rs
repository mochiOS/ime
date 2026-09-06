#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
	pub reading: String,
	pub surface: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
	pub text: String,
	pub cost: i32,
	pub segments: Vec<Segment>,
}