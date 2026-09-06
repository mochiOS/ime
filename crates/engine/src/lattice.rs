#[derive(Debug, Clone)]
pub(crate) struct Node<'a> {
	pub surface: &'a str,
	pub right_id: u16,
	pub total_cost: i32,
	pub previous: Option<usize>,
}
