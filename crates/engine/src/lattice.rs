#[derive(Debug, Clone)]
pub(crate) struct Node<'a> {
	pub(crate) reading: &'a str,
	pub(crate) surface: &'a str,
	pub(crate) right_id: u16,
	pub(crate) total_cost: i32,
	pub(crate) previous: Option<usize>,
}