use std::path::Path;

use crate::lattice::Node;
use crate::{Candidate, Dictionary, Error};

const UNKNOWN_WORD_COST: i32 = 10_000;
const UNKNOWN_CONTEXT_ID: u16 = 0;
const BOS_EOS_CONTEXT_ID: u16 = 0;

#[derive(Debug, Clone)]
pub struct Engine {
	dictionary: Dictionary,
}

impl Engine {
	pub fn new(dictionary: Dictionary) -> Self {
		Self { dictionary }
	}

	pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
		Ok(Self::new(Dictionary::open(path)?))
	}

	pub fn convert(&self, reading: &str) -> Vec<Candidate> {
		if reading.is_empty() {
			return Vec::new();
		}

		match self.best_path(reading) {
			Some(candidate) => vec![candidate],
			None => Vec::new(),
		}
	}

	fn best_path(&self, reading: &str) -> Option<Candidate> {
		let boundaries = char_boundaries(reading);
		let mut nodes: Vec<Node<'_>> = Vec::new();
		let mut ending_at: Vec<Vec<usize>> = vec![Vec::new(); reading.len() + 1];

		nodes.push(Node {
			surface: "",
			right_id: BOS_EOS_CONTEXT_ID,
			total_cost: 0,
			previous: None,
		});
		ending_at[0].push(0);

		for &start in &boundaries[..boundaries.len() - 1] {
			if ending_at[start].is_empty() {
				continue;
			}

			let suffix = &reading[start..];
			let matches = self.dictionary.lookup_prefix(suffix);
			if matches.is_empty() {
				let end = next_boundary(&boundaries, start)?;
				let surface = &reading[start..end];
				let (previous, total_cost) = self.best_previous(
					&nodes,
					&ending_at[start],
					UNKNOWN_CONTEXT_ID,
					UNKNOWN_WORD_COST,
				)?;
				let index = nodes.len();
				nodes.push(Node {
					surface,
					right_id: UNKNOWN_CONTEXT_ID,
					total_cost,
					previous: Some(previous),
				});
				ending_at[end].push(index);
				continue;
			}

			for entry in matches {
				let end = start + entry.reading.len();
				let (previous, total_cost) = self.best_previous(
					&nodes,
					&ending_at[start],
					entry.left_id,
					entry.cost as i32,
				)?;
				let index = nodes.len();
				nodes.push(Node {
					surface: &entry.surface,
					right_id: entry.right_id,
					total_cost,
					previous: Some(previous),
				});
				ending_at[end].push(index);
			}
		}

		let final_node = ending_at[reading.len()]
			.iter()
			.copied()
			.min_by_key(|&index| {
				let node = &nodes[index];
				node.total_cost
					+ self.dictionary.matrix().cost(node.right_id, BOS_EOS_CONTEXT_ID) as i32
			})?;

		let final_cost = nodes[final_node].total_cost
			+ self.dictionary.matrix().cost(nodes[final_node].right_id, BOS_EOS_CONTEXT_ID) as i32;
		let mut pieces = Vec::new();
		let mut current = final_node;
		while current != 0 {
			let node = &nodes[current];
			pieces.push(node.surface);
			current = node.previous?;
		}
		pieces.reverse();

		Some(Candidate {
			text: pieces.concat(),
			cost: final_cost,
		})
	}

	fn best_previous(
		&self,
		nodes: &[Node<'_>],
		indices: &[usize],
		next_left_id: u16,
		word_cost: i32,
	) -> Option<(usize, i32)> {
		indices
			.iter()
			.copied()
			.map(|index| {
				let previous = &nodes[index];
				let connection = self.dictionary.matrix().cost(previous.right_id, next_left_id) as i32;
				(index, previous.total_cost + connection + word_cost)
			})
			.min_by_key(|(_, cost)| *cost)
	}
}

fn char_boundaries(input: &str) -> Vec<usize> {
	let mut boundaries = input.char_indices().map(|(index, _)| index).collect::<Vec<_>>();
	boundaries.push(input.len());
	boundaries
}

fn next_boundary(boundaries: &[usize], current: usize) -> Option<usize> {
	boundaries.iter().copied().find(|&boundary| boundary > current)
}
