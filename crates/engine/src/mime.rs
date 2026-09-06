pub(crate) const MAGIC: [u8; 4] = *b"MIME";
pub(crate) const VERSION: u16 = 1;
pub(crate) const HEADER_SIZE: usize = 48;
pub(crate) const ENTRY_SIZE: usize = 24;

pub(crate) fn read_u16(bytes: &[u8], offset: usize) -> Option<u16> {
	let value = bytes.get(offset..offset + 2)?;
	Some(u16::from_le_bytes([value[0], value[1]]))
}

pub(crate) fn read_i16(bytes: &[u8], offset: usize) -> Option<i16> {
	let value = bytes.get(offset..offset + 2)?;
	Some(i16::from_le_bytes([value[0], value[1]]))
}

pub(crate) fn read_u32(bytes: &[u8], offset: usize) -> Option<u32> {
	let value = bytes.get(offset..offset + 4)?;
	Some(u32::from_le_bytes([value[0], value[1], value[2], value[3]]))
}

pub(crate) fn read_u64(bytes: &[u8], offset: usize) -> Option<u64> {
	let value = bytes.get(offset..offset + 8)?;
	Some(u64::from_le_bytes([
		value[0], value[1], value[2], value[3],
		value[4], value[5], value[6], value[7],
	]))
}
