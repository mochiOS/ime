use std::fmt;
use std::io;

#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    InvalidDictionary(&'static str),
    UnsupportedVersion(u16),
    InvalidUtf8,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::InvalidDictionary(message) => write!(f, "invalid .mime dictionary: {message}"),
            Self::UnsupportedVersion(version) => write!(f, "unsupported .mime version: {version}"),
            Self::InvalidUtf8 => write!(f, "dictionary contains invalid UTF-8"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}
