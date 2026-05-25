#[cfg(feature = "extension-module")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "extension-module")]
use pyo3::prelude::*;
use thiserror::Error;

/// Enumeration of possible errors that can occur during embedding normalization.
#[derive(Debug, Error)]
pub enum PreprocessingError {
    /// Returned when the input matrix is empty.
    #[error("input matrix must have at least 1 row and 1 column")]
    EmptyMatrix,

    /// Returned when the input matrix shape is invalid or inconsistent.
    #[error("expected a 2D embedding matrix with consistent row lengths")]
    InvalidShape,

    /// Returned when standardizing a column with zero standard deviation, preventing division by zero.
    #[error("cannot standardize column {column}: standard deviation is zero")]
    ZeroStdDev {
        /// The index of the column with zero standard deviation.
        column: usize,
    },

    /// Returned when an invalid number of principal components is requested for PCA or whitening.
    #[error("requested {requested} principal components, but only {available} are available")]
    InvalidComponentCount {
        /// The number of components requested.
        requested: usize,
        /// The maximum number of components available based on the matrix dimensions.
        available: usize,
    },

    /// Wraps errors originating from linear algebra operations (e.g., eigenvalue decomposition).
    #[error("linear algebra error: {0}")]
    Linalg(String),
}

impl From<ndarray_linalg::error::LinalgError> for PreprocessingError {
    fn from(value: ndarray_linalg::error::LinalgError) -> Self {
        Self::Linalg(value.to_string())
    }
}

/// Convert PreprocessingError into a Python Exception (PyErr)
#[cfg(feature = "extension-module")]
impl From<PreprocessingError> for PyErr {
    fn from(err: PreprocessingError) -> Self {
        PyValueError::new_err(err.to_string())
    }
}
