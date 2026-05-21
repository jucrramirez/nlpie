use pyo3::prelude::*;

pub mod embeddings;
pub mod errors;

/// A Python module implemented in Rust.
#[pymodule]
mod nlpie_core {
    use pyo3::prelude::*;

    /// Formats the sum of two numbers as string.
    #[pyfunction]
    fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
        Ok((a + b).to_string())
    }
}
