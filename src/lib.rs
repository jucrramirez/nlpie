pub mod core;
pub mod errors;

#[cfg(feature = "extension-module")]
pub mod bindings;

/// Python module registration — only compiled when maturin builds the extension.
#[cfg(feature = "extension-module")]
mod python_module {
    use super::bindings;
    use pyo3::prelude::*;

    #[pymodule]
    pub fn _nlpie_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<bindings::normalization::PyFitStats>()?;
        m.add_class::<bindings::normalization::PyWhitenModel>()?;
        m.add_class::<bindings::normalization::PyEmbeddingPreprocessor>()?;

        m.add_function(wrap_pyfunction!(
            bindings::normalization::cosine_similarity,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(
            bindings::normalization::l2_normalize_rows,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(bindings::normalization::mean_center, m)?)?;
        m.add_function(wrap_pyfunction!(
            bindings::normalization::standardize_columns,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(bindings::normalization::whiten_pca, m)?)?;
        m.add_function(wrap_pyfunction!(
            bindings::normalization::remove_top_principal_components,
            m
        )?)?;

        Ok(())
    }
}
