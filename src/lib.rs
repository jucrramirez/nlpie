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

        m.add_function(wrap_pyfunction!(
            bindings::metrics::cosine_similarity_matrix,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(
            bindings::metrics::pairwise_cosine_stats,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(bindings::metrics::pearson_correlation, m)?)?;
        m.add_function(wrap_pyfunction!(
            bindings::metrics::spearman_correlation,
            m
        )?)?;

        m.add_function(wrap_pyfunction!(bindings::metrics::adjusted_rand_index, m)?)?;
        m.add_function(wrap_pyfunction!(
            bindings::metrics::normalized_mutual_info,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(bindings::metrics::purity_score, m)?)?;
        m.add_function(wrap_pyfunction!(
            bindings::metrics::calinski_harabasz_score,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(bindings::metrics::silhouette_score, m)?)?;

        m.add_function(wrap_pyfunction!(bindings::metrics::effective_rank, m)?)?;
        m.add_function(wrap_pyfunction!(
            bindings::metrics::similarity_to_global_mean,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(bindings::metrics::compute_hubness, m)?)?;

        // Projection quality metrics (TASK-005)
        m.add_function(wrap_pyfunction!(bindings::metrics::trustworthiness, m)?)?;
        m.add_function(wrap_pyfunction!(bindings::metrics::continuity, m)?)?;
        m.add_function(wrap_pyfunction!(bindings::metrics::projection_quality, m)?)?;

        // Retrieval and ranking metrics (TASK-006)
        m.add_function(wrap_pyfunction!(bindings::metrics::recall_at_k, m)?)?;
        m.add_function(wrap_pyfunction!(bindings::metrics::precision_at_k, m)?)?;
        m.add_function(wrap_pyfunction!(
            bindings::metrics::mean_reciprocal_rank,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(bindings::metrics::ndcg_at_k, m)?)?;
        m.add_function(wrap_pyfunction!(bindings::metrics::coverage_at_k, m)?)?;

        Ok(())
    }
}
