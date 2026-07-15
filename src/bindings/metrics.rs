use crate::core::metrics::basic::{
    cosine_similarity_matrix as core_cosine_similarity_matrix,
    pearson_correlation as core_pearson_correlation,
    spearman_correlation as core_spearman_correlation,
};
use crate::core::metrics::clustering::labels::{
    adjusted_rand_index as core_ari, normalized_mutual_info as core_nmi, purity_score as core_purity,
};
use crate::core::metrics::clustering::quality::{
    calinski_harabasz_score as core_ch_score, silhouette_score as core_silhouette,
};
use crate::core::metrics::geometry::anisotropy::{
    compute_covariance_and_eigenvalues as core_cov_eig, effective_rank as core_effective_rank,
    similarity_to_global_mean as core_sim_mean,
};
use crate::core::metrics::geometry::hubness::compute_hubness as core_hubness;
use crate::core::metrics::projection::trustworthiness::{
    trustworthiness as core_trustworthiness,
    continuity as core_continuity,
};
use crate::core::metrics::retrieval::ranking::{
    recall_at_k as core_recall_at_k,
    precision_at_k as core_precision_at_k,
    mean_reciprocal_rank as core_mrr,
    ndcg_at_k as core_ndcg_at_k,
    coverage_at_k as core_coverage_at_k,
};

use crate::core::normalization::DEFAULT_EPS;
use ndarray::Array2;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;

fn to_ndarray(py_array: PyReadonlyArray2<f32>) -> Array2<f32> {
    py_array.as_array().to_owned()
}

fn to_vec(arr: Array2<f32>) -> Vec<Vec<f32>> {
    let mut vec = Vec::with_capacity(arr.nrows());
    for row in arr.rows() {
        vec.push(row.to_vec());
    }
    vec
}

#[pyfunction]
#[pyo3(signature = (embeddings, eps=DEFAULT_EPS))]
pub fn cosine_similarity_matrix(
    embeddings: PyReadonlyArray2<f32>,
    eps: f32,
) -> PyResult<Vec<Vec<f32>>> {
    let arr = to_ndarray(embeddings);
    let sim_matrix = core_cosine_similarity_matrix(&arr, eps);
    Ok(to_vec(sim_matrix))
}

#[pyfunction]
pub fn pearson_correlation(x: Vec<f32>, y: Vec<f32>) -> PyResult<f32> {
    core_pearson_correlation(&x, &y).map_err(Into::into)
}

#[pyfunction]
pub fn spearman_correlation(x: Vec<f32>, y: Vec<f32>) -> PyResult<f32> {
    core_spearman_correlation(&x, &y).map_err(Into::into)
}

#[pyfunction]
pub fn adjusted_rand_index(labels_true: Vec<i32>, labels_pred: Vec<i32>) -> PyResult<f64> {
    core_ari(&labels_true, &labels_pred).map_err(Into::into)
}

#[pyfunction]
pub fn normalized_mutual_info(labels_true: Vec<i32>, labels_pred: Vec<i32>) -> PyResult<f64> {
    core_nmi(&labels_true, &labels_pred).map_err(Into::into)
}

#[pyfunction]
pub fn purity_score(labels_true: Vec<i32>, labels_pred: Vec<i32>) -> PyResult<f64> {
    core_purity(&labels_true, &labels_pred).map_err(Into::into)
}

#[pyfunction]
pub fn calinski_harabasz_score(
    embeddings: PyReadonlyArray2<f32>,
    labels: Vec<i32>,
) -> PyResult<f32> {
    let arr = to_ndarray(embeddings);
    core_ch_score(&arr, &labels).map_err(Into::into)
}

#[pyfunction]
pub fn silhouette_score(embeddings: PyReadonlyArray2<f32>, labels: Vec<i32>) -> PyResult<f32> {
    let arr = to_ndarray(embeddings);
    core_silhouette(&arr, &labels).map_err(Into::into)
}

#[pyfunction]
pub fn effective_rank(embeddings: PyReadonlyArray2<f32>) -> PyResult<f32> {
    let arr = to_ndarray(embeddings);
    let (_, eigenvalues) = core_cov_eig(&arr)?;
    core_effective_rank(&eigenvalues).map_err(Into::into)
}

#[pyfunction]
pub fn similarity_to_global_mean(embeddings: PyReadonlyArray2<f32>) -> PyResult<Vec<f32>> {
    let arr = to_ndarray(embeddings);
    let sims = core_sim_mean(&arr)?;
    Ok(sims.to_vec())
}

#[pyfunction]
pub fn compute_hubness(embeddings: PyReadonlyArray2<f32>, k: usize) -> PyResult<(Vec<usize>, f32)> {
    let arr = to_ndarray(embeddings);
    core_hubness(&arr, k).map_err(Into::into)
}

#[pyfunction]
#[pyo3(signature = (high_dim, low_dim, k = 10))]
pub fn trustworthiness(
    high_dim: PyReadonlyArray2<f32>,
    low_dim: PyReadonlyArray2<f32>,
    k: usize,
) -> PyResult<f32> {
    let high = to_ndarray(high_dim);
    let low = to_ndarray(low_dim);
    core_trustworthiness(&high, &low, k).map_err(Into::into)
}

#[pyfunction]
#[pyo3(signature = (high_dim, low_dim, k = 10))]
pub fn continuity(
    high_dim: PyReadonlyArray2<f32>,
    low_dim: PyReadonlyArray2<f32>,
    k: usize,
) -> PyResult<f32> {
    let high = to_ndarray(high_dim);
    let low = to_ndarray(low_dim);
    core_continuity(&high, &low, k).map_err(Into::into)
}

#[pyfunction]
pub fn recall_at_k(
    retrieved: Vec<usize>,
    relevant: Vec<usize>,
    k: usize,
) -> PyResult<f64> {
    core_recall_at_k(&retrieved, &relevant, k).map_err(Into::into)
}

#[pyfunction]
pub fn precision_at_k(
    retrieved: Vec<usize>,
    relevant: Vec<usize>,
    k: usize,
) -> PyResult<f64> {
    core_precision_at_k(&retrieved, &relevant, k).map_err(Into::into)
}

#[pyfunction]
pub fn mean_reciprocal_rank(retrieved: Vec<usize>, relevant: Vec<usize>) -> PyResult<f64> {
    core_mrr(&retrieved, &relevant).map_err(Into::into)
}

#[pyfunction]
pub fn ndcg_at_k(
    retrieved: Vec<usize>,
    relevant: Vec<usize>,
    k: usize,
) -> PyResult<f64> {
    core_ndcg_at_k(&retrieved, &relevant, k).map_err(Into::into)
}

#[pyfunction]
pub fn coverage_at_k(
    all_retrieved: Vec<Vec<usize>>,
    all_relevant: Vec<Vec<usize>>,
    k: usize,
) -> PyResult<f64> {
    core_coverage_at_k(&all_retrieved, &all_relevant, k).map_err(Into::into)
}
