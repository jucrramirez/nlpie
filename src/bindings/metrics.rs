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

use crate::core::normalization::DEFAULT_EPS;
use ndarray::Array2;
use pyo3::prelude::*;

/// Converts a Python 2D nested list (`Vec<Vec<f32>>`) to a 2D ndarray (`Array2<f32>`).
fn to_ndarray(vec: Vec<Vec<f32>>) -> PyResult<Array2<f32>> {
    if vec.is_empty() {
        return Ok(Array2::zeros((0, 0)));
    }
    let nrows = vec.len();
    let ncols = vec[0].len();
    if ncols == 0 {
        return Ok(Array2::zeros((nrows, 0)));
    }
    let mut flat = Vec::with_capacity(nrows * ncols);
    for row in vec {
        if row.len() != ncols {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "expected a 2D embedding matrix with consistent row lengths",
            ));
        }
        flat.extend(row);
    }
    Array2::from_shape_vec((nrows, ncols), flat)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Converts a 2D ndarray back to a nested list.
fn to_vec(arr: Array2<f32>) -> Vec<Vec<f32>> {
    let mut vec = Vec::with_capacity(arr.nrows());
    for row in arr.rows() {
        vec.push(row.to_vec());
    }
    vec
}

// ================= Basic Metrics =================

/// Computes the N x N cosine similarity matrix for an N x D embedding matrix.
#[pyfunction]
#[pyo3(signature = (embeddings, eps=DEFAULT_EPS))]
pub fn cosine_similarity_matrix(embeddings: Vec<Vec<f32>>, eps: f32) -> PyResult<Vec<Vec<f32>>> {
    let arr = to_ndarray(embeddings)?;
    let sim_matrix = core_cosine_similarity_matrix(&arr, eps);
    Ok(to_vec(sim_matrix))
}

/// Computes the Pearson correlation coefficient between two 1D vectors.
#[pyfunction]
pub fn pearson_correlation(x: Vec<f32>, y: Vec<f32>) -> PyResult<f32> {
    core_pearson_correlation(&x, &y).map_err(Into::into)
}

/// Computes the Spearman rank correlation coefficient between two 1D vectors.
#[pyfunction]
pub fn spearman_correlation(x: Vec<f32>, y: Vec<f32>) -> PyResult<f32> {
    core_spearman_correlation(&x, &y).map_err(Into::into)
}

// ================= Clustering Metrics =================

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
pub fn calinski_harabasz_score(embeddings: Vec<Vec<f32>>, labels: Vec<i32>) -> PyResult<f32> {
    let arr = to_ndarray(embeddings)?;
    core_ch_score(&arr, &labels).map_err(Into::into)
}

#[pyfunction]
pub fn silhouette_score(embeddings: Vec<Vec<f32>>, labels: Vec<i32>) -> PyResult<f32> {
    let arr = to_ndarray(embeddings)?;
    core_silhouette(&arr, &labels).map_err(Into::into)
}

// ================= Geometry & Hubness Metrics =================

#[pyfunction]
pub fn effective_rank(embeddings: Vec<Vec<f32>>) -> PyResult<f32> {
    let arr = to_ndarray(embeddings)?;
    let (_, eigenvalues) = core_cov_eig(&arr)?;
    core_effective_rank(&eigenvalues).map_err(Into::into)
}

#[pyfunction]
pub fn similarity_to_global_mean(embeddings: Vec<Vec<f32>>) -> PyResult<Vec<f32>> {
    let arr = to_ndarray(embeddings)?;
    let sims = core_sim_mean(&arr)?;
    Ok(sims.to_vec())
}

#[pyfunction]
pub fn compute_hubness(embeddings: Vec<Vec<f32>>, k: usize) -> PyResult<(Vec<usize>, f32)> {
    let arr = to_ndarray(embeddings)?;
    core_hubness(&arr, k).map_err(Into::into)
}

