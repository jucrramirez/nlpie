use crate::core::normalization::{
    DEFAULT_EPS, EmbeddingPreprocessor, cosine_similarity as core_cosine_similarity,
    l2_normalize_rows as core_l2_normalize_rows, mean_center as core_mean_center,
    remove_top_principal_components as core_remove_top_principal_components,
    standardize_columns as core_standardize_columns, whiten_pca as core_whiten_pca,
};
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

/// Python-exposed version of `FitStats` holding mean and standard deviation.
#[pyclass(name = "FitStats", from_py_object)]
#[derive(Debug, Clone)]
pub struct PyFitStats {
    #[pyo3(get)]
    pub mean: Vec<f32>,
    #[pyo3(get)]
    pub std: Option<Vec<f32>>,
}

/// Python-exposed version of `WhitenModel` holding PCA projection data.
#[pyclass(name = "WhitenModel", from_py_object)]
#[derive(Debug, Clone)]
pub struct PyWhitenModel {
    #[pyo3(get)]
    pub mean: Vec<f32>,
    #[pyo3(get)]
    pub projection: Vec<Vec<f32>>,
    #[pyo3(get)]
    pub eigenvalues: Vec<f32>,
    #[pyo3(get)]
    pub eps: f32,
}

/// A convenience preprocessor struct that wraps various normalization routines.
#[pyclass(name = "EmbeddingPreprocessor", from_py_object)]
#[derive(Clone)]
pub struct PyEmbeddingPreprocessor {
    inner: EmbeddingPreprocessor,
}

#[pymethods]
impl PyEmbeddingPreprocessor {
    #[new]
    #[pyo3(signature = (eps = None))]
    pub fn new(eps: Option<f32>) -> Self {
        Self {
            inner: EmbeddingPreprocessor::new(eps.unwrap_or(DEFAULT_EPS)),
        }
    }

    /// Applies L2 normalization to each row of the embedding matrix.
    pub fn l2_normalize_rows(&self, embeddings: PyReadonlyArray2<f32>) -> PyResult<Vec<Vec<f32>>> {
        let arr = to_ndarray(embeddings);
        let result = self.inner.l2_normalize_rows(&arr);
        Ok(to_vec(result))
    }

    /// Centers the embedding matrix by subtracting the mean of each column.
    pub fn mean_center(
        &self,
        embeddings: PyReadonlyArray2<f32>,
    ) -> PyResult<(Vec<Vec<f32>>, PyFitStats)> {
        let arr = to_ndarray(embeddings);
        let (centered, stats) = self.inner.mean_center(&arr)?;
        let py_stats = PyFitStats {
            mean: stats.mean.to_vec(),
            std: stats.std.map(|s| s.to_vec()),
        };
        Ok((to_vec(centered), py_stats))
    }

    /// Standardizes the embeddings (zero mean, unit variance for each column).
    pub fn standardize_columns(
        &self,
        embeddings: PyReadonlyArray2<f32>,
    ) -> PyResult<(Vec<Vec<f32>>, PyFitStats)> {
        let arr = to_ndarray(embeddings);
        let (standardized, stats) = self.inner.standardize_columns(&arr)?;
        let py_stats = PyFitStats {
            mean: stats.mean.to_vec(),
            std: stats.std.map(|s| s.to_vec()),
        };
        Ok((to_vec(standardized), py_stats))
    }

    /// Applies PCA whitening to the embeddings.
    #[pyo3(signature = (embeddings, n_components = None))]
    pub fn whiten_pca(
        &self,
        embeddings: PyReadonlyArray2<f32>,
        n_components: Option<usize>,
    ) -> PyResult<(Vec<Vec<f32>>, PyWhitenModel)> {
        let arr = to_ndarray(embeddings);
        let (whitened, model) = self.inner.whiten_pca(&arr, n_components)?;
        let py_model = PyWhitenModel {
            mean: model.mean.to_vec(),
            projection: to_vec(model.projection),
            eigenvalues: model.eigenvalues.to_vec(),
            eps: model.eps,
        };
        Ok((to_vec(whitened), py_model))
    }

    /// Removes the top principal components from the embeddings.
    pub fn remove_top_principal_components(
        &self,
        embeddings: PyReadonlyArray2<f32>,
        n_components: usize,
    ) -> PyResult<Vec<Vec<f32>>> {
        let arr = to_ndarray(embeddings);
        let result = self
            .inner
            .remove_top_principal_components(&arr, n_components)?;
        Ok(to_vec(result))
    }
}

/// Standalone bindings for functional execution.

#[pyfunction]
#[pyo3(signature = (lhs, rhs))]
pub fn cosine_similarity(lhs: Vec<f32>, rhs: Vec<f32>) -> PyResult<f32> {
    core_cosine_similarity(&lhs, &rhs).map_err(Into::into)
}

#[pyfunction]
#[pyo3(signature = (embeddings, eps = DEFAULT_EPS))]
pub fn l2_normalize_rows(embeddings: PyReadonlyArray2<f32>, eps: f32) -> PyResult<Vec<Vec<f32>>> {
    let arr = to_ndarray(embeddings);
    let normalized = core_l2_normalize_rows(&arr, eps);
    Ok(to_vec(normalized))
}

#[pyfunction]
#[pyo3(signature = (embeddings))]
pub fn mean_center(embeddings: PyReadonlyArray2<f32>) -> PyResult<(Vec<Vec<f32>>, PyFitStats)> {
    let arr = to_ndarray(embeddings);
    let (centered, stats) = core_mean_center(&arr)?;
    let py_stats = PyFitStats {
        mean: stats.mean.to_vec(),
        std: stats.std.map(|s| s.to_vec()),
    };
    Ok((to_vec(centered), py_stats))
}

#[pyfunction]
#[pyo3(signature = (embeddings, eps = DEFAULT_EPS))]
pub fn standardize_columns(
    embeddings: PyReadonlyArray2<f32>,
    eps: f32,
) -> PyResult<(Vec<Vec<f32>>, PyFitStats)> {
    let arr = to_ndarray(embeddings);
    let (standardized, stats) = core_standardize_columns(&arr, eps)?;
    let py_stats = PyFitStats {
        mean: stats.mean.to_vec(),
        std: stats.std.map(|s| s.to_vec()),
    };
    Ok((to_vec(standardized), py_stats))
}

#[pyfunction]
#[pyo3(signature = (embeddings, n_components = None, eps = DEFAULT_EPS))]
pub fn whiten_pca(
    embeddings: PyReadonlyArray2<f32>,
    n_components: Option<usize>,
    eps: f32,
) -> PyResult<(Vec<Vec<f32>>, PyWhitenModel)> {
    let arr = to_ndarray(embeddings);
    let (whitened, model) = core_whiten_pca(&arr, n_components, eps)?;
    let py_model = PyWhitenModel {
        mean: model.mean.to_vec(),
        projection: to_vec(model.projection),
        eigenvalues: model.eigenvalues.to_vec(),
        eps: model.eps,
    };
    Ok((to_vec(whitened), py_model))
}

#[pyfunction]
#[pyo3(signature = (embeddings, n_components))]
pub fn remove_top_principal_components(
    embeddings: PyReadonlyArray2<f32>,
    n_components: usize,
) -> PyResult<Vec<Vec<f32>>> {
    let arr = to_ndarray(embeddings);
    let result = core_remove_top_principal_components(&arr, n_components)?;
    Ok(to_vec(result))
}
