use super::centering::{mean_center, FitStats};
use super::standardization::standardize_columns;
use super::utils::{l2_normalize_rows, DEFAULT_EPS};
use super::whitening::{remove_top_principal_components, whiten_pca, WhitenModel};
use crate::errors::PreprocessingError;
use ndarray::Array2;

/// A convenience preprocessor struct that wraps various normalization routines.
///
/// It holds an epsilon (`eps`) value used across various methods to avoid division by zero.
#[derive(Debug, Clone)]
pub struct EmbeddingPreprocessor {
    eps: f32,
}

impl Default for EmbeddingPreprocessor {
    fn default() -> Self {
        Self { eps: DEFAULT_EPS }
    }
}

impl EmbeddingPreprocessor {
    /// Creates a new `EmbeddingPreprocessor` with a custom epsilon value.
    pub fn new(eps: f32) -> Self {
        Self { eps }
    }

    /// Applies L2 normalization to each row of the embedding matrix.
    pub fn l2_normalize_rows(&self, embeddings: &Array2<f32>) -> Array2<f32> {
        l2_normalize_rows(embeddings, self.eps)
    }

    /// Centers the embedding matrix by subtracting the mean of each column.
    pub fn mean_center(
        &self,
        embeddings: &Array2<f32>,
    ) -> Result<(Array2<f32>, FitStats), PreprocessingError> {
        mean_center(embeddings)
    }

    /// Standardizes the embeddings (zero mean, unit variance for each column).
    pub fn standardize_columns(
        &self,
        embeddings: &Array2<f32>,
    ) -> Result<(Array2<f32>, FitStats), PreprocessingError> {
        standardize_columns(embeddings, self.eps)
    }

    /// Applies PCA whitening to the embeddings.
    pub fn whiten_pca(
        &self,
        embeddings: &Array2<f32>,
        n_components: Option<usize>,
    ) -> Result<(Array2<f32>, WhitenModel), PreprocessingError> {
        whiten_pca(embeddings, n_components, self.eps)
    }

    /// Removes the top principal components from the embeddings.
    pub fn remove_top_principal_components(
        &self,
        embeddings: &Array2<f32>,
        n_components: usize,
    ) -> Result<Array2<f32>, PreprocessingError> {
        remove_top_principal_components(embeddings, n_components)
    }
}
