use ndarray::{Array1, Array2, Axis};
use super::error::PreprocessingError;
use super::utils::ensure_non_empty;

/// Statistics gathered during the fitting phase of normalization operations.
#[derive(Debug, Clone)]
pub struct FitStats {
    /// The mean of each column across the dataset.
    pub mean: Array1<f32>,
    /// The standard deviation of each column (optional, as not all operations compute it).
    pub std: Option<Array1<f32>>,
}

/// Computes the column-wise mean of the given embeddings and subtracts it from each row.
///
/// # Arguments
/// * `embeddings` - A 2D array of embeddings to mean-center.
///
/// # Returns
/// * `Ok((centered_embeddings, FitStats))` containing the mean-centered embeddings and the calculated mean.
/// * `Err(PreprocessingError::EmptyMatrix)` if the input matrix is empty.
pub fn mean_center(
    embeddings: &Array2<f32>,
) -> Result<(Array2<f32>, FitStats), PreprocessingError> {
    ensure_non_empty(embeddings)?;

    let mean = embeddings
        .mean_axis(Axis(0))
        .ok_or(PreprocessingError::EmptyMatrix)?;
    let centered = embeddings - &mean;

    Ok((
        centered,
        FitStats {
            mean,
            std: None,
        },
    ))
}
