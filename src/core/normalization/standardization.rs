use super::centering::FitStats;
use super::utils::ensure_non_empty;
use crate::errors::PreprocessingError;
use ndarray::{Array1, Array2, Axis};

/// Standardizes the columns of the given embeddings to have zero mean and unit variance.
///
/// # Arguments
/// * `embeddings` - A 2D array of embeddings to standardize.
/// * `eps` - A small value to prevent division by zero in case of zero variance.
///
/// # Returns
/// * `Ok((standardized_embeddings, FitStats))` containing the standardized embeddings and the computed mean/std.
/// * `Err(PreprocessingError::ZeroStdDev)` if any column has zero standard deviation.
/// * `Err(PreprocessingError::EmptyMatrix)` if the input matrix is empty.
pub fn standardize_columns(
    embeddings: &Array2<f32>,
    eps: f32,
) -> Result<(Array2<f32>, FitStats), PreprocessingError> {
    ensure_non_empty(embeddings)?;

    let mean = embeddings
        .mean_axis(Axis(0))
        .ok_or(PreprocessingError::EmptyMatrix)?;
    let centered = embeddings - &mean;

    let n_rows = embeddings.nrows() as f32;
    let mut std = Array1::<f32>::zeros(embeddings.ncols());

    for (col_idx, col) in centered.axis_iter(Axis(1)).enumerate() {
        let variance = col.iter().map(|x| x * x).sum::<f32>() / n_rows;
        let sigma = variance.sqrt();
        if sigma <= eps {
            return Err(PreprocessingError::ZeroStdDev { column: col_idx });
        }
        std[col_idx] = sigma;
    }

    let standardized = &centered / &std;
    Ok((
        standardized,
        FitStats {
            mean,
            std: Some(std),
        },
    ))
}
