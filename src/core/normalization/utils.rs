use crate::errors::PreprocessingError;
use ndarray::{Array2, Axis};

/// The default epsilon value used to prevent division by zero in numerical operations.
pub const DEFAULT_EPS: f32 = 1e-12;

/// Computes the cosine similarity between two 1D slice vectors.
///
/// # Arguments
/// * `lhs` - A slice representing the first vector.
/// * `rhs` - A slice representing the second vector.
///
/// # Returns
/// * `Ok(f32)` containing the cosine similarity.
/// * `Err(PreprocessingError::InvalidShape)` if the slices have different lengths or are empty.
pub fn cosine_similarity(lhs: &[f32], rhs: &[f32]) -> Result<f32, PreprocessingError> {
    if lhs.len() != rhs.len() || lhs.is_empty() {
        return Err(PreprocessingError::InvalidShape);
    }

    let mut dot = 0.0_f32;
    let mut lhs_norm_sq = 0.0_f32;
    let mut rhs_norm_sq = 0.0_f32;

    for (&a, &b) in lhs.iter().zip(rhs.iter()) {
        dot += a * b;
        lhs_norm_sq += a * a;
        rhs_norm_sq += b * b;
    }

    let denom = lhs_norm_sq.sqrt() * rhs_norm_sq.sqrt();
    if denom <= DEFAULT_EPS {
        return Ok(0.0);
    }

    Ok(dot / denom)
}

/// Applies L2 normalization to each row of the embedding matrix independently.
///
/// # Arguments
/// * `embeddings` - A 2D array of embeddings to normalize.
/// * `eps` - A small value to avoid division by zero; if the row norm is below `eps`, the row is set to zeros.
///
/// # Returns
/// A new 2D array with L2-normalized rows.
pub fn l2_normalize_rows(embeddings: &Array2<f32>, eps: f32) -> Array2<f32> {
    let mut output = embeddings.clone();

    for mut row in output.axis_iter_mut(Axis(0)) {
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > eps {
            row /= norm;
        } else {
            row.fill(0.0);
        }
    }

    output
}

/// Helper function to ensure the embedding matrix is not empty.
pub(crate) fn ensure_non_empty(embeddings: &Array2<f32>) -> Result<(), PreprocessingError> {
    if embeddings.nrows() == 0 || embeddings.ncols() == 0 {
        return Err(PreprocessingError::EmptyMatrix);
    }
    Ok(())
}
