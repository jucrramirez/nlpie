use crate::core::normalization::mean_center;
use crate::errors::PreprocessingError;
use ndarray::{Array1, Array2};
use ndarray_linalg::{Eigh, UPLO};

/// Computes the covariance matrix and its eigenvalues.
/// Returns (covariance_matrix, eigenvalues).
pub fn compute_covariance_and_eigenvalues(
    embeddings: &Array2<f32>,
) -> Result<(Array2<f32>, Array1<f32>), PreprocessingError> {
    let n_samples = embeddings.nrows();
    if n_samples < 2 {
        return Err(PreprocessingError::InvalidShape);
    }

    let (centered, _) = mean_center(embeddings)?;
    let cov = centered.t().dot(&centered) / ((n_samples - 1) as f32);
    let (eigenvalues, _) = cov.eigh(UPLO::Upper)?;

    Ok((cov, eigenvalues))
}

/// Computes the effective rank of an embedding space based on its eigenvalues.
/// Effective Rank = exp(Entropy(normalized_eigenvalues)).
pub fn effective_rank(eigenvalues: &Array1<f32>) -> Result<f32, PreprocessingError> {
    let mut sum = 0.0;
    for &val in eigenvalues.iter() {
        if val > 0.0 {
            sum += val;
        }
    }

    if sum <= 0.0 {
        // Degenerate space (no variance in any direction): there is no
        // well-defined spectral distribution, so the space effectively
        // uses zero dimensions.
        return Ok(0.0);
    }

    let mut entropy = 0.0;
    for &val in eigenvalues.iter() {
        if val > 0.0 {
            let p = val / sum;
            entropy -= p * p.ln();
        }
    }

    Ok(entropy.exp())
}

/// Computes the similarity of each point to the global mean vector.
pub fn similarity_to_global_mean(
    embeddings: &Array2<f32>,
) -> Result<Array1<f32>, PreprocessingError> {
    let n_samples = embeddings.nrows();
    let n_features = embeddings.ncols();

    if n_samples == 0 || n_features == 0 {
        return Err(PreprocessingError::InvalidShape);
    }

    let mut mean_vec = Array1::<f32>::zeros(n_features);
    for row in embeddings.rows() {
        mean_vec = mean_vec + row;
    }
    mean_vec /= n_samples as f32;

    let mut mean_norm = 0.0;
    for &v in mean_vec.iter() {
        mean_norm += v * v;
    }
    mean_norm = mean_norm.sqrt();
    if mean_norm < 1e-12 {
        return Ok(Array1::zeros(n_samples));
    }

    let mut similarities = Array1::<f32>::zeros(n_samples);
    for (i, row) in embeddings.rows().into_iter().enumerate() {
        let mut dot = 0.0;
        let mut row_norm = 0.0;
        for j in 0..n_features {
            dot += row[j] * mean_vec[j];
            row_norm += row[j] * row[j];
        }
        row_norm = row_norm.sqrt();
        let sim = if row_norm < 1e-12 {
            0.0
        } else {
            dot / (row_norm * mean_norm)
        };
        similarities[i] = sim;
    }

    Ok(similarities)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, array};

    #[test]
    fn test_effective_rank_degenerate_spectrum_returns_zero() {
        // All-zero eigenvalues (e.g. constant embeddings) have no entropy.
        let eigenvalues = Array1::zeros(4);
        assert_eq!(effective_rank(&eigenvalues).unwrap(), 0.0);
    }

    #[test]
    fn test_effective_rank_uniform_spectrum_equals_dimension() {
        // Uniform spectrum → maximum entropy → effective rank == n_dims.
        let eigenvalues = array![1.0f32, 1.0, 1.0, 1.0];
        let rank = effective_rank(&eigenvalues).unwrap();
        assert!((rank - 4.0).abs() < 1e-4, "expected ~4.0, got {rank}");
    }

    #[test]
    fn test_effective_rank_single_dominant_component() {
        let eigenvalues = array![100.0f32, 1e-8, 1e-8, 1e-8];
        let rank = effective_rank(&eigenvalues).unwrap();
        assert!(rank < 1.1, "expected rank near 1.0, got {rank}");
    }
}
