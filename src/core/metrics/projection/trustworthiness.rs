use crate::errors::PreprocessingError;
use ndarray::Array2;

/// Builds an exact K-nearest-neighbour rank matrix for every point.
///
/// Returns a matrix `ranks` of shape `(n_samples, n_samples)` where `ranks[i][j]`
/// is the rank (1-indexed) of point `j` in the sorted neighbourhood of `i`
/// (rank 1 = closest). Points are excluded from their own neighbourhood (`i == j`).
fn knn_ranks(embeddings: &Array2<f32>) -> Vec<Vec<usize>> {
    let n = embeddings.nrows();
    let d = embeddings.ncols();

    let mut ranks = vec![vec![0usize; n]; n];

    for i in 0..n {
        let row_i = embeddings.row(i);

        // Compute squared Euclidean distances to all other points.
        let mut dists: Vec<(usize, f32)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let row_j = embeddings.row(j);
                let dist_sq = (0..d).map(|f| {
                    let diff = row_i[f] - row_j[f];
                    diff * diff
                }).sum::<f32>();
                (j, dist_sq)
            })
            .collect();

        // Sort ascending by distance.
        dists.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Assign 1-based ranks.
        for (rank, &(j, _)) in dists.iter().enumerate() {
            ranks[i][j] = rank + 1;
        }
    }

    ranks
}

/// Computes **Trustworthiness** of a low-dimensional projection.
///
/// Measures whether the K nearest neighbours in the projection space were also
/// neighbours in the original high-dimensional space. A value of 1.0 is perfect.
///
/// # Arguments
/// * `high_dim` – embedding matrix in the original space `(n_samples, d_high)`.
/// * `low_dim`  – embedding matrix in the projected space `(n_samples, d_low)`.
/// * `k`        – number of neighbours to consider (default 10 is a common choice).
///
/// # Errors
/// Returns [`PreprocessingError::InvalidShape`] if dimensions are inconsistent or
/// `k` is zero or `>= n_samples - 1`.
pub fn trustworthiness(
    high_dim: &Array2<f32>,
    low_dim: &Array2<f32>,
    k: usize,
) -> Result<f32, PreprocessingError> {
    let n = high_dim.nrows();
    if n < 2 || low_dim.nrows() != n || k == 0 || k >= n {
        return Err(PreprocessingError::InvalidShape);
    }

    let ranks_high = knn_ranks(high_dim);
    let ranks_low = knn_ranks(low_dim);

    // For each point i, iterate over its K nearest neighbours in the *projection*.
    // Penalise those whose rank in the *original* space is > K.
    let mut penalty = 0.0f32;
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            // Is j among the K nearest in the projected space?
            if ranks_low[i][j] <= k {
                let r = ranks_high[i][j] as f32;
                if r > k as f32 {
                    penalty += r - k as f32;
                }
            }
        }
    }

    let normalisation = 2.0 / (n as f32 * k as f32 * (2.0 * n as f32 - 3.0 * k as f32 - 1.0));
    Ok(1.0 - normalisation * penalty)
}

/// Computes **Continuity** of a low-dimensional projection.
///
/// Measures whether the K nearest neighbours in the original space are preserved
/// in the projection. A value of 1.0 is perfect.
///
/// # Arguments
/// * `high_dim` – embedding matrix in the original space `(n_samples, d_high)`.
/// * `low_dim`  – embedding matrix in the projected space `(n_samples, d_low)`.
/// * `k`        – number of neighbours to consider.
///
/// # Errors
/// Returns [`PreprocessingError::InvalidShape`] if dimensions are inconsistent or
/// `k` is zero or `>= n_samples`.
pub fn continuity(
    high_dim: &Array2<f32>,
    low_dim: &Array2<f32>,
    k: usize,
) -> Result<f32, PreprocessingError> {
    let n = high_dim.nrows();
    if n < 2 || low_dim.nrows() != n || k == 0 || k >= n {
        return Err(PreprocessingError::InvalidShape);
    }

    let ranks_high = knn_ranks(high_dim);
    let ranks_low = knn_ranks(low_dim);

    // For each point i, iterate over its K nearest neighbours in the *original* space.
    // Penalise those whose rank in the *projection* is > K.
    let mut penalty = 0.0f32;
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            if ranks_high[i][j] <= k {
                let r = ranks_low[i][j] as f32;
                if r > k as f32 {
                    penalty += r - k as f32;
                }
            }
        }
    }

    let normalisation = 2.0 / (n as f32 * k as f32 * (2.0 * n as f32 - 3.0 * k as f32 - 1.0));
    Ok(1.0 - normalisation * penalty)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_perfect_projection_trustworthiness() {
        // When high-dim == low-dim the projection is perfect → score ≈ 1.0.
        let emb = array![
            [1.0f32, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [5.0, 0.0]
        ];
        let score = trustworthiness(&emb, &emb, 2).unwrap();
        assert!((score - 1.0).abs() < 1e-5, "expected 1.0, got {score}");
    }

    #[test]
    fn test_perfect_projection_continuity() {
        let emb = array![
            [1.0f32, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [5.0, 0.0]
        ];
        let score = continuity(&emb, &emb, 2).unwrap();
        assert!((score - 1.0).abs() < 1e-5, "expected 1.0, got {score}");
    }

    #[test]
    fn test_invalid_k_returns_error() {
        let emb = array![[1.0f32, 0.0], [2.0, 0.0]];
        // k == n is invalid
        assert!(trustworthiness(&emb, &emb, 2).is_err());
    }

    #[test]
    fn test_mismatched_rows_returns_error() {
        let high = array![[1.0f32, 0.0], [2.0, 0.0], [3.0, 0.0]];
        let low = array![[1.0f32, 0.0], [2.0, 0.0]];
        assert!(trustworthiness(&high, &low, 1).is_err());
    }

    #[test]
    fn test_score_in_range() {
        // A reversed ordering should give a score < 1.0 but still in [0, 1].
        let high = array![
            [1.0f32, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [10.0, 0.0],
            [11.0, 0.0]
        ];
        // Reverse the ordering in 2D
        let low = array![
            [11.0f32, 0.0],
            [10.0, 0.0],
            [3.0, 0.0],
            [2.0, 0.0],
            [1.0, 0.0]
        ];
        let t = trustworthiness(&high, &low, 2).unwrap();
        let c = continuity(&high, &low, 2).unwrap();
        assert!(t >= 0.0 && t <= 1.0, "trustworthiness out of range: {t}");
        assert!(c >= 0.0 && c <= 1.0, "continuity out of range: {c}");
    }
}
