use crate::errors::PreprocessingError;
use ndarray::Array2;
use rayon::prelude::*;

/// Builds an exact K-nearest-neighbour rank matrix for every point.
///
/// Returns a matrix `ranks` of shape `(n_samples, n_samples)` where `ranks[i][j]`
/// is the rank (1-indexed) of point `j` in the sorted neighbourhood of `i`
/// (rank 1 = closest). Points are excluded from their own neighbourhood (`i == j`).
///
/// The outer loop is parallelised with Rayon.
fn knn_ranks(embeddings: &Array2<f32>) -> Vec<Vec<usize>> {
    let n = embeddings.nrows();
    let d = embeddings.ncols();

    (0..n)
        .into_par_iter()
        .with_min_len(100)
        .map(|i| {
            let row_i = embeddings.row(i);
            let mut ranks_i = vec![0usize; n];

            // Compute squared Euclidean distances to all other points.
            let mut dists: Vec<(usize, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let row_j = embeddings.row(j);
                    let dist_sq = (0..d)
                        .map(|f| {
                            let diff = row_i[f] - row_j[f];
                            diff * diff
                        })
                        .sum::<f32>();
                    (j, dist_sq)
                })
                .collect();

            // Sort ascending by distance.
            dists.sort_unstable_by(|a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });

            // Assign 1-based ranks.
            for (rank, &(j, _)) in dists.iter().enumerate() {
                ranks_i[j] = rank + 1;
            }
            ranks_i
        })
        .collect()
}

/// Validates the neighbourhood size for projection-quality metrics.
///
/// `k` must satisfy `1 <= k < n` and keep the normalisation denominator
/// positive: `2*n - 3*k - 1 > 0`. Outside this domain the
/// Venna & Kaski normalisation is degenerate (zero or negative), which would
/// produce NaN or scores outside `[0, 1]`.
fn validate_projection_k(n: usize, k: usize) -> Result<(), PreprocessingError> {
    if k == 0 || k >= n || 2 * n < 3 * k + 2 {
        return Err(PreprocessingError::InvalidK { k, n_samples: n });
    }
    Ok(())
}

/// Trustworthiness computed from precomputed rank matrices.
///
/// For each point i, penalises the K nearest neighbours in the *projection*
/// whose rank in the *original* space is > K.
fn trustworthiness_from_ranks(
    ranks_high: &[Vec<usize>],
    ranks_low: &[Vec<usize>],
    n: usize,
    k: usize,
) -> f32 {
    let mut penalty = 0.0f32;
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            if ranks_low[i][j] <= k {
                let r = ranks_high[i][j] as f32;
                if r > k as f32 {
                    penalty += r - k as f32;
                }
            }
        }
    }

    let normalisation = 2.0 / (n as f32 * k as f32 * (2.0 * n as f32 - 3.0 * k as f32 - 1.0));
    1.0 - normalisation * penalty
}

/// Continuity computed from precomputed rank matrices.
///
/// For each point i, penalises the K nearest neighbours in the *original*
/// space whose rank in the *projection* is > K.
fn continuity_from_ranks(
    ranks_high: &[Vec<usize>],
    ranks_low: &[Vec<usize>],
    n: usize,
    k: usize,
) -> f32 {
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
    1.0 - normalisation * penalty
}

/// Validates the common shape preconditions for projection-quality metrics.
fn validate_shapes(
    high_dim: &Array2<f32>,
    low_dim: &Array2<f32>,
) -> Result<usize, PreprocessingError> {
    let n = high_dim.nrows();
    if n < 2 || low_dim.nrows() != n {
        return Err(PreprocessingError::InvalidShape);
    }
    Ok(n)
}

/// Computes **Trustworthiness** of a low-dimensional projection.
///
/// Measures whether the K nearest neighbours in the projection space were also
/// neighbours in the original high-dimensional space. A value of 1.0 is perfect.
///
/// # Errors
/// Returns [`PreprocessingError::InvalidShape`] if dimensions are inconsistent,
/// or [`PreprocessingError::InvalidK`] if `k` is outside the valid domain
/// (`1 <= k < n` and `2*n - 3*k - 1 > 0`).
pub fn trustworthiness(
    high_dim: &Array2<f32>,
    low_dim: &Array2<f32>,
    k: usize,
) -> Result<f32, PreprocessingError> {
    let n = validate_shapes(high_dim, low_dim)?;
    validate_projection_k(n, k)?;

    let ranks_high = knn_ranks(high_dim);
    let ranks_low = knn_ranks(low_dim);
    Ok(trustworthiness_from_ranks(&ranks_high, &ranks_low, n, k))
}

/// Computes **Continuity** of a low-dimensional projection.
///
/// Measures whether the K nearest neighbours in the original space are preserved
/// in the projection. A value of 1.0 is perfect.
///
/// # Errors
/// Same error conditions as [`trustworthiness`].
pub fn continuity(
    high_dim: &Array2<f32>,
    low_dim: &Array2<f32>,
    k: usize,
) -> Result<f32, PreprocessingError> {
    let n = validate_shapes(high_dim, low_dim)?;
    validate_projection_k(n, k)?;

    let ranks_high = knn_ranks(high_dim);
    let ranks_low = knn_ranks(low_dim);
    Ok(continuity_from_ranks(&ranks_high, &ranks_low, n, k))
}

/// Computes trustworthiness and continuity for several `k` values in one pass.
///
/// Both O(N²) k-NN rank matrices are built exactly once and reused for every
/// `k`, instead of being rebuilt per metric per `k`.
///
/// Returns a vector of `(k, trustworthiness, continuity)` triples in the same
/// order as `k_values`.
///
/// # Errors
/// Returns [`PreprocessingError::InvalidShape`] if dimensions are inconsistent
/// or `k_values` is empty, and [`PreprocessingError::InvalidK`] if any `k` is
/// outside the valid domain.
pub fn projection_quality(
    high_dim: &Array2<f32>,
    low_dim: &Array2<f32>,
    k_values: &[usize],
) -> Result<Vec<(usize, f32, f32)>, PreprocessingError> {
    let n = validate_shapes(high_dim, low_dim)?;
    if k_values.is_empty() {
        return Err(PreprocessingError::InvalidShape);
    }
    for &k in k_values {
        validate_projection_k(n, k)?;
    }

    let ranks_high = knn_ranks(high_dim);
    let ranks_low = knn_ranks(low_dim);

    Ok(k_values
        .iter()
        .map(|&k| {
            (
                k,
                trustworthiness_from_ranks(&ranks_high, &ranks_low, n, k),
                continuity_from_ranks(&ranks_high, &ranks_low, n, k),
            )
        })
        .collect())
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
    fn test_degenerate_k_returns_error_instead_of_nan() {
        // n=2, k=1 → normalisation denominator is 0 → must error, not NaN.
        let emb = array![[1.0f32, 0.0], [2.0, 0.0]];
        assert!(trustworthiness(&emb, &emb, 1).is_err());
        assert!(continuity(&emb, &emb, 1).is_err());
        // n=5, k=3 → denominator is 0 as well (2*5 - 3*3 - 1 = 0).
        let emb5 = array![
            [1.0f32, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [5.0, 0.0]
        ];
        assert!(trustworthiness(&emb5, &emb5, 3).is_err());
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
        assert!(
            (0.0..=1.0).contains(&t),
            "trustworthiness out of range: {t}"
        );
        assert!((0.0..=1.0).contains(&c), "continuity out of range: {c}");
    }

    #[test]
    fn test_projection_quality_matches_individual_calls() {
        let high = array![
            [1.0f32, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [10.0, 0.0],
            [11.0, 0.0],
            [12.0, 0.0]
        ];
        let low = array![
            [12.0f32, 0.0],
            [11.0, 0.0],
            [10.0, 0.0],
            [3.0, 0.0],
            [2.0, 0.0],
            [1.0, 0.0]
        ];
        let results = projection_quality(&high, &low, &[1, 2, 3]).unwrap();
        assert_eq!(results.len(), 3);
        for (k, t, c) in &results {
            assert_eq!(
                *t,
                trustworthiness(&high, &low, *k).unwrap(),
                "trustworthiness mismatch at k={k}"
            );
            assert_eq!(
                *c,
                continuity(&high, &low, *k).unwrap(),
                "continuity mismatch at k={k}"
            );
        }
    }

    #[test]
    fn test_projection_quality_rejects_empty_k_values() {
        let emb = array![[1.0f32, 0.0], [2.0, 0.0], [3.0, 0.0]];
        assert!(projection_quality(&emb, &emb, &[]).is_err());
    }

    #[test]
    fn test_projection_quality_rejects_invalid_k() {
        let emb = array![
            [1.0f32, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [5.0, 0.0]
        ];
        assert!(projection_quality(&emb, &emb, &[1, 3]).is_err());
    }
}
