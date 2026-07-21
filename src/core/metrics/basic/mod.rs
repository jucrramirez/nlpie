use crate::core::normalization::{DEFAULT_EPS, l2_normalize_rows};
use crate::errors::PreprocessingError;
use ndarray::Array2;
use std::cmp::Ordering;

/// Computes the N x N cosine similarity matrix for an N x D embedding matrix.
///
/// # Arguments
/// * `embeddings` - A 2D array of embeddings.
/// * `eps` - A small value to avoid division by zero during row normalization.
///
/// # Returns
/// An N x N array containing pairwise cosine similarities.
pub fn cosine_similarity_matrix(embeddings: &Array2<f32>, eps: f32) -> Array2<f32> {
    let normalized = l2_normalize_rows(embeddings, eps);
    normalized.dot(&normalized.t())
}

/// Summary statistics for a cosine similarity matrix.
#[derive(Debug, Clone, Copy)]
pub struct SimilarityStats {
    pub mean: f32,
    pub std: f32,
    pub min_val: f32,
    pub max_val: f32,
}

/// Computes the pairwise cosine similarities of the upper triangle (i < j)
/// together with summary statistics (mean, std, min, max) in a single pass.
///
/// Returns the flattened upper triangle in row-major order (exactly
/// `n * (n - 1) / 2` values) instead of the full N x N matrix, so no
/// O(N²) matrix ever crosses the Python FFI boundary. Use
/// [`cosine_similarity_matrix`] when the full matrix is actually needed.
pub fn pairwise_cosine_stats(embeddings: &Array2<f32>, eps: f32) -> (Vec<f32>, SimilarityStats) {
    let gram = cosine_similarity_matrix(embeddings, eps);
    let n = gram.nrows();

    if n <= 1 {
        return (
            Vec::new(),
            SimilarityStats {
                mean: 0.0,
                std: 0.0,
                min_val: 0.0,
                max_val: 0.0,
            },
        );
    }

    let mut triangle = Vec::with_capacity(n * (n - 1) / 2);
    let mut sum: f64 = 0.0;
    let mut min_val = gram[[0, 1]];
    let mut max_val = gram[[0, 1]];

    for i in 0..n {
        for j in (i + 1)..n {
            let val = gram[[i, j]];
            triangle.push(val);
            sum += val as f64;
            if val < min_val {
                min_val = val;
            }
            if val > max_val {
                max_val = val;
            }
        }
    }

    let count = triangle.len() as f64;
    let mean = (sum / count) as f32;

    let mut var_sum: f64 = 0.0;
    for &val in &triangle {
        let diff = val as f64 - mean as f64;
        var_sum += diff * diff;
    }
    let std = (var_sum / count).sqrt() as f32;

    (
        triangle,
        SimilarityStats {
            mean,
            std,
            min_val,
            max_val,
        },
    )
}

/// Computes the Pearson correlation coefficient between two 1D slices.
pub fn pearson_correlation(x: &[f32], y: &[f32]) -> Result<f32, PreprocessingError> {
    if x.len() != y.len() || x.is_empty() {
        return Err(PreprocessingError::InvalidShape);
    }

    let n = x.len() as f32;
    let sum_x: f32 = x.iter().sum();
    let sum_y: f32 = y.iter().sum();

    let mean_x = sum_x / n;
    let mean_y = sum_y / n;

    let mut num = 0.0;
    let mut den_x = 0.0;
    let mut den_y = 0.0;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        num += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }

    let den = (den_x * den_y).sqrt();
    if den <= DEFAULT_EPS {
        return Ok(0.0);
    }

    Ok(num / den)
}

/// Helper function to compute ranks (handling ties by averaging their ranks).
fn compute_ranks(x: &[f32]) -> Vec<f32> {
    let mut indexed_x: Vec<(usize, f32)> = x.iter().cloned().enumerate().collect();
    // Sort by value. We can use partial_cmp safely if no NaNs, but let's handle it
    indexed_x.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

    let mut ranks = vec![0.0; x.len()];
    let mut i = 0;
    let n = indexed_x.len();

    while i < n {
        let mut j = i + 1;
        while j < n && (indexed_x[j].1 - indexed_x[i].1).abs() < 1e-9 {
            j += 1;
        }

        let count = (j - i) as f32;
        // Sum of ranks: if i=0, j=2, ranks are 1 and 2. Average is (1+2)/2 = 1.5
        let sum_ranks = ((i + 1 + j) as f32) * count / 2.0;
        let avg_rank = sum_ranks / count;

        for k in i..j {
            ranks[indexed_x[k].0] = avg_rank;
        }

        i = j;
    }

    ranks
}

/// Computes the Spearman rank correlation coefficient between two 1D slices.
pub fn spearman_correlation(x: &[f32], y: &[f32]) -> Result<f32, PreprocessingError> {
    if x.len() != y.len() || x.is_empty() {
        return Err(PreprocessingError::InvalidShape);
    }

    let rank_x = compute_ranks(x);
    let rank_y = compute_ranks(y);

    pearson_correlation(&rank_x, &rank_y)
}
