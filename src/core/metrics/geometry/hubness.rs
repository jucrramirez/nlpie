use crate::errors::PreprocessingError;
use ndarray::Array2;
use rayon::prelude::*;

/// Computes exact K-Nearest Neighbors for each point, and returns
/// the hubness count for each point (how many times it appears in a K-NN set).
/// Also returns the skewness of the hubness distribution as a scalar index.
///
/// The outer loop is parallelised with Rayon. Each thread maintains its own
/// local count accumulator via `fold`, then `reduce` sums them element-wise.
pub fn compute_hubness(
    embeddings: &Array2<f32>,
    k: usize,
) -> Result<(Vec<usize>, f32), PreprocessingError> {
    let n_samples = embeddings.nrows();
    let n_features = embeddings.ncols();

    if n_samples == 0 || k == 0 || k >= n_samples {
        return Err(PreprocessingError::InvalidShape);
    }

    let hubness_counts: Vec<usize> = (0..n_samples)
        .into_par_iter()
        .with_min_len(100)
        .flat_map_iter(|i| {
            let row_i = embeddings.row(i);

            let mut distances: Vec<(usize, f32)> = Vec::with_capacity(n_samples);
            for j in 0..n_samples {
                if i == j {
                    continue;
                }
                let row_j = embeddings.row(j);
                let mut dist_sq = 0.0;
                for f in 0..n_features {
                    let d = row_i[f] - row_j[f];
                    dist_sq += d * d;
                }
                distances.push((j, dist_sq));
            }

            distances.sort_unstable_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let k_neighbors: Vec<usize> = distances.iter().take(k).map(|&(idx, _)| idx).collect();
            k_neighbors
        })
        .fold(
            || vec![0; n_samples],
            |mut acc, neighbor_idx| {
                acc[neighbor_idx] += 1;
                acc
            },
        )
        .reduce(
            || vec![0; n_samples],
            |mut a, b| {
                for (i, count) in b.into_iter().enumerate() {
                    a[i] += count;
                }
                a
            },
        );

    // Compute skewness of hubness counts (commonly used as hubness index)
    // Skewness = (mean( (x - mean)^3 ) ) / std^3
    let mean = k as f32;

    let mut variance = 0.0;
    let mut m3 = 0.0;

    for &count in &hubness_counts {
        let diff = count as f32 - mean;
        variance += diff * diff;
        m3 += diff * diff * diff;
    }

    variance /= n_samples as f32;
    m3 /= n_samples as f32;

    let std_dev = variance.sqrt();
    let skewness = if std_dev > 1e-12 {
        m3 / (std_dev * std_dev * std_dev)
    } else {
        0.0
    };

    Ok((hubness_counts, skewness))
}
