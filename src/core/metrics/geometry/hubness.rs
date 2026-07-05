use crate::errors::PreprocessingError;
use ndarray::Array2;

/// Computes exact K-Nearest Neighbors for each point, and returns
/// the hubness count for each point (how many times it appears in a K-NN set).
/// Also returns the skewness of the hubness distribution as a scalar index.
pub fn compute_hubness(
    embeddings: &Array2<f32>,
    k: usize,
) -> Result<(Vec<usize>, f32), PreprocessingError> {
    let n_samples = embeddings.nrows();
    let n_features = embeddings.ncols();

    if n_samples == 0 || k == 0 || k >= n_samples {
        return Err(PreprocessingError::InvalidShape);
    }

    let mut hubness_counts = vec![0; n_samples];

    // Compute pairwise distances and find top K neighbors
    // Note: For large N, this O(N^2 * D) is slow but exact.
    for i in 0..n_samples {
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

        // Sort distances to find K nearest neighbors
        // We can just sort by distance and take the first k
        distances.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        for &(neighbor_idx, _) in distances.iter().take(k) {
            hubness_counts[neighbor_idx] += 1;
        }
    }

    // Compute skewness of hubness counts (commonly used as hubness index)
    // Skewness = (mean( (x - mean)^3 ) ) / std^3
    let mean = (n_samples as f32 * k as f32) / (n_samples as f32); // Average is exactly k
    
    let mut variance = 0.0;
    let mut m3 = 0.0; // 3rd central moment

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
