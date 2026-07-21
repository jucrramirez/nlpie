use crate::errors::PreprocessingError;
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use std::collections::HashMap;

/// Computes the Calinski-Harabasz index (Variance Ratio Criterion).
/// Higher values indicate better clustering.
pub fn calinski_harabasz_score(
    embeddings: &Array2<f32>,
    labels: &[i32],
) -> Result<f32, PreprocessingError> {
    let n_samples = embeddings.nrows();
    if n_samples != labels.len() || n_samples == 0 {
        return Err(PreprocessingError::InvalidShape);
    }

    let mut cluster_sums: HashMap<i32, Array1<f32>> = HashMap::new();
    let mut cluster_counts: HashMap<i32, usize> = HashMap::new();

    let n_features = embeddings.ncols();
    let mut global_sum = Array1::<f32>::zeros(n_features);

    for (i, &label) in labels.iter().enumerate() {
        let row = embeddings.row(i);
        global_sum = global_sum + row;

        *cluster_counts.entry(label).or_insert(0) += 1;

        let sum = cluster_sums
            .entry(label)
            .or_insert_with(|| Array1::zeros(n_features));
        *sum = &*sum + &row;
    }

    let n_clusters = cluster_counts.len();
    if n_clusters < 2 || n_clusters >= n_samples {
        // Undefined for a single cluster or one point per cluster; by
        // convention (matching scikit-learn) we report 0.0.
        return Ok(0.0);
    }

    let global_mean = global_sum / (n_samples as f32);

    let mut wgss = 0.0; // within-cluster dispersion
    let mut bgss = 0.0; // between-cluster dispersion

    for (&label, sum) in &cluster_sums {
        let count = cluster_counts[&label] as f32;
        let mean = sum / count;

        // bgss += count * || mean - global_mean ||^2
        let mut bg_dist = 0.0;
        for j in 0..n_features {
            let diff = mean[j] - global_mean[j];
            bg_dist += diff * diff;
        }
        bgss += count * bg_dist;
    }

    // Now compute wgss
    for (i, &label) in labels.iter().enumerate() {
        let row = embeddings.row(i);
        let count = cluster_counts[&label] as f32;
        let mean = &cluster_sums[&label] / count;

        let mut w_dist = 0.0;
        for j in 0..n_features {
            let diff = row[j] - mean[j];
            w_dist += diff * diff;
        }
        wgss += w_dist;
    }

    if wgss == 0.0 {
        return Ok(1.0);
    }

    let ch_score = (bgss * (n_samples - n_clusters) as f32) / (wgss * (n_clusters - 1) as f32);
    Ok(ch_score)
}

/// Computes the mean Silhouette Coefficient of all samples.
pub fn silhouette_score(
    embeddings: &Array2<f32>,
    labels: &[i32],
) -> Result<f32, PreprocessingError> {
    let n_samples = embeddings.nrows();
    if n_samples != labels.len() || n_samples < 2 {
        return Err(PreprocessingError::InvalidShape);
    }

    let mut cluster_counts: HashMap<i32, usize> = HashMap::new();
    for &label in labels {
        *cluster_counts.entry(label).or_insert(0) += 1;
    }

    let n_clusters = cluster_counts.len();
    if n_clusters < 2 || n_clusters == n_samples {
        return Ok(0.0);
    }

    let n_features = embeddings.ncols();

    // O(N²·D) pairwise distances; the per-point loop is embarrassingly
    // parallel, so it is distributed over threads with Rayon.
    let total_silhouette: f32 = (0..n_samples)
        .into_par_iter()
        .with_min_len(100)
        .map(|i| {
            let label_i = labels[i];
            let count_i = cluster_counts[&label_i];

            if count_i == 1 {
                // Silhouette is 0 for points in clusters of size 1.
                return 0.0;
            }

            let mut a_i = 0.0;
            let mut b_sums: HashMap<i32, f32> = HashMap::new();

            let row_i = embeddings.row(i);

            for (j, &label_j) in labels.iter().enumerate() {
                if i == j {
                    continue;
                }
                let row_j = embeddings.row(j);

                let mut dist_sq = 0.0;
                for k in 0..n_features {
                    let d = row_i[k] - row_j[k];
                    dist_sq += d * d;
                }
                let dist = dist_sq.sqrt();

                if label_i == label_j {
                    a_i += dist;
                } else {
                    *b_sums.entry(label_j).or_insert(0.0) += dist;
                }
            }

            a_i /= (count_i - 1) as f32;

            let mut b_i = f32::MAX;
            for (&label_j, sum_dist) in &b_sums {
                let mean_dist = sum_dist / (cluster_counts[&label_j] as f32);
                if mean_dist < b_i {
                    b_i = mean_dist;
                }
            }

            let max_ab = a_i.max(b_i);
            if max_ab > 0.0 {
                (b_i - a_i) / max_ab
            } else {
                0.0
            }
        })
        .sum();

    Ok(total_silhouette / (n_samples as f32))
}
