use crate::core::metrics::distance::{DistanceMetric, distance, normalize_centroid};
use crate::errors::PreprocessingError;
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use std::collections::HashMap;

fn cluster_counts(labels: &[i32]) -> HashMap<i32, usize> {
    let mut counts = HashMap::new();
    for &label in labels {
        *counts.entry(label).or_insert(0) += 1;
    }
    counts
}

fn validate_cluster_inputs(
    embeddings: &Array2<f32>,
    labels: &[i32],
) -> Result<HashMap<i32, usize>, PreprocessingError> {
    if embeddings.nrows() != labels.len() || embeddings.nrows() < 2 {
        return Err(PreprocessingError::InvalidShape);
    }
    Ok(cluster_counts(labels))
}

/// Computes the mean Silhouette Coefficient with the requested distance.
pub fn silhouette_score(
    embeddings: &Array2<f32>,
    labels: &[i32],
    metric: DistanceMetric,
) -> Result<f32, PreprocessingError> {
    let cluster_counts = validate_cluster_inputs(embeddings, labels)?;
    let n_samples = embeddings.nrows();
    let n_clusters = cluster_counts.len();
    if n_clusters < 2 || n_clusters == n_samples {
        return Ok(0.0);
    }

    let prepared = metric.prepare(embeddings);
    let total_silhouette: f32 = (0..n_samples)
        .into_par_iter()
        .with_min_len(100)
        .map(|i| {
            let label_i = labels[i];
            let count_i = cluster_counts[&label_i];
            if count_i == 1 {
                return 0.0;
            }

            let mut a_i = 0.0;
            let mut b_sums: HashMap<i32, f32> = HashMap::new();
            let row_i = prepared.row(i);

            for (j, &label_j) in labels.iter().enumerate() {
                if i == j {
                    continue;
                }
                let dist = distance(row_i, prepared.row(j), metric);
                if label_i == label_j {
                    a_i += dist;
                } else {
                    *b_sums.entry(label_j).or_insert(0.0) += dist;
                }
            }

            a_i /= (count_i - 1) as f32;
            let b_i = b_sums
                .iter()
                .map(|(label, sum)| sum / cluster_counts[label] as f32)
                .fold(f32::MAX, f32::min);
            let max_ab = a_i.max(b_i);
            if max_ab > 0.0 {
                (b_i - a_i) / max_ab
            } else {
                0.0
            }
        })
        .sum();

    Ok(total_silhouette / n_samples as f32)
}

/// Computes the Davies–Bouldin index with the requested distance.
/// Lower values indicate better separated clusters.
pub fn davies_bouldin_score(
    embeddings: &Array2<f32>,
    labels: &[i32],
    metric: DistanceMetric,
) -> Result<f32, PreprocessingError> {
    let cluster_counts = validate_cluster_inputs(embeddings, labels)?;
    let n_samples = embeddings.nrows();
    let cluster_labels: Vec<i32> = cluster_counts.keys().copied().collect();
    if cluster_labels.len() < 2 || cluster_labels.len() == n_samples {
        return Ok(0.0);
    }

    let prepared = metric.prepare(embeddings);
    let n_features = prepared.ncols();
    let mut cluster_sums: HashMap<i32, Array1<f32>> = HashMap::new();
    for (i, &label) in labels.iter().enumerate() {
        let sum = cluster_sums
            .entry(label)
            .or_insert_with(|| Array1::zeros(n_features));
        *sum = &*sum + &prepared.row(i);
    }

    let mut centroids = HashMap::new();
    for (&label, sum) in &cluster_sums {
        let mut centroid = sum / cluster_counts[&label] as f32;
        normalize_centroid(&mut centroid, metric);
        centroids.insert(label, centroid);
    }

    let mut scatter = HashMap::new();
    for &label in &cluster_labels {
        let centroid = centroids[&label].view();
        let total: f32 = labels
            .iter()
            .enumerate()
            .filter(|(_, point_label)| **point_label == label)
            .map(|(i, _)| distance(prepared.row(i), centroid, metric))
            .sum();
        scatter.insert(label, total / cluster_counts[&label] as f32);
    }

    let mut total_ratio = 0.0;
    for &label_i in &cluster_labels {
        let mut worst_ratio: f32 = 0.0;
        for &label_j in &cluster_labels {
            if label_i == label_j {
                continue;
            }
            let separation = distance(
                centroids[&label_i].view(),
                centroids[&label_j].view(),
                metric,
            );
            let ratio = if separation <= f32::EPSILON {
                f32::INFINITY
            } else {
                (scatter[&label_i] + scatter[&label_j]) / separation
            };
            worst_ratio = worst_ratio.max(ratio);
        }
        total_ratio += worst_ratio;
    }

    Ok(total_ratio / cluster_labels.len() as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn cosine_silhouette_is_invariant_to_row_scaling() {
        let embeddings = array![[1.0f32, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, 2.0]];
        let scaled = array![[10.0f32, 0.0], [0.5, 0.0], [0.0, 3.0], [0.0, 0.25]];
        let labels = [0, 0, 1, 1];
        let first = silhouette_score(&embeddings, &labels, DistanceMetric::Cosine).unwrap();
        let second = silhouette_score(&scaled, &labels, DistanceMetric::Cosine).unwrap();
        assert!((first - second).abs() < 1e-6);
    }

    #[test]
    fn davies_bouldin_is_zero_for_degenerate_cluster_layouts() {
        let embeddings = array![[1.0f32, 0.0], [0.0, 1.0]];
        assert_eq!(
            davies_bouldin_score(&embeddings, &[0, 1], DistanceMetric::Cosine).unwrap(),
            0.0
        );
    }
}
