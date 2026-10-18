use crate::core::normalization::{DEFAULT_EPS, l2_normalize_rows};
use crate::errors::PreprocessingError;
use ndarray::{Array1, Array2, ArrayView1};

/// Distance functions supported by embedding metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
}

impl DistanceMetric {
    /// Parses the public metric name used by the Python API.
    pub fn parse(value: &str) -> Result<Self, PreprocessingError> {
        match value {
            "cosine" => Ok(Self::Cosine),
            "euclidean" => Ok(Self::Euclidean),
            _ => Err(PreprocessingError::InvalidMetric {
                metric: value.to_owned(),
            }),
        }
    }

    /// Prepares rows for repeated distance calculations.
    pub fn prepare(self, embeddings: &Array2<f32>) -> Array2<f32> {
        match self {
            Self::Cosine => l2_normalize_rows(embeddings, DEFAULT_EPS),
            Self::Euclidean => embeddings.clone(),
        }
    }
}

/// Computes a non-negative distance between two rows prepared for `metric`.
pub fn distance(lhs: ArrayView1<'_, f32>, rhs: ArrayView1<'_, f32>, metric: DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::Cosine => {
            let similarity = lhs.dot(&rhs).clamp(-1.0, 1.0);
            1.0 - similarity
        }
        DistanceMetric::Euclidean => lhs
            .iter()
            .zip(rhs.iter())
            .map(|(&a, &b)| {
                let diff = a - b;
                diff * diff
            })
            .sum::<f32>()
            .sqrt(),
    }
}

/// Normalizes a centroid when cosine geometry is active.
pub fn normalize_centroid(centroid: &mut Array1<f32>, metric: DistanceMetric) {
    if metric == DistanceMetric::Cosine {
        let norm = centroid.dot(centroid).sqrt();
        if norm > DEFAULT_EPS {
            *centroid /= norm;
        } else {
            centroid.fill(0.0);
        }
    }
}
