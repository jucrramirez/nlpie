use super::centering::mean_center;
use super::utils::ensure_non_empty;
use crate::errors::PreprocessingError;
use ndarray::{Array1, Array2, s};
use ndarray_linalg::{Eigh, UPLO};

/// Holds the model state after fitting PCA whitening.
#[derive(Debug, Clone)]
pub struct WhitenModel {
    /// The mean vector used to center the data before projection.
    pub mean: Array1<f32>,
    /// The projection matrix (eigenvectors) used to transform the data.
    pub projection: Array2<f32>,
    /// The eigenvalues corresponding to the principal components.
    pub eigenvalues: Array1<f32>,
    /// The epsilon value used to avoid division by zero.
    pub eps: f32,
}

/// Computes the PCA whitening transformation on the given embeddings.
///
/// # Arguments
/// * `embeddings` - A 2D array of embeddings.
/// * `n_components` - The number of principal components to keep. If `None`, keeps all available.
/// * `eps` - A small value to prevent division by zero during scaling.
///
/// # Returns
/// * `Ok((whitened_embeddings, WhitenModel))` containing the transformed embeddings and the model.
/// * `Err` if linear algebra operations fail or the component count is invalid.
pub fn whiten_pca(
    embeddings: &Array2<f32>,
    n_components: Option<usize>,
    eps: f32,
) -> Result<(Array2<f32>, WhitenModel), PreprocessingError> {
    ensure_non_empty(embeddings)?;

    let (centered, stats) = mean_center(embeddings)?;
    let n_samples = centered.nrows();
    let n_features = centered.ncols();
    let available = n_samples.min(n_features);
    let k = n_components.unwrap_or(available);

    if k == 0 || k > available {
        return Err(PreprocessingError::InvalidComponentCount {
            requested: k,
            available,
        });
    }

    let cov = centered.t().dot(&centered) / ((n_samples as f32) - 1.0);
    let (eigenvalues, eigenvectors) = cov.eigh(UPLO::Upper)?;

    let mut indices: Vec<usize> = (0..eigenvalues.len()).collect();
    indices.sort_by(|&i, &j| eigenvalues[j].total_cmp(&eigenvalues[i]));
    let selected = &indices[..k];

    let mut selected_vals = Array1::<f32>::zeros(k);
    let mut projection = Array2::<f32>::zeros((n_features, k));

    for (dst, &src) in selected.iter().enumerate() {
        selected_vals[dst] = eigenvalues[src];
        let column = eigenvectors.slice(s![.., src]).to_owned();
        projection.slice_mut(s![.., dst]).assign(&column);
    }

    let projected = centered.dot(&projection);
    let scales = selected_vals.mapv(|v| 1.0 / (v.max(eps)).sqrt());
    let whitened = &projected * &scales;

    Ok((
        whitened,
        WhitenModel {
            mean: stats.mean,
            projection,
            eigenvalues: selected_vals,
            eps,
        },
    ))
}

/// Removes the top `n_components` principal components from the embeddings to debias them.
///
/// # Arguments
/// * `embeddings` - A 2D array of embeddings.
/// * `n_components` - The number of top principal components to remove.
///
/// # Returns
/// * `Ok(debiased_embeddings)` containing the reconstructed embeddings without the top components.
/// * `Err` if linear algebra operations fail or the component count is invalid.
pub fn remove_top_principal_components(
    embeddings: &Array2<f32>,
    n_components: usize,
) -> Result<Array2<f32>, PreprocessingError> {
    ensure_non_empty(embeddings)?;

    if n_components == 0 {
        return Ok(embeddings.clone());
    }

    let (centered, stats) = mean_center(embeddings)?;
    let n_samples = centered.nrows();
    let n_features = centered.ncols();
    let available = n_samples.min(n_features);

    if n_components > available {
        return Err(PreprocessingError::InvalidComponentCount {
            requested: n_components,
            available,
        });
    }

    let cov = centered.t().dot(&centered) / ((n_samples as f32) - 1.0);
    let (eigenvalues, eigenvectors) = cov.eigh(UPLO::Upper)?;

    let mut indices: Vec<usize> = (0..eigenvalues.len()).collect();
    indices.sort_by(|&i, &j| eigenvalues[j].total_cmp(&eigenvalues[i]));

    let mut top_components = Array2::<f32>::zeros((n_features, n_components));
    for (dst, &src) in indices.iter().take(n_components).enumerate() {
        let column = eigenvectors.slice(s![.., src]).to_owned();
        top_components.slice_mut(s![.., dst]).assign(&column);
    }

    let top_projection = centered.dot(&top_components);
    let reconstructed_top = top_projection.dot(&top_components.t());
    let debiased = centered - reconstructed_top;
    let restored = debiased + &stats.mean;

    Ok(restored)
}
