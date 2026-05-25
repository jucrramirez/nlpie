use _nlpie_core::core::normalization::{
    cosine_similarity, l2_normalize_rows, mean_center, standardize_columns, whiten_pca,
};
use _nlpie_core::errors::PreprocessingError;
use ndarray::array;

#[test]
fn test_mean_center() {
    let embeddings = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let (centered, stats) = mean_center(&embeddings).unwrap();

    // Mean of columns: [3.0, 4.0]
    assert_eq!(stats.mean, array![3.0, 4.0]);
    assert_eq!(centered, array![[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]]);
}

#[test]
fn test_standardize_columns() {
    let embeddings = array![[1.0, 2.0], [3.0, 4.0]];
    let (standardized, stats) = standardize_columns(&embeddings, 1e-5).unwrap();

    // Mean: [2.0, 3.0]
    // Deviation from mean: [-1.0, -1.0], [1.0, 1.0]
    // Variance: 1.0, Standard Deviation: 1.0
    assert_eq!(stats.mean, array![2.0, 3.0]);
    assert_eq!(stats.std.unwrap(), array![1.0, 1.0]);
    assert_eq!(standardized, array![[-1.0, -1.0], [1.0, 1.0]]);
}

#[test]
fn test_standardize_zero_std_dev() {
    let embeddings = array![[1.0, 2.0], [1.0, 4.0]];
    let result = standardize_columns(&embeddings, 1e-5);
    // Column 0 has std dev = 0.0 which is <= eps
    match result {
        Err(PreprocessingError::ZeroStdDev { column }) => assert_eq!(column, 0),
        _ => panic!("Expected ZeroStdDev error"),
    }
}

#[test]
fn test_l2_normalize_rows() {
    let embeddings = array![[3.0, 4.0], [0.0, 0.0]];
    let normalized = l2_normalize_rows(&embeddings, 1e-5);

    // Norm of row 0 is 5.0, normalized is [0.6, 0.8]
    // Norm of row 1 is 0.0 <= eps, filled with 0.0
    assert!((normalized[[0, 0]] - 0.6).abs() < 1e-6);
    assert!((normalized[[0, 1]] - 0.8).abs() < 1e-6);
    assert_eq!(normalized[[1, 0]], 0.0);
    assert_eq!(normalized[[1, 1]], 0.0);
}

#[test]
fn test_cosine_similarity() {
    let a = vec![1.0, 0.0];
    let b = vec![0.0, 1.0];
    let sim = cosine_similarity(&a, &b).unwrap();
    assert!((sim - 0.0).abs() < 1e-6);

    let c = vec![1.0, 2.0];
    let d = vec![2.0, 4.0];
    let sim2 = cosine_similarity(&c, &d).unwrap();
    assert!((sim2 - 1.0).abs() < 1e-6);
}

#[test]
fn test_whiten_pca() {
    let embeddings = array![[1.0, 2.0], [3.0, 5.0], [5.0, 6.0]];
    let (whitened, model) = whiten_pca(&embeddings, Some(2), 1e-5).unwrap();
    assert_eq!(whitened.nrows(), 3);
    assert_eq!(whitened.ncols(), 2);
    assert_eq!(model.mean.len(), 2);
}
