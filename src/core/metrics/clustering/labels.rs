use crate::errors::PreprocessingError;
use std::collections::HashMap;

type LabelCounts = HashMap<i32, usize>;
type ContingencyTable = (HashMap<(i32, i32), usize>, LabelCounts, LabelCounts);

/// Builds a contingency table from two label vectors.
/// Returns a map from (label_u, label_v) to counts, and maps of marginals.
fn build_contingency_table(u: &[i32], v: &[i32]) -> ContingencyTable {
    let mut contingency = HashMap::new();
    let mut a_sums = HashMap::new();
    let mut b_sums = HashMap::new();

    for (&a, &b) in u.iter().zip(v.iter()) {
        *contingency.entry((a, b)).or_insert(0) += 1;
        *a_sums.entry(a).or_insert(0) += 1;
        *b_sums.entry(b).or_insert(0) += 1;
    }

    (contingency, a_sums, b_sums)
}

fn comb2(n: usize) -> f64 {
    (n as f64) * ((n - 1) as f64) / 2.0
}

/// Computes the Adjusted Rand Index (ARI) between two clustering assignments.
pub fn adjusted_rand_index(
    labels_true: &[i32],
    labels_pred: &[i32],
) -> Result<f64, PreprocessingError> {
    if labels_true.len() != labels_pred.len() || labels_true.is_empty() {
        return Err(PreprocessingError::InvalidShape);
    }

    let n = labels_true.len();
    if n < 2 {
        return Ok(1.0);
    }

    let (contingency, a_sums, b_sums) = build_contingency_table(labels_true, labels_pred);

    let mut sum_comb_c = 0.0;
    for &count in contingency.values() {
        sum_comb_c += comb2(count);
    }

    let mut sum_comb_a = 0.0;
    for &count in a_sums.values() {
        sum_comb_a += comb2(count);
    }

    let mut sum_comb_b = 0.0;
    for &count in b_sums.values() {
        sum_comb_b += comb2(count);
    }

    let comb_n = comb2(n);
    let expected_index = sum_comb_a * sum_comb_b / comb_n;
    let max_index = (sum_comb_a + sum_comb_b) / 2.0;
    let index = sum_comb_c;

    if (max_index - expected_index).abs() < 1e-12 {
        Ok(1.0)
    } else {
        Ok((index - expected_index) / (max_index - expected_index))
    }
}

/// Computes the Normalized Mutual Information (NMI) between two clustering assignments.
pub fn normalized_mutual_info(
    labels_true: &[i32],
    labels_pred: &[i32],
) -> Result<f64, PreprocessingError> {
    if labels_true.len() != labels_pred.len() || labels_true.is_empty() {
        return Err(PreprocessingError::InvalidShape);
    }

    let n = labels_true.len() as f64;
    let (contingency, a_sums, b_sums) = build_contingency_table(labels_true, labels_pred);

    let mut mi = 0.0;
    for (&(a, b), &count) in &contingency {
        if count > 0 {
            let p_xy = (count as f64) / n;
            let p_x = (a_sums[&a] as f64) / n;
            let p_y = (b_sums[&b] as f64) / n;
            mi += p_xy * (p_xy / (p_x * p_y)).ln();
        }
    }

    let mut h_true = 0.0;
    for &count in a_sums.values() {
        if count > 0 {
            let p_x = (count as f64) / n;
            h_true -= p_x * p_x.ln();
        }
    }

    let mut h_pred = 0.0;
    for &count in b_sums.values() {
        if count > 0 {
            let p_y = (count as f64) / n;
            h_pred -= p_y * p_y.ln();
        }
    }

    let max_h = h_true.max(h_pred);
    if max_h < 1e-12 {
        Ok(1.0)
    } else {
        // We use arithmetic mean for NMI
        Ok(2.0 * mi / (h_true + h_pred))
    }
}

/// Computes the Purity score between true and predicted clustering assignments.
pub fn purity_score(labels_true: &[i32], labels_pred: &[i32]) -> Result<f64, PreprocessingError> {
    if labels_true.len() != labels_pred.len() || labels_true.is_empty() {
        return Err(PreprocessingError::InvalidShape);
    }

    let (contingency, _, _) = build_contingency_table(labels_true, labels_pred);

    // For each cluster in predicted, find the max overlap with any true class
    let mut max_overlaps = HashMap::new();
    for (&(_a, b), &count) in &contingency {
        let current_max = max_overlaps.entry(b).or_insert(0);
        if count > *current_max {
            *current_max = count;
        }
    }

    let mut total_max = 0;
    for &max_val in max_overlaps.values() {
        total_max += max_val;
    }

    Ok((total_max as f64) / (labels_true.len() as f64))
}
