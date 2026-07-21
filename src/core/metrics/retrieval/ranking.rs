use crate::errors::PreprocessingError;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Returns `true` when `k` is a valid cut-off (non-zero, within list length).
#[inline]
fn check_k(k: usize, list_len: usize) -> Result<(), PreprocessingError> {
    if k == 0 || list_len == 0 {
        Err(PreprocessingError::InvalidShape)
    } else {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Public metric functions
// ---------------------------------------------------------------------------

/// Computes **Recall\@K** for a single query.
///
/// Recall\@K = |relevant ∩ retrieved\@K| / |relevant|.
///
/// # Arguments
/// * `retrieved`  – ranked list of document indices, ordered most-relevant first.
/// * `relevant`   – set of ground-truth relevant document indices.
/// * `k`          – cut-off rank.
///
/// Returns `0.0` when the relevant set is empty.
pub fn recall_at_k(
    retrieved: &[usize],
    relevant: &[usize],
    k: usize,
) -> Result<f64, PreprocessingError> {
    check_k(k, retrieved.len())?;
    if relevant.is_empty() {
        return Ok(0.0);
    }
    let top_k: std::collections::HashSet<usize> = retrieved.iter().take(k).copied().collect();
    let hits = relevant.iter().filter(|r| top_k.contains(r)).count();
    Ok(hits as f64 / relevant.len() as f64)
}

/// Computes **Precision\@K** for a single query.
///
/// Precision\@K = |relevant ∩ retrieved\@K| / K.
///
/// # Arguments
/// * `retrieved` – ranked list of document indices.
/// * `relevant`  – ground-truth relevant document indices.
/// * `k`         – cut-off rank.
pub fn precision_at_k(
    retrieved: &[usize],
    relevant: &[usize],
    k: usize,
) -> Result<f64, PreprocessingError> {
    check_k(k, retrieved.len())?;
    if relevant.is_empty() {
        return Ok(0.0);
    }
    let relevant_set: std::collections::HashSet<usize> = relevant.iter().copied().collect();
    let hits = retrieved
        .iter()
        .take(k)
        .filter(|r| relevant_set.contains(r))
        .count();
    Ok(hits as f64 / k as f64)
}

/// Computes **Mean Reciprocal Rank (MRR)** for a single query.
///
/// MRR = 1 / rank_of_first_relevant_item, or 0 if no relevant item appears.
///
/// # Arguments
/// * `retrieved` – ranked list of document indices (position 0 = rank 1).
/// * `relevant`  – ground-truth relevant document indices.
pub fn mean_reciprocal_rank(
    retrieved: &[usize],
    relevant: &[usize],
) -> Result<f64, PreprocessingError> {
    if retrieved.is_empty() {
        return Err(PreprocessingError::InvalidShape);
    }
    if relevant.is_empty() {
        return Ok(0.0);
    }
    let relevant_set: std::collections::HashSet<usize> = relevant.iter().copied().collect();
    for (rank_idx, &doc) in retrieved.iter().enumerate() {
        if relevant_set.contains(&doc) {
            return Ok(1.0 / (rank_idx + 1) as f64);
        }
    }
    Ok(0.0)
}

/// Computes **nDCG\@K** (Normalised Discounted Cumulative Gain) for a single query.
///
/// Uses binary relevance (1 if relevant, 0 otherwise).
///
/// # Arguments
/// * `retrieved` – ranked list of document indices.
/// * `relevant`  – ground-truth relevant document indices.
/// * `k`         – cut-off rank.
pub fn ndcg_at_k(
    retrieved: &[usize],
    relevant: &[usize],
    k: usize,
) -> Result<f64, PreprocessingError> {
    check_k(k, retrieved.len())?;
    if relevant.is_empty() {
        return Ok(0.0);
    }
    let relevant_set: std::collections::HashSet<usize> = relevant.iter().copied().collect();

    // Compute DCG@K
    let dcg: f64 = retrieved
        .iter()
        .take(k)
        .enumerate()
        .filter(|(_, doc)| relevant_set.contains(*doc))
        .map(|(idx, _)| 1.0 / (idx as f64 + 2.0).log2()) // rank is idx+1, log2(rank+1)
        .sum();

    // Compute ideal DCG@K (best possible ordering).
    let ideal_hits = relevant.len().min(k);
    let idcg: f64 = (0..ideal_hits)
        .map(|idx| 1.0 / (idx as f64 + 2.0).log2())
        .sum();

    if idcg == 0.0 {
        return Ok(0.0);
    }
    Ok(dcg / idcg)
}

/// Computes **Coverage\@K** across multiple queries.
///
/// Coverage\@K = fraction of the total relevant item space that appears in
/// *at least one* query's top-K retrieved list.
///
/// # Arguments
/// * `all_retrieved` – one `Vec<usize>` per query, each a ranked list.
/// * `all_relevant`  – one `Vec<usize>` per query, each the relevant set.
/// * `k`             – cut-off rank.
///
/// Returns `0.0` when the union of all relevant items is empty.
pub fn coverage_at_k(
    all_retrieved: &[Vec<usize>],
    all_relevant: &[Vec<usize>],
    k: usize,
) -> Result<f64, PreprocessingError> {
    if all_retrieved.is_empty() || all_retrieved.len() != all_relevant.len() {
        return Err(PreprocessingError::InvalidShape);
    }
    if k == 0 {
        return Err(PreprocessingError::InvalidShape);
    }

    let all_relevant_items: std::collections::HashSet<usize> =
        all_relevant.iter().flatten().copied().collect();

    if all_relevant_items.is_empty() {
        return Ok(0.0);
    }

    let covered: std::collections::HashSet<usize> = all_retrieved
        .iter()
        .zip(all_relevant.iter())
        .flat_map(|(retrieved, relevant)| {
            let rel_set: std::collections::HashSet<usize> = relevant.iter().copied().collect();
            retrieved
                .iter()
                .take(k)
                .filter(move |doc| rel_set.contains(doc))
                .copied()
                .collect::<Vec<_>>()
        })
        .collect();

    Ok(covered.len() as f64 / all_relevant_items.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recall_at_k_perfect() {
        let retrieved = vec![0, 1, 2, 3, 4];
        let relevant = vec![0, 1, 2];
        let r = recall_at_k(&retrieved, &relevant, 3).unwrap();
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_recall_at_k_partial() {
        let retrieved = vec![0, 5, 6, 1, 2];
        let relevant = vec![0, 1, 2];
        let r = recall_at_k(&retrieved, &relevant, 3).unwrap();
        // Only doc 0 is in top-3; recall = 1/3
        assert!((r - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_precision_at_k() {
        let retrieved = vec![0, 1, 5, 6];
        let relevant = vec![0, 1, 2];
        let p = precision_at_k(&retrieved, &relevant, 4).unwrap();
        // 2 hits in top-4; precision = 2/4 = 0.5
        assert!((p - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_mrr_first_hit() {
        let retrieved = vec![5, 0, 1];
        let relevant = vec![0];
        // First hit at rank 2 → MRR = 0.5
        let mrr = mean_reciprocal_rank(&retrieved, &relevant).unwrap();
        assert!((mrr - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_mrr_no_hit() {
        let retrieved = vec![5, 6, 7];
        let relevant = vec![0];
        let mrr = mean_reciprocal_rank(&retrieved, &relevant).unwrap();
        assert!((mrr - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_ndcg_perfect() {
        let retrieved = vec![0, 1, 2];
        let relevant = vec![0, 1, 2];
        let score = ndcg_at_k(&retrieved, &relevant, 3).unwrap();
        assert!((score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_ndcg_partial() {
        // Only first item is relevant → DCG = 1/log2(2) = 1.0; IDCG also 1.0 → nDCG = 1.0
        let retrieved = vec![0, 5, 6];
        let relevant = vec![0];
        let score = ndcg_at_k(&retrieved, &relevant, 3).unwrap();
        assert!((score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_coverage_at_k() {
        let all_retrieved = vec![vec![0, 1], vec![2, 3]];
        let all_relevant = vec![vec![0, 1], vec![2, 3]];
        let cov = coverage_at_k(&all_retrieved, &all_relevant, 2).unwrap();
        assert!((cov - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_coverage_partial() {
        let all_retrieved = vec![vec![0, 5], vec![6, 7]];
        let all_relevant = vec![vec![0, 1], vec![2, 3]];
        // Only doc 0 is covered across all queries; 4 total relevant → coverage = 1/4
        let cov = coverage_at_k(&all_retrieved, &all_relevant, 2).unwrap();
        assert!((cov - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_invalid_k_zero() {
        let r = recall_at_k(&[0, 1], &[0], 0);
        assert!(r.is_err());
    }
}
