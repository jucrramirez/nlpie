import math

import pytest

from nlpie.interpret.hubness import explain_hubness, HubnessExplanation


class TestExplainHubness:
    def test_returns_hubness_explanation(self):
        counts = [5, 5, 5, 5, 5]
        result = explain_hubness(counts, skewness=0.1, k=5, top_n=2)
        assert isinstance(result, HubnessExplanation)

    def test_negligible_hubness(self):
        counts = [5, 5, 5, 5, 5]
        result = explain_hubness(counts, skewness=0.1, k=5)
        assert result.severity == "none"
        assert "Negligible" in result.summary

    def test_low_hubness(self):
        counts = [5, 5, 6, 4, 5]
        result = explain_hubness(counts, skewness=0.3, k=5)
        assert result.severity == "low"
        assert "Low hubness" in result.summary

    def test_moderate_hubness(self):
        counts = [5, 10, 5, 2, 5]
        result = explain_hubness(counts, skewness=0.7, k=5)
        assert result.severity == "moderate"
        assert "Moderate" in result.summary

    def test_severe_hubness(self):
        counts = [5, 20, 5, 0, 5]
        result = explain_hubness(counts, skewness=1.5, k=5)
        assert result.severity == "severe"
        assert "Severe" in result.summary

    def test_severe_with_hubs_counted(self):
        counts = [5, 15, 15, 0, 5]
        result = explain_hubness(counts, skewness=1.5, k=5)
        assert "point(s) appear in more than twice" in result.summary

    def test_top_hubs_ordered_by_count(self):
        counts = [10, 2, 5, 1, 3]
        result = explain_hubness(counts, skewness=0.8, k=5, top_n=3)
        assert len(result.top_hubs) == 3
        assert result.top_hubs[0].count >= result.top_hubs[1].count >= result.top_hubs[2].count

    def test_top_hubs_only_positive(self):
        counts = [0, 0, 5, 0, 0]
        result = explain_hubness(counts, skewness=0.5, k=5, top_n=5)
        assert len(result.top_hubs) == 1
        assert result.top_hubs[0].index == 2

    def test_share_of_total(self):
        counts = [10, 2, 2, 2, 2]
        result = explain_hubness(counts, skewness=0.8, k=5, top_n=1)
        total_slots = 5 * 5
        assert math.isclose(result.top_hubs[0].share_of_total, 10 / total_slots)

    def test_mean_count_equals_k(self):
        counts = [5, 5, 5, 5, 5]
        result = explain_hubness(counts, skewness=0.0, k=7)
        assert result.mean_count == 7.0

    def test_max_count(self):
        counts = [3, 10, 4, 2, 1]
        result = explain_hubness(counts, skewness=0.6, k=5)
        assert result.max_count == 10

    def test_custom_thresholds(self):
        counts = [5, 5, 5, 5, 5, 5]
        result = explain_hubness(counts, skewness=0.4, k=5,
                                 severe_threshold=0.3, moderate_threshold=0.2)
        assert result.severity == "severe"

    def test_interpretation_present(self):
        counts = [5, 5, 5, 5, 5]
        result = explain_hubness(counts, skewness=0.1, k=5)
        assert isinstance(result.interpretation, str)
        assert len(result.interpretation) > 0

    def test_skewness_zero_variance(self):
        counts = [5, 5, 5, 5, 5]
        result = explain_hubness(counts, skewness=0.0, k=5)
        assert result.severity == "none"

    def test_single_element(self):
        counts = [0]
        result = explain_hubness(counts, skewness=0.0, k=1, top_n=5)
        assert len(result.top_hubs) == 0
        assert result.max_count == 0

    def test_empty_counts_raises(self):
        with pytest.raises(ValueError, match="at least one element"):
            explain_hubness([], skewness=0.0, k=5)

    def test_smoke_with_rust_hubness(self):
        from nlpie import compute_hubness
        embeddings = [[float(i + j) for j in range(4)] for i in range(30)]
        counts, skewness = compute_hubness(embeddings, k=5)
        result = explain_hubness(counts, skewness, k=5)
        assert result.severity in ("severe", "moderate", "low", "none")
        assert len(result.top_hubs) <= 5
