"""Tests for neurogen/analysis/weight_analysis.py."""

import math

import pytest
import torch

from neurogen.analysis.weight_analysis import (
    analyze_model_weights,
    analyze_weight_dict,
    compare_weight_sets,
    condition_number,
    effective_rank,
    frobenius_norm,
    spectral_norm,
    sparsity,
    weight_statistics,
)
from neurogen.config import GPTConfig
from neurogen.model.gpt import GPT


class TestSpectralNorm:
    """Tests for spectral_norm."""

    def test_spectral_norm_known_matrix(self):
        """Known matrix: identity has spectral norm 1.0."""
        identity = torch.eye(10, dtype=torch.float32)
        sn = spectral_norm(identity)
        assert abs(sn - 1.0) < 1e-4, (
            f"Identity spectral norm should be 1.0, got {sn}"
        )

    def test_spectral_norm_scaled_identity(self):
        """Scaled identity: spectral norm equals the scaling factor."""
        scale = 3.0
        mat = scale * torch.eye(10, dtype=torch.float32)
        sn = spectral_norm(mat)
        assert abs(sn - scale) < 1e-4, (
            f"Scaled identity spectral norm should be {scale}, got {sn}"
        )

    def test_spectral_norm_rank_1(self):
        """Rank-1 matrix: spectral norm equals product of vector norms."""
        u = torch.tensor([1.0, 0.0, 0.0]).unsqueeze(1)
        v = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0)
        mat = u @ v  # Rank-1 matrix
        sn = spectral_norm(mat)
        assert abs(sn - 1.0) < 1e-4, (
            f"Rank-1 matrix spectral norm should be 1.0, got {sn}"
        )


class TestEffectiveRank:
    """Tests for effective_rank."""

    def test_effective_rank_full(self):
        """Full-rank matrix should have high effective rank."""
        torch.manual_seed(42)
        mat = torch.randn(20, 20, dtype=torch.float32)
        er = effective_rank(mat)
        assert er > 10, (
            f"Full-rank 20x20 matrix should have effective rank > 10, got {er:.2f}"
        )

    def test_effective_rank_one(self):
        """Rank-1 matrix should have effective rank near 1."""
        u = torch.randn(20, 1, dtype=torch.float32)
        v = torch.randn(1, 20, dtype=torch.float32)
        mat = u @ v
        er = effective_rank(mat)
        assert er < 2.0, (
            f"Rank-1 matrix should have effective rank near 1, got {er:.2f}"
        )

    def test_effective_rank_zeros(self):
        """All-zero matrix should have effective rank 0."""
        mat = torch.zeros(10, 10, dtype=torch.float32)
        er = effective_rank(mat)
        assert er == 0.0, f"Zero matrix effective rank should be 0.0, got {er}"


class TestSparsity:
    """Tests for sparsity."""

    def test_sparsity_all_zeros(self):
        """All zeros -> sparsity 1.0."""
        mat = torch.zeros(10, 10, dtype=torch.float32)
        sp = sparsity(mat)
        assert sp == 1.0, f"All-zero matrix sparsity should be 1.0, got {sp}"

    def test_sparsity_no_zeros(self):
        """Dense matrix with no small values -> sparsity ~0.0."""
        mat = torch.ones(10, 10, dtype=torch.float32)
        sp = sparsity(mat)
        assert sp == 0.0, f"All-ones matrix sparsity should be 0.0, got {sp}"

    def test_sparsity_mixed(self):
        """Matrix with 50% zeros -> sparsity ~0.5."""
        mat = torch.zeros(10, 10, dtype=torch.float32)
        mat[:5, :] = 1.0
        sp = sparsity(mat)
        assert abs(sp - 0.5) < 0.01, (
            f"Half-zero matrix sparsity should be ~0.5, got {sp}"
        )


class TestFrobeniusNorm:
    """Tests for frobenius_norm."""

    def test_frobenius_norm_known(self):
        """Known matrix: 2x2 matrix of ones has Frobenius norm 2."""
        mat = torch.ones(2, 2, dtype=torch.float32)
        fn = frobenius_norm(mat)
        assert abs(fn - 2.0) < 1e-4, (
            f"2x2 ones Frobenius norm should be 2.0, got {fn}"
        )

    def test_frobenius_norm_identity(self):
        """Identity: Frobenius norm = sqrt(n)."""
        n = 10
        mat = torch.eye(n, dtype=torch.float32)
        fn = frobenius_norm(mat)
        expected = math.sqrt(n)
        assert abs(fn - expected) < 1e-4, (
            f"Identity({n}) Frobenius norm should be {expected:.4f}, got {fn}"
        )


class TestConditionNumber:
    """Tests for condition_number."""

    def test_condition_number_identity(self):
        """Identity matrix has condition number 1.0."""
        mat = torch.eye(10, dtype=torch.float32)
        cn = condition_number(mat)
        assert abs(cn - 1.0) < 1e-4, (
            f"Identity condition number should be 1.0, got {cn}"
        )

    def test_condition_number_singular(self):
        """Singular matrix has infinite condition number."""
        mat = torch.zeros(10, 10, dtype=torch.float32)
        mat[0, 0] = 1.0
        cn = condition_number(mat)
        assert cn == float("inf"), (
            f"Near-singular matrix condition number should be inf, got {cn}"
        )


class TestWeightStatistics:
    """Tests for weight_statistics."""

    def test_weight_statistics_keys(self):
        """Returns all expected keys."""
        mat = torch.randn(10, 10, dtype=torch.float32)
        stats = weight_statistics(mat)
        expected_keys = [
            "mean", "std", "min", "max", "frobenius", "sparsity",
            "spectral_norm", "effective_rank",
        ]
        for key in expected_keys:
            assert key in stats, f"weight_statistics should contain '{key}'"

    def test_weight_statistics_values(self):
        """Values should be finite floats."""
        mat = torch.randn(10, 10, dtype=torch.float32)
        stats = weight_statistics(mat)
        for key, value in stats.items():
            assert isinstance(value, float), f"'{key}' should be a float"
            assert math.isfinite(value), f"'{key}' should be finite, got {value}"


class TestCompareWeightSets:
    """Tests for compare_weight_sets."""

    def test_compare_weight_sets_identical(self):
        """Comparing identical weight sets: distance 0, cosine 1."""
        weights = {
            "layer1": torch.randn(10, 10, dtype=torch.float32),
            "layer2": torch.randn(20, 20, dtype=torch.float32),
        }
        result = compare_weight_sets(weights, weights)
        assert "aggregate" in result, "Should have 'aggregate' entry"
        assert "layer1" in result, "Should have 'layer1' entry"
        assert "layer2" in result, "Should have 'layer2' entry"
        # L2 distance should be 0 for identical weights
        assert result["layer1"]["l2_distance"] < 1e-6, (
            "L2 distance should be 0 for identical weights"
        )
        # Cosine similarity should be 1 for identical weights
        assert abs(result["layer1"]["cosine_similarity"] - 1.0) < 1e-4, (
            "Cosine similarity should be 1.0 for identical weights"
        )

    def test_compare_weight_sets_different(self):
        """Comparing different weight sets: distance > 0."""
        w1 = {"layer1": torch.ones(10, 10, dtype=torch.float32)}
        w2 = {"layer1": torch.zeros(10, 10, dtype=torch.float32)}
        result = compare_weight_sets(w1, w2)
        assert result["layer1"]["l2_distance"] > 0, (
            "L2 distance should be > 0 for different weights"
        )


class TestAnalyzeModelWeights:
    """Tests for analyze_model_weights."""

    def test_analyze_model_weights(self, tiny_model):
        """Works on a model, returns per-layer and aggregate analysis."""
        result = analyze_model_weights(tiny_model)
        assert "aggregate" in result, "Should contain 'aggregate' summary"
        model_weights = tiny_model.get_weight_tensors()
        for name in model_weights:
            assert name in result, f"Should contain analysis for '{name}'"
        agg = result["aggregate"]
        assert "n_tensors" in agg, "Aggregate should contain 'n_tensors'"
        assert agg["n_tensors"] == float(len(model_weights)), (
            f"n_tensors should be {len(model_weights)}"
        )
