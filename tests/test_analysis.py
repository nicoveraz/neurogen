"""Tests for weight analysis utilities."""

import pytest
import torch

from neurogen.analysis.weight_analysis import (
    compare_weight_sets,
    condition_number,
    effective_rank,
    frobenius_norm,
    singular_value_spectrum,
    sparsity,
    spectral_norm,
    weight_statistics,
)


class TestSpectralNorm:
    def test_known_value(self):
        # Identity matrix has spectral norm 1
        w = torch.eye(10)
        assert abs(spectral_norm(w) - 1.0) < 1e-5

    def test_scaled_identity(self):
        w = 3.0 * torch.eye(10)
        assert abs(spectral_norm(w) - 3.0) < 1e-5

    def test_matches_svdvals(self):
        torch.manual_seed(42)
        w = torch.randn(64, 128)
        expected = torch.linalg.svdvals(w)[0].item()
        assert abs(spectral_norm(w) - expected) < 1e-4


class TestEffectiveRank:
    def test_full_rank(self):
        torch.manual_seed(42)
        w = torch.randn(32, 32)
        rank = effective_rank(w)
        # Full rank random matrix should have high effective rank
        assert rank > 20

    def test_rank_one(self):
        v = torch.randn(32, 1)
        w = v @ v.T
        rank = effective_rank(w)
        assert rank < 2.0

    def test_zero_matrix(self):
        w = torch.zeros(10, 10)
        rank = effective_rank(w)
        assert rank == 0.0


class TestSparsity:
    def test_all_zeros(self):
        w = torch.zeros(10, 10)
        assert sparsity(w) == 1.0

    def test_no_zeros(self):
        w = torch.ones(10, 10)
        assert sparsity(w) == 0.0

    def test_half_sparse(self):
        w = torch.zeros(10, 10)
        w[:5, :] = 1.0
        s = sparsity(w)
        assert abs(s - 0.5) < 0.01


class TestFrobeniusNorm:
    def test_known_value(self):
        w = torch.ones(3, 4)
        expected = (3.0 * 4.0) ** 0.5
        assert abs(frobenius_norm(w) - expected) < 1e-5

    def test_identity(self):
        w = torch.eye(5)
        expected = 5.0**0.5
        assert abs(frobenius_norm(w) - expected) < 1e-5


class TestConditionNumber:
    def test_identity(self):
        w = torch.eye(10)
        assert abs(condition_number(w) - 1.0) < 1e-5

    def test_singular_matrix(self):
        w = torch.zeros(10, 10)
        w[0, 0] = 1.0
        assert condition_number(w) == float("inf")


class TestSingularValueSpectrum:
    def test_length(self):
        w = torch.randn(32, 64)
        svs = singular_value_spectrum(w)
        assert len(svs) == 32  # min(m, n)

    def test_descending(self):
        w = torch.randn(32, 64)
        svs = singular_value_spectrum(w)
        for i in range(len(svs) - 1):
            assert svs[i] >= svs[i + 1] - 1e-6


class TestWeightStatistics:
    def test_has_all_keys(self):
        w = torch.randn(32, 64)
        stats = weight_statistics(w)
        expected_keys = [
            "mean", "std", "min", "max", "frobenius_norm",
            "sparsity", "spectral_norm", "effective_rank", "condition_number",
        ]
        for key in expected_keys:
            assert key in stats, f"Missing key: {key}"


class TestCompareWeightSets:
    def test_comparison(self):
        weights_a = {"layer.0": torch.randn(32, 64)}
        weights_b = {"layer.0": torch.randn(32, 64)}
        result = compare_weight_sets(weights_a, weights_b, "init_a", "init_b")
        assert "per_layer" in result
        assert "aggregate" in result
        assert "layer.0" in result["per_layer"]
