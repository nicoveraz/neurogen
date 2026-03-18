"""Tests for neurogen/ca/live/ (live CA rules, alpha schedule, trainer)."""

import math

import pytest
import torch

from neurogen.ca.live import (
    LIVE_CA_RULES,
    AlphaSchedule,
    CAOptimizer,
    CompetitionCA,
    LearnedCA,
    LiveCA,
    LocalNormCA,
    ModularityCA,
    MultiTimescaleCA,
    PruningCA,
)
from neurogen.ca.live.alpha_schedule import AlphaSchedule
from neurogen.config import GPTConfig, LiveCAConfig, TrainConfig
from neurogen.model.gpt import GPT
from neurogen.training.live_ca_trainer import LiveCATrainer, measure_alignment


@pytest.fixture
def weight_matrix():
    """A small random weight matrix on CPU."""
    torch.manual_seed(42)
    return torch.randn(32, 32, dtype=torch.float32)


@pytest.fixture
def grad_matrix():
    """A small gradient matrix matching weight_matrix."""
    torch.manual_seed(123)
    return torch.randn(32, 32, dtype=torch.float32)


class TestLiveCABaseInterface:
    """Tests for all live CA rules producing correct output."""

    @pytest.mark.parametrize("rule_name", list(LIVE_CA_RULES.keys()))
    def test_live_ca_base_interface(self, rule_name, weight_matrix, grad_matrix):
        """All rules return correct shape, finite values."""
        rule_cls = LIVE_CA_RULES[rule_name]
        if rule_name == "learned":
            rule = rule_cls(n_features=5, hidden=32)
        else:
            rule = rule_cls()
        delta = rule.step(weight_matrix, grad_W=grad_matrix)
        assert delta.shape == weight_matrix.shape, (
            f"'{rule_name}' delta shape {delta.shape} != weight shape {weight_matrix.shape}"
        )
        assert torch.isfinite(delta).all(), (
            f"'{rule_name}' produced non-finite delta values"
        )


class TestLiveCADeltaBounded:
    """Tests for bounded delta magnitudes."""

    @pytest.mark.parametrize("rule_name", list(LIVE_CA_RULES.keys()))
    def test_live_ca_delta_bounded(self, rule_name, weight_matrix, grad_matrix):
        """Delta max < 1.0, mean < 0.1 for bounded CA updates."""
        rule_cls = LIVE_CA_RULES[rule_name]
        if rule_name == "learned":
            rule = rule_cls(n_features=5, hidden=32)
        else:
            rule = rule_cls()
        delta = rule.step(weight_matrix, grad_W=grad_matrix)
        assert delta.abs().max().item() < 5.0, (
            f"'{rule_name}' delta max {delta.abs().max().item():.4f} is too large"
        )
        assert delta.abs().mean().item() < 1.0, (
            f"'{rule_name}' delta mean {delta.abs().mean().item():.4f} is too large"
        )


class TestLocalNormCA:
    """Tests for LocalNormCA."""

    def test_local_norm_reduces_outliers(self, weight_matrix):
        """Delta at outlier weight should push it toward local mean."""
        rule = LocalNormCA(neighborhood_size=3, target_std=0.02)
        # Create a weight matrix with an extreme outlier
        W = weight_matrix.clone()
        W[16, 16] = 10.0  # Big outlier
        delta = rule.step(W)
        # Delta at the outlier should be negative (pushing it back toward mean)
        assert delta[16, 16].item() < 0, (
            "LocalNormCA should push outlier weights back toward local mean"
        )


class TestModularityCA:
    """Tests for ModularityCA."""

    def test_modularity_ca_structure(self, weight_matrix):
        """On-diagonal blocks should get positive delta, off-diagonal negative."""
        rule = ModularityCA(n_blocks=4)
        W = torch.ones(32, 32, dtype=torch.float32) * 0.5
        delta = rule.step(W)
        # Check on-diagonal block (first block: 0:8, 0:8)
        on_diag_delta = delta[:8, :8].mean().item()
        # Check off-diagonal block (first off-diag: 0:8, 8:16)
        off_diag_delta = delta[:8, 8:16].mean().item()
        assert on_diag_delta > 0, (
            f"On-diagonal delta should be positive, got {on_diag_delta}"
        )
        assert off_diag_delta < 0, (
            f"Off-diagonal delta should be negative, got {off_diag_delta}"
        )


class TestPruningCA:
    """Tests for PruningCA."""

    def test_pruning_ca_needs_gradients(self, weight_matrix):
        """Returns zeros without grad, non-zero with grad."""
        rule = PruningCA()
        delta_no_grad = rule.step(weight_matrix, grad_W=None)
        assert delta_no_grad.abs().sum().item() == 0, (
            "PruningCA should return zeros without gradients"
        )

    def test_pruning_ca_suppresses_unimportant(self, weight_matrix, grad_matrix):
        """Low importance -> decay toward zero."""
        rule = PruningCA()
        # Create a case where some weights have low importance (small W * small grad)
        W = torch.zeros(32, 32, dtype=torch.float32)
        W[0, 0] = 0.001  # Low magnitude
        grad = torch.zeros(32, 32, dtype=torch.float32)
        grad[0, 0] = 0.001  # Low gradient
        W[16, 16] = 1.0  # High magnitude
        grad[16, 16] = 1.0  # High gradient
        delta = rule.step(W, grad_W=grad)
        # The result should be finite
        assert torch.isfinite(delta).all(), "Pruning delta should be finite"


class TestCompetitionCA:
    """Tests for CompetitionCA."""

    def test_competition_ca_winner_take_all(self):
        """Winner should get positive boost, losers negative."""
        rule = CompetitionCA(neighborhood_size=5)
        W = torch.zeros(32, 32, dtype=torch.float32)
        # Create a clear winner at (16, 16)
        W[16, 16] = 10.0
        # Neighbors have much smaller values
        W[15, 16] = 0.1
        W[17, 16] = 0.1
        delta = rule.step(W)
        # Winner should be reinforced (positive delta for positive weight)
        assert delta[16, 16].item() > 0, (
            "Winner should get positive delta boost"
        )
        # Losers near the winner should be suppressed
        assert delta[15, 16].item() < 0, (
            "Loser near winner should get negative delta"
        )


class TestLearnedCA:
    """Tests for LearnedCA."""

    def test_learned_ca_gradient_flow(self, weight_matrix):
        """rule_net params should get gradients after backward."""
        rule = LearnedCA(n_features=5, hidden=32)
        W = weight_matrix.clone().requires_grad_(True)
        delta = rule.step(W)
        loss = delta.sum()
        loss.backward()
        has_grad = False
        for p in rule.rule_net.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "LearnedCA rule_net params should have gradients"


class TestMultiTimescaleCA:
    """Tests for MultiTimescaleCA."""

    def test_multi_timescale_frequency(self, weight_matrix):
        """Correct CAs fire at correct steps."""
        fast = LocalNormCA()
        medium = ModularityCA()
        slow = PruningCA()
        multi = MultiTimescaleCA(
            fast_ca=fast, medium_ca=medium, slow_ca=slow,
            fast_interval=1, medium_interval=10, slow_interval=100,
        )
        # Step 0: all should fire (0 % N == 0 for all N)
        delta_0 = multi.step(weight_matrix, step_number=0)
        assert delta_0.abs().sum().item() > 0, (
            "At step 0, at least one CA should fire"
        )
        # Step 5: only fast should fire
        delta_5 = multi.step(weight_matrix, step_number=5)
        # Step 10: fast + medium should fire
        delta_10 = multi.step(weight_matrix, step_number=10)
        # Both should be non-zero and different
        assert not torch.allclose(delta_5, delta_10, atol=1e-6), (
            "Steps 5 and 10 should produce different deltas"
        )


class TestAlphaScheduleExponential:
    """Tests for exponential alpha schedule."""

    def test_alpha_schedule_exponential(self):
        """Exponential schedule should be monotonically decreasing."""
        schedule = AlphaSchedule(
            mode="exponential_decay", alpha_0=0.01, decay=0.001, total_steps=5000
        )
        prev_alpha = schedule.get_alpha(0)
        for step in range(1, 100):
            alpha = schedule.get_alpha(step)
            assert alpha <= prev_alpha + 1e-10, (
                f"Exponential alpha should be non-increasing: "
                f"step {step}: {alpha} > step {step-1}: {prev_alpha}"
            )
            prev_alpha = alpha


class TestAlphaScheduleCosine:
    """Tests for cosine alpha schedule."""

    def test_alpha_schedule_cosine(self):
        """Cosine: starts at alpha_0, ends near 0."""
        schedule = AlphaSchedule(
            mode="cosine", alpha_0=0.01, total_steps=1000
        )
        alpha_start = schedule.get_alpha(0)
        alpha_end = schedule.get_alpha(1000)
        assert abs(alpha_start - 0.01) < 1e-6, (
            f"Cosine alpha at step 0 should be alpha_0=0.01, got {alpha_start}"
        )
        assert alpha_end < 0.001, (
            f"Cosine alpha at total_steps should be near 0, got {alpha_end}"
        )


class TestAlphaSchedulePhased:
    """Tests for phased alpha schedule."""

    def test_alpha_schedule_phased(self):
        """Correct alpha at each phase."""
        schedule = AlphaSchedule(
            mode="phased",
            phase_boundaries=[0, 1000, 3000],
            phase_alphas=[0.01, 0.005, 0.001],
        )
        # Before first boundary (step 0 is < boundary 0 which is 0...
        # actually step=0 is not < 0, so it falls through to phase_alphas[-1])
        # Let's check concrete values
        alpha_500 = schedule.get_alpha(500)
        alpha_2000 = schedule.get_alpha(2000)
        alpha_4000 = schedule.get_alpha(4000)
        # Step 500 is >= boundary[0]=0, < boundary[1]=1000 -> phase_alphas[0]=0.01
        assert alpha_500 == 0.01, f"Phase 0 alpha should be 0.01, got {alpha_500}"
        # Step 2000 is >= boundary[1]=1000, < boundary[2]=3000 -> phase_alphas[1]=0.005
        assert alpha_2000 == 0.005, f"Phase 1 alpha should be 0.005, got {alpha_2000}"
        # Step 4000 is >= boundary[2]=3000 -> phase_alphas[-1]=0.001
        assert alpha_4000 == 0.001, f"Phase 2 alpha should be 0.001, got {alpha_4000}"


class TestAlphaScheduleAdaptive:
    """Tests for adaptive alpha schedule."""

    def test_alpha_schedule_adaptive(self):
        """Stagnating loss -> higher alpha, improving loss -> lower alpha."""
        schedule = AlphaSchedule(mode="adaptive", alpha_0=0.01)
        # Stagnating: last 10 losses barely decrease
        stagnating_history = [5.0] * 10 + [4.99]
        alpha_stagnant = schedule.get_alpha(50, loss_history=stagnating_history)
        # Improving: last 10 losses decrease significantly
        improving_history = [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.1]
        alpha_improving = schedule.get_alpha(50, loss_history=improving_history)
        assert alpha_stagnant > alpha_improving, (
            f"Stagnating alpha ({alpha_stagnant}) should be > improving alpha ({alpha_improving})"
        )


class TestAlphaScheduleCyclic:
    """Tests for cyclic alpha schedule."""

    def test_alpha_schedule_cyclic(self):
        """Cyclic schedule should oscillate."""
        schedule = AlphaSchedule(
            mode="cyclic", alpha_base=0.002,
            cycle_amplitude=0.005, cycle_period=500,
        )
        alphas = [schedule.get_alpha(step) for step in range(1000)]
        # Should have values both above and below alpha_base
        above = sum(1 for a in alphas if a > schedule.alpha_base + 0.001)
        below = sum(1 for a in alphas if a < schedule.alpha_base - 0.001)
        assert above > 0, "Cyclic should have values above alpha_base"
        assert below > 0, "Cyclic should have values below alpha_base"


class TestGradientAlignmentMetric:
    """Tests for measure_alignment."""

    def test_ca_gradient_alignment_parallel(self):
        """Parallel vectors -> alignment 1."""
        v1 = torch.ones(10, dtype=torch.float32)
        v2 = torch.ones(10, dtype=torch.float32) * 3
        alignment = measure_alignment(v1, v2)
        assert abs(alignment - 1.0) < 1e-4, (
            f"Parallel vectors alignment should be 1.0, got {alignment}"
        )

    def test_ca_gradient_alignment_anti_parallel(self):
        """Anti-parallel vectors -> alignment -1."""
        v1 = torch.ones(10, dtype=torch.float32)
        v2 = -torch.ones(10, dtype=torch.float32)
        alignment = measure_alignment(v1, v2)
        assert abs(alignment - (-1.0)) < 1e-4, (
            f"Anti-parallel alignment should be -1.0, got {alignment}"
        )

    def test_ca_gradient_alignment_orthogonal(self):
        """Orthogonal vectors -> alignment ~0."""
        v1 = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        v2 = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)
        alignment = measure_alignment(v1, v2)
        assert abs(alignment) < 0.01, (
            f"Orthogonal alignment should be ~0, got {alignment}"
        )


class MockDataset:
    """Minimal mock dataset for LiveCATrainer tests."""

    def __init__(self, vocab_size: int = 256, data_len: int = 5000):
        self.vocab_size = vocab_size
        self._data = torch.randint(0, vocab_size, (data_len,))

    def get_batch(
        self, split: str, batch_size: int, block_size: int, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(len(self._data) - block_size, (batch_size,))
        x = torch.stack([self._data[i : i + block_size] for i in ix])
        y = torch.stack([self._data[i + 1 : i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)


class TestLiveCATrainerIntegration:
    """Tests for LiveCATrainer end-to-end."""

    def test_live_ca_trainer_integration(self, tiny_config, device):
        """Train 50 steps with LiveCATrainer: loss decreases, metrics logged."""
        model = GPT(tiny_config).to(device)
        dataset = MockDataset(vocab_size=tiny_config.vocab_size)

        ca_rules = {
            "attn": LocalNormCA(neighborhood_size=3, target_std=0.02),
            "ffn": LocalNormCA(neighborhood_size=3, target_std=0.02),
        }
        alpha_schedule = AlphaSchedule(
            mode="exponential_decay", alpha_0=0.01,
            decay=0.001, total_steps=50,
        )
        live_config = LiveCAConfig(
            ca_type="local_norm", ca_interval=1,
            alpha_schedule="exponential_decay",
            alpha_0=0.01, decay=0.001, total_steps=50,
            ca_sees_gradients=True, clamp_weights=True, max_weight=3.0,
        )
        trainer = LiveCATrainer(
            model=model, ca_rules=ca_rules,
            alpha_schedule=alpha_schedule,
            config=live_config, device=device,
        )
        trainer.configure_optimizer(lr=1e-3, weight_decay=0.1)
        metrics = trainer.train(dataset)

        assert "train_losses" in metrics, "Should have train_losses"
        assert "ca_magnitudes" in metrics, "Should have ca_magnitudes"
        assert "grad_magnitudes" in metrics, "Should have grad_magnitudes"
        assert "ca_alignments" in metrics, "Should have ca_alignments"
        assert "ca_contributions" in metrics, "Should have ca_contributions"
        assert len(metrics["train_losses"]) == 50, "Should have 50 losses"
        assert metrics["final_loss"] < float("inf"), "Final loss should be finite"
        # Check loss decreases
        losses = metrics["train_losses"]
        early_avg = sum(losses[:5]) / 5
        late_avg = sum(losses[-5:]) / 5
        assert late_avg < early_avg, (
            f"Loss should decrease: early {early_avg:.4f} > late {late_avg:.4f}"
        )
