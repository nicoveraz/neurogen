"""Tests for research/ (engine, registry, report)."""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from research.engine import load_experiment_config, load_raw_yaml
from research.registry import ExperimentRegistry
from research.report import generate_phase_report


@pytest.fixture
def phase1_yaml_path():
    """Path to the phase1 baselines YAML config."""
    path = Path(__file__).resolve().parent.parent / "research" / "experiments" / "phase1_baselines.yaml"
    if not path.exists():
        pytest.skip("phase1_baselines.yaml not found")
    return str(path)


@pytest.fixture
def temp_yaml():
    """Create a temporary YAML experiment file."""
    config = {
        "name": "Test Experiment",
        "hypothesis": "Testing YAML loading",
        "model": {
            "block_size": 32,
            "n_layer": 2,
            "n_head": 2,
            "n_embd": 64,
            "dropout": 0.0,
        },
        "dataset": "shakespeare_char",
        "inits": ["xavier_normal", "kaiming_normal"],
        "training": {
            "max_steps": 100,
            "eval_interval": 50,
            "lr": 1e-3,
            "batch_size": 4,
        },
        "metrics": ["train_loss", "val_loss"],
        "seeds": [42, 123],
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        yaml.dump(config, f)
        return f.name


@pytest.fixture
def temp_registry():
    """Create a temporary registry file path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield str(Path(tmpdir) / "test_registry.json")


class TestExperimentYAMLLoading:
    """Tests for YAML experiment config loading."""

    def test_experiment_yaml_loading(self, phase1_yaml_path):
        """Loads phase1 yaml correctly into ExperimentConfig."""
        config = load_experiment_config(phase1_yaml_path)
        assert config.name == "Phase 1: Baseline Initialization Sweep", (
            f"Expected phase 1 name, got '{config.name}'"
        )
        assert config.model_config.n_layer == 6, "n_layer should be 6"
        assert config.model_config.n_head == 6, "n_head should be 6"
        assert config.model_config.n_embd == 384, "n_embd should be 384"
        assert config.train_config.max_steps == 5000, "max_steps should be 5000"
        # YAML may load scientific notation as string; check type-flexibly
        lr_val = config.train_config.lr
        if isinstance(lr_val, str):
            lr_val = float(lr_val)
        assert abs(lr_val - 3e-4) < 1e-8, f"lr should be 3e-4, got {lr_val}"

    def test_experiment_yaml_loading_custom(self, temp_yaml):
        """Loads custom temp yaml correctly."""
        config = load_experiment_config(temp_yaml)
        assert config.name == "Test Experiment", (
            f"Expected 'Test Experiment', got '{config.name}'"
        )
        assert config.model_config.block_size == 32, "block_size should be 32"
        assert config.init_method == "xavier_normal", (
            "First init method should be xavier_normal"
        )

    def test_load_raw_yaml(self, phase1_yaml_path):
        """load_raw_yaml returns dict with expected keys."""
        raw = load_raw_yaml(phase1_yaml_path)
        assert isinstance(raw, dict), "load_raw_yaml should return a dict"
        assert "name" in raw, "Raw YAML should have 'name'"
        assert "inits" in raw, "Raw YAML should have 'inits'"
        assert "seeds" in raw, "Raw YAML should have 'seeds'"

    def test_experiment_yaml_missing_file(self):
        """Missing YAML file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_experiment_config("/nonexistent/path/config.yaml")


class TestExperimentRegistry:
    """Tests for ExperimentRegistry."""

    def test_experiment_registry(self, temp_registry):
        """Register, update status, get status lifecycle."""
        registry = ExperimentRegistry(temp_registry)
        assert len(registry) == 0, "Fresh registry should be empty"

        # Register
        registry.register("exp_001", {"name": "test", "method": "xavier"})
        assert "exp_001" in registry, "Should contain registered experiment"
        assert len(registry) == 1, "Should have 1 experiment"

        # Get status
        status = registry.get_status("exp_001")
        assert status == "pending", f"Initial status should be 'pending', got '{status}'"

        # Update status
        registry.update_status("exp_001", "running")
        assert registry.get_status("exp_001") == "running", (
            "Status should be 'running' after update"
        )

        # Complete with results
        registry.update_status(
            "exp_001", "complete",
            results={"val_loss": 2.5, "train_loss": 2.0},
        )
        assert registry.get_status("exp_001") == "complete", (
            "Status should be 'complete'"
        )
        results = registry.get_results("exp_001")
        assert results["val_loss"] == 2.5, "Should store results correctly"

    def test_registry_persistence(self, temp_registry):
        """Registry data persists across instances."""
        reg1 = ExperimentRegistry(temp_registry)
        reg1.register("exp_persist", {"name": "persist_test"})

        # Create new instance pointing to same file
        reg2 = ExperimentRegistry(temp_registry)
        assert "exp_persist" in reg2, "Data should persist to disk"
        assert reg2.get_status("exp_persist") == "pending", (
            "Status should be persisted"
        )

    def test_registry_duplicate_raises(self, temp_registry):
        """Registering duplicate experiment_id should raise ValueError."""
        registry = ExperimentRegistry(temp_registry)
        registry.register("dup_test", {"name": "test"})
        with pytest.raises(ValueError, match="already registered"):
            registry.register("dup_test", {"name": "test2"})

    def test_registry_invalid_status_raises(self, temp_registry):
        """Invalid status should raise ValueError."""
        registry = ExperimentRegistry(temp_registry)
        with pytest.raises(ValueError, match="Invalid status"):
            registry.register("bad_status", {"name": "test"}, status="invalid")

    def test_registry_list_experiments(self, temp_registry):
        """list_experiments returns correct results with optional filter."""
        registry = ExperimentRegistry(temp_registry)
        registry.register("exp_a", {"name": "a"})
        registry.register("exp_b", {"name": "b"})
        registry.update_status("exp_a", "running")

        all_exps = registry.list_experiments()
        assert len(all_exps) == 2, "Should list all experiments"

        running_exps = registry.list_experiments(status="running")
        assert len(running_exps) == 1, "Should list only running experiments"
        assert running_exps[0]["id"] == "exp_a", "Running experiment should be exp_a"


class TestReportGeneration:
    """Tests for generate_phase_report."""

    def test_report_generation(self):
        """generate_phase_report produces markdown file."""
        phase_results = {
            "xavier_normal": {
                "final_val_loss_mean": 2.5,
                "final_val_loss_std": 0.1,
                "best_val_loss_mean": 2.3,
                "best_val_loss_std": 0.05,
                "final_train_loss_mean": 2.0,
                "total_time_mean_s": 120.0,
                "n_seeds": 3,
            },
            "kaiming_normal": {
                "final_val_loss_mean": 2.6,
                "final_val_loss_std": 0.15,
                "best_val_loss_mean": 2.4,
                "best_val_loss_std": 0.08,
                "final_train_loss_mean": 2.1,
                "total_time_mean_s": 115.0,
                "n_seeds": 3,
            },
        }
        config = {
            "name": "Test Phase Report",
            "hypothesis": "Testing report generation",
            "model": {"n_layer": 2, "n_head": 2, "n_embd": 64, "block_size": 32},
            "training": {"max_steps": 100, "lr": 1e-3, "batch_size": 4},
            "inits": ["xavier_normal", "kaiming_normal"],
            "seeds": [42, 123],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = str(Path(tmpdir) / "test_report.md")
            generate_phase_report(
                phase_results=phase_results,
                output_path=report_path,
                config=config,
            )
            report_file = Path(report_path)
            assert report_file.exists(), "Report file should be created"
            content = report_file.read_text()
            assert "Test Phase Report" in content, "Report should contain experiment name"
            assert "xavier_normal" in content, "Report should contain init method"
            assert "Results" in content, "Report should have Results section"
            assert "Analysis" in content, "Report should have Analysis section"
            assert len(content) > 100, "Report should have substantial content"
