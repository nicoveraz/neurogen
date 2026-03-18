"""Tests for the auto-research engine."""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from research.engine import load_experiment_config, run_single_experiment
from research.registry import ExperimentRegistry, ExperimentStatus
from research.report import generate_phase_report
from neurogen.config import GPTConfig, TrainConfig, get_device
from neurogen.data.shakespeare import ShakespeareDataset


@pytest.fixture(scope="module")
def dataset():
    return ShakespeareDataset()


@pytest.fixture
def tiny_model_config(dataset):
    return GPTConfig(
        block_size=32, vocab_size=dataset.vocab_size,
        n_layer=2, n_head=2, n_embd=64, dropout=0.0,
    )


@pytest.fixture
def tiny_train_cfg(device):
    return TrainConfig(
        max_steps=10, eval_interval=10, eval_steps=2,
        batch_size=4, lr=1e-3, device=device, log_interval=10,
    )


class TestExperimentYAML:
    def test_load_valid_yaml(self):
        config = load_experiment_config("research/experiments/phase1_baselines.yaml")
        assert "name" in config
        assert "inits" in config
        assert "training" in config
        assert "model" in config

    def test_missing_field_raises(self, tmp_path):
        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text(yaml.dump({"name": "test"}))
        with pytest.raises(ValueError, match="Missing"):
            load_experiment_config(yaml_file)


class TestExperimentRunner:
    def test_single_experiment(self, dataset, tiny_model_config, tiny_train_cfg):
        metrics = run_single_experiment(
            "xavier_normal", tiny_model_config, tiny_train_cfg, dataset, seed=42
        )
        assert "val_loss" in metrics
        assert "total_train_time_s" in metrics
        assert metrics["init_method"] == "xavier_normal"
        assert metrics["seed"] == 42


class TestExperimentRegistry:
    def test_register_and_transitions(self, tmp_path):
        reg = ExperimentRegistry(tmp_path / "reg.json")
        reg.register("exp1", "Test Experiment")
        assert reg.get_status("exp1") == ExperimentStatus.PENDING

        reg.mark_running("exp1")
        assert reg.get_status("exp1") == ExperimentStatus.RUNNING

        reg.mark_complete("exp1", {"val_loss": 2.5})
        assert reg.get_status("exp1") == ExperimentStatus.COMPLETE

    def test_persistence(self, tmp_path):
        reg1 = ExperimentRegistry(tmp_path / "reg.json")
        reg1.register("exp1", "Test")
        reg1.mark_complete("exp1")

        reg2 = ExperimentRegistry(tmp_path / "reg.json")
        assert reg2.get_status("exp1") == ExperimentStatus.COMPLETE

    def test_failed_status(self, tmp_path):
        reg = ExperimentRegistry(tmp_path / "reg.json")
        reg.register("exp1", "Test")
        reg.mark_failed("exp1", "Something broke")
        record = reg.get_record("exp1")
        assert record.status == ExperimentStatus.FAILED
        assert record.error_message == "Something broke"


class TestReportGeneration:
    def test_generate_report(self, tmp_path):
        results = {
            "xavier_normal": [
                {"final_val_loss": 2.5, "best_val_loss": 2.3, "total_train_time_s": 10, "seed": 42},
                {"final_val_loss": 2.6, "best_val_loss": 2.4, "total_train_time_s": 11, "seed": 137},
            ],
            "kaiming_normal": [
                {"final_val_loss": 2.4, "best_val_loss": 2.2, "total_train_time_s": 10, "seed": 42},
            ],
        }
        report_path = tmp_path / "report.md"
        generate_phase_report(results, "Test Phase", report_path)
        assert report_path.exists()
        text = report_path.read_text()
        assert "Test Phase" in text
        assert "xavier_normal" in text
        assert "kaiming_normal" in text
        assert "Best performer" in text
