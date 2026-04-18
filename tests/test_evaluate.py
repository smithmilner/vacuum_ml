import pytest
from vacuum_ml.training.train import train
from vacuum_ml.training.evaluate import evaluate


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("models") / "tiny")
    train(total_timesteps=2000, n_envs=1, save_path=path)
    return path


def test_evaluate_returns_coverage_and_steps(tiny_model):
    results = evaluate(model_path=tiny_model, episodes=3)
    assert "mean_coverage" in results
    assert "mean_steps" in results
    assert 0.0 <= results["mean_coverage"] <= 1.0
    assert results["mean_steps"] > 0
