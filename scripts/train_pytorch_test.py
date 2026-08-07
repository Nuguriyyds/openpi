import pytest

from openpi.training import config as _config

from . import train_pytorch


@pytest.mark.parametrize(
    "config_name",
    ["pi05_agilex_breakfast_ttrtc_s1_action", "pi05_agilex_breakfast_ttrtc_s2_completion_head"],
)
def test_pytorch_trainer_rejects_staged_completion_configs_before_setup(monkeypatch, config_name):
    config = _config.get_config(config_name)
    monkeypatch.setattr(train_pytorch, "setup_ddp", lambda: pytest.fail("DDP setup must not run"))

    with pytest.raises(NotImplementedError, match="only supported by the JAX trainer"):
        train_pytorch.train_loop(config)
