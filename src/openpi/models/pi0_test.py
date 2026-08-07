import flax.nnx as nnx
import jax

import openpi.models.pi0_config as _pi0_config


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


def test_s1_parameter_audit_freezes_vlm_and_trains_action_using_actual_nnx_paths():
    config = _pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))
    audit = _pi0_config.audit_frozen_vlm_parameters(
        abstract_model,
        config.get_vlm_freeze_filter(),
        trainable_groups=("action",),
    )

    assert audit.frozen_vlm
    assert audit.trainable_action
    assert not audit.frozen_action
    assert not audit.trainable_completion
    assert all(path.startswith("PaliGemma/") for path in audit.frozen_vlm)
    assert all(not path.startswith("PaliGemma/img/") for path in audit.trainable_action)


def test_s2_parameter_audit_freezes_vlm_and_action_and_trains_only_head():
    config = _pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        completion_head=_pi0_config.CompletionHeadConfig(enabled=True),
    )
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))
    audit = _pi0_config.audit_frozen_vlm_parameters(
        abstract_model,
        config.get_completion_head_only_freeze_filter(),
        trainable_groups=("completion",),
    )

    assert audit.frozen_vlm
    assert audit.frozen_action
    assert not audit.trainable_action
    assert audit.trainable_completion
    assert all(path.startswith("completion_head/") for path in audit.trainable_completion)


def test_completion_disabled_preserves_legacy_model_structure():
    config = _pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))
    paths = [tuple(path) for path in nnx.state(abstract_model, nnx.Param).flat_state()]

    assert not hasattr(abstract_model, "completion_head")
    assert all(path[0] != "completion_head" for path in paths)
    assert len(_get_frozen_state(config)) == 0
