# TTRTC Migration Notes

This copy was created from the official `Physical-Intelligence/openpi` `main`
branch at commit `15a9616a00943ada6c20a0f158e3adb39df2ccac`.

The original local TTRTC backup under the Desktop was used as a read-only
reference.  All edits are in this new copy.

## Configs

- Regular pi0.5 AgileX:
  `pi05_agilex_empty_the_box_all_470`
- pi0.5 AgileX with training-time RTC enabled:
  `pi05_agilex_empty_the_box_all_470_ttrtc`

Both configs are registered in `src/openpi/training/config.py`.

## Switch

The switch is `TrainConfig.training_time_rtc.enabled`.

Extra TTRTC parameters live in `src/openpi/training/ttrtc.py`:

- `simulated_delay`
- `delay_sampling`
- `fixed_delay`
- `clean_timestep`
- `loss_normalization`

The normal training path keeps calling the original pi0/pi0.5 loss.  When the
switch is enabled, `scripts/train.py` passes the TTRTC config into `Pi0.compute_loss`.

## Example

```bash
python scripts/train.py pi05_agilex_empty_the_box_all_470 --exp-name regular_pi05
python scripts/train.py pi05_agilex_empty_the_box_all_470_ttrtc --exp-name ttrtc_pi05
```

## Completion-head overfit diagnostic

Before another full completion-head run, use the balanced 20-episode diagnostic:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi05_agilex_breakfast_frozen_head_s2_completion_overfit \
  --exp-name=s2_completion_overfit --overwrite
```

Every train batch contains 16 positive frames, 16 terminal-near negatives, and
32 ordinary negatives. Look for `completion_positive_count=16`, then compare
`train_overfit/best_f1` and `train_overfit/auc` with the naturally distributed
`val/*` metrics. If the train-overfit metrics cannot approach 1.0, the frozen
single-frame prefix and/or final-two-frame label is not separable.
