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
