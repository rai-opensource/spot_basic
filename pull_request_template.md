## Change Overview

Please summarize the change and link the relevant issue. Include why the change is being made, not just what changed.

## Testing Done

Please list the tests you ran and check off the ones that passed. Note that unit tests use mocked ROS 2 interfaces, so changes affecting robot behavior should also be exercised against hardware.

- [ ] `pre-commit run --all-files` passes.
- [ ] Unit tests pass (`colcon test --packages-select spot_basic`, or `pytest test --ignore=test/system`).
- [ ] Hardware-in-the-loop tests run, if applicable (`pytest --manual -v -s test/system`).
- [ ] Affected examples in `examples/` still run.

If you tested against a robot, please note its configuration (arm or no arm, and the driver launch arguments used).

## Documentation

- [ ] README, `docs/`, and docstrings updated for any user-visible change.
- [ ] Not applicable.
