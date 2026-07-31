<!-- Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved. -->

# Contributing to `spot_basic`

Thanks for your interest in `spot_basic`. This document explains what kinds of contributions the
project accepts, and how to get a change reviewed and merged.

## Table of contents

- [Maintenance expectations](#maintenance-expectations)
- [Reporting a bug](#reporting-a-bug)
- [Requesting a feature](#requesting-a-feature)
- [Setting up a development environment](#setting-up-a-development-environment)
- [Making a change](#making-a-change)
- [Running the tests](#running-the-tests)
- [Coding conventions](#coding-conventions)
- [Opening a pull request](#opening-a-pull-request)
- [Review and merge process](#review-and-merge-process)

## Maintenance expectations

> [!IMPORTANT]
> `spot_basic` is under **light maintenance**. No feature development is guaranteed. If you file a
> bug report, or open a pull request that fixes a bug, expect a maintainer to respond within a few
> weeks. Feature requests may be closed without implementation.

Please read this before investing significant effort in a change. If you need guarantees this
project does not offer, forking is a reasonable option and the [MIT license](LICENSE) permits it.

## Reporting a bug

Open an issue using the [bug report template][1].

Good reports are much faster to act on. Please include:

- Your Ubuntu, ROS 2, and Python versions.
- Your Spot SDK version (`pip list | grep bosdyn` — all entries should read `5.0.1`).
- Whether your robot has an arm, since arm APIs behave differently on armless robots.
- The `spot_driver` launch arguments you used, including whether `controllable:=True` was set.
- A minimal script that reproduces the problem.
- The complete error output, not just the final line.

Before filing, please check the [Requirements](README.md#requirements) and
[Limitations](README.md#limitations) sections of the README. A large share of reported problems turn
out to be version mismatches or functionality that is intentionally out of scope.

## Requesting a feature

Feature requests are welcome as discussion, but under light maintenance they will generally not be
implemented by the maintainers. A feature request accompanied by a working pull request has a far
better chance of landing.

## Setting up a development environment

Fork the repository, then follow the [Installation](README.md#installation) instructions, cloning
your fork instead of the upstream repository:

```bash
mkdir -p ~/spot_ws/src && cd ~/spot_ws
git clone https://github.com/<your-username>/spot_basic.git src/spot_basic
./src/spot_basic/install_spot_basic.sh
source install/setup.bash
```

Then install the pre-commit hooks, which run the same checks CI does:

```bash
cd ~/spot_ws/src/spot_basic
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Making a change

1. Create a branch off `main` for your change.
2. Keep the change focused. Unrelated fixes are easier to review as separate pull requests.
3. Add or update tests covering the behavior you changed.
4. Update the documentation — the [README](README.md), the guides under [`docs/`](docs), and any
   affected [`examples/`](examples) — when your change affects user-visible behavior.

## Running the tests

Unit tests run through `colcon`:

```bash
cd ~/spot_ws
colcon test --packages-select spot_basic
colcon test-result --verbose
```

You can also run them directly with `pytest`, which is what CI does. Note the `--ignore`, which
excludes the hardware-in-the-loop suite:

```bash
source install/setup.bash
pytest src/spot_basic/test --ignore=src/spot_basic/test/system -vv
```

The hardware-in-the-loop suite is interactive and requires a real Spot with the driver running. It
is never run in CI, so please exercise it manually for changes that affect robot behavior:

```bash
pytest --manual -v -s src/spot_basic/test/system
```

## Coding conventions

Formatting and linting are enforced automatically by pre-commit and CI, so the fastest path is to
run `pre-commit run --all-files` before pushing.

- **Formatting**: `black` and `isort`, both at a 120 character line length.
- **Linting**: `flake8`, configured in [`setup.cfg`](setup.cfg).
- **Type hints**: annotate public APIs; the codebase is broadly typed.
- **Docstrings**: all public functions, classes, and modules should carry a docstring. Document
  raised exceptions, especially the `RuntimeError` that arm-dependent APIs raise on robots without
  an arm.
- **Copyright header**: every source file begins with the RAI copyright line. Copy the header from a
  neighboring file when adding a new one.
- **No magic numbers**: give numeric constants names or explain them in a comment.

## Opening a pull request

Push your branch to your fork and open a pull request against `main`. The
[pull request template](pull_request_template.md) asks you to describe:

- **What** changed.
- **Why** it changed, linking the relevant issue.
- **How** you tested it, including whether you ran the hardware-in-the-loop suite and on what robot
  configuration.

> [!NOTE]
> CI workflows do not run automatically for first-time contributors. A maintainer must approve the
> run, so expect a delay before checks appear on your pull request.

## Review and merge process

- Every pull request requires review and approval from a maintainer listed in
  [`.github/CODEOWNERS`](.github/CODEOWNERS) before it can be merged.
- CI — lint, build, and unit tests — must pass.
- Maintainers may ask for changes, or decline a contribution that falls outside the project's scope
  or maintenance commitment.

Thanks for contributing.

<!-- References -->

[1]: https://github.com/rai-opensource/spot_basic/issues/new?template=bug_report.md
