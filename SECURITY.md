<!-- Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved. -->

# Security Policy

## Supported versions

`spot_basic` is under [light maintenance](README.md#maintenance). Security fixes, when they are
made, are applied to the `main` branch only. There are no long-term support branches, and older
releases are not patched.

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, report them privately through either of the following:

- Using GitHub's [private vulnerability reporting][1]
- Email to [opensource@rai-inst.com](mailto:opensource@rai-inst.com)

Please include as much of the following as you can:

- The type of issue, and the affected source files.
- The version or commit of `spot_basic` affected.
- Steps to reproduce, including any configuration required.
- The impact, including how an attacker might exploit the issue.

You can expect an acknowledgement within a few weeks, consistent with the project's light
maintenance commitment. Please do not disclose the issue publicly until a maintainer has had a
chance to respond.

## Scope

`spot_basic` is a client library for the [`spot_ros2`][2] driver, and it operates on robot hardware.
When judging whether something is in scope, note the following.

- Vulnerabilities in the Spot SDK, `spot_ros2`, `bosdyn_msgs`, or ROS 2 itself should be reported to
  those projects rather than here.
- ROS 2 offers no authentication or encryption by default. Exposing a Spot driver on an untrusted
  network is a deployment risk rather than a `spot_basic` vulnerability.
- Robot credentials belong in driver configuration that you keep outside version control.
  `spot_basic` never stores them.

## Dependency management

Dependency updates are monitored with [Dependabot](.github/dependabot.yml). Pull requests it opens
are reviewed by the maintainers listed in [`.github/CODEOWNERS`](.github/CODEOWNERS).

<!-- References -->

[1]: https://github.com/rai-opensource/spot_basic/security/advisories/new
[2]: https://github.com/rai-opensource/spot_ros2
