# CLAUDE.md: AUV hull inspection (simulation)

Production-grade underwater inspection software, developed and tested entirely in simulation:
scenario generation with exact defect ground truth, survey planning, perception, defect mapping,
reporting, and the production layer (simulation tests in CI, versioned models, fault injection,
observability). See README.md for the milestones.

## Hard rules (never break these)

1. **Git: the user does all writes.** Never run `git init/add/commit/push/pull/fetch/clone/remote/
   rebase` or any `gh` command. Read-only git (`status`, `diff`, `log`, `show`) is fine. After each
   finished step, print the exact commands for the user with explicit paths.
2. **No AI attribution anywhere**: no co-author trailers, no "generated with" lines, in commits,
   docs or code.
3. **No sudo.** Give the user system install commands and wait.
4. **Safety:** LLMs and agents never command thrusters or low-level control; they act only through
   validated high-level tools.
5. **Leave other projects alone.** `~/yaad_ws/auv_research_cvpr2027` (frozen paper results),
   `mujoco_marine_sim`, `auv_navigation_fusion`, `camera_sonar_perception` and other folders are
   separate repos: do not edit them without asking. Never stop other projects' processes/containers.
6. **Downloads:** PyPI, Docker Hub and apt mirrors are fine. Ask before anything else (GitHub,
   Hugging Face models, datasets).
7. **Memory (15 GB shared machine):** at most 6 heavy workers, each inside
   `systemd-run --user --scope -p MemoryMax=<N> -p MemorySwapMax=0`; do not launch when MemAvailable
   is below 4 GB.

## Conventions

- Python via `.venv/bin/python` (uv venv, Python 3.10, numpy pinned 1.26.4 for later ROS 2 Humble
  compatibility, pytest 7.4.4). Rendering is headless: `MUJOCO_GL=egl`.
- Every number lives in `src/auvinspect/config/default.yaml`; code reads configuration, never
  hard-codes physical or sampling constants.
- Determinism: every random draw comes from `auvinspect.rng.stream(seed, name)`. The same seed and
  configuration give the same scenario digest and the same images.
- Ground truth is computed from the scenario, never from a model's output. Coverage and defect
  findings are scored against it (the referee), not against what the pipeline claims.
- Check pytest's own exit code before calling a step green.
- Simulation results are simulation results: never describe them as real-world performance.
