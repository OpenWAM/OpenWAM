#!/usr/bin/env bash
set -euo pipefail

uv run python -m open_wam.training.train --cfg configs/experiments/backbone_only_robotwin.yaml
