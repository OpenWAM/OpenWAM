#!/usr/bin/env bash
set -euo pipefail

uv run python -m open_wam.training.train --cfg configs/experiments/contract_only_robotwin.yaml
