#!/usr/bin/env bash
set -e

N_RUNS=${1:-5}
CFG=${2:-configs/sim_default.yaml}

accelerate launch --config_file accelerate_config.yaml run_sim_suite.py \
  --n_runs ${N_RUNS} --config ${CFG}

