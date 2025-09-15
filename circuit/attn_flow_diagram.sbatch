#!/bin/bash
#SBATCH -J attn-flow-diagram
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p res-gpu-small
#SBATCH --qos=long-high-prio
#SBATCH --time=5-00:00:00
#SBATCH --gres=gpu:turing:1
#SBATCH -o /home3/wclv88/data/adversarial-steering/logs/slurm-%A_%a.out

source /etc/profile
module load cuda/12.5

mkdir -p "cache"

# Ensure uv is available (train.py uses uv script metadata for deps)
if ! command -v uv >/dev/null 2>&1; then
  echo "Error: 'uv' not found in PATH. Install uv or load the appropriate module." >&2
  exit 127
fi

uv run attn_flow_diagram.py