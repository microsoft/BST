#!/bin/bash
set -e

# Get directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
REPO_ROOT="$(dirname $DIR)"
echo "REPO_ROOT=$REPO_ROOT"

# Activate venv
source "$REPO_ROOT/venv/bin/activate"

# Hardcoded
GPUS_PER_NODE=8

# Get number of nodes allocated by Volcano
NUM_NODES=$(cat /etc/volcano/VC_*_NUM)
echo "NUM_NODES=$NUM_NODES"

# Get first node's network name
MASTER_ADDR=$(cat /etc/volcano/VC_*_HOSTS | awk -F, '{print $1}')
echo "MASTER_ADDR=$MASTER_ADDR"

# Get rank of this node from Volcano
# If not found, try fallback to MPI
NODE_RANK=$VC_TASK_INDEX
[[ -z "$NODE_RANK" ]] && NODE_RANK=$OMPI_COMM_WORLD_RANK
[[ -z "$NODE_RANK" ]] && echo "ERROR: Could not determine node rank!" && exit 1
echo "NODE_RANK=$NODE_RANK"

# Launch python processes for each GPU on this node
cd "$REPO_ROOT"
fabric --help

fabric run \
    --node-rank=$NODE_RANK \
    --main-address=$MASTER_ADDR \
    --num-nodes=$NUM_NODES \
    --devices=$GPUS_PER_NODE \
    --strategy=ddp \
    --precision=bf16-mixed \
    train.py --config config/bst_phi_subsample.yaml --no_pbar
