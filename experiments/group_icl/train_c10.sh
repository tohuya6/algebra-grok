#!/bin/bash

# Constant-C10 fixed_p run. fixed_p (the fraction of the vocabulary pinned to a canonical
# token every run) is the first argument: 0.0 = all-variable (symbolic), 1.0 = fully
# pinned (parametric). Grokking lives at intermediate values.
#
#   bash experiments/group_icl/train_c10.sh 0.5
#
# The main experiment is the sweep over p (p=1 excluded, degenerate):
#   for p in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do bash experiments/group_icl/train_c10.sh $p; done

cd "$(dirname "$0")/../.." || exit 1   # run from the repo root, wherever this is invoked from

if [ -z "$1" ]; then
    echo "usage: bash experiments/group_icl/train_c10.sh <fixed_p> [name] [n_steps]"
    exit 1
fi

fixed_p="$1"
name="${2:-c10-p$1}"
n_steps="${3:-200000}"

echo "training fixed_p=$fixed_p (constant C10) -> outputs/$name"

# min_order == max_order == 10 with mix=0 pins one constant C10 every run, so each 0.1 of
# fixed_p pins exactly one more element. He et al. grokking hparams (wd=2.0, lr=1.5e-4, bf16).
# k_shots=100 (~10 examples/element for C10); bump toward 200 if acc_full plateaus.
python experiments/group_icl/train_fixed_p.py \
    --name="$name" --fixed_p="$fixed_p" \
    --task_name=mixcyclic --num_symbols=16 --max_order=10 --min_order=10 --mix=0 --k_shots=100 \
    --weight_decay=2.0 --lr=1.5e-4 --bf16 --checkpoint_steps=10000 --evaluation_steps=1000 \
    --n_steps="$n_steps"
