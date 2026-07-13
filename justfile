# algebra-grok task runner. Recipes call `python` in the ACTIVE conda env — activate it
# first (see README: `conda env create -f environment.yml` or `-cpu.yml`, then activate).
# Run `just` (or `just --list`) to see everything.
#
# Recipe args are POSITIONAL (name, then steps), e.g.
#   just train-symbolic big 20000
# Override a GLOBAL (device/task/steps) by assigning it BEFORE the recipe, e.g.
#   just device=cuda task=mixcyclic train-symbolic
# Append any extra raw flags at the end (no `--` needed); they pass straight through:
#   just train-symbolic big 20000 --d_model 256 --n_heads 8 --bf16

# --- overridable globals -------------------------------------------------------
device := "auto"        # auto | cpu | cuda | cuda:N
task   := "mixcyclic"   # mixrosette | mixcyclic | mixdihedral | mixmonoid
steps  := "3000"        # training steps for the quick presets

# show all recipes (default when you just type `just`)
default:
    @just --list

# create the conda env (cluster / CUDA); run from the repo root
setup:
    conda env create -f environment.yml

# create the conda env for local macOS / CPU dev
setup-local:
    conda env create -f environment-cpu.yml

# --- training presets ----------------------------------------------------------
# SYMBOLIC regime: all-variable vocabulary (fixed_p=0) -> model must solve in-context
train-symbolic name="symbolic" steps=steps *extra="":
    python experiments/train_fixed_p.py \
        --name {{name}} --fixed_p 0.0 \
        --task_name {{task}} --device {{device}} --n_steps {{steps}} {{extra}}

# PARAMETRIC regime: fully-pinned vocabulary (fixed_p=1) -> memorizable token meaning
train-parametric name="parametric" steps=steps *extra="":
    python experiments/train_fixed_p.py \
        --name {{name}} --fixed_p 1.0 \
        --task_name {{task}} --device {{device}} --n_steps {{steps}} {{extra}}

# arbitrary fixed_p sweep point (e.g. `just train 0.5 mid 8000`)
# CONSTANT-GROUP sweep (recommended): pin ONE group every run with --min_order == --max_order
# and --mix 0, so order is constant (10 -> each 0.1 step of p adds exactly one fixed element).
# Sweep p over 0.0..0.9 (p=1 is degenerate) on a constant C10:
#   for p in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do \
#     just train $p c10-$p 20000 \
#       --task_name mixcyclic --num_symbols 16 --max_order 10 --min_order 10 --mix 0; done
train p name="run" steps=steps *extra="":
    python experiments/train_fixed_p.py \
        --name {{name}} --fixed_p {{p}} \
        --task_name {{task}} --device {{device}} --n_steps {{steps}} {{extra}}

# GROKKING run: He et al. weight_decay=2.0 + lr=1.5e-4, long + bf16 (bf16 no-ops off-CUDA)
grok name="grok" steps="50000" fixed_p="0.0" *extra="":
    python experiments/train_fixed_p.py \
        --name {{name}} --fixed_p {{fixed_p}} \
        --task_name {{task}} --device {{device}} --n_steps {{steps}} \
        --weight_decay 2.0 --lr 1.5e-4 --bf16 {{extra}}

# FULL constant-C10 sweep: trains p in {0.0..0.9} (p=1 excluded, degenerate). Long -> GPU/cluster.
# NOT run by default; invoke explicitly, e.g. `just device=cuda sweep-c10` (steps default 200k).
sweep-c10 steps="200000" *extra="":
    for p in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do \
        python experiments/train_fixed_p.py \
            --name c10-p$p --fixed_p $p \
            --task_name mixcyclic --num_symbols 16 --max_order 10 --min_order 10 --mix 0 \
            --device {{device}} --n_steps {{steps}} {{extra}}; \
    done

# --- evaluation ----------------------------------------------------------------
# run the symbolic-reliance readout on a trained checkpoint dir
eval dir *extra="":
    python experiments/eval.py {{dir}} --device {{device}} {{extra}}

# --- housekeeping --------------------------------------------------------------
# remove stray bytecode caches, notebook checkpoints, and local wandb logs
clean:
    find . -type d -name __pycache__ -not -path './.git/*' -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name .ipynb_checkpoints -exec rm -rf {} + 2>/dev/null || true
    rm -rf wandb *.egg-info
    @echo "cleaned (outputs/ weights left untouched)"
