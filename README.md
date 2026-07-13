# algebra-grok

Parametric-vs-symbolic solving in small transformers on in-context group/monoid arithmetic.
`fixed_p` interpolates the training data between an **all-variable** vocabulary (symbolic:
token meaning must be inferred in-context) and a **fully-pinned** one (parametric:
token meaning is memorizable). The `symbolic_reliance` readout reports which regime a
trained model is in. See the concept notes at the bottom for the full picture.

## Setup (conda)

Create the environment **from the repo root** (so the editable `pip install -e .` resolves):

**Cluster — Northeastern Explorer / any Linux + NVIDIA box:**
```bash
module load anaconda3            # on NURC Explorer
conda env create -f environment.yml
source activate alg-grok         # NURC recommends `source activate` over `conda activate`
```
Per NURC guidance, create envs under `/projects/<lab>/...` (not `/home`) for storage quota.
If `pytorch-cuda=12.4` exceeds the node's driver (`nvidia-smi`), lower it in `environment.yml`.

**Local — macOS / CPU:**
```bash
conda env create -f environment-cpu.yml
conda activate alg-grok
```
The only difference is the cluster env pins `pytorch-cuda` (Linux-only); local uses a CPU build.

## Common runs (`just`)

With the env active (`just` also prints this list):
```bash
just                       # list all recipes
just train-symbolic        # fixed_p=0  (symbolic regime)
just train-parametric      # fixed_p=1  (parametric regime)
just train 0.5 mid 8000    # arbitrary fixed_p sweep point (p, name, steps)
just grok                  # weight_decay=2.0, lr=1.5e-4, long + bf16
just eval outputs/<run>    # symbolic-reliance readout on a checkpoint
```
Args are **positional**; override a global **before** the recipe (`just device=cuda train-symbolic`);
extra flags pass straight through (`just train-symbolic big 20000 --d_model 256 --bf16`).

## Layout
- `src/` — model, tasks (data generation), trainer, readout, device utils
- `experiments/` — `train_fixed_p.py` (training CLI), `eval.py` (readout CLI), notebooks
- `outputs/` — training runs + checkpoints (gitignored)
- `environment.yml` / `environment-cpu.yml` — conda envs (cluster / local)

---

## Concept notes

The task (Eric's setup): one group, in-context learning. Each run shows solved facts (a b = c) using a symbol→element vocab, holds some facts out, and tests the model on a held-out fact. Symbols carry no built-in meaning (even digits are just letters) — meaning comes from the group structure.

two ways to solve:

Symbolic — read the context, infer this run's mapping, apply group structure.
Parametric — recall memorized symbol meanings from weights, ignore context.
The dials:

p = how many elements are fixed (pinned to a canonical symbol every run → memorizable). p controls the regime.
fixed_perm (π) = which ones — a frozen random priority order, so the fixed set isn't biased toward the identity/low slots. Stable across runs, nested as p grows.


the regime axis:

p=0 → all symbols reshuffled each run → symbolic (valid baseline: fresh vocab = real generalization).
p→1 → more pinned → parametric.
p=1 → degenerate (eval = train, nothing to generalize) → exclude. Range = [0, 1).
Grokking = the gap (memorize first, generalize later) needs both a memorization shortcut and something unseen to generalize to:

p=0: generalizes, but no shortcut → no gap → no grok.
p=1: shortcut only, nothing to generalize → no grok.
Middle p: both present → grokking lives here.
The readout (verify which regime): take a batch and relabel symbols via a per-example lookup table (element tokens only; =, , untouched).

local shuffle (context only, question left normal) → symbolic dies, parametric survives → this is symbolic_reliance.
global relabel (context + question, consistently) → parametric dies, symbolic survives → the control.



Extensions later: raise mix for multiple groups; add a global fact-level holdout if you ever want a grokking signal near p=1.
