# algebra-grok

Parametric-vs-symbolic solving in small transformers on in-context group arithmetic.
`fixed_p` sets how much of the vocabulary is pinned each run: `0` = all-variable (symbolic),
`1` = fully fixed (parametric). The `symbolic_reliance` readout reports which regime a
trained model is in.

## Setup

Create the conda env from the repo root, then activate it:

```bash
conda env create -f env/environment.yml       # cluster / Linux + NVIDIA
conda activate algebra-grok
```

Cluster access (Explorer / Bau Lab machines) is covered in the
[onramp wiki](https://github.com/thebaulab/onramp/wiki/Accessing-Clusters); once you're on a
node, create the env as above.

## Run

The main experiment is the constant-C10 `fixed_p` sweep, run one point at a time with
`experiments/train_c10.sh` (the argument is fixed_p). Start with the `fixed_p=0` control:

```bash
bash experiments/train_c10.sh 0.0        # all-variable control (symbolic; no grokking)

# the full sweep (p=1 excluded — degenerate):
for p in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do bash experiments/train_c10.sh $p; done
```

Each run logs `symbolic_reliance` to `outputs/<run>/metrics.json` every 1000 steps. To see the
p=0 control's dynamics, open `experiments/control_p0.ipynb` and run its plot cell — it reads
`outputs/c10-p0.0/`. (Its run cell just calls `train_c10.sh 0.0`, so you can Run All instead of
using the terminal, but that retrains — don't do both.) `experiments/verify_solution.ipynb` runs
the static readout on a trained checkpoint. `python experiments/train_fixed_p.py --help` lists
all training options.
