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

The main experiment is the constant-C10 `fixed_p` sweep. `experiments/train_c10.sh` wraps one
sweep point (fixed_p is the argument):

```bash
bash experiments/train_c10.sh 0.0        # one point (all-variable / symbolic)

# the full sweep (p=1 excluded — degenerate):
for p in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do bash experiments/train_c10.sh $p; done
```

Or call `python experiments/train_fixed_p.py --help` directly for all training options.

The `symbolic_reliance` readout is logged to `outputs/<run>/metrics.json` every eval step during
training. Two notebooks cover it: `experiments/control_p0.ipynb` trains the all-variable
(`fixed_p=0`) control and plots its dynamics, and `experiments/verify_solution.ipynb` runs the
static readout on a trained checkpoint (validated on the released reference model).
