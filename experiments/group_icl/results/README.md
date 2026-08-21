# Initial experiments — constant-C10 `fixed_p` sweep (archived)

Frozen record of the first experimental setup, run 2026-08-20 on the Bau cluster (nagoya, A100).
Archived because the approach is being replaced by the Bietti assign-and-add structure. The
compute artifacts (checkpoints, logs) were deleted from the cluster; only the metrics survive here.

## What the experiment was

Train one model per `fixed_p`, on a **constant C10** (one cyclic group of order 10 in every
sequence), and read off `symbolic_reliance` to see which regime the model is in.

Each sequence is 100 shots of `, a b = c`, one held-out fact at the end. `fixed_p` sets the
fraction of the 10 element-slots pinned to a canonical token across sequences (memorizable /
parametric); the rest are reshuffled every sequence (only inferable in-context / symbolic).

## Resolved config

`train_c10.sh` set these explicitly:

| | |
|---|---|
| task | mixcyclic, `num_symbols=16`, `max_order=min_order=10`, `mix=0` |
| `k_shots` | 100 (→ 499 input tokens) |
| `weight_decay` | 2.0 |
| `lr` | 1.5e-4 |
| `bf16` | on |
| `n_steps` | 200000 |
| `evaluation_steps` / `checkpoint_steps` | 1000 / 10000 |

These came from argparse defaults, not from the script — worth recording because they were never
a deliberate choice and `src/config.py` holds *different* (upstream) values that argparse silently
overrides:

| | argparse (in effect) | `src/config.py` (dead) |
|---|---|---|
| `n_layers` | 2 | 4 |
| `d_model` | 128 | 1024 |
| `n_heads` | 4 | 8 |
| `batch_size` | 64 | 128 |
| `lr_warmup_steps` | 100 | 1000 |
| `holdout_zero` | False | (upstream used True) |

LR schedule: linear warmup 100 steps → cosine anneal to **0** over 200k. Optimizer AdamW,
betas (0.9, 0.98), decoupled decay on matrices only. Model: **0.40M params**.

## Result: null

`c10-p0.0` ran the full 200k steps and never learned.

| step | acc_full | train_loss |
|---|---|---|
| 1000 | 0.047 | 1.4539 |
| 27000 | 0.031 | 1.4075 |
| 139000 | 0.125 | 1.4042 |
| 199000 | 0.156 | 1.4038 |

Chance is 0.100 (order 10). Loss sits at the "format learned, arithmetic not learned" floor:
per 5-token shot `, a b = c`, predicting `,` and `=` is deterministic and `a`,`b` are irreducible
at ~log(10), so format-only ≈ `(0 + 2.303 + 2.303 + 0 + 2.303)/5 = 1.382`. Learning the answer
token would drop it toward 0.921. It never moved off ~1.404.

A direct split of the answer positions confirmed the model was not even retrieving facts present
in its own context:

```
in-context answer accuracy : 0.0947   (train-like)
held-out answer accuracy   : 0.0938   (generalization)
chance                     : 0.1000
```

So this is not a memorize-but-don't-generalize plateau — there is no learning on either side.
`symbolic_reliance` in `metrics.json` is division noise at these accuracies (it goes negative at
several steps) and should not be read as signal.

## The task is not at fault

The released In-Context Algebra reference model (arXiv 2512.16902, `algebra.baulab.info/weights`,
50.42M params) evaluated on this *exact* constant-C10 task:

```
acc_full            = 0.9375
acc_context_shuffle = 0.1250
acc_global_relabel  = 0.9062
symbolic_reliance   = 0.8667
```

94% accuracy, with the expected symbolic signature. The data pipeline and readout were separately
verified by hand (sequence format, input/target alignment, prediction mask, held-out integrity,
`fixed_p` semantics all check out). So the failure is in model size and/or optimization, not the task.

## Why the hyperparameters were suspect

The config was a hybrid of two papers that were never meant to be mixed:

- **Eric / In-Context Algebra** (arXiv 2512.16902) — the codebase, model architecture, task
  generator, and LR scheduler. Trained its reference at `lr=1e-5` with **no weight decay**.
- **He et al. / Learning to grok** (arXiv 2406.02550) — where `lr=1.5e-4`, `wd=2.0`, and the AdamW
  betas came from. But their recipe used `batch_size=1024–1536`, `warmup=10000`, and a LR decay
  that **floors at 0.1× peak** rather than reaching 0.

We took two numbers from He et al. and left behind the batch size, warmup, and schedule they were
calibrated against. `wd=2.0` at batch 64 is a very different regularization regime than at 1536.

## Known open issues at time of archiving

- 2 layers is the theoretical minimum for induction heads, and this task needs more than induction
  (the held-out fact cannot be copied — it must be composed), so depth was plausibly below the floor.
- `holdout_zero=False` let the identity element be labelled `'0'`, a parametric confound the
  upstream setup deliberately excluded.
- C10 is composite (subgroups of order 2 and 5); grokking work standardly uses prime moduli.
- `fixed_p=1` is degenerate (constant vocabulary ⇒ eval ≡ train), which is what motivated moving to
  the assign-and-add structure where both sweep endpoints stay well-defined.
