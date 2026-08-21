# assign_add — variable assignment + modular addition

Workspace for the Bietti et al. setting ("Assign and Add: A Mechanistic Study of Compositional
Arithmetic", [arXiv:2605.31497](https://arxiv.org/abs/2605.31497)).

A sequence is a run of `variable = value` assignments followed by an addition query whose operands
may be literal constants or variables needing lookup:

```
c=1, b=17, a=42, b+19=?          (mod N)
```

Sequences are classified by how many of the two operands are variables (0-, 1-, or 2-variable).
The motivation for moving here from `../group_icl/` is that memorizable structure (the fixed mod-N
addition table), in-context structure (bindings resampled every sequence), and the generalization
target (a held-out split over addition pairs) come from three *independent* mechanisms — so the
sweep parameter can be pushed to either endpoint without one of them collapsing. In the `fixed_p`
setting a single knob controlled both memorizability and eval novelty, which made `p=1` degenerate.

## Reference hyperparameters (from the paper, verbatim)

AdamW · `lr=1e-3` · `weight_decay=2e-2` · 30000 steps · 2-layer transformer · single attention
head · `d_model = d_head = 128` · `d_mlp = 512` · sequence length `T=16` · biases and normalization
layers omitted · layer-1 MLP removed. LR schedule is not stated in the paper.

Note these differ sharply from `group_icl/`, which inherited `lr=1.5e-4` / `wd=2.0` from He et al.
(a different paper, calibrated at batch 1024–1536).

## Open design questions

- `V` (variables) and `N` (modulus): paper uses 12 and 59, with no stated justification and no
  sensitivity analysis. `N` should stay prime.
- Sweep parameterization: the paper's `r` is a *ratio* (`#2var / #0var`) studied over `(0, 1]`, so
  `r=1` is a 50/50 mix rather than all-variable. A fraction `#2var / (#2var + #0var)` over `[0, 1]`
  gives evenly spaced points, but everything above 0.5 is outside the paper's tested range.
- Held-out split: remove ~30% of addition pairs transitively (a removed pair must also be
  unreachable via assignment, e.g. dropping `(1,3)` also drops `c=1, c+3`), plus the paper's
  positional restrictions on which variables may occupy which operand slot.
