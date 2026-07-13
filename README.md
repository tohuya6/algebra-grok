temp notes:

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