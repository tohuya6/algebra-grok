"""Run the symbolic-reliance readout on a trained checkpoint.

Loads the model + its task from a run's output dir (via metadata.json) and prints
the readout: acc_full, acc_context_shuffle, acc_global_relabel, symbolic_reliance.
A high symbolic_reliance (~1) means the model solves in-context (symbolic);
near 0 means it leans on memorized token meaning (parametric).
"""
import argparse

from src.device import Compute
from src.load_utils import load_gpt, load_task, load_metadata
from src.readout import symbolic_reliance


def main(a):
    ctx = Compute.resolve(a.device)
    print(f"device: {ctx}")
    model = load_gpt(a.dir, iternum=a.iternum, device=str(ctx.device),
                     disable_flash_attention=ctx.disable_flash_attention)
    model.to(ctx.device).eval()
    task, desc = load_task(a.dir)

    # Default k_shots AND fixed_p to the values the model was trained with, so the readout
    # stays in-distribution (matching the reliance the trainer logged); then clamp k_shots so
    # 5*k_shots-1 fits the model's block_size (else sample_batch would raise). Both keep
    # `just eval` from silently probing off-distribution or over-running the model.
    meta_args = load_metadata(a.dir)['args']
    k_shots = a.k_shots if a.k_shots is not None else meta_args.get('k_shots', 40)
    fixed_p = a.fixed_p if a.fixed_p is not None else meta_args.get('fixed_p', 0.0)
    max_k = (model.config.block_size + 1) // 5
    if k_shots > max_k:
        print(f"warning: k_shots={k_shots} over block_size={model.config.block_size}; clamping to {max_k}")
        k_shots = max_k

    stats = symbolic_reliance(model, task, ctx.device,
                              batch_size=a.batch_size, k_shots=k_shots, fixed_p=fixed_p)
    print(f"\nreadout for {desc!r}  (fixed_p={fixed_p}, k_shots={k_shots}):")
    for k, v in stats.items():
        print(f"  {k:22s} {v:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dir", help="run output dir, e.g. outputs/mixrosette-facts-10-16-8heads")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    p.add_argument("--iternum", type=int, default=None,
                   help="load newest checkpoint <= this step (default: latest)")
    p.add_argument("--fixed_p", type=float, default=None,
                   help="pinned-vocabulary fraction of the eval batch (default: the model's "
                        "trained fixed_p, i.e. in-distribution; 0 = all-variable)")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--k_shots", type=int, default=None,
                   help="context length in shots (default: the model's trained k_shots; "
                        "clamped to fit block_size)")
    main(p.parse_args())
