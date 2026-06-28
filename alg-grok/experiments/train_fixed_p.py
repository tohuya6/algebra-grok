import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # alg-grok/
sys.path.insert(0, ROOT)  # make `src` importable regardless of cwd

import torch

from src.device import Compute
from src.models.model import GPT, GPTConfig, GPTConfigNoFlashAttention
from src.trainer import Trainer
from src.config import TrainingParams
from src.constants import TASK_MAP


def build_model(d_model, n_layers, n_heads, block_size, vocab_size, use_flash):
    params = dict(n_embd=d_model, n_layer=n_layers, n_head=n_heads,
                  block_size=block_size, vocab_size=vocab_size, positional_encoding="rope")
    config_cls = GPTConfig if use_flash else GPTConfigNoFlashAttention
    return GPT(config_cls(**params)), params


def main(a):
    ctx = Compute.resolve(a.device)
    print(f"device: {ctx}")

    task_config = {"num_symbols": a.num_symbols, "max_order": a.max_order,
                   "mix": a.mix, "holdout_zero": a.holdout_zero}
    task = TASK_MAP[a.task_name](**task_config)
    vocab_size = -(-task.vocab_size // 32) * 32          # round up to a multiple of 32 (as training.py does)

    torch.manual_seed(a.seed)
    model, model_params = build_model(a.d_model, a.n_layers, a.n_heads,
                                      a.block_size, vocab_size, ctx.use_flash)

    cfg = TrainingParams(
        task_name=a.task_name, task_config=task_config, k_shots=a.k_shots, fixed_p=a.fixed_p,
        n_layers=a.n_layers, n_heads=a.n_heads, d_model=a.d_model, d_mlp=4 * a.d_model,
        block_size=a.block_size, positional_encoding="rope",
        output_dir=os.path.join(ROOT, "outputs", a.name), leftpad=False, batch_size=a.batch_size,
        n_steps=a.n_steps, lr=a.lr, weight_decay=a.weight_decay, lr_warmup_steps=a.lr_warmup_steps,
        evaluation_steps=a.evaluation_steps, checkpoint_steps=0, final_token_only=False,
        seed=a.seed, evaluation_size=a.evaluation_size, use_wandb=False, bf16=a.bf16,
    )
    metadata = {"args": {"task_name": a.task_name, "task_config": task_config,
                         "k_shots": a.k_shots, "fixed_p": a.fixed_p,
                         "weight_decay": a.weight_decay, "lr": a.lr, "bf16": a.bf16},
                "model_params": model_params}

    print(f"training {a.name!r}: {a.n_steps} steps, fixed_p={a.fixed_p} "
          f"wd={a.weight_decay} lr={a.lr} -> {cfg.output_dir}")
    Trainer(cfg, ctx).fit(model, task, metadata, cfg)
    print("done")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    p.add_argument("--name", default="poc-parametric-tiny", help="output subfolder under outputs/")
    p.add_argument("--fixed_p", type=float, default=1.0)
    p.add_argument("--task_name", default="mixrosette",
                   choices=["mixrosette", "mixcyclic", "mixdihedral", "mixmonoid"])
    p.add_argument("--num_symbols", type=int, default=8)
    p.add_argument("--max_order", type=int, default=4)
    p.add_argument("--mix", type=float, default=0.0, help="prob. of adding another group per sequence")
    p.add_argument("--holdout_zero", action="store_true", help="forbid a variable identity mapping to '0'")
    p.add_argument("--k_shots", type=int, default=40)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=2.0,
                   help="AdamW weight decay (He et al. grokking value; pair with lr~1.5e-4)")
    p.add_argument("--bf16", action="store_true",
                   help="bf16 autocast on CUDA (~2x on A100; no-op on T4/CPU)")
    p.add_argument("--lr_warmup_steps", type=int, default=100)
    p.add_argument("--evaluation_steps", type=int, default=250)
    p.add_argument("--evaluation_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
