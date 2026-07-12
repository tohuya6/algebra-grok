import os
import json
import time

import torch
import torch.nn.functional as F
try:
    import wandb
except ImportError:  # only needed when use_wandb=True; local/CPU runs can train without it
    wandb = None
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import random
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import asdict

from .readout import symbolic_reliance
from .device import Compute

def loss_fn(outputs, targets):
    loss = F.cross_entropy(
        outputs.view(-1, outputs.size(-1)),
        targets.to(torch.long).view(-1)
    )
    return loss


def accuracy_fn(outputs, targets):
    preds = outputs.argmax(-1).view(-1)
    truth = targets.to(torch.long).view(-1)
    correct = (preds == truth).sum().item()
    total = truth.size(0)
    return correct, total


class Trainer:

    def __init__(self, config, compute=None):
        self.compute = compute or Compute.resolve("auto")
        self.device = self.compute.device

    def train_step(self, model, task, config, optimizer, scheduler):
        context_length = model.config.block_size

        # Sample training batch
        train_batch = task.sample_batch(
            batch_size=config.batch_size,
            k_shots=config.k_shots,
            max_length=context_length,
            fixed_p=config.fixed_p,
        )

        # Move input data to the correct device
        train_batch = {k: v.to(self.device)
                for k, v in train_batch.items() if isinstance(v, torch.Tensor)}

        # Apply padding if needed
        if config.leftpad:
            pad_token_id = task.pad_token_id
            input_padding = context_length - train_batch["inputs"].size(1)
            target_padding = context_length - train_batch["targets"].size(1)
            padded_inputs = F.pad(train_batch["inputs"], (input_padding, 0), value=pad_token_id)
            padded_targets = F.pad(train_batch["targets"], (target_padding, 0), value=pad_token_id)
        else:
            padded_inputs = train_batch["inputs"]
            padded_targets = train_batch["targets"]

        # Forward pass. bf16 autocast on CUDA when config.bf16 (matches He et al.'s
        # --dtype bfloat16); a no-op on CPU/Turing or when bf16 is off. Weights/optimizer
        # stay fp32, and bf16 has fp32's range, so no GradScaler is needed.
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                            enabled=getattr(config, "bf16", False) and self.device.type == "cuda"):
            outputs = model(padded_inputs)
            if not isinstance(outputs, torch.Tensor):
                outputs = outputs[0]

            if config.final_token_only:
                # Only compute loss on final token
                masked_outputs = outputs[:, -1, :].reshape(-1, outputs.size(-1))
                masked_targets = padded_targets[:, -1].reshape(-1)
            else:
                # Create mask for non-padding tokens
                mask = (padded_targets != task.pad_token_id)

                # Reshape outputs and targets, applying mask
                masked_outputs = outputs.reshape(-1, outputs.size(-1))[mask.reshape(-1)]
                masked_targets = padded_targets.reshape(-1)[mask.reshape(-1)]

            # Compute loss only on non-padded positions
            loss = loss_fn(masked_outputs, masked_targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Compute metrics
        return loss.item()

    def fit(self, model, task, metadata, config):
        # Set random seeds for reproducibility
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)     
        torch.cuda.manual_seed(config.seed)   

        model = model.to(self.device)

        # Setup optimizer and learning rate scheduler
        # Decoupled weight decay, matching He et al. (icl/_src/scheduler.py): decay only the
        # weight matrices, not biases / LayerNorm (param.ndim <= 1); embeddings are decayed.
        # betas=(0.9, 0.98) also matches their AdamW.
        decay, no_decay = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if (p.ndim <= 1 or name.endswith(".bias")) else decay).append(p)
        optimizer = torch.optim.AdamW(
            [{"params": no_decay, "weight_decay": 0.0},
             {"params": decay, "weight_decay": config.weight_decay}],
            lr=config.lr, betas=(0.9, 0.98),
        )
        train_scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=config.n_steps,
            eta_min=0
        )
        warmup_scheduler = LinearLR(optimizer,
                start_factor=1e-4, end_factor=1, total_iters=config.lr_warmup_steps)
        scheduler = SequentialLR(optimizer,
                [warmup_scheduler, train_scheduler], [config.lr_warmup_steps])

        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=config.wandb_run_name,
                config=asdict(config),
                dir="./outputs"
            )

        # Create output directory (with wandb run name if available)
        output_dir = config.output_dir
        if config.use_wandb:
            output_dir = os.path.join(config.output_dir, wandb.run.name)

        # Create results directory if it doesn't exist
        os.makedirs(f'{output_dir}/progress', exist_ok=True)

        # Write metadata file
        self.write_metadata(output_dir, metadata)

        # Training loop
        losses = []
        history = []  # per-eval metrics (reliance/acc/loss vs step), persisted to metrics.json
        eval_accuracy = 0.0
        best_eval_accuracy = 0.0
        step_wall = 0.0           # summed wall-time of train steps since last record
        steps_since_record = 0    # -> accurate s/it (tqdm's own rate is skewed by eval steps)
        s_per_it = float('nan')
        pbar = tqdm(range(config.n_steps), desc="Training", unit="it")
        for step in pbar:
            log = {}
            # Periodic evaluation + metrics record, every evaluation_steps (He-style cadence)
            if step == 0 or step % config.evaluation_steps == 0:
                stats = symbolic_reliance(model, task, self.device,
                                          batch_size=config.batch_size,
                                          k_shots=config.k_shots, fixed_p=config.fixed_p)
                eval_accuracy = stats["acc_full"]
                # Accurate seconds/iteration over the train steps since the last record
                # (decoupled from tqdm's smoothed rate, which the eval step skews).
                s_per_it = step_wall / steps_since_record if steps_since_record else float('nan')
                step_wall, steps_since_record = 0.0, 0
                summary = (f"reliance={stats['symbolic_reliance']:.3f} "
                           f"ctx_shuf={stats['acc_context_shuffle']:.3f} "
                           f"glob={stats['acc_global_relabel']:.3f} "
                           f"{s_per_it:.4f}s/it")

                # Task-specific logging
                pbar.write(f'Step: {step}, Acc: {round(eval_accuracy * 100, 4)}; {summary}')
                log |= {
                    "eval_accuracy": eval_accuracy,
                    "s_per_it": s_per_it,
                }
                log |= stats

                # Persist the metrics history so reliance/acc/loss vs step can be plotted
                # later without wandb (e.g. on Colab).
                history.append({"step": step, "train_loss": (losses[-1] if losses else None),
                                "s_per_it": s_per_it, **stats})
                with open(f'{output_dir}/metrics.json', 'w', encoding='utf-8') as f:
                    json.dump(history, f, indent=2)

                # Plot and save training loss curve
                plt.figure(figsize=(3, 3))
                plt.plot(losses)
                plt.grid(True)
                plt.title('Training Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.savefig(f'{output_dir}/progress/train_loss.png', bbox_inches='tight', dpi=300)
                plt.close()

                # Save best model checkpoint
                os.makedirs(f'{output_dir}/models', exist_ok=True)
                if eval_accuracy > best_eval_accuracy:
                    best_eval_accuracy = eval_accuracy
                    torch.save(model.state_dict(), f'{output_dir}/models/algebra_gpt_best.pt')

            if config.checkpoint_steps and (step % config.checkpoint_steps == 0):
                torch.save(model.state_dict(), f'{output_dir}/models/algebra_gpt_{step}.pt')

            t0 = time.perf_counter()
            train_loss = self.train_step(model, task, config, optimizer, scheduler)
            step_wall += time.perf_counter() - t0
            steps_since_record += 1
            pbar.set_postfix({'Step': step, 'Train Loss': round(train_loss, 4),
                              's/it': f'{s_per_it:.4f}'})
            losses.append(train_loss)
            log |= {
                "train_loss": train_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "step": step
            }

            if config.use_wandb:
                wandb.log(log)

        if config.use_wandb:
            wandb.finish()

        return losses

    def write_metadata(self, output_dir, metadata, filename="metadata.json"):
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

