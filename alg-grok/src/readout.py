import torch

@torch.no_grad()
def final_token_correct(model, inputs, targets, device):
    
    logits = model(inputs.to(device))
    logits = logits if isinstance(logits, torch.Tensor) else logits[0]
    return (logits[:, -1].argmax(-1).cpu() == targets[:, -1]).float()


def relabel_table(batch_size, num_symbols, vocab_size):
    table = torch.arange(vocab_size).repeat(batch_size, 1)
    table[:, :num_symbols] = torch.stack(
        [torch.randperm(num_symbols) for _ in range(batch_size)])
    return table


def symbolic_reliance(model, task, device, *, batch_size=128, k_shots=100, fixed_p=0.0):
    batch = task.sample_batch(batch_size=batch_size, k_shots=k_shots,
                              max_length=model.config.block_size, hold_out=True, fixed_p=fixed_p)
    inp, tgt = batch["inputs"], batch["targets"]
    ns, vs = task.num_symbols, task.vocab_size

    # Final fact is ',a b = c'; inputs end in [',', a, b, '=']. Keep those 4 tokens fixed.
    context = inp.size(1) - 4
    ctx_shuf = inp.clone()
    ctx_shuf[:, :context] = torch.gather(relabel_table(batch_size, ns, vs), 1, inp[:, :context])
    g = relabel_table(batch_size, ns, vs)
    glob_inp, glob_tgt = torch.gather(g, 1, inp), torch.gather(g, 1, tgt)

    acc_full = final_token_correct(model, inp, tgt, device).mean().item()
    acc_ctx = final_token_correct(model, ctx_shuf, tgt, device).mean().item()
    acc_glob = final_token_correct(model, glob_inp, glob_tgt, device).mean().item()
    return {
        "acc_full": acc_full,
        "acc_context_shuffle": acc_ctx,
        "acc_global_relabel": acc_glob,
        "symbolic_reliance": (acc_full - acc_ctx) / acc_full if acc_full > 0 else float("nan"),
    }
