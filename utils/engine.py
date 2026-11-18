"""训练引擎：学习率调度、一轮训练、验证、权重存取。"""

import math
import time
from pathlib import Path

import torch


class NoamOpt:
    """论文里的学习率策略：预热阶段线性上升，之后按步数平方根衰减。

    为什么需要它：Transformer 用 Adam 直接训练容易前期炸、后期学不动，
    预热 + 衰减是让训练稳定的关键小技巧。
    """

    def __init__(self, optimizer, d_model: int, warmup_steps: int = 2000, scale: float = 2.0):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup = warmup_steps
        self.scale = scale
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.scale * (self.d_model ** -0.5) * min(
            self.step_num ** -0.5, self.step_num * self.warmup ** -1.5
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.step()
        return lr


def train_one_epoch(model, batches, opt: NoamOpt, criterion, device, log_every: int = 20):
    """跑一轮。返回平均 loss 和平均困惑度（perplexity）。"""
    model.train()
    total_loss, n_batches, n_tokens = 0.0, 0, 0
    t0 = time.time()

    for src, tgt_in, tgt_out in batches:
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        logits = model(src, tgt_in)                       # (B, T, V)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        opt.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # 梯度裁剪，防止爆炸
        lr = opt.step()

        n_tokens += (tgt_out != 0).sum().item()
        total_loss += loss.item()
        n_batches += 1

        if n_batches % log_every == 0:
            print(
                f"  step {n_batches:5d} | loss {total_loss / n_batches:.3f} | "
                f"lr {lr:.2e} | {(time.time() - t0) / n_batches:.2f}s/step",
                flush=True,
            )

    return {"loss": total_loss / max(n_batches, 1), "tokens": n_tokens}


@torch.no_grad()
def evaluate(model, batches, criterion, device):
    """验证集上的 loss（不算 BLEU，BLEU 在 evaluate.py 里单独算）。"""
    model.eval()
    total, n = 0.0, 0
    for src, tgt_in, tgt_out in batches:
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        logits = model(src, tgt_in)
        total += criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1)).item()
        n += 1
    return total / max(n, 1)


def save_checkpoint(path: Path, model, optimizer, model_cfg: dict, meta: dict):
    """把权重 + 重建模型所需的一切都存下来，这样推理时不用再问“结构是什么”。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "model_cfg": model_cfg,        # 层数、维度等
            "meta": meta,                  # 词表大小、特殊 id、语料统计
        },
        path,
    )


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return ckpt["model"], ckpt["model_cfg"], ckpt["meta"]
