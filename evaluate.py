"""
评测自己训练的模型（在 dev 集上算 BLEU）

用法：
    python evaluate.py                        # 默认 weights/best.pt，取 dev 前 200 句
    python evaluate.py --weights weights/last.pt --limit 100 --beam-size 4
"""

import argparse
import sys
import time
from pathlib import Path

import torch

from models.transformer import build_model
from utils.dataset import read_lines
from utils.engine import load_checkpoint
from utils.metrics import corpus_bleu
from utils.vocab import Vocab

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description="评测翻译模型（BLEU）")
    p.add_argument("--weights", default=str(ROOT / "weights" / "best.pt"))
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--limit", type=int, default=200, help="评测多少句，0 表示全部")
    p.add_argument("--beam-size", type=int, default=4)
    p.add_argument("--show", type=int, default=3, help="打印几个例子")
    args = p.parse_args()

    if not Path(args.weights).exists():
        sys.exit(f"权重不存在：{args.weights}（先运行 python train.py）")

    state, model_cfg, meta = load_checkpoint(args.weights, "cpu")
    model = build_model(model_cfg, meta["vocab_sizes"], meta["special_ids"]).eval()
    model.load_state_dict(state)
    src_vocab = Vocab(meta["src_spm"])
    tgt_vocab = Vocab(meta["tgt_spm"])
    print(f"[模型] {args.weights}（参数量 {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M）")

    data_dir = Path(args.data_dir)
    src_lines = read_lines(data_dir / "dev.en")
    ref_lines = read_lines(data_dir / "dev.zh")
    if args.limit:
        src_lines, ref_lines = src_lines[: args.limit], ref_lines[: args.limit]
    print(f"[数据] dev {len(src_lines)} 句")

    hyps, t0 = [], time.time()
    for i, text in enumerate(src_lines, 1):
        print(f"\r[解码] {i}/{len(src_lines)}  已用 {time.time() - t0:.0f}s", end="", flush=True)
        ids = torch.tensor([src_vocab.encode(text, model_cfg["max_len"])], dtype=torch.long)
        out = model.beam_search(ids, max_new_tokens=model_cfg["max_len"], beam_size=args.beam_size)
        hyps.append(tgt_vocab.decode(out[0]))
    print()

    bleu = corpus_bleu(hyps, ref_lines)
    print(f"[BLEU] {bleu:.2f}")

    for i in range(min(args.show, len(hyps))):
        print(f"\n原文：{src_lines[i]}\n参考：{ref_lines[i]}\n译文：{hyps[i]}")


if __name__ == "__main__":
    sys.exit(main())
