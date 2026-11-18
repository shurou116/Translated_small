"""
本地训练脚本（从零训练一个翻译用 Transformer）

前提：data/ 下要有 train.en / train.zh（以及 dev.en / dev.zh）。
      如果没有，先运行：python scripts/get_corpus.py

用法：
    python train.py                          # 默认参数，CPU 也能跑
    python train.py --epochs 10 --batch-size 32
    python train.py --resume weights/last.pt # 断点续训

产物：
    data/spm_en.model / spm_zh.model   词表（自动训练）
    weights/last.pt                    最新权重（每轮覆盖）
    weights/best.pt                    验证集 loss 最好的权重
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

from models.transformer import build_model, count_parameters
from utils.dataset import ParallelDataset, make_batches
from utils.engine import NoamOpt, evaluate, load_checkpoint, save_checkpoint, train_one_epoch
from utils.vocab import PAD_ID, Vocab, train_tokenizer

ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description="训练翻译 Transformer")
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--save-dir", default=str(ROOT / "weights"))
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-len", type=int, default=128, help="句子最大子词数")
    p.add_argument("--vocab-size", type=int, default=8000)
    # 模型结构（小模型，CPU 友好）
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--enc-layers", type=int, default=3)
    p.add_argument("--dec-layers", type=int, default=3)
    p.add_argument("--ffn", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--lr-scale", type=float, default=2.0)
    p.add_argument("--threads", type=int, default=0, help="CPU 线程数，0 表示用默认")
    p.add_argument("--resume", default="", help="从某个 .pt 断点续训")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def ensure_tokenizers(data_dir: Path, vocab_size: int):
    """词表不存在就先训练一个（几秒钟的事）。"""
    for lang in ("en", "zh"):
        model_file = data_dir / f"spm_{lang}.model"
        if not model_file.exists():
            print(f"[词表] 训练 {model_file.name}（vocab={vocab_size}）")
            train_tokenizer(data_dir / f"train.{lang}", data_dir / f"spm_{lang}", vocab_size)
    return Vocab(data_dir / "spm_en.model"), Vocab(data_dir / "spm_zh.model")


def main():
    args = parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)

    data_dir, save_dir = Path(args.data_dir), Path(args.save_dir)
    need = [data_dir / "train.en", data_dir / "train.zh", data_dir / "dev.en", data_dir / "dev.zh"]
    missing = [str(p) for p in need if not p.exists()]
    if missing:
        sys.exit("缺少语料文件：\n  " + "\n  ".join(missing) + "\n请先运行：python scripts/get_corpus.py")

    print("[数据] 准备词表与数据集 ...")
    src_vocab, tgt_vocab = ensure_tokenizers(data_dir, args.vocab_size)
    train_ds = ParallelDataset(data_dir / "train.en", data_dir / "train.zh", src_vocab, tgt_vocab, args.max_len)
    dev_ds = ParallelDataset(data_dir / "dev.en", data_dir / "dev.zh", src_vocab, tgt_vocab, args.max_len)
    print(f"[数据] 训练 {len(train_ds)} 对 / 验证 {len(dev_ds)} 对")

    model_cfg = {
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_encoder_layers": args.enc_layers,
        "num_decoder_layers": args.dec_layers,
        "dim_feedforward": args.ffn,
        "dropout": args.dropout,
        "max_len": args.max_len,
    }
    meta = {
        "vocab_sizes": {"src": src_vocab.size, "tgt": tgt_vocab.size},
        "special_ids": {
            "src_pad": PAD_ID, "tgt_pad": PAD_ID,
            "tgt_bos": 2, "tgt_eos": 3,
        },
        "src_lang": "en",
        "tgt_lang": "zh",
        "src_spm": str(data_dir / "spm_en.model"),
        "tgt_spm": str(data_dir / "spm_zh.model"),
    }

    model = build_model(model_cfg, meta["vocab_sizes"], meta["special_ids"]).to(device)
    print(f"[模型] 参数量 {count_parameters(model) / 1e6:.2f} M")

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    opt = NoamOpt(optimizer, args.d_model, args.warmup, args.lr_scale)
    # label_smoothing：不要逼模型把正确答案的概率推到 1.0，泛化更好
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.1)

    start_epoch, best_loss = 0, float("inf")
    if args.resume:
        state, _, _ = load_checkpoint(args.resume, device)
        model.load_state_dict(state)
        print(f"[续训] 已加载 {args.resume}")

    save_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, args.epochs):
        print(f"\n===== Epoch {epoch + 1}/{args.epochs} =====")
        t0 = time.time()
        batches = make_batches(train_ds, args.batch_size, shuffle=True)
        stats = train_one_epoch(model, batches, opt, criterion, device)
        dev_batches = make_batches(dev_ds, args.batch_size, shuffle=False)
        dev_loss = evaluate(model, dev_batches, criterion, device)
        ppl = 2.718 ** dev_loss
        print(f"[Epoch {epoch + 1}] train loss {stats['loss']:.3f} | "
              f"dev loss {dev_loss:.3f} (ppl {ppl:.1f}) | {time.time() - t0:.0f}s")

        save_checkpoint(save_dir / "last.pt", model, optimizer, model_cfg, meta)
        if dev_loss < best_loss:
            best_loss = dev_loss
            save_checkpoint(save_dir / "best.pt", model, optimizer, model_cfg, meta)
            print(f"[保存] best.pt（dev loss {best_loss:.3f}）")

    print(f"\n训练结束，最好 dev loss = {best_loss:.3f}")
    print("接着可以：python evaluate.py   或   python translate.py --weights weights/best.pt --file a.txt")


if __name__ == "__main__":
    sys.exit(main())
