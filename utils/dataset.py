"""数据集与批处理。

翻译任务的批处理有两个讲究：
1. 每个 batch 里句子要 pad 到同一长度，所以按长度排序后再组 batch，能少做很多无用计算；
2. 训练时把 batch 顺序打乱，但 batch 内部仍保持长度接近，兼顾效率与随机性。
"""

import random
from pathlib import Path

import torch

from .vocab import PAD_ID, Vocab


def read_lines(path: Path) -> list[str]:
    return [line.rstrip("\n") for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


class ParallelDataset:
    """平行语料：第 i 行源句对应第 i 行目标句。"""

    def __init__(self, src_file: Path, tgt_file: Path, src_vocab: Vocab, tgt_vocab: Vocab, max_len: int = 128):
        src_lines = read_lines(src_file)
        tgt_lines = read_lines(tgt_file)
        assert len(src_lines) == len(tgt_lines), f"源/目标行数不一致：{len(src_lines)} vs {len(tgt_lines)}"
        self.pairs = [
            (src_vocab.encode(s, max_len), tgt_vocab.encode(t, max_len))
            for s, t in zip(src_lines, tgt_lines)
        ]
        self.src_texts, self.tgt_texts = src_lines, tgt_lines

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        return self.pairs[i]


def pad_batch(seqs: list[list[int]], pad_id: int = PAD_ID) -> torch.Tensor:
    """把变长 id 列表补成 (B, T) 的矩形张量。"""
    max_len = max(len(s) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


def make_batches(dataset: ParallelDataset, batch_size: int, shuffle: bool = True, bucket: bool = True):
    """生成 (src_ids, tgt_in_ids, tgt_out_ids) 批次。

    tgt_in  是输入给解码器的（以 <s> 开头，不含 </s>）
    tgt_out 是监督信号（去掉了开头的 <s>），两者错开一位，所以叫“教师强制”
    """
    indices = list(range(len(dataset)))
    if bucket:
        indices.sort(key=lambda i: len(dataset[i][0]))   # 长度相近的放一起
    batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
    if shuffle:
        random.shuffle(batches)                          # 只打乱 batch 顺序

    for idx in batches:
        src = [dataset[i][0] for i in idx]
        tgt = [dataset[i][1] for i in idx]
        src_ids = pad_batch(src)
        tgt_ids = pad_batch(tgt)
        yield src_ids, tgt_ids[:, :-1], tgt_ids[:, 1:]
