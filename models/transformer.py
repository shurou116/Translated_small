"""
从零实现的 Seq2Seq Transformer（用于机器翻译）

结构就是论文《Attention Is All You Need》里的标准 encoder-decoder：
    源句 -> Embedding + 位置编码 -> Encoder 若干层 -> 编码结果
                                             ↓（cross-attention 看编码结果）
    目标句(左移一位) -> Embedding + 位置编码 -> Decoder 若干层 -> Linear -> 词表概率

为什么用 nn.TransformerEncoderLayer / DecoderLayer 而不是完全手写：
PyTorch 已经把这些层（多头注意力 + 残差 + LayerNorm + FFN）实现好了并且经过大量验证，
手写一遍既容易出错，又不会改变模型结构。我们要自己掌控的是：词嵌入、位置编码、
掩码（mask）的构造、解码策略（贪心 / 束搜索）——这些才是翻译任务的关键。
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """正弦位置编码：让模型知道词的先后顺序（注意力本身是无序的）。"""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # (1, max_len, d_model)，方便和 batch 广播
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


def build_pad_mask(ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """True 表示该位置是 padding，注意力要忽略它。"""
    return ids.eq(pad_id)


def build_causal_mask(size: int, device) -> torch.Tensor:
    """上三角为 True：解码时第 i 个位置不能看到 i 之后的词（防止偷看答案）。"""
    return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)


class TransformerMT(nn.Module):
    """翻译用的 Transformer。词表大小由数据决定，结构尺寸由参数决定。"""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_pad_id: int,
        tgt_pad_id: int,
        tgt_bos_id: int,
        tgt_eos_id: int,
        d_model: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id
        self.tgt_bos_id = tgt_bos_id
        self.tgt_eos_id = tgt_eos_id
        self.max_len = max_len

        # 两个词表大小不同，所以嵌入不共享权重
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=src_pad_id)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=tgt_pad_id)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=True
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_encoder_layers, enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers)

        self.generator = nn.Linear(d_model, tgt_vocab_size)

        # 参数初始化（论文做法：正态分布，越小越稳）
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ---------- 训练时用：一次前向算出所有位置的输出 ----------
    def forward(self, src_ids: torch.Tensor, tgt_in_ids: torch.Tensor) -> torch.Tensor:
        src_pad = build_pad_mask(src_ids, self.src_pad_id)
        tgt_pad = build_pad_mask(tgt_in_ids, self.tgt_pad_id)
        tgt_causal = build_causal_mask(tgt_in_ids.size(1), tgt_in_ids.device)

        memory = self.encode(src_ids, src_pad)
        out = self.decode(memory, tgt_in_ids, src_pad, tgt_pad, tgt_causal)
        return self.generator(out)          # (B, T, tgt_vocab)

    def encode(self, src_ids: torch.Tensor, src_pad_mask: torch.Tensor) -> torch.Tensor:
        x = self.pos_enc(self.src_embed(src_ids) * math.sqrt(self.d_model))
        return self.encoder(x, src_key_padding_mask=src_pad_mask)

    def decode(
        self,
        memory: torch.Tensor,
        tgt_in_ids: torch.Tensor,
        src_pad_mask: torch.Tensor,
        tgt_pad_mask: torch.Tensor = None,
        tgt_causal_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        y = self.pos_enc(self.tgt_embed(tgt_in_ids) * math.sqrt(self.d_model))
        return self.decoder(
            y,
            memory,
            tgt_mask=tgt_causal_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )

    # ---------- 推理时用：一个词一个词往外蹦 ----------
    @torch.no_grad()
    def greedy_decode(self, src_ids: torch.Tensor, max_new_tokens: int = 256) -> torch.Tensor:
        """贪心解码：每步取概率最大的词。最快，但容易卡在重复词上。"""
        src_pad = build_pad_mask(src_ids, self.src_pad_id)
        memory = self.encode(src_ids, src_pad)
        batch = src_ids.size(0)
        ys = torch.full((batch, 1), self.tgt_bos_id, dtype=torch.long, device=src_ids.device)
        finished = torch.zeros(batch, dtype=torch.bool, device=src_ids.device)

        for _ in range(max_new_tokens):
            tgt_causal = build_causal_mask(ys.size(1), ys.device)
            out = self.decode(memory, ys, src_pad, None, tgt_causal)
            next_id = self.generator(out[:, -1]).argmax(-1)      # (B,)
            ys = torch.cat([ys, next_id.unsqueeze(1)], dim=1)
            finished |= next_id.eq(self.tgt_eos_id)
            if finished.all():
                break
        return ys

    @torch.no_grad()
    def beam_search(self, src_ids: torch.Tensor, max_new_tokens: int = 256, beam_size: int = 4,
                    len_penalty: float = 1.0) -> torch.Tensor:
        """束搜索：同时保留 beam_size 个候选，最后按“对数概率 / 长度^α”选最好的。

        比贪心慢 beam_size 倍，但译文明显更顺，是机器翻译的常规做法。
        """
        src_pad = build_pad_mask(src_ids, self.src_pad_id)
        memory = self.encode(src_ids, src_pad)                   # (1, S, D)
        device = src_ids.device
        beams = [(0.0, [self.tgt_bos_id])]                       # (累计对数概率, 词序列)
        finished = []

        for _ in range(max_new_tokens):
            candidates = []
            for score, seq in beams:
                ys = torch.tensor([seq], dtype=torch.long, device=device)
                tgt_causal = build_causal_mask(ys.size(1), device)
                out = self.decode(memory, ys, src_pad, None, tgt_causal)
                log_probs = torch.log_softmax(self.generator(out[0, -1]), dim=-1)
                topk = torch.topk(log_probs, beam_size)
                for log_p, idx in zip(topk.values.tolist(), topk.indices.tolist()):
                    candidates.append((score + log_p, seq + [idx]))

            # 先按长度归一化再排序，避免一味偏好短句
            candidates.sort(key=lambda c: c[0] / (len(c[1]) ** len_penalty), reverse=True)
            beams = []
            for score, seq in candidates:
                if seq[-1] == self.tgt_eos_id:
                    finished.append((score, seq))
                else:
                    beams.append((score, seq))
                if len(beams) >= beam_size:
                    break
            if not beams:
                break

        pool = finished or beams
        pool.sort(key=lambda c: c[0] / (len(c[1]) ** len_penalty), reverse=True)
        return torch.tensor([pool[0][1]], dtype=torch.long, device=device)


def build_model(config: dict, vocab_sizes: dict, special_ids: dict) -> TransformerMT:
    """根据配置文件重建模型（训练和推理共用，保证结构一致）。"""
    return TransformerMT(
        src_vocab_size=vocab_sizes["src"],
        tgt_vocab_size=vocab_sizes["tgt"],
        src_pad_id=special_ids["src_pad"],
        tgt_pad_id=special_ids["tgt_pad"],
        tgt_bos_id=special_ids["tgt_bos"],
        tgt_eos_id=special_ids["tgt_eos"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_encoder_layers=config["num_encoder_layers"],
        num_decoder_layers=config["num_decoder_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
        max_len=config["max_len"],
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
