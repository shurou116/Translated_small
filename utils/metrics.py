"""BLEU 评测（不依赖第三方库的轻量实现）。

BLEU 的思路：把译文和参考译文的 n 元组（n=1..4）做匹配，算精确率，
再乘一个长度惩罚（译文太短会被罚），最后取几何平均。分数 0~100，越高越好。
"""

import math
from collections import Counter


def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(hypotheses: list[str], references: list[str], max_n: int = 4) -> float:
    """语料级 BLEU。中文按字符切分，英文按空格切分，所以这里用“字符级”的通用切法。"""
    matches = [0] * max_n
    totals = [0] * max_n
    hyp_len = ref_len = 0

    for hyp, ref in zip(hypotheses, references):
        h_tok = list(hyp) if _is_cjk(hyp) else hyp.split()
        r_tok = list(ref) if _is_cjk(ref) else ref.split()
        hyp_len += len(h_tok)
        ref_len += len(r_tok)
        for n in range(1, max_n + 1):
            h_ng, r_ng = _ngrams(h_tok, n), _ngrams(r_tok, n)
            total = max(sum(h_ng.values()), 0)
            match = sum(min(c, r_ng[g]) for g, c in h_ng.items())
            matches[n - 1] += match
            totals[n - 1] += total

    if hyp_len == 0:
        return 0.0

    # 加一平滑，避免某个 n 元组命中数为 0 时整体变 0
    precisions = [(matches[i] + 1) / (totals[i] + 1) for i in range(max_n)]
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
    score = bp * math.exp(sum(math.log(p) for p in precisions) / max_n)
    return score * 100


def _is_cjk(text: str) -> bool:
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return cjk > len(text) * 0.3
