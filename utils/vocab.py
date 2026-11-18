"""句子分词器（SentencePiece BPE）的封装：训练词表 + 编码/解码。"""

from pathlib import Path

import sentencepiece as spm

# 固定的特殊符号 id，训练和推理必须一致
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3


def train_tokenizer(txt_file: Path, model_prefix: Path, vocab_size: int = 8000):
    """在语料上训练一个 BPE 词表。产出 model_prefix.model 和 .vocab 两个文件。

    语料太小时，sentencepiece 会报“词表太大”，这里自动减半重试，
    避免直接崩掉（正式训练用 2 万句以上不会触发）。
    """
    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    size = vocab_size
    while True:
        try:
            spm.SentencePieceTrainer.train(
                input=str(txt_file),
                model_prefix=str(model_prefix),
                vocab_size=size,
                model_type="bpe",
                character_coverage=0.9995,      # 中文用 0.9995，保证常用汉字全覆盖
                pad_id=PAD_ID,
                unk_id=UNK_ID,
                bos_id=BOS_ID,
                eos_id=EOS_ID,
                pad_piece="<pad>",
                unk_piece="<unk>",
                bos_piece="<s>",
                eos_piece="</s>",
                num_threads=4,
                minloglevel=2,
            )
            return
        except RuntimeError as e:
            if "Vocabulary size too high" not in str(e) or size <= 64:
                raise
            size //= 2
            print(f"[词表] 语料较小，词表大小降为 {size} 重试")


class Vocab:
    """包装 SentencePiece 模型，提供“文本 <-> id 序列”的转换。"""

    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))
        self.size = self.sp.get_piece_size()
        # 交叉验证一下特殊 id，防止词表文件和代码约定不一致
        assert self.sp.pad_id() == PAD_ID, "词表 pad_id 与代码约定不一致"

    def encode(self, text: str, max_len: int = 128) -> list[int]:
        """文本 -> [BOS] + 子词 id + [EOS]，超过 max_len 截断。"""
        ids = self.sp.encode(text, out_type=int)[: max_len - 2]
        return [BOS_ID] + ids + [EOS_ID]

    def decode(self, ids) -> str:
        """id 序列 -> 文本，自动去掉特殊符号。"""
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        keep = [i for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID)]
        return self.sp.decode(keep)
