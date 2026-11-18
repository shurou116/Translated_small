"""
文章翻译（推理 / 应用）

两种用法：
  1) 用现成预训练权重（推荐，质量好，开箱即用）：
        python translate.py --file samples/sample_en.txt
        python translate.py --model opus --file a.txt        # 小模型，更快
  2) 用自己训练的权重（train.py 的产物）：
        python translate.py --weights weights/best.pt --file a.txt

其他：
        python translate.py --text "Attention is all you need."
        python translate.py --interactive                    # 交互模式，边输边翻
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

# 国内直连 huggingface.co 不通，走镜像（必须在 import transformers 之前设置）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch

ROOT = Path(__file__).resolve().parent
MAX_CHARS = 400          # 每块最多多少字符，避免超模型长度上限

PRETRAINED = {
    "nllb": {"name": "facebook/nllb-200-distilled-600M",
             "langs": {"en2zh": ("eng_Latn", "zho_Hans"), "zh2en": ("zho_Hans", "eng_Latn")}},
    "opus": {"name": {"en2zh": "Helsinki-NLP/opus-mt-en-zh", "zh2en": "Helsinki-NLP/opus-mt-zh-en"},
             "langs": None},
}

# 机器翻译常见的术语误译修正（仅英译中）
GLOSSARY = [
    (r"变压器架构|变换器结构|变换器架构", "Transformer 架构"),
    (r"自我关注", "自注意力"),
    (r"统计排列|统计校正", "统计对齐"),
    (r"培训", "训练"),
    (r"推论", "推理"),
    (r"矩阵倍增", "矩阵乘法"),
    (r"重现性|重复性|反复性", "循环结构"),
]


# ---------------- 文本切分 ----------------
def detect_direction(text: str) -> str:
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    letters = sum(1 for ch in text if ch.isalpha())
    return "zh2en" if cjk / max(letters, 1) > 0.3 else "en2zh"


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?。！？；;:])\s*", text)
    return [p for p in (s.strip() for s in parts) if p]


def build_chunks(sentences: list[str], max_chars: int = MAX_CHARS) -> list[str]:
    """把句子拼成不超过 max_chars 的块，长文章也不会被截断。"""
    chunks, buf, size = [], [], 0
    for sent in sentences:
        if len(sent) > max_chars:                       # 单句超长：硬切
            if buf:
                chunks.append(" ".join(buf))
                buf, size = [], 0
            chunks.extend(sent[i:i + max_chars] for i in range(0, len(sent), max_chars))
            continue
        if buf and size + len(sent) > max_chars:
            chunks.append(" ".join(buf))
            buf, size = [], 0
        buf.append(sent)
        size += len(sent)
    if buf:
        chunks.append(" ".join(buf))
    return chunks


# ---------------- 译文后处理 ----------------
def to_chinese_punct(text: str) -> str:
    text = re.sub(r"(?<!\d)\.", "。", text)          # 3.5 这类小数点不动
    text = re.sub(r"(?<!\d),|,(?!\d)", "，", text)
    text = re.sub(r";", "；", text)
    text = re.sub(r":(?!\d)", "：", text)
    text = re.sub(r"(?<=[。，；：！？])\s+", "", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])[ \t]+(?=[\u4e00-\u9fff])", "", text)
    return text.strip()


def post_edit(text: str, direction: str) -> str:
    if direction != "en2zh":
        return text
    for pattern, repl in GLOSSARY:
        text = re.sub(pattern, repl, text)
    return to_chinese_punct(text)


# ---------------- 翻译引擎 ----------------
class PretrainedEngine:
    """直接用 Hugging Face 上的预训练模型（Marian / NLLB）。"""

    def __init__(self, backend: str, direction: str, beams: int = 2):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        cfg = PRETRAINED[backend]
        name = cfg["name"] if isinstance(cfg["name"], str) else cfg["name"][direction]
        self.direction, self.beams = direction, beams
        print(f"[模型] {name}", flush=True)
        if cfg["langs"]:
            self.tokenizer = AutoTokenizer.from_pretrained(name, src_lang=cfg["langs"][direction][0])
            self.bos_id = self.tokenizer.convert_tokens_to_ids(cfg["langs"][direction][1])
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(name)
            self.bos_id = None
        self.model = AutoModelForSeq2SeqLM.from_pretrained(name, dtype=torch.float32).eval()

    @torch.no_grad()
    def translate(self, chunk: str) -> str:
        inputs = self.tokenizer(chunk, return_tensors="pt", truncation=True, max_length=1024)
        kwargs = {"forced_bos_token_id": self.bos_id} if self.bos_id is not None else {}
        ids = self.model.generate(
            **inputs, max_new_tokens=512, num_beams=self.beams,
            no_repeat_ngram_size=3, **kwargs,
        )
        return self.tokenizer.decode(ids[0], skip_special_tokens=True)


class LocalEngine:
    """加载 train.py 训练出来的权重（自带词表，完全离线）。"""

    def __init__(self, weights: str, beam_size: int = 4):
        from models.transformer import build_model
        from utils.engine import load_checkpoint
        from utils.vocab import Vocab

        print(f"[模型] 本地权重 {weights}", flush=True)
        state, model_cfg, meta = load_checkpoint(weights, "cpu")
        self.model = build_model(model_cfg, meta["vocab_sizes"], meta["special_ids"]).eval()
        self.model.load_state_dict(state)
        self.src_vocab = Vocab(meta["src_spm"])
        self.tgt_vocab = Vocab(meta["tgt_spm"])
        self.beam_size = beam_size
        self.max_len = model_cfg.get("max_len", 128)

    @torch.no_grad()
    def translate(self, chunk: str) -> str:
        src_ids = torch.tensor([self.src_vocab.encode(chunk, self.max_len)], dtype=torch.long)
        out = self.model.beam_search(src_ids, max_new_tokens=self.max_len, beam_size=self.beam_size)
        return self.tgt_vocab.decode(out[0])


def translate_text(text: str, engine, direction: str) -> str:
    chunks = build_chunks(split_sentences(text))
    results, t0 = [], time.time()
    for i, chunk in enumerate(chunks, 1):
        print(f"\r[翻译] {i}/{len(chunks)} 段  已用 {time.time() - t0:.0f}s", end="", flush=True)
        results.append(engine.translate(chunk))
    print(f"\r[翻译] 完成，共 {len(chunks)} 段，耗时 {time.time() - t0:.0f}s")
    joiner = "" if direction == "en2zh" else " "
    return post_edit(joiner.join(results), direction)


# ---------------- 命令行 ----------------
def main():
    ap = argparse.ArgumentParser(description="文章翻译（Transformer）")
    ap.add_argument("--file", help="待翻译的文章（.txt / .md）")
    ap.add_argument("--text", help="直接给一段文字")
    ap.add_argument("--out", help="译文保存路径，默认 <原名>_zh.txt / <原名>_en.txt")
    ap.add_argument("--direction", choices=["en2zh", "zh2en"], help="默认自动判断")
    ap.add_argument("--model", choices=list(PRETRAINED), default="nllb", help="预训练模型，默认 nllb")
    ap.add_argument("--weights", help="使用自己训练的权重（如 weights/best.pt），此时忽略 --model")
    ap.add_argument("--beams", type=int, default=2, help="束搜索宽度，越大越准越慢")
    ap.add_argument("--interactive", action="store_true", help="交互模式：逐句输入翻译")
    args = ap.parse_args()

    engine = LocalEngine(args.weights, beam_size=max(args.beams, 4)) if args.weights else None

    if args.interactive:
        direction = args.direction or "en2zh"
        engine = engine or PretrainedEngine(args.model, direction, args.beams)
        print(f"交互模式（{'英译中' if direction == 'en2zh' else '中译英'}），输入空行退出\n")
        while True:
            try:
                line = input("原文> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                break
            print("译文>", translate_text(line, engine, direction), "\n")
        return

    if args.file:
        src = Path(args.file)
        if not src.exists():
            sys.exit(f"文件不存在：{src}")
        text = src.read_text(encoding="utf-8")
    elif args.text:
        src, text = None, args.text
    else:
        ap.error("请用 --file / --text / --interactive 指定要翻译的内容")

    direction = args.direction or detect_direction(text)
    if engine is not None and direction != "en2zh":
        sys.exit("本地权重只支持英译中；如需反向翻译请去掉 --weights，改用 --model nllb")
    if engine is None:
        engine = PretrainedEngine(args.model, direction, args.beams)

    print(f"[方向] {'英译中' if direction == 'en2zh' else '中译英'}，字数 {len(text)}")
    result = translate_text(text, engine, direction)

    if args.out:
        out_path = Path(args.out)
    elif src:
        suffix = "_zh" if direction == "en2zh" else "_en"
        out_path = src.with_name(src.stem + suffix + src.suffix)
    else:
        print("\n" + result)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result, encoding="utf-8")
    print(f"[完成] 译文已保存：{out_path}")
    print("-" * 60)
    print(result[:600] + ("..." if len(result) > 600 else ""))


if __name__ == "__main__":
    sys.exit(main())
