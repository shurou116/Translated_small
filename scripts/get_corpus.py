"""
下载并整理中英平行语料（Tatoeba en-zh，来自 OPUS）

运行：python scripts/get_corpus.py
产物：data/train.en / train.zh / dev.en / dev.zh
"""

import io
import random
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

try:                                  # 用 certifi 的根证书，避免 Windows 上 SSL 校验失败
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = ssl.create_default_context()


def _open(url: str, timeout: int = 120):
    """打开 URL；若本机缺少根证书导致校验失败，退回不校验（公共语料，可接受）。"""
    try:
        return urllib.request.urlopen(url, context=SSL_CTX, timeout=timeout)
    except urllib.error.URLError as e:
        if "CERTIFICATE_VERIFY_FAILED" not in str(e):
            raise
        print("[提示] 本机缺少根证书，改为跳过 SSL 校验下载")
        return urllib.request.urlopen(url, context=ssl._create_unverified_context(), timeout=timeout)

URL = "https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/moses/en-zh.txt.zip"
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

MAX_PAIRS = 20000      # 保留多少对（CPU 训练够用，太大跑不动）
MAX_LEN = 100          # 句子最大字符数，过滤超长句
DEV_RATIO = 0.05       # 验证集比例


def fetch_zip() -> Path:
    """下载语料压缩包（带缓存，第二次直接复用）。"""
    DATA.mkdir(parents=True, exist_ok=True)
    zip_path = DATA / "tatoeba-en-zh.zip"
    if zip_path.exists() and zip_path.stat().st_size > 1_000_000:
        print(f"[跳过] 已有压缩包 {zip_path}")
        return zip_path
    print(f"[下载] {URL}")
    with _open(URL) as resp, open(zip_path, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    print(f"[下载] 完成 {zip_path.stat().st_size / 1e6:.1f} MB")
    return zip_path


def load_pairs(zip_path: Path):
    """从 zip 里读 en / zh 两个文本文件，按行对齐。"""
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        en_name = next(n for n in names if n.endswith(".en"))
        zh_name = next(n for n in names if n.endswith(".zh"))
        en = zf.read(en_name).decode("utf-8").splitlines()
        zh = zf.read(zh_name).decode("utf-8").splitlines()
    print(f"[语料] 原始 {len(en)} 对")
    return en, zh


def clean(en, zh):
    """过滤空行、超长句、长度严重不对齐的句子对。"""
    pairs = []
    for e, z in zip(en, zh):
        e, z = e.strip(), z.strip()
        if not e or not z or len(e) > MAX_LEN or len(z) > MAX_LEN:
            continue
        ratio = len(e) / max(len(z), 1)
        if ratio < 0.2 or ratio > 5:      # 中英文长度比异常，多半是错对齐
            continue
        pairs.append((e, z))
    print(f"[语料] 过滤后 {len(pairs)} 对")
    return pairs


def main():
    zip_path = fetch_zip()
    en, zh = load_pairs(zip_path)
    pairs = clean(en, zh)

    random.seed(0)
    random.shuffle(pairs)
    pairs = pairs[:MAX_PAIRS]
    n_dev = max(200, int(len(pairs) * DEV_RATIO))
    dev, train = pairs[:n_dev], pairs[n_dev:]

    for split, rows in (("train", train), ("dev", dev)):
        (DATA / f"{split}.en").write_text("\n".join(r[0] for r in rows) + "\n", encoding="utf-8")
        (DATA / f"{split}.zh").write_text("\n".join(r[1] for r in rows) + "\n", encoding="utf-8")
        print(f"[写入] data/{split}.en / {split}.zh  {len(rows)} 对")

    print("\n示例：")
    for e, z in train[:3]:
        print(f"  EN: {e}\n  ZH: {z}")


if __name__ == "__main__":
    sys.exit(main())
