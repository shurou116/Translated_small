# Transformer 文章翻译（训练 / 推理 / 应用）

一套可以直接用的机器翻译项目：**用现成预训练权重翻译文章**，也**可以从零训练自己的 Transformer**。  
纯 CPU 可跑，不需要 GPU、不需要 API Key。

```
E:\PythonProject1\
├── translate.py          ← 推理/应用：文章翻译（主力脚本，日常就用它）
├── train.py              ← 本地训练：从零训练 Transformer（可选）
├── evaluate.py           ← 评测：在自己训练的权重上算 BLEU
├── scripts\
│   └── get_corpus.py     ← 下载中英平行语料（Tatoeba，约 2 万句）
├── models\
│   └── transformer.py    ← 从零实现的 Seq2Seq Transformer（含束搜索解码）
├── utils\
│   ├── vocab.py          ← SentencePiece 词表：训练 + 编解码
│   ├── dataset.py        ← 数据集、按长度分桶组 batch
│   ├── engine.py         ← 训练循环、Noam 学习率、权重存取
│   └── metrics.py        ← BLEU 评测（无第三方依赖）
├── data\                 ← 语料与词表（get_corpus.py 自动生成）
├── weights\              ← 自己训练的权重（train.py 自动生成）
└── samples\              ← 示例文章与译文
```

---

## 一、环境（新建环境后一次性安装）

```bash
conda create -p E:\conda-envs\transformer python=3.10 -y
E:\conda-envs\transformer\python.exe -m pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

PyCharm / VSCode 里把解释器选成 `E:\conda-envs\transformer`，工作目录设为 `E:\PythonProject1`。

> PyTorch 建议单独装 CPU 版，更稳：  
> `pip install torch --index-url https://download.pytorch.org/whl/cpu`

---

## 二、翻译文章（日常用法，不需要训练）

```bash
python translate.py --file 你的文章.txt              # 英文→中文，译文存 你的文章_zh.txt
python translate.py --file 中文文章.txt              # 自动判断方向，中文→英文
python translate.py --text "Attention is all you need."
python translate.py --interactive                    # 交互模式，边输边翻
python translate.py --model opus --file a.txt        # 小模型，快 3 倍，术语差一些
```

| 参数                  | 说明                                    |
| ------------------- | ------------------------------------- |
| `--file` / `--text` | 待翻译的文章或一段文字                           |
| `--out`             | 输出路径，默认 `<原名>_zh.txt` / `<原名>_en.txt` |
| `--direction`       | `en2zh` / `zh2en`，默认自动判断              |
| `--model`           | `nllb`（默认，质量好，4.7GB）/ `opus`（599MB，快） |
| `--beams`           | 束搜索宽度，默认 2，调 4 更准更慢                   |
| `--weights`         | 换成自己训练的权重，如 `weights/best.pt`         |

**权重在哪**：预训练权重第一次运行会自动下到 `C:\Users\26938\.cache\huggingface\hub`（走 hf-mirror 镜像），之后完全离线可用。

---

## 三、本地训练（可选，想自己训一个模型时再用）

```bash
python scripts/get_corpus.py     # 1. 下载并整理语料：data/train.en|zh、dev.en|zh
python train.py                  # 2. 训练（词表自动训练，权重存 weights/）
python evaluate.py               # 3. 在 dev 集上算 BLEU
python translate.py --weights weights/best.pt --file a.txt    # 4. 用自己训的权重翻译
```

常用参数：

| 参数                              | 默认      | 说明                       |
| ------------------------------- | ------- | ------------------------ |
| `--epochs`                      | 5       | 训练轮数                     |
| `--batch-size`                  | 32      | 批大小                      |
| `--d-model` / `--nhead`         | 256 / 4 | 模型宽度 / 注意力头数             |
| `--enc-layers` / `--dec-layers` | 3 / 3   | 编码器 / 解码器层数              |
| `--vocab-size`                  | 8000    | 词表大小（语料太小时会自动降级）         |
| `--resume`                      | 无       | 断点续训，如 `weights/last.pt` |

**CPU 训练的现实预期**：默认小模型约 10M 参数，2 万句话一轮大概十几分钟到半小时。  
小语料 + 小模型只能学会“简单句 + 常见词”，质量明显不如预训练模型 —— 它的价值是可复现、能讲清原理、完全离线。

---

## 四、模型原理（代码里的对应关系）

```
源句 "I love NLP"
  → SentencePiece 切成子词 → id 序列
  → Embedding + 正弦位置编码
  → Encoder（自注意力，每层 = 多头注意力 + FFN + 残差 + LayerNorm）
       ↓ 输出 (B, S, D) 的语义表示
  → Decoder（自注意力 + 交叉注意力看 Encoder 输出 + FFN）
  → Linear 到目标词表 → 每个位置的概率分布
  → 解码：训练用教师强制，推理用贪心 / 束搜索
```

对应文件：

- `models/transformer.py`：`TransformerMT`（结构）、`build_causal_mask`（防止偷看未来词）、`beam_search`（束搜索）
- `utils/dataset.py`：`tgt_in` / `tgt_out` 错开一位 —— 这就是“教师强制”
- `utils/engine.py`：`NoamOpt`（预热 + 衰减，Transformer 训练的标配）

---

## 五、实测记录（本机 Intel Iris Xe，纯 CPU）

| 项目              | 结果                           |
| --------------- | ---------------------------- |
| NLLB 翻译 700 字英文 | 约 25 秒                       |
| opus 翻译 700 字英文 | 约 10 秒                       |
| 预训练权重翻译样例       | 见 `samples/sample_en_zh.txt` |
| 训练→评测→推理全链路     | 合成小语料验证通过（BLEU 100，仅验证流程）    |

## 六、已知限制

- 通用翻译模型，专业领域（法律、医学）和长难句质量一般，重要内容需人工校对。
- 术语表 `GLOSSARY`（在 `translate.py` 里）只修了常见的几个 CS 词，可按格式自行添加。
- 从零训练的小模型不能替代预训练模型，别拿它的输出去交正式译文。
