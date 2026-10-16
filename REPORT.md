# 图嵌入在语义本体图上到底有没有用

2026-10-16 · PyKEEN 1.11 · WN18RR / FB15k-237 · M5 Pro 64 GB
网页版（含图表）：https://claude.ai/artifact/AaLxK1ERkFzTmJxJapXfqA

所有数字来自 `results/*.json`，每个结论都和一个不用嵌入的基线在同一套评估协议下比较
（filtered ranking，头尾双向，并列取期望排名）。

## 结论

| 主张 | 结果 | 关键数字 |
|---|---|---|
| 嵌入能发现图搜索找不到的关系 | 成立，幅度有限 | RotatE MRR 0.487（3 seeds，sd 0.0014）vs 路径规则 0.424；差值 95% CI [+0.055, +0.070]。53% 的隐藏边无 ≤2 跳路径，规则 Hits@10 3%，RotatE 30% |
| 嵌入给出的连接建议可靠 | 只有高置信度部分可靠 | 前 10% 建议 87% 正确，全部约 51%；错误多为"语义邻域对、关系错"的近似答案 |
| 嵌入让大图查询更快 | 对相关性查询成立 | 300 万节点 3 跳 p99 1.5 s；HNSW 0.01 ms、真实嵌入召回 99.9%；训练约 48 s/百万三元组/epoch，直推式模型需重训 |
| 嵌入会自然跨本体对齐 | 不成立 | 零锚点时 Hits@1 = 随机（1/N）。30% 锚点联合训练 Hits@1 0.47，关系 11/11 对齐；结构 + 文本 0.66 |

## 实验 1：WN18RR 链接预测（`experiments/exp1_link_prediction.py`）

| 方法 | MRR | Hits@1 | Hits@10 |
|---|---|---|---|
| RotatE（seed 0/1/2） | 0.487 / 0.485 / 0.489 | 0.434 | 0.586 |
| 规则优先 → RotatE 兜底 | 0.451 | 0.411 | 0.540 |
| 路径规则（AnyBURL 风格，≤2 跳） | 0.424 | 0.405 | 0.459 |
| TransE | 0.214 | 0.012 | 0.541 |
| 流行度 | 0.026 | 0.015 | 0.045 |

- 提升集中在层级关系：hypernym 0.054 → 0.178，instance_hypernym 0.102 → 0.336，member_meronym 0.060 → 0.175，has_part 0.078 → 0.188。
  对称或可直接推导的关系（派生词、verb_group、region）规则更好。
- 可达性：路径可达的测试边（46.8%）上规则 Hits@10 0.946、RotatE 0.917；不可达的（53.2%）上规则 0.031、RotatE 0.296。
  Hits@10 互补：两者都中 43.0%，只有嵌入 15.6%，只有规则 2.9%，都没中 38.5%。
- 建议核对（隐藏集 + 完整 WordNet 3.0）：RotatE 前 10% 87.3%，前 50% 78.2%，全部 50.7%；规则前 25% 100%，全部 46.2%。
  真正"WordNet 有、WN18RR 各划分都没有"的新发现只占 0.1%，可推出的占 2.8%。

## 实验 2：规模与性能（`exp2_scaling.py`, `exp2b_ann_realistic.py`）

| 节点数 | 暴力 kNN | 2 跳 p99 | 3 跳 p50 | 3 跳 p99 | 3 跳最大覆盖 |
|---|---|---|---|---|---|
| 1 万 | 0.07 ms | 0.8 ms | 0.6 ms | 1.5 ms | 95% |
| 10 万 | 0.66 ms | 7.2 ms | 5.1 ms | 15 ms | 95% |
| 100 万 | 6.6 ms | 124 ms | 43 ms | 270 ms | 94% |
| 300 万 | 19 ms | 704 ms | 268 ms | 1,535 ms | 94% |

- HNSW：真实 RotatE 嵌入 0.009 ms、召回 99.9%；100 万聚簇向量 0.013 ms、召回 100%。纯随机高斯向量召回只有 10–47%，是 ANN 的最坏情况，不代表真实嵌入。
- 训练：6.8 万 / 13.6 万 / 27.2 万三元组每 epoch 3.4 / 6.5 / 12.7 s（线性）。1 亿实体 × 256 维复数 ≈ 205 GB 参数。
- 小图上（WN18RR）回答 (h, r, ?)：RotatE 1.7 ms，规则索引 0.19 ms。

## 实验 3：跨本体对齐（`exp3_alignment.py`）

同一张图各取 70% 的边得到 A、B，B 的实体和关系全部改名，部分实体作为已知锚点。Hits@1：

| 锚点 | WN 独立+Procrustes | WN 锚点邻域(无嵌入) | WN 联合训练 | WN 关系对齐 | FB 独立+Procrustes | FB 锚点邻域 | FB 联合训练* | FB 关系对齐 |
|---|---|---|---|---|---|---|---|---|
| 0% | 0.00003（随机） | – | – | – | 0.00007（随机） | – | – | – |
| 5% | 0.045 | 0.049 | 0.036 | 4/11 | 0.256 | 0.287 | 0.018 | 39/237 |
| 10% | 0.067 | 0.116 | 0.111 | 5/11 | 0.315 | 0.343 | 0.069 | 94/237 |
| 30% | 0.112 | 0.362 | 0.466 | 11/11 | 0.403 | 0.411 | 0.346 | 164/237 |

\* FB 联合训练只跑了 20 个 epoch，低锚点时可能欠训练。

同义词标签（A 用一个同义词、B 用另一个，WN 中 ≥2 个同义词的概念）：

| 方法 | Hits@1 @5% | Hits@1 @30% | Hits@10 @30% |
|---|---|---|---|
| 只用结构 | 0.049 | 0.502 | 0.777 |
| 字符 3-gram | 0.328 | 0.354 | 0.554 |
| MiniLM 文本嵌入 | 0.342 | 0.373 | 0.589 |
| 结构 + MiniLM | 0.383 | 0.664 | 0.918 |

## 局限

- 超参只小范围搜索；RotatE 达到论文水平（0.476），ComplEx 在 PyKEEN LCWA 设置下未能训练成功，未纳入。
- 规则只到长度 2；完整 AnyBURL 在 WN18RR 上更强，真实差距可能更小。
- 对齐的两张图来自同一源图，比真实的独立本体更相似。
- 遍历用单线程 scipy，大规模训练耗时为线性外推。
- WordNet 核对器对 also_see / similar_to / member_of_domain_usage 的映射一致率只有 53–71%。

## 复现

```bash
uv sync
uv run python experiments/exp1_link_prediction.py
uv run python experiments/exp3_alignment.py --dataset wn18rr
uv run python experiments/exp3_alignment.py --dataset fb15k237 --epochs 20 --fractions 0.05,0.1,0.3
uv run python experiments/exp2_scaling.py
uv run python experiments/exp2b_ann_realistic.py
```

`kgv/` 是共享代码：`data.py` 数据加载，`evaluate.py` 统一评估，`baselines.py` 流行度和路径规则，
`kge.py` PyKEEN 训练封装，`wordnet.py` 开放世界核对（含 WordNet 偏移量的词性消歧）。
`run_all.sh` 是本次串行跑实验用的脚本；faiss 和 torch 同进程会因 libomp 死锁，所以 ANN 部分拆成了子进程。
