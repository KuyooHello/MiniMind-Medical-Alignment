# MiniMind Medical Alignment

基于 [MiniMind](https://github.com/jingyaogong/minimind) 的 **64M 中文医疗问答微调与偏好对齐实验**，围绕嵌入噪声正则化、偏好目标的长度处理和弱偏好数据质量开展对照研究。

[English](README_en.md) · [实验配置](configs/experiments.json) · [完整结果](results/experiments.json) · [数据说明](dataset/dataset.md) · [生成样例](examples/eval_cases_sample.json) · [100 对匿名评审](examples/answer_review.csv)

模型为 8 层、hidden size 768，共 **63,912,192 个独立参数**。本项目实现了 SFT、NEFTune、求和／均值 DPO 与 chosen NLL 的可切换训练，以及数据隔离、抽样审阅和统一评测。模型结构、tokenizer 和基础训练框架来自 MiniMind；NEFTune、DPO 均按原论文注明来源。

## 实验设计

| 环节 | 设置 |
| :--- | :--- |
| SFT 对照 | 普通 SFT 与 NEFTune（α=5），随机种子 42、43、44，各训练 2 epochs |
| 偏好目标消融 | 序列 log probability 求和／按回复长度取均值 × chosen NLL 关闭／λ=0.1 |
| 数据质量对照 | 在均值 DPO + NLL 上比较原始偏好数据与过滤后的数据 |
| 数据隔离 | 规范化后的首个用户问题分组，train/val/test 精确交集为 0；DPO 问题仅来自 train |
| 模型选择 | 固定代表种子 42，只按验证集回复 NLL 选择默认模型；其他 SFT 种子用于重复性分析 |
| 最终评测 | 封存的 1,000 个新问题；同时报告全对话与 assistant 回复 PPL |
| 生成质量评估 | 从测试集抽取此前未展示的 100 个问题，固定候选后进行匿名 A/B 模型评审 |

SFT 使用学习率 5e-5、batch size 32、最大长度 512。全部 DPO 从同一个 NEFTune seed-42 权重开始，使用 β=0.15、学习率 2e-6、batch size 8、2 epochs。四组目标消融共享相同数据和超参数；过滤数据对照单独改变数据。完整设置见 [配置文件](configs/experiments.json)，实际环境见 [环境记录](results/environment.json)。

## 结果

![多随机种子、DPO 消融与训练过程](images/medical_experiments.png)

PPL 越低，模型对参考文本的拟合越好；**PPL 不等于医学正确率或安全性**。回复 PPL 仅统计 assistant 回复及其结束标记，按有效 token 汇总；全对话 PPL 还包含问题与模板。所有模型评测相同的 1,000 条样本，共 277,845 个有效全对话 token、208,321 个回复 token，采用 FP32 推理、最大长度 512。

| SFT 方法 | 全对话 PPL，均值 ± 样本标准差 | 回复 PPL，均值 ± 样本标准差 |
| :--- | ---: | ---: |
| 普通 SFT，3 seeds | 8.96153 ± 0.16667 | 4.84430 ± 0.00080 |
| NEFTune，3 seeds | 8.92699 ± 0.10205 | 4.84461 ± 0.00190 |

三个随机种子中，NEFTune 的回复 PPL 两次降低、一次升高；均值与样本标准差见上表。

| 固定 seed-42 对照 | NLL 权重 | 偏好对数 | 全对话 PPL | 回复 PPL |
| :--- | ---: | ---: | ---: | ---: |
| 预训练基座 | — | — | 28.8676 | 21.6621 |
| 普通 SFT | — | — | 9.1521 | 4.8452 |
| NEFTune SFT | — | — | 8.9430 | 4.8449 |
| 求和 DPO | 0 | 2,498 | 10.6655 | 5.7613 |
| 均值 DPO | 0 | 2,498 | 683,028.1613 | 3,304,363.0095 |
| 求和 DPO + NLL | 0.1 | 2,498 | 9.8280 | 5.2950 |
| 均值 DPO + NLL | 0.1 | 2,498 | 9.3796 | 4.9911 |
| 均值 DPO + NLL，过滤数据 | 0.1 | 2,450 | 9.3737 | 4.9908 |

上述测试集结果中，均值 DPO 加入 NLL 后，回复 PPL 从 3,304,363.0095 降至 4.9911；过滤数据后为 4.9908。表中各组 DPO 的回复 PPL 均高于 NEFTune SFT 的 4.8449。

默认推理权重 `medical_model_768.pth` 由验证集选为 **NEFTune SFT，seed 42**；DPO 对照 `medical_dpo_768.pth` 为过滤数据的均值 DPO + NLL。选择记录见 [验证集结果](results/validation.json) 与 [选择依据](results/selection.json)，测试集没有参与选择。

图中的训练曲线来自实际日志中的稀疏 minibatch 统计，不代表整轮平均值；原始数值见 [training_metrics.csv](results/training_metrics.csv)。[20 条生成样例](examples/eval_cases_sample.json) 由固定随机抽样得到，完整保留错误与截断输出，未用于报告胜率。

### 匿名 A/B 评审

对比验证集已选定的 NEFTune SFT seed-42 与过滤数据的均值 DPO + NLL。排除此前展示过的问题后，从 980 个候选问题中固定抽取 100 个；抽样和生成种子均为 73。两模型使用相同生成参数：最多 384 个新 token、temperature=0.7、top-p=0.9、top-k=50、batch size=20，GPU BF16 推理。A/B 位置各占一半，先按事实与安全、相关性、完整性逐条判断，锁定全部判断后再揭示身份。

| DPO 胜 | 平局 | SFT 胜 | DPO 胜率，排除平局 | Wilson 95% 区间 | 平局计半分 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 27 | 36 | 37 | 42.19%（27/64） | 30.87%–54.39% | 45.00% |

本轮由一个具有项目上下文的模型评审者完成，不向其展示参考答案或当前样本的模型身份。36 个平局中，29 个为双方均有明显问题、6 个为质量接近、1 个为证据不足。明显问题标记为 SFT 70/100、DPO 71/100，包括事实错误、编造病情和严重偏题；这些是审阅标记，不能换算成医学正确率。相对胜出也不代表答案合格。

SFT 全部生成 EOS；DPO 有 1 条达到 384-token 上限且未生成 EOS，该条原样纳入统计，未剔除。EOS 只表示生成结束标记，不能证明内容完整。区间包含 50%，**本轮未提供 DPO 回答质量优于 SFT 的证据**，默认推理模型仍为验证集选定的 SFT。

[全部匿名答案与判分](examples/answer_review.csv) · [逐条依据及揭示后的身份](results/answer_review_judgments.json) · [统计](results/answer_review.json) · [固定方案](results/answer_review_protocol.json) · [评审来源](results/answer_review_provenance.json)

### DPO 训练诊断

[诊断配置](configs/dpo_diagnostics.json) 固定同一初始 SFT 权重、数据与超参数，覆盖四组目标消融和过滤数据对照。JSONL 日志记录每 20 步窗口均值及整轮汇总；固定探针每 100 步及每轮末评测同一批 32 对训练样本，使用 FP32、eval 模式并移除空思考标签。

记录包括两侧及参考模型的 NLL、逐 token／序列 log probability 变化、reward margin、相对偏好排序、梯度裁剪前后范数与裁剪比例，以及 teacher-forced token KL。KL 为诊断量；训练目标由偏好项与设定权重的 chosen NLL 构成。

| 目标 | Probe Chosen NLL | Probe Rejected NLL | 验证集回复 PPL |
| :--- | ---: | ---: | ---: |
| 初始 NEFTune SFT | 1.3572 | 1.0918 | 4.6800 |
| 求和 DPO | 1.5412 | 1.5774 | 5.5625 |
| 均值 DPO | 16.5532 | 23.5017 | 2,459,034.5898 |
| 求和 DPO + NLL | 1.4263 | 1.4089 | 5.1041 |
| 均值 DPO + NLL | 1.3231 | 1.2233 | 4.8059 |
| 均值 DPO + NLL，过滤数据 | 1.3259 | 1.2162 | 4.8047 |

Probe NLL 为固定训练对上的逐样本平均；验证 PPL 使用相同的 300 条验证样本，按有效回复 token 加权计算。五组运行的 Prompt、Mask、EOS 异常及非有限损失计数均为 0。纯均值组的 Probe Chosen token KL 最终为 11.9253，均值 + NLL 组为 0.0456。

[完整诊断与参数比较](results/dpo_diagnostics.json) · [逐窗口／探针指标](results/dpo_diagnostic_metrics.csv) · [诊断曲线](images/dpo_diagnostics.png)


## 数据与偏好质量

医疗语料来自 [HuatuoGPT-sft-data-v1](https://huggingface.co/datasets/FreedomIntelligence/HuatuoGPT-sft-data-v1)。核心 25,000 条样本经问题分组后，使用 train 22,501 条、val 1,248 条；另从核心语料以外的候选问题中固定抽样 1,000 条作为最终测试集。核心划分中剩余的 1,251 条不进入本次最终评测。实际未混入通用 SFT 数据。

偏好对使用训练集参考答案作 chosen、SFT 采样答案作 rejected，属于**弱标注**。全量规则检查删除 23 条依赖缺失图片的样本；在审阅前按固定种子均匀随机抽取 50 对作模型语义审阅，得到参考答案相对更好 36 对、采样答案相对更好 4 对、不确定 10 对。参考答案存在主要问题、标签反转或无法确定的已审样本予以排除，不翻转标签。规则与抽样审阅合计删除 48 对，保留 2,450 对。

[审阅判断](results/preference_sample_review.json) 和 [过滤摘要](results/preference_quality.json) 保存逐条记录与统计。

## 复现

从项目根目录运行。建议 Python 3.10，并安装与 GPU 驱动匹配的 PyTorch；本次实测环境为 RTX 4090 D，具体版本见 [environment.json](results/environment.json)。

```bash
pip install -r requirements-train.txt
```

准备以下输入，数据来源和文件格式见 [dataset/dataset.md](dataset/dataset.md)：

- 医疗 JSONL：`dataset/raw/FreedomIntelligence/HuatuoGPT-sft-data-v1/HuatuoGPT_sft_data_v1.jsonl`。
- MiniMind 预训练权重：`out/pretrain_768.pth`，来源为 [MiniMind 权重仓库](https://huggingface.co/jingyaogong/minimind-3-pytorch)。
- tokenizer：仓库内 `model/` 文件。

```bash
# 清洗、问题分组并封存核心语料以外的新测试集；已有数据时跳过
PYTHONPATH=. python scripts/prepare_vertical_data.py --fresh-test-size 1000
PYTHONPATH=. python scripts/check_data_leakage.py

# 3 个种子 × 普通 SFT / NEFTune；同时生成 seed-42 推理别名
PYTHONPATH=. python scripts/run_experiments.py --stage sft

# 仅从 train 采样，构建偏好对并检查隔离
PYTHONPATH=. python scripts/build_onpolicy_dpo.py
PYTHONPATH=. python scripts/check_data_leakage.py --dpo-path dataset/dpo_medical.jsonl

# 重放对应数据的抽样审阅，并生成过滤数据
PYTHONPATH=. python scripts/audit_preferences.py \
  --semantic-review results/preference_sample_review.json

# 4 组目标消融 + 1 组数据质量对照
PYTHONPATH=. python scripts/run_experiments.py --stage dpo

# 先验证集选择，再统一评测最终测试集
PYTHONPATH=. python scripts/evaluate_experiments.py --phase val
PYTHONPATH=. python scripts/evaluate_experiments.py --phase test
PYTHONPATH=. python scripts/eval_generation.py --num-samples 20
PYTHONPATH=. python scripts/export_results.py

# 生成报告图，可选
pip install -r requirements-report.txt
PYTHONPATH=. python scripts/plot_results.py
```

脚本自动校验数据、权重与审阅记录的一致性。若提示数据与审阅记录不匹配，需要重新审阅当前样本。规则过滤可不带 `--semantic-review` 单独执行，作为单独的数据条件记录。训练器保存运行参数与校验记录；已经完成且参数、数据、权重一致的实验会跳过。

数据、完整训练权重和运行日志保存在本地，不随 Git 提交；公开仓库包含配置、结果、摘要、图和固定生成样例。完整上游功能和可选依赖见 [README_upstream.md](README_upstream.md) 与 `requirements.txt`。

```bash
# 使用验证集选定的默认模型
python eval_llm.py --weight medical_model --hidden_size 768 \
  --num_hidden_layers 8 --max_new_tokens 220 --historys 0

# 运行数据隔离、偏好目标、回复掩码、KV Cache 与评审完整性检查
PYTHONPATH=. python -m unittest discover -s tests -v

# 使用当前 SFT / DPO 权重比较完整前向、缓存续写和贪心生成
PYTHONPATH=. python scripts/check_kv_cache.py --device cpu --dtype float32

# 记录完整诊断并重跑固定 DPO 对照，再绘制轨迹
PYTHONPATH=. python scripts/run_dpo_diagnostics.py
PYTHONPATH=. python scripts/plot_dpo_diagnostics.py
```

KV Cache 测试覆盖逐 token 续写、分块续写、左填充批量输入、缓存复用和贪心生成，并检查缓存长度及内容。实际权重的 CPU FP32 检查使用 `atol=1e-4`、`rtol=1e-5`，记录见 [kv_cache_consistency.json](results/kv_cache_consistency.json)。

诊断配置绑定本次输入文件；更换数据或初始权重时，应按实际文件更新配置中的校验记录。

本轮匿名评审使用以下流程。生成脚本不自动判分；先只读取匿名表，将逐条判断写入 `out/answer_blind_review/judgments/batch_*.json`，字段见公开的逐条记录。所有判断完成后，`lock` 保存内容校验记录，`publish` 校验锁定记录后才读取身份映射。更换问题或答案后需要重新评审，不能套用本次判断。

```bash
# 已有本轮生成结果时跳过；生成后先完成全部匿名判分
PYTHONPATH=. python scripts/prepare_answer_review.py
PYTHONPATH=. python scripts/finalize_answer_review.py lock
PYTHONPATH=. python scripts/finalize_answer_review.py publish
```

如需另行组织人工评审，`scripts/blind_review.py` 提供通用的 `prepare` / `summarize`，支持单独记录评审来源。

## 核心实现

| 文件 | 职责 |
| :--- | :--- |
| `model/model_minimind.py` | MiniMind 模型与训练态 NEFTune 嵌入噪声 |
| `trainer/train_full_sft.py` | 普通 SFT / NEFTune 开关与训练记录 |
| `trainer/dpo_diagnostics.py` | 窗口汇总、固定样本探针、数据约束与条件 token KL |
| `trainer/train_dpo.py` | 求和／均值偏好分数、独立 NLL 权重、FP32 log-softmax 与诊断指标 |
| `scripts/data_splits.py` | 问题规范化、分组划分、文件一致性与交集验证 |
| `scripts/build_onpolicy_dpo.py` | 训练集内偏好采样与批量生成 |
| `scripts/audit_preferences.py` | 规则过滤、固定抽样与审阅重放 |
| `scripts/run_experiments.py` | 执行配置中的训练矩阵并记录完成状态 |
| `scripts/run_dpo_diagnostics.py` | 固定 DPO 诊断矩阵、验证集评测与参数比较 |
| `scripts/evaluate_experiments.py` | 验证集选择、统一测试与多种子汇总 |
| `scripts/eval_ppl.py` | 全对话及 assistant 回复的 token 加权 NLL/PPL |
| `scripts/check_kv_cache.py` | 完整前向与缓存前向的 logits、缓存结构及贪心生成一致性检查 |
| `scripts/prepare_answer_review.py` | 固定候选与未展示问题，生成相同参数的匿名 A/B 表 |
| `scripts/finalize_answer_review.py` | 完整判分校验、揭示前锁定、身份解码与结果导出 |

项目用于微调与对齐研究，未进行临床验证。

## 来源与许可

- [MiniMind](https://github.com/jingyaogong/minimind)：模型结构、tokenizer 与基础训练框架；保留 [上游说明](README_upstream.md) 和 [MIT LICENSE](LICENSE)。
- [NEFTune: Noisy Embeddings Improve Instruction Finetuning](https://arxiv.org/abs/2310.05914)。
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)。
- [HuatuoGPT](https://github.com/FreedomIntelligence/HuatuoGPT) 及其 [SFT 数据集](https://huggingface.co/datasets/FreedomIntelligence/HuatuoGPT-sft-data-v1)：按数据源许可使用，数据与模型权重的使用条件分别遵循对应来源。
