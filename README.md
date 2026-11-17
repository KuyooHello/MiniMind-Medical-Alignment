# 轻量级垂直医疗大语言模型微调（NEFTune）与偏好对齐（DPO + NLL）

本项目基于 104M 参数量（`hidden_size=768, num_hidden_layers=8`）的现代 Decoder-only Transformer 架构（集成 RoPE 旋转位置编码、GQA 分组查询注意力、SwiGLU 激活函数与 RMSNorm），针对小参数模型在垂直医疗问诊场景下**易过拟合、长文本复读退化、高危用药幻觉**三大痛点进行了全流程算法改进与消融验证。

## 1. 核心代码改动与优化点

1. **SFT 阶段嵌入噪声正则化（NEFTune）**
   - 修改位置：`model/model_minimind.py` (`MiniMindModel.forward`) & `trainer/train_full_sft.py`
   - 原理与实现：在训练阶段向 Token Embedding 注入缩放因子为 $\frac{\alpha}{\sqrt{L \cdot d}}$（$\alpha=5.0$）的均匀分布随机噪声，打破小参数模型对表面话术的死记硬背；同时通过配置开关实现训练阶段隔离，确保仅在 SFT 阶段生效，避免干扰 Pretrain 与 DPO 的概率分布对比。
2. **长度归一化 DPO + NLL 混合对齐目标**
   - 修改位置：`trainer/train_dpo.py` (`dpo_loss`)
   - 原理与实现：针对 104M 小模型运行原生 DPO 时易发生的似然坍塌（Likelihood Displacement，`chosen` 与 `rejected` 概率同步下降引发输出乱码）问题，使用序列有效 Token 均值替代求和以消除 Length Bias，并叠加 $\lambda=0.1$ 权重的 `chosen` 序列负对数似然（NLL）正则项稳定生成流畅度。
3. **On-Policy 偏好数据构建与批量推理加速**
   - 修改位置：`scripts/prepare_vertical_data.py`、`scripts/build_onpolicy_dpo.py`、`scripts/eval_ppl.py`、`scripts/eval_generation.py`
   - 原理与实现：基于 `full_sft_neftune` 模型在未见医疗样本上高温采样（`T=1.0`）构造 2,494 对 On-Policy 偏好数据；通过**按提问长度分桶排序 + Left-Padding 批处理 + BF16 推理**消除 CPU Dispatch 瓶颈，采样吞吐从 `2.03 it/s` 提升至 `50+ samples/s`（提速超 25 倍）。

## 2. 四阶段消融实验结果（验证集 300 条样本）

| 实验阶段 | 模型权重 | Val Loss | Val PPL | 核心结论 |
| :--- | :--- | :---: | :---: | :--- |
| Pretrain Baseline | `pretrain_768.pth` | 3.4380 | 31.13 | 104M 通用预训练基座 |
| Full SFT (Base) | `full_sft_base_768.pth` | 2.2513 | 9.50 | 验证集 PPL 较基线大幅下降 **69.5%** |
| Full SFT + NEFTune | `full_sft_neftune_768.pth` | **2.2488** | **9.48** | PPL 进一步微降，增强开放生成泛化性 |
| **DPO + NLL ($\lambda=0.1$)** | `dpo_nll_768.pth` | 2.2797 | 9.77 | **50 样本盲测有效胜率 58.5%（24胜/9平/17负）**，无似然坍塌 |

## 3. 典型生成效果对比（完整 20 组样例见 `examples/eval_cases_sample.json`）

- **抑制复读死循环（Case #21 宫颈癌筛查）**：NEFTune SFT 输出陷入连续重复 30 次“宫颈涂片”的退化循环；DPO+NLL 恢复清晰的结构化分点说明。
- **消除药物幻觉（Case #35 泌尿与前列腺症状）**：NEFTune SFT 错误将抗过敏药“氯雷他定”作为抗生素推荐；DPO+NLL 准确提示前列腺炎可能性并建议先行尿常规检查。
- **建立临床追问习惯（Case #8 肛周不适）**：NEFTune SFT 直接给出武断病因；DPO+NLL 主动追问瘙痒、渗液、排便困难等伴随症状后再给分诊建议。

## 4. 快速复现流程

```bash
# 1. 清洗并混合垂直医疗与通用 SFT 数据
PYTHONPATH=. python scripts/prepare_vertical_data.py

# 2. 启动带 NEFTune 正则化的全量微调
cd trainer && python train_full_sft.py --data_path ../dataset/sft_vertical_mix.jsonl --neftune_alpha 5.0 --epochs 2 --batch_size 32 --learning_rate 5e-5

# 3. 批量采样构建 On-Policy 医疗 DPO 偏好集
cd .. && PYTHONPATH=. python scripts/build_onpolicy_dpo.py

# 4. 启动长度归一化 DPO + NLL 偏好对齐
cd trainer && python train_dpo.py --data_path ../dataset/dpo_vertical.jsonl --from_weight full_sft_neftune --save_weight dpo_nll --epochs 2 --batch_size 8 --learning_rate 2e-6

# 5. 运行 PPL 评测与 50 组盲测生成对比
cd .. && PYTHONPATH=. python scripts/eval_ppl.py && PYTHONPATH=. python scripts/eval_generation.py
