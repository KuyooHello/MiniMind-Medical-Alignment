# 医疗数据与实验输入

本项目使用 [HuatuoGPT-sft-data-v1](https://huggingface.co/datasets/FreedomIntelligence/HuatuoGPT-sft-data-v1)，来源项目为 [HuatuoGPT](https://github.com/FreedomIntelligence/HuatuoGPT)。本地下载的数据卡标注 Apache-2.0；使用及再分发应遵循数据源许可。完整医疗语料不提交到本仓库。

## 输入文件

从数据集页面下载 `HuatuoGPT_sft_data_v1.jsonl`，保存为：

```text
dataset/raw/FreedomIntelligence/HuatuoGPT-sft-data-v1/HuatuoGPT_sft_data_v1.jsonl
```

本次原始文件 SHA-256：

```text
bfbaca0810171e8afb4b0c62163ccd0804aacb30a575c2590e6e0fd0679123c9
```

处理器支持 Huatuo 字符串问答列表、`role/content` 对话，以及 Alpaca `instruction/input/output` 格式。输出每行示例：

```json
{"conversations": [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "参考回答"}]}
```

预训练权重来自 [MiniMind PyTorch 权重仓库](https://huggingface.co/jingyaogong/minimind-3-pytorch)，放在 `out/pretrain_768.pth`。本次权重 SHA-256 为 `0191958e7a96ef7cd1e443d379dc2ebea90cfea72cc4450fbcb4a7b87ff945f6`。仓库 `model/` 中 tokenizer 的 SHA-256 为 `71f32c68cf63a15355a8fc171b7594b3d41870fe0ddb54fc6aefa55f73a4a668`。

## 清洗与划分

从按路径排序的原始文件中依次收集前 25,000 条有效医疗样本：移除空内容与完全相同的对话，保留模板化后长度 32–512 token 的样本。按 NFKC、连续空白和开头角色前缀规范化后的首个用户问题分组；保留否定信息、数值和剂量。不同答案的同一问题不会跨集合。

核心语料按问题组约 90%/5%/5% 划分，seed=42。其中 train 和 val 用于本次训练及选择，核心测试部分的 1,251 条不进入最终评测。为最终测试，排除整个核心语料的问题，从后续最先收集的 5,000 条有效候选中按首个问题去重，再用 seed=10042 固定抽取 1,000 个新问题。该测试集是同来源、受读取顺序影响的留出集，不是外部医学基准。

| 集合 | 样本数 | 首个问题组数 | 用途 |
| :--- | ---: | ---: | :--- |
| train | 22,501 | 22,470 | SFT 训练与 DPO 问题采样 |
| val | 1,248 | 1,248 | 固定前 300 条用于统一选择 |
| test | 1,000 | 1,000 | 设置确定后的最终评测 |

本次 `general_rows=0`，未混入通用 SFT 数据。准备新数据时运行：

```bash
PYTHONPATH=. python scripts/prepare_vertical_data.py --fresh-test-size 1000
PYTHONPATH=. python scripts/check_data_leakage.py
```

生成目录为 `dataset/medical/`。`manifest.json` 绑定原始文件、tokenizer、样本计数和输出哈希；数据已存在时默认拒绝覆盖。公开摘要与测试问题哈希见 [results/data_manifest.json](../results/data_manifest.json)。三个集合的规范化精确问题交集均为 0，最终测试问题与整个核心语料交集为 0。尚未执行语义近重复、患者身份分组或基座预训练去污染检查。

| 文件 | SHA-256 |
| :--- | :--- |
| train.jsonl | `95641e8106cbb467ec529e16798c490657f264df15dec8df2d564e9f8e01cad4` |
| val.jsonl | `a4993e29ad4baad646244de28efac2537351df73d01d4e5cbb16155f44032f5a` |
| test.jsonl | `38b25dd57df1bf99795148d6151cae910eff82c1e7012685f445f42890a18784` |

## 偏好数据

`scripts/build_onpolicy_dpo.py` 从 train 采样最多 2,500 个医疗问题，参考答案作为 chosen，NEFTune seed-42 模型采样作为 rejected。长度、空回答等过滤后得到 2,498 对。这一构造假设参考答案相对更好，属于弱标签；它不等价于人工偏好或专家医学答案。

`scripts/audit_preferences.py` 检查空回答、同文回答、文本损坏、缺失图片上下文、异常重复等可追溯规则。均匀随机抽样 50 对后作模型语义审阅；参考与采样身份可见，不属于盲评。对已审阅的反转、不确定或参考存在主要问题的偏好对予以删除，不翻转标签。规则与抽样审阅共删除 48 对，得到 2,450 对过滤对照数据。

- 未过滤：`dataset/dpo_medical.jsonl`，SHA-256 `e3182342cab9313b529e1fa36c76381e06c8ad3a7907cde263c888c8c15c1668`。
- 过滤后：`dataset/dpo_medical_filtered.jsonl`，SHA-256 `4bacd06a772e4a953c443215bc5a1ad47502757a64bbecb879c5020452de0577`。
- [抽样判断与依据](../results/preference_sample_review.json)、[过滤统计](../results/preference_quality.json)。

已有审阅只允许重放到相同输入哈希；重新采样得到不同数据时必须重新审阅。未被审阅的保留样本仍是弱标签，本次没有独立医学专家标签。

公开的 [20 条生成样例](../examples/eval_cases_sample.json) 在模型选择完成后固定抽样，未筛选成功案例。其 [生成设置与哈希](../results/generation_metadata.json) 包含采样参数和模型指纹；输出含明显错误，仅供实验分析。
