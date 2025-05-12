# ourLLMPlayground

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## intro

欢迎来到 [@aokimi0](https://github.com/aokimi0) 和 [@Frederick2313072](https://github.com/Frederick2313072) 的 LLM 游乐场！
我们致力于利用现有硬件资源，尽量只依赖 PyTorch,datasets,accelerate,huggingface_hub,transformers(仅基础构件)等基础开源库，手工实现 Transformer，从零构建大语言模型，涵盖预训练、指令微调（对齐）、下游任务应用等全流程。未来还计划尝试用强化学习打造推理模型。

## refs

- [bert4torch](https://github.com/Tongjilibo/bert4torch)
- [build_MiniLLM_from_scratch](https://github.com/Tongjilibo/build_MiniLLM_from_scratch)

## log

### codeparrot-ds (2025-05-12) [[model]](https://ai.gitee.com/aokimi/codeparrot-ds)

在V100-16G*8机器上复现[huggingface教程](https://huggingface.co/learn/llm-course/chapter7/6?fw=pt)

#### deps

- PyTorch
- transformers
- datasets
- accelerate
- huggingface_hub
- 其他依赖详见 requirements.txt（如有）
- 环境变量配置（详见`config.json`）：
  - Hugging Face 镜像站点：`https://hf-api.gitee.com`
  - HF_HOME：`/root/data-tmp`
  - Token 通过环境变量注入

#### train config

- **`default_config.yaml` 的 accelerate 配置**：

  | 配置项 | 说明 | 示例值 |
  |--------|------|--------|
  | compute_environment | 计算环境 | LOCAL_MACHINE |
  | debug | 是否开启调试 | false |
  | distributed_type | 分布式类型 | MULTI_GPU |
  | downcast_bf16 | 是否降精度到bf16 | 'no' |
  | dynamo_config.dynamo_backend | torch dynamo后端 | INDUCTOR |
  | enable_cpu_affinity | 是否绑定CPU亲和性 | true |
  | gpu_ids | 使用的GPU编号 | all |
  | machine_rank | 当前机器编号 | 0 |
  | main_training_function | 主训练函数名 | main |
  | mixed_precision | 混合精度类型 | fp16 |
  | num_machines | 机器总数 | 1 |
  | num_processes | 进程数 | 8 |
  | rdzv_backend | rendezvous后端 | static |
  | same_network | 是否同一网络 | true |
  | tpu_env | TPU环境变量 | [] |
  | tpu_use_cluster | 是否使用TPU集群 | false |
  | tpu_use_sudo | TPU是否用sudo | false |
  | use_cpu | 是否仅用CPU | false |

- **Trainer 训练参数**：

  | 参数名 | 值 | 说明 |
  |--------|------|--------|
  | batch_size | 32 | 每GPU批次大小 |
  | gradient_accumulation_steps | 8 | 梯度累积步数 |
  | num_train_epochs | 1 | 训练轮数 |
  | weight_decay | 0.1 | 权重衰减 |
  | warmup_steps | 500 | 预热步数 |
  | learning_rate | 5e-4 | 学习率 |
  | lr_scheduler_type | cosine | 学习率调度器 |
  | fp16 | True | 混合精度训练 |
  | eval_steps | 500 | 评估频率 |
  | logging_steps | 100 | 日志记录频率 |
  | save_steps | 500 | 保存检查点频率 |
  | save_total_limit | 3 | 最大保存检查点数 |
  | dataloader_num_workers | 4 | 数据加载线程数 |

#### quickstart

```bash
bash ./train
```

#### data prep

- 训练集：`huggingface-course/codeparrot-ds-train`
- 验证集：`huggingface-course/codeparrot-ds-valid`
- 数据字段：repo_name、path、copies、size、content、license
- 样本量：训练集 606,720，验证集 3,322
- 分词器：`AutoTokenizer`（`huggingface-course/code-search-net-tokenizer`），context_length=128，pad_token 设为 eos_token
- 仅保留长度等于 context_length 的片段以提升训练效率

#### model arch

- 基于 `GPT2LMHeadModel`，使用 `gpt2` 基础配置
- vocab_size 与 tokenizer 一致，n_ctx=128
- 模型参数量约 117M（GPT-2 小型版）

#### train proc

训练流程时间线：

1. 数据准备阶段 (约46分钟)：
   - 数据集加载：2分钟
   - Tokenizing 处理：606,720个样本，44分钟 (处理速度：~150 examples/s)
   - 数据预处理后数据集大小：训练集 ~16GB

2. 模型初始化阶段 (约5分钟)：
   - 模型加载与配置：2分钟
   - 优化器和学习率调度器初始化：3分钟

3. 训练阶段 (总计约96分钟)：
   - 总批次：8155步
   - 平均训练速度：1.42 steps/s (2901.64 samples/s)
   - 每500步评估一次：约17秒/次
   - 每500步保存检查点：约4分钟/次
   - 单epoch总训练时间：5756秒 (约96分钟)

4. 日志记录：
   - 实时输出到 `training_log_[TIMESTAMP].log`
   - 每100步记录一次训练指标
   - 支持断点续训

#### train results

1. 训练损失变化：
   - 起始损失：~3.5
   - 最终训练损失：1.64
   - 最终步的loss：1.217

2. 验证集评估：
   - 起始验证损失：~2.0
   - 最终验证损失：1.154
   - 验证损失稳定下降，无过拟合迹象

3. 优化器表现：
   - 梯度范数稳定在 0.22-0.26 之间
   - 学习率从 5e-4 平稳下降至接近 0
   - warmup阶段损失曲线平稳

4. 资源利用：
   - GPU显存利用率: 75-85%
   - 8卡V100并行训练，单卡负载均衡
   - 吞吐量: ~2.9K samples/s

5. 最终模型：
   - 保存位置：/root/data-tmp/codeparrot-ds
   - 模型大小：~450MB
   - perplexity：~3.17 (e^1.154)

整体训练过程顺利完成，无明显错误或异常。模型在代码生成任务上表现良好，训练和验证损失曲线平滑下降，验证集损失持续改善，表明模型具有良好的泛化能力。