# Training a causal language model from scratch (PyTorch)

# export HF_ENDPOINT=https://hf-api.gitee.com ; export HF_HOME=/root/data-tmp ; huggingface-cli login 
# 输入token
# accelerate launch train.py
# 安装依赖和环境配置请在命令行手动完成。

import os
import json
from collections import defaultdict
from tqdm import tqdm # 如果自定义循环中使用了 tqdm
from datasets import Dataset, load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    AutoConfig,
    DataCollatorForLanguageModeling,
    Trainer, # 如果你仍计划使用 Trainer (需要 accelerate launch 或 torchrun)
    TrainingArguments, # 同上
    get_scheduler # 如果自定义循环
)
from torch.nn import CrossEntropyLoss
import torch
from torch.utils.data.dataloader import DataLoader # 如果自定义循环
from torch.optim import AdamW # 如果自定义循环
from accelerate import Accelerator # 如果自定义循环
from huggingface_hub import Repository, get_full_repo_name, login
import argparse

# 命令行参数解析
parser = argparse.ArgumentParser(description='训练代码生成模型')
parser.add_argument('--output_dir', type=str, default="codeparrot-ds", help='模型输出目录')
args = parser.parse_args()

# --- 配置读取 ---
with open("config.json", "r") as f:
    config_file_content = json.load(f)
# os.environ["HF_DATASETS_CACHE"] = "/root/data-tmp/datasets"
# os.environ["TRANSFORMERS_CACHE"] = "/root/data-tmp/models"
# os.environ["HF_DATASETS_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"  # 注释掉此行，允许tokenizer下载
os.environ["HF_ENDPOINT"] = config_file_content["hf_endpoint"]
os.environ["HF_HOME"] = config_file_content["hf_home"]
token = config_file_content["hf_token"] 
os.environ["HUGGINGFACE_TOKEN"] = token
os.environ["HF_TOKEN"] = token
os.environ["HF_API_TOKEN"] = token  

# 使用环境变量HF_HOME作为输出目录的基础路径
hf_home = os.environ["HF_HOME"]
output_dir = os.path.join(hf_home, args.output_dir)
print(f"已设置Hugging Face镜像站环境变量，模型输出路径: {output_dir}")

# --- 辅助函数 ---
def any_keyword_in_string(string, keywords):
    for keyword in keywords:
        if keyword in string:
            return True
    return False

def filter_streaming_dataset(dataset, filters): # 注意：如果数据集很大，这个函数可能依然很慢
    filtered_dict = defaultdict(list)
    total = 0
    # 使用 iter(dataset) 如果 dataset 是可迭代的流式数据集
    for sample in tqdm(iter(dataset)): # 确保 dataset 是可迭代的
        total += 1
        if any_keyword_in_string(sample["content"], filters):
            for k, v in sample.items():
                filtered_dict[k].append(v)
    if total > 0:
        print(f"{len(filtered_dict['content'])/total:.2%} of data after filtering.")
    else:
        print("No data to filter or dataset was empty.")
    return Dataset.from_dict(filtered_dict)

# --- 数据加载和预处理 ---
print("加载数据集中...")
ds_train = load_dataset(
    "huggingface-course/codeparrot-ds-train", 
    split="train"
)
ds_valid = load_dataset(
    "huggingface-course/codeparrot-ds-valid", 
    split="validation"
)

raw_datasets = DatasetDict(
    {
        "train": ds_train,
        "valid": ds_valid,
    }
)
print("原始数据集信息:")
print(raw_datasets)

# 打印一个样本看看
if len(raw_datasets["train"]) > 0:
    print("\n训练集样本示例:")
    for key in raw_datasets["train"][0]:
        content_preview = raw_datasets['train'][0][key]
        if isinstance(content_preview, str):
            content_preview = content_preview[:200]
        print(f"{key.upper()}: {content_preview}")
else:
    print("训练集为空。")


print("\n初始化 Tokenizer...")
context_length = 128
tokenizer = AutoTokenizer.from_pretrained(
    "huggingface-course/code-search-net-tokenizer"
)
tokenizer.pad_token = tokenizer.eos_token # 关键：设置 pad_token

# Tokenize 函数
def tokenize_function(element):
    outputs = tokenizer(
        element["content"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length: # 只保留固定长度的片段
            input_batch.append(input_ids)
    return {"input_ids": input_batch}

print("Tokenizing 数据集 (这可能需要一些时间)...")
tokenized_datasets = raw_datasets.map(
    tokenize_function, batched=True, remove_columns=raw_datasets["train"].column_names
)
print("Tokenized 数据集信息:")
print(tokenized_datasets)

# --- 模型配置和初始化 ---
print("\n配置和初始化模型...")
model_config = AutoConfig.from_pretrained(
    "gpt2", # 使用 gpt2 作为基础配置
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
model = GPT2LMHeadModel(model_config)
model_size = sum(t.numel() for t in model.parameters())
print(f"GPT-2 模型大小: {model_size/1000**2:.1f}M 参数")

# --- 数据整理器 ---
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# --- 选择训练方式 ---
# 方式一: 使用 Hugging Face Trainer (需要用 accelerate launch 或 torchrun 启动此脚本以实现多卡)
print("\n使用 Hugging Face Trainer 进行训练...")
training_args = TrainingArguments(
    output_dir=output_dir, # 使用命令行参数指定的输出目录
    per_device_train_batch_size=32, # 根据你的 GPU 显存调整
    per_device_eval_batch_size=32,
    eval_strategy="steps",  # 修正了这里的参数名
    eval_steps=500,
    logging_steps=100,
    gradient_accumulation_steps=8, # 根据你的 GPU 数量和 per_device_batch_size 调整
    num_train_epochs=1, # 仅为示例，可调整
    weight_decay=0.1,
    warmup_steps=500,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=500,
    save_total_limit=3,
    fp16=True, # 如果 accelerate config 中配置了 fp16
    push_to_hub=False, # gitee镜像不支持
    dataloader_num_workers=4, # 根据你的 CPU 和 IO 调整
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
)
print("开始 Trainer 训练...")
trainer.train()
print("Trainer 训练完成.")
if training_args.should_save: # trainer.save_model() 也可以
   trainer.save_model() # 保存最终模型
   tokenizer.save_pretrained(training_args.output_dir)


# 方式二: 使用 Accelerate 自定义训练循环 (推荐，更灵活，可以直接 python train.py 运行多卡)
# print("\n使用 Accelerate 自定义训练循环...")
# # --- 自定义训练循环的参数 ---
# learning_rate = 5e-4
# num_epochs = 1 # 仅为示例
# train_batch_size = 32 # 这是 per_device_batch_size
# eval_batch_size = 32
# gradient_accumulation_steps = 8 # 调整以适应你的总 batch size 需求
# output_dir_accelerate = os.path.join(hf_home, "codeparrot-ds-accelerate") # 同样使用hf_home作为基础路径
# log_interval = 100 # 每多少步打印一次日志
# eval_interval = 500 # 每多少步评估一次
# save_interval = 500 # 每多少步保存一次模型

# # 创建 DataLoader
# train_dataloader = DataLoader(
#     tokenized_datasets["train"], shuffle=True, collate_fn=data_collator, batch_size=train_batch_size
# )
# eval_dataloader = DataLoader(
#     tokenized_datasets["valid"], collate_fn=data_collator, batch_size=eval_batch_size
# )

# # 优化器
# optimizer = AdamW(model.parameters(), lr=learning_rate)

# # Accelerate 初始化
# accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, fp16=True) # 确保与 accelerate config 一致

# model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
#     model, optimizer, train_dataloader, eval_dataloader
# )

# # 学习率调度器
# num_update_steps_per_epoch = len(train_dataloader) // gradient_accumulation_steps
# num_training_steps = num_epochs * num_update_steps_per_epoch

# lr_scheduler = get_scheduler(
#     "cosine",
#     optimizer=optimizer,
#     num_warmup_steps=500, # 与 TrainingArguments 保持一致
#     num_training_steps=num_training_steps,
# )

# print("开始 Accelerate 自定义训练循环...")
# for epoch in range(num_epochs):
#     model.train()
#     total_loss = 0
#     completed_steps_in_epoch = 0
#     progress_bar = tqdm(range(num_training_steps // num_epochs), disable=not accelerator.is_local_main_process, desc=f"Epoch {epoch+1}")

#     for step, batch in enumerate(train_dataloader):
#         with accelerator.accumulate(model): # 处理梯度累积
#             outputs = model(**batch)
#             loss = outputs.loss
#             total_loss += loss.detach().float() # 累积 loss 用于日志
#             accelerator.backward(loss)
#             optimizer.step()
#             lr_scheduler.step()
#             optimizer.zero_grad()

#         if (step + 1) % gradient_accumulation_steps == 0:
#             progress_bar.update(1)
#             completed_steps_in_epoch += 1
#             current_global_step = epoch * (num_training_steps // num_epochs) + completed_steps_in_epoch

#             if current_global_step % log_interval == 0:
#                 avg_loss = total_loss / (log_interval * gradient_accumulation_steps) if step > 0 else loss.item() # 平均的是累积步内的 loss
#                 accelerator.print(f"Epoch {epoch+1}, Step {current_global_step}, LR {lr_scheduler.get_last_lr()[0]:.2e}, Loss: {avg_loss:.4f}")
#                 total_loss = 0 # 重置 loss 累加器

#             if current_global_step % eval_interval == 0:
#                 model.eval()
#                 eval_losses = []
#                 for eval_batch in eval_dataloader:
#                     with torch.no_grad():
#                         eval_outputs = model(**eval_batch)
#                     eval_losses.append(accelerator.gather(eval_outputs.loss)) # 收集所有 GPU 的 loss
                
#                 eval_loss = torch.mean(torch.cat(eval_losses))
#                 try:
#                     perplexity = torch.exp(eval_loss)
#                 except OverflowError:
#                     perplexity = float("inf")
#                 accelerator.print(f"Validation after step {current_global_step}: Loss: {eval_loss:.4f}, Perplexity: {perplexity:.2f}")
#                 model.train() # 切换回训练模式

#             if current_global_step % save_interval == 0:
#                 accelerator.wait_for_everyone()
#                 if accelerator.is_main_process:
#                     save_path = os.path.join(output_dir_accelerate, f"checkpoint-step-{current_global_step}")
#                     os.makedirs(save_path, exist_ok=True)
#                     unwrapped_model = accelerator.unwrap_model(model)
#                     unwrapped_model.save_pretrained(save_path)
#                     tokenizer.save_pretrained(save_path)
#                     accelerator.print(f"Checkpoint saved to {save_path}")
#     progress_bar.close()

# # 训练结束后保存最终模型
# accelerator.wait_for_everyone()
# if accelerator.is_main_process:
#     final_save_path = os.path.join(output_dir_accelerate, "final_model")
#     os.makedirs(final_save_path, exist_ok=True)
#     unwrapped_model = accelerator.unwrap_model(model)
#     unwrapped_model.save_pretrained(final_save_path)
#     tokenizer.save_pretrained(final_save_path)
#     accelerator.print(f"Final model saved to {final_save_path}")

# print("Accelerate 自定义训练循环完成.")


# --- 推送模型到 Hub (如果需要，确保在主进程执行) ---
print("\n准备推送模型到 Hub...")
try:
    # 导入自定义的push_to_hub函数
    from model_utils import push_to_hub
    
    repo_name_on_hub = "aokimi/codeparrot-ds" # 确保这个仓库存在或你有权限创建
    local_model_dir = training_args.output_dir # 如果使用 Trainer
    
    # 使用model_utils.py中的push_to_hub函数，它已经处理了镜像站的适配
    success = push_to_hub(
        model_path=local_model_dir,
        repo_id=repo_name_on_hub,
        token=token,
        use_mirror=True  # 启用镜像站支持
    )
    
    if success:
        print(f"模型成功推送到 {repo_name_on_hub}")
    else:
        print(f"模型推送失败，请查看上方日志")

except Exception as e:
    print(f"推送到 Hub 失败: {e}")
    import traceback
    traceback.print_exc()
    
    # 提供手动推送的命令
    print("\n如果需要手动推送，可以执行以下命令：")
    print(f"cd {local_model_dir}")
    print("git init")
    print("# 创建.gitattributes文件支持大文件")
    print("echo '*.bin filter=lfs diff=lfs merge=lfs -text' > .gitattributes")
    print("echo '*.safetensors filter=lfs diff=lfs merge=lfs -text' >> .gitattributes")
    
    # 根据镜像站配置选择正确的URL
    hf_endpoint = os.environ.get("HF_ENDPOINT", "")
    if "gitee.com" in hf_endpoint:
        print(f"git remote add origin https://gitee.com/huggingface-hub/{repo_name_on_hub}")
    else:
        print(f"git remote add origin https://huggingface.co/{repo_name_on_hub}")
        
    print("git lfs install")
    print("git lfs track '*.bin' '*.safetensors'")
    print("git add --all")
    print('git commit -m "Add model"')
    print("git push -u origin main")


# --- 推理示例 (如果需要) ---
print("\n进行推理测试...")
from transformers import pipeline
try:
    device_for_pipeline = 0 if torch.cuda.is_available() else -1 # pipeline 的 device 参数
    model_to_load_for_pipeline = training_args.output_dir # 使用训练结束后的模型
    pipe = pipeline(
        "text-generation", model=model_to_load_for_pipeline, tokenizer=model_to_load_for_pipeline, device=device_for_pipeline
    )
    txt_example = "# create a scatter plot with x, y\nimport matplotlib.pyplot as plt\n"
    generated_code = pipe(txt_example, num_return_sequences=1, max_length=50)
    print("生成的代码:")
    print(generated_code[0]["generated_text"])
except Exception as e:
    print(f"推理失败: {e}")


print("\n脚本执行完毕。")