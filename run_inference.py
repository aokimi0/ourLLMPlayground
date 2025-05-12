#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简单的模型推理脚本 - 用于测试模型生成能力
"""

import os
import argparse
import torch
import json

# 设置Hugging Face镜像站环境变量（如果存在配置文件）
try:
    if os.path.exists("config.json"):
        with open("config.json", "r") as f:
            config = json.load(f)
            if "hf_endpoint" in config:
                os.environ["HF_ENDPOINT"] = config["hf_endpoint"]
                print(f"使用Hugging Face镜像站: {config['hf_endpoint']}")
            if "hf_home" in config:
                os.environ["HF_HOME"] = config["hf_home"]
                print(f"使用HF_HOME: {config['hf_home']}")
except Exception as e:
    print(f"读取配置文件时出错: {e}")

from transformers import AutoTokenizer, pipeline

def main():
    parser = argparse.ArgumentParser(description="模型推理测试")
    parser.add_argument("--model_path", type=str, default="/root/data-tmp/codeparrot-ds",
                       help="模型目录路径")
    parser.add_argument("--prompt", type=str, 
                       default="# create a scatter plot with x, y\nimport matplotlib.pyplot as plt\n",
                       help="提示文本")
    parser.add_argument("--max_length", type=int, default=100,
                       help="生成的最大长度")
    parser.add_argument("--use_mirror", action="store_true",
                       help="是否使用Hugging Face镜像站")
    args = parser.parse_args()
    
    # 如果指定了使用镜像站但环境变量未设置，设置默认值
    if args.use_mirror and "HF_ENDPOINT" not in os.environ:
        os.environ["HF_ENDPOINT"] = "https://hf-api.gitee.com"
        print(f"命令行指定使用镜像站: https://hf-api.gitee.com")
    
    # 打印目录内容
    print(f"检查模型目录 {args.model_path}:")
    if os.path.exists(args.model_path):
        files = os.listdir(args.model_path)
        print(f"目录中有 {len(files)} 个文件:")
        for f in files:
            print(f" - {f}")
    else:
        print(f"错误: 目录 {args.model_path} 不存在")
        return
    
    # 设置设备
    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU" if device == 0 else "CPU"
    print(f"使用 {device_name} 进行推理")
    
    # 创建pipeline
    try:
        print("初始化推理pipeline...")
        pipe = pipeline(
            "text-generation",
            model=args.model_path,
            tokenizer=args.model_path,
            device=device,
            max_length=args.max_length,
            num_return_sequences=1,
            truncation=True
        )
        
        # 执行推理
        print("\n提示:")
        print(args.prompt)
        print("\n生成中...")
        
        generated = pipe(args.prompt)
        
        print("\n生成的代码:")
        print("-" * 50)
        print(generated[0]["generated_text"])
        print("-" * 50)
        
    except Exception as e:
        print(f"推理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 