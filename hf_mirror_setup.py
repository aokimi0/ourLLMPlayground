#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hugging Face镜像站配置工具

此脚本用于帮助用户管理Hugging Face镜像站配置，提供了以下功能：
1. 验证镜像站连接状态
2. 设置环境变量
3. 更新config.json配置
4. 验证登录状态
"""

import os
import json
import argparse
import subprocess
import sys
import requests
from urllib.parse import urlparse

# 常用的镜像站点
MIRROR_SITES = {
    "官方站点": "https://huggingface.co",
    "国内镜像1 (hf-mirror.com)": "https://hf-mirror.com",
    "国内镜像2 (gitee.com)": "https://hf-api.gitee.com",
    "自定义": "custom"
}

def create_config_if_not_exists():
    """如果配置文件不存在，创建一个默认的配置文件"""
    if not os.path.exists("config.json"):
        default_config = {
            "hf_token": "",
            "hf_endpoint": "https://hf-api.gitee.com",
            "hf_home": "/root/data-tmp"
        }
        with open("config.json", "w") as f:
            json.dump(default_config, f, indent=4)
        print("已创建默认配置文件: config.json")
    return True

def load_config():
    """加载配置文件"""
    try:
        with open("config.json", "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"读取配置文件时出错: {e}")
        return {}

def save_config(config):
    """保存配置文件"""
    try:
        with open("config.json", "w") as f:
            json.dump(config, f, indent=4)
        print("配置已保存到 config.json")
        return True
    except Exception as e:
        print(f"保存配置文件时出错: {e}")
        return False

def check_mirror_status(endpoint):
    """检查镜像站点连接状态"""
    print(f"正在检查镜像站点连接状态: {endpoint}")
    try:
        # 移除API路径，只获取基础URL
        parsed_url = urlparse(endpoint)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        # 尝试连接镜像站
        response = requests.get(base_url, timeout=5)
        if response.status_code == 200:
            print(f"✅ 镜像站点连接成功: {base_url}")
            return True
        else:
            print(f"❌ 镜像站点连接失败: {base_url}, 状态码: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 镜像站点连接失败: {endpoint}, 错误: {e}")
        return False

def set_environment_variables(config):
    """设置环境变量并显示当前设置"""
    os.environ["HF_ENDPOINT"] = config.get("hf_endpoint", "")
    os.environ["HF_HOME"] = config.get("hf_home", "")
    os.environ["HF_TOKEN"] = config.get("hf_token", "")
    os.environ["HUGGINGFACE_TOKEN"] = config.get("hf_token", "")
    
    print("\n当前环境变量设置:")
    print(f"HF_ENDPOINT = {os.environ.get('HF_ENDPOINT', '未设置')}")
    print(f"HF_HOME = {os.environ.get('HF_HOME', '未设置')}")
    
    # 检查token是否设置，不显示实际token值
    if os.environ.get("HF_TOKEN", ""):
        print("HF_TOKEN = [已设置]")
    else:
        print("HF_TOKEN = [未设置]")
    
    return True

def login_to_huggingface(token=None):
    """使用huggingface-cli登录"""
    if token is None:
        config = load_config()
        token = config.get("hf_token", "")
    
    if not token:
        print("❌ 未提供Hugging Face令牌，无法登录")
        return False
    
    try:
        # 使用subprocess运行huggingface-cli login
        result = subprocess.run(
            ["huggingface-cli", "login", "--token", token],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ 成功登录到Hugging Face")
            return True
        else:
            print(f"❌ 登录失败: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ 登录过程中出错: {e}")
        return False

def list_repositories(token=None):
    """列出用户的仓库"""
    if token is None:
        config = load_config()
        token = config.get("hf_token", "")
    
    if not token:
        print("❌ 未提供Hugging Face令牌，无法获取仓库列表")
        return False
    
    try:
        from huggingface_hub import HfApi
        
        # 使用当前环境变量中设置的endpoint
        endpoint = os.environ.get("HF_ENDPOINT", None)
        api = HfApi(token=token, endpoint=endpoint)
        
        # 获取用户信息
        user_info = api.whoami()
        username = user_info.get("name", "未知用户")
        
        # 获取用户的模型仓库
        models = api.list_models(author=username)
        datasets = api.list_datasets(author=username)
        
        print(f"\n用户 {username} 的仓库列表:")
        
        if models:
            print("\n模型仓库:")
            for model in models:
                print(f" - {model.modelId}")
        else:
            print("\n无模型仓库")
            
        if datasets:
            print("\n数据集仓库:")
            for dataset in datasets:
                print(f" - {dataset.id}")
        else:
            print("\n无数据集仓库")
            
        return True
    except Exception as e:
        print(f"❌ 获取仓库列表时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def interactive_setup():
    """交互式设置Hugging Face镜像站"""
    config = load_config()
    
    # 选择镜像站点
    print("\n请选择Hugging Face镜像站点:")
    for i, (name, url) in enumerate(MIRROR_SITES.items(), 1):
        print(f"{i}. {name}: {url if url != 'custom' else '自定义URL'}")
    
    while True:
        try:
            choice = int(input("\n请输入选项序号 [1-4]: ").strip())
            if 1 <= choice <= len(MIRROR_SITES):
                break
            else:
                print(f"无效的选项，请输入 1 到 {len(MIRROR_SITES)} 之间的数字")
        except ValueError:
            print("请输入有效的数字")
    
    # 获取选择的镜像站
    selected_mirror_name = list(MIRROR_SITES.keys())[choice-1]
    selected_mirror = MIRROR_SITES[selected_mirror_name]
    
    # 如果选择自定义，需要输入URL
    if selected_mirror == "custom":
        selected_mirror = input("请输入自定义的镜像站URL: ").strip()
    
    # 设置HF_HOME
    default_hf_home = config.get("hf_home", "/root/data-tmp")
    hf_home_input = input(f"\n请输入HF_HOME路径 [默认: {default_hf_home}]: ").strip()
    hf_home = hf_home_input if hf_home_input else default_hf_home
    
    # 设置token
    default_token = config.get("hf_token", "")
    masked_token = "********" if default_token else "未设置"
    token_input = input(f"\n请输入Hugging Face令牌 [当前: {masked_token}]: ").strip()
    token = token_input if token_input else default_token
    
    # 更新配置
    config["hf_endpoint"] = selected_mirror
    config["hf_home"] = hf_home
    config["hf_token"] = token
    
    # 保存配置
    if save_config(config):
        print(f"\n已选择镜像站点: {selected_mirror_name} ({selected_mirror})")
        set_environment_variables(config)
        
        # 检查镜像站连接状态
        check_mirror_status(selected_mirror)
        
        # 尝试登录
        if token:
            login_to_huggingface(token)
        
        print("\n配置完成! 您现在可以使用Hugging Face镜像站了。")
        print("\n使用以下命令将环境变量添加到您的终端会话:")
        print(f"export HF_ENDPOINT={selected_mirror}")
        print(f"export HF_HOME={hf_home}")
        if token:
            print(f"export HF_TOKEN={token}")
            print(f"export HUGGINGFACE_TOKEN={token}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Hugging Face镜像站配置工具")
    
    # 设置子命令
    subparsers = parser.add_subparsers(dest="command", help="要执行的命令")
    
    # 创建check命令
    check_parser = subparsers.add_parser("check", help="检查镜像站连接状态")
    
    # 创建set命令
    set_parser = subparsers.add_parser("set", help="设置镜像站配置")
    set_parser.add_argument("--endpoint", type=str, help="设置HF_ENDPOINT环境变量")
    set_parser.add_argument("--home", type=str, help="设置HF_HOME环境变量")
    set_parser.add_argument("--token", type=str, help="设置Hugging Face令牌")
    
    # 创建login命令
    login_parser = subparsers.add_parser("login", help="登录到Hugging Face")
    
    # 创建list命令
    list_parser = subparsers.add_parser("list", help="列出用户的仓库")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 如果配置文件不存在，创建一个
    create_config_if_not_exists()
    
    # 加载当前配置
    config = load_config()
    
    # 根据命令执行相应的操作
    if args.command == "check":
        # 检查当前配置的镜像站连接状态
        if "hf_endpoint" in config:
            set_environment_variables(config)
            check_mirror_status(config["hf_endpoint"])
        else:
            print("❌ 未配置镜像站点，请使用 'set' 命令设置")
    
    elif args.command == "set":
        # 如果提供了参数，更新配置
        if args.endpoint or args.home or args.token:
            if args.endpoint:
                config["hf_endpoint"] = args.endpoint
            if args.home:
                config["hf_home"] = args.home
            if args.token:
                config["hf_token"] = args.token
            save_config(config)
            set_environment_variables(config)
            
            # 检查镜像站连接状态
            if "hf_endpoint" in config:
                check_mirror_status(config["hf_endpoint"])
        else:
            # 如果没有提供参数，进入交互式设置
            interactive_setup()
    
    elif args.command == "login":
        # 设置环境变量
        set_environment_variables(config)
        # 登录到Hugging Face
        login_to_huggingface()
    
    elif args.command == "list":
        # 设置环境变量
        set_environment_variables(config)
        # 列出用户的仓库
        list_repositories()
    
    else:
        # 如果没有提供命令，默认进入交互式设置
        interactive_setup()

if __name__ == "__main__":
    main() 