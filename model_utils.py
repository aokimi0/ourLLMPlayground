#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型工具脚本 - 用于加载已训练的模型、推送到Hub和执行推理
"""

import os
import json
import argparse
import subprocess
import torch
import time
import sys
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel, pipeline
from huggingface_hub import Repository, HfApi

def load_model(model_path):
    """加载模型和tokenizer"""
    print(f"正在加载模型和tokenizer: {model_path}")
    
    # 检查模型目录是否存在
    if not os.path.exists(model_path):
        raise ValueError(f"模型路径不存在: {model_path}")
    
    # 列出目录内容
    print("目录内容:")
    for item in os.listdir(model_path):
        print(f" - {item}")
    
    # 检查必要文件是否存在
    required_files = ["config.json"]
    model_file_exists = any(f in os.listdir(model_path) for f in ["pytorch_model.bin", "model.safetensors"])
    
    if not model_file_exists:
        raise ValueError(f"模型文件(pytorch_model.bin或model.safetensors)不存在于: {model_path}")
    
    for req_file in required_files:
        if not os.path.exists(os.path.join(model_path, req_file)):
            raise ValueError(f"必要文件{req_file}不存在于: {model_path}")
    
    # 加载模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    
    print("模型和tokenizer加载成功!")
    return model, tokenizer

def push_to_hub(model_path, repo_id, token=None, use_mirror=True):
    """将模型推送到Hugging Face Hub或镜像站"""
    print(f"准备将模型推送到 {repo_id}")
    
    # 如果未提供token，尝试从配置中读取
    if token is None:
        try:
            with open("config.json", "r") as f:
                config = json.load(f)
                token = config.get("hf_token", None)
                # 检查是否有HF镜像站配置
                if use_mirror and "hf_endpoint" in config:
                    hf_endpoint = config.get("hf_endpoint", "https://hf-api.gitee.com")
                    os.environ["HF_ENDPOINT"] = hf_endpoint
                    print(f"使用镜像站: {hf_endpoint}")
        except Exception as e:
            print(f"无法从配置文件读取token: {e}")
    
    if token is None:
        raise ValueError("需要提供Hugging Face token才能推送到Hub")
    
    # 确定URL基础
    if use_mirror:
        # 检查环境变量
        hf_endpoint = os.environ.get("HF_ENDPOINT", "https://hf-api.gitee.com")
        print(f"使用镜像站点: {hf_endpoint}")
        # 从环境变量中提取域名部分
        if "gitee.com" in hf_endpoint:
            # 更新为Gitee AI的正确URL格式
            hub_url = "https://ai.gitee.com"
            print(f"使用Gitee AI镜像URL: {hub_url}")
        else:
            # 默认使用与API相同的域名
            from urllib.parse import urlparse
            parsed_url = urlparse(hf_endpoint)
            domain = parsed_url.netloc
            hub_url = f"https://{domain}"
        print(f"使用Hub URL: {hub_url}")
    else:
        hub_url = "https://huggingface.co"
    
    try:
        # 检查目录是否已经是git仓库
        is_git_repo = os.path.exists(os.path.join(model_path, ".git"))
        if not is_git_repo:
            print(f"正在将 {model_path} 初始化为git仓库...")
            # 初始化git仓库
            subprocess.run(["git", "init"], cwd=model_path, check=True)
            print("Git仓库初始化完成")
            
            # 添加.gitattributes文件以支持LFS
            gitattributes_content = "*.bin filter=lfs diff=lfs merge=lfs -text\n*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
            with open(os.path.join(model_path, ".gitattributes"), "w") as f:
                f.write(gitattributes_content)
            print("创建.gitattributes文件以支持LFS")
            
            # 检查git lfs是否安装
            try:
                subprocess.run(["git", "lfs", "version"], check=True, capture_output=True)
                print("Git LFS已安装")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("警告: Git LFS可能未安装，大文件可能无法正确推送")
            
            # 添加远程仓库
            remote_url = f"{hub_url}/{repo_id}"
            print(f"添加远程仓库: {remote_url}")
            subprocess.run(["git", "remote", "add", "origin", remote_url], 
                          cwd=model_path, check=True)
            
            # 配置git用户信息
            subprocess.run(["git", "config", "user.email", "user@example.com"], cwd=model_path)
            subprocess.run(["git", "config", "user.name", "HF Model Training"], cwd=model_path)
            print("Git用户信息已配置")
        else:
            print("目录已经是Git仓库")
        
        # 在异常处理前定义默认分支名
        branch_name = "main"
        
        # 跳过HF API创建仓库 (在镜像站环境下可能不生效)
        if not use_mirror:
            print("通过API确保远程仓库存在...")
            try:
                api = HfApi(token=token, endpoint=os.environ.get("HF_ENDPOINT", None))
                api.create_repo(repo_id=repo_id, exist_ok=True)
                print(f"已确保仓库 {repo_id} 存在")
            except Exception as e:
                print(f"创建仓库时出错(可能已存在): {e}")
        else:
            print("使用镜像站，跳过API创建仓库步骤")
        
        # 手动添加、提交和推送，避免使用Repository类
        print("使用命令行git操作进行推送(可能需要几分钟)...")
        # 添加所有文件
        print("添加文件到git...")
        if os.path.exists(os.path.join(model_path, ".gitattributes")):
            subprocess.run(["git", "add", ".gitattributes"], cwd=model_path, check=True)
        
        # 检查大文件
        large_files = []
        for root, _, files in os.walk(model_path):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, model_path)
                if os.path.getsize(file_path) > 10*1024*1024:  # 大于10MB
                    large_files.append(relative_path)
        
        if large_files:
            print(f"检测到{len(large_files)}个大文件，尝试使用git lfs跟踪...")
            for file in large_files:
                try:
                    subprocess.run(["git", "lfs", "track", file], cwd=model_path, check=True)
                    print(f"使用LFS跟踪: {file}")
                except subprocess.CalledProcessError:
                    print(f"LFS跟踪失败: {file}，可能无法推送大文件")
        
        # 添加所有文件
        subprocess.run(["git", "add", "--all"], cwd=model_path, check=True)
        print("文件添加完成")
        
        # 提交更改
        try:
            # 配置Git用户信息（如果未配置）
            git_config_check = subprocess.run(["git", "config", "user.name"], 
                                           cwd=model_path, capture_output=True, check=False, text=True)
            if not git_config_check.stdout.strip():
                print("配置Git用户信息...")
                subprocess.run(["git", "config", "user.email", "user@example.com"], 
                               cwd=model_path, check=False)
                subprocess.run(["git", "config", "user.name", "HF Model Training"], 
                               cwd=model_path, check=False)
            
            # 检查Git状态，看是否有更改需要提交
            status_result = subprocess.run(["git", "status", "--porcelain"], 
                                       cwd=model_path, capture_output=True, check=False, text=True)
            
            if not status_result.stdout.strip():
                print("工作区干净，没有需要提交的更改")
                # 这里不需要抛出异常，继续执行后续的分支切换逻辑
            else:
                print(f"检测到需要提交的更改: {status_result.stdout.strip()}")
                result = subprocess.run(["git", "commit", "-m", "Update model from model_utils.py"], 
                                       cwd=model_path, check=False, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("文件提交完成")
                else:
                    stderr = result.stderr.strip()
                    
                    if "nothing to commit" in stderr:
                        print("没有新的更改需要提交")
                    elif "Please tell me who you are" in stderr:
                        print("Git需要用户信息...")
                        # 更完善的Git用户配置
                        subprocess.run(["git", "config", "user.email", "user@example.com"], 
                                       cwd=model_path, check=True)
                        subprocess.run(["git", "config", "user.name", "HF Model Training"], 
                                       cwd=model_path, check=True)
                        
                        # 重试提交
                        print("重试提交...")
                        retry_result = subprocess.run(["git", "commit", "-m", "Update model from model_utils.py"], 
                                                    cwd=model_path, check=False, capture_output=True, text=True)
                        
                        if retry_result.returncode == 0:
                            print("文件提交完成")
                        else:
                            retry_stderr = retry_result.stderr.strip() if retry_result.stderr else ""
                            print(f"重试提交失败: {retry_stderr}")
                            # 不抛出异常，继续尝试后续的分支切换和推送
                    else:
                        print(f"提交文件时出错: {stderr}")
                        # 打印更详细的信息但不终止执行
                        print(f"  - 标准输出: {result.stdout.strip()}")
                        print(f"  - 错误输出: {stderr}")
                        print("  - 继续执行后续步骤...")
        except Exception as e:
            print(f"执行Git提交时发生异常: {e}")
            print("将尝试继续执行后续步骤...")
        
        # 推送到hub
        print("推送到Hub中，可能需要几分钟...")
        env = os.environ.copy()
        env["HUGGINGFACE_TOKEN"] = token
        env["HF_TOKEN"] = token
        
        # 强制使用main分支
        try:
            print("确保使用main分支...")
            # 检查当前分支
            result = subprocess.run(["git", "branch", "--show-current"], 
                                  cwd=model_path, check=False, capture_output=True, text=True)
            
            if result.returncode == 0:
                current_branch = result.stdout.strip()
                
                if current_branch == "master":
                    # 当前是master分支，创建main分支并切换
                    print("检测到master分支，尝试切换到main分支...")
                    try:
                        # 先查看是否已有main分支
                        branch_check = subprocess.run(["git", "branch"], 
                                                    cwd=model_path, check=False, capture_output=True, text=True)
                        branches = branch_check.stdout
                        
                        if "main" in branches:
                            # main分支已存在，直接切换
                            print("main分支已存在，直接切换")
                            subprocess.run(["git", "checkout", "main"], 
                                          cwd=model_path, check=False)
                        else:
                            # 创建main分支并切换
                            subprocess.run(["git", "branch", "-m", "master", "main"], 
                                          cwd=model_path, check=False)
                            print("已将master分支重命名为main")
                    except Exception as e:
                        print(f"切换分支时出错: {e}")
                        print("将尝试继续使用当前分支")
                    
                elif current_branch != "main" and current_branch:
                    # 不是main也不是master的其他分支
                    print(f"当前分支为: {current_branch}，尝试切换到main分支...")
                    try:
                        # 先查看是否已有main分支
                        branch_check = subprocess.run(["git", "branch"], 
                                                     cwd=model_path, check=False, capture_output=True, text=True)
                        branches = branch_check.stdout
                        
                        if "main" in branches:
                            # main分支已存在，直接切换
                            subprocess.run(["git", "checkout", "main"], 
                                          cwd=model_path, check=False)
                        else:
                            # 创建并切换到main分支
                            subprocess.run(["git", "checkout", "-b", "main"], 
                                          cwd=model_path, check=False)
                        print("已切换到main分支")
                    except Exception as e:
                        print(f"切换分支时出错: {e}")
                        print("将尝试继续使用当前分支")
                
                elif current_branch == "main":
                    print("当前已经是main分支")
                
                else:
                    # 未检测到有效分支名
                    print("未检测到有效分支名，尝试创建main分支...")
                    try:
                        subprocess.run(["git", "checkout", "-b", "main"], 
                                      cwd=model_path, check=False)
                        print("已创建并切换到main分支")
                    except Exception as e:
                        print(f"创建main分支时出错: {e}")
                        print("将尝试继续使用当前分支")
            else:
                print("获取分支信息失败，尝试创建main分支...")
                try:
                    subprocess.run(["git", "checkout", "-b", "main"], 
                                  cwd=model_path, check=False)
                    print("已创建并切换到main分支")
                except Exception as e:
                    print(f"创建main分支时出错: {e}")
                    print("将继续使用当前分支")
            
            # 再次检查当前分支
            result = subprocess.run(["git", "branch", "--show-current"], 
                                   cwd=model_path, check=False, capture_output=True, text=True)
            if result.returncode == 0:
                branch_name = result.stdout.strip() or "main"
            else:
                branch_name = "main"  # 默认使用main
            
            print(f"将使用分支: {branch_name}")
            
        except Exception as e:
            print(f"处理分支时出错: {e}")
            branch_name = "main"  # 默认使用main
            print(f"将使用默认分支: {branch_name}")
        
        # 执行推送操作
        try:
            print(f"执行: git push -u origin {branch_name}")
            
            # 检查远程仓库是否存在
            check_remote = subprocess.run(["git", "remote", "-v"], 
                                       cwd=model_path, check=False, capture_output=True, text=True)
            
            if "origin" not in check_remote.stdout:
                print("未找到远程仓库，尝试添加...")
                remote_url = f"{hub_url}/{repo_id}"
                subprocess.run(["git", "remote", "add", "origin", remote_url], 
                              cwd=model_path, check=False)
                print(f"已添加远程仓库: {remote_url}")
            
            # 使用一个带有进度条的推送过程
            print("开始推送，请耐心等待...")
            
            # 创建一个进度条
            progress_bar = tqdm(total=100, desc="推送进度", unit="%")
            
            # 使用Popen而不是run来实时获取输出
            process = subprocess.Popen(
                ["git", "push", "-u", "origin", branch_name],
                cwd=model_path,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # 跟踪进度
            start_time = time.time()
            timeout = 600  # 10分钟超时
            last_update = 0
            
            # 模拟阶段和进度
            stages = [
                "压缩对象", "计算对象数量", "枚举对象", 
                "检测文件变化", "发送数据", "接收索引", "解析索引"
            ]
            stage_weights = [5, 10, 15, 20, 30, 15, 5]  # 各阶段的权重
            current_stage_idx = 0
            stage_progress = 0
            
            # 读取输出并更新进度条
            while process.poll() is None:
                # 检查是否超时
                if time.time() - start_time > timeout:
                    process.terminate()
                    print("\n推送操作超时，但可能仍在后台进行")
                    break
                
                # 读取一行输出
                line = process.stdout.readline().strip()
                if line:
                    # 打印详细信息，但不干扰进度条
                    tqdm.write(f"  {line}")
                    
                    # 根据输出更新当前阶段
                    for i, stage in enumerate(stages):
                        if stage.lower() in line.lower():
                            current_stage_idx = i
                            stage_progress = 0
                    
                    # 如果发现关键词，更新进度
                    if "%" in line:
                        try:
                            # 尝试从输出中提取百分比
                            percent = int(line.split("%")[0].split()[-1])
                            stage_progress = percent
                        except:
                            # 如果无法提取，增加一定的进度
                            stage_progress = min(100, stage_progress + 5)
                
                # 计算总体进度
                completed_weights = sum(stage_weights[:current_stage_idx])
                current_weight = stage_weights[current_stage_idx]
                total_progress = completed_weights + (current_weight * stage_progress / 100)
                
                # 每秒更新一次进度条
                current_time = time.time()
                if current_time - last_update >= 1:
                    progress_bar.n = int(total_progress)
                    progress_bar.refresh()
                    last_update = current_time
                    
                    # 模拟进度
                    if stage_progress < 100:
                        stage_progress += 2
                
                # 让CPU休息一下
                time.sleep(0.1)
            
            # 获取最终返回码
            return_code = process.wait()
            
            # 完成进度条
            progress_bar.n = 100
            progress_bar.refresh()
            progress_bar.close()
            
            # 检查推送结果
            if return_code == 0:
                print(f"✅ 模型成功推送到 {repo_id}")
            else:
                print(f"❌ 推送失败，返回码: {return_code}")
                raise subprocess.CalledProcessError(return_code, ["git", "push"])
                
        except subprocess.TimeoutExpired:
            print("推送操作超时，但可能仍在后台进行")
            print("您可以手动检查推送状态: cd " + model_path + " && git status")
        except Exception as e:
            print(f"推送操作失败: {e}")
            print("您可以尝试手动推送")
        
        return True
    
    except Exception as e:
        print(f"推送到Hub失败: {e}")
        import traceback
        traceback.print_exc()
        print("\n如果需要手动推送，可以执行以下命令：")
        print(f"cd {model_path}")
        print("git init")
        print("# 创建.gitattributes文件支持大文件")
        print("echo '*.bin filter=lfs diff=lfs merge=lfs -text' > .gitattributes")
        print("echo '*.safetensors filter=lfs diff=lfs merge=lfs -text' >> .gitattributes")
        print("git add .gitattributes")
        
        if use_mirror:
            if "gitee.com" in hub_url:
                mirror_url = f"{hub_url}/{repo_id}"
                print(f"git remote add origin {mirror_url}")
            else:
                print(f"git remote add origin {hub_url}/{repo_id}")
        else:
            print(f"git remote add origin https://huggingface.co/{repo_id}")
            
        print("git lfs install")
        print("git lfs track '*.bin' '*.safetensors'")
        print("git add --all")
        print('git commit -m "Add model"')
        print(f"git push -u origin main")  # 始终使用main作为分支名
        return False

def run_inference(model_path, prompts=None):
    """使用模型进行推理"""
    if prompts is None:
        prompts = [
            "# create a scatter plot with x, y\nimport matplotlib.pyplot as plt\n",
            "def fibonacci(n):\n    ",
            "# function to calculate factorial\ndef factorial(n):\n    "
        ]
    
    print(f"使用模型 {model_path} 进行推理测试")
    
    try:
        # 检查是否有GPU
        device = 0 if torch.cuda.is_available() else -1
        device_name = "GPU" if device == 0 else "CPU"
        print(f"使用 {device_name} 进行推理")
        
        # 创建pipeline
        pipe = pipeline(
            "text-generation",
            model=model_path,
            tokenizer=model_path,
            device=device,
            max_length=100,
            num_return_sequences=1,
            truncation=True
        )
        
        # 对每个提示进行推理
        for i, prompt in enumerate(prompts):
            print(f"\n示例 {i+1}:")
            print(f"提示: {prompt[:50]}..." if len(prompt) > 50 else f"提示: {prompt}")
            
            generated = pipe(prompt)
            print("生成的代码:")
            print(generated[0]["generated_text"])
            print("-" * 50)
        
        return True
    
    except Exception as e:
        print(f"推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hub_connection(repo_id=None, token=None, use_mirror=True):
    """测试与Hugging Face Hub/镜像站的连接和仓库状态"""
    print("开始测试Hugging Face Hub/镜像站连接...")
    
    # 加载配置
    if token is None:
        try:
            with open("config.json", "r") as f:
                config = json.load(f)
                token = config.get("hf_token", None)
                # 检查是否有HF镜像站配置
                if use_mirror and "hf_endpoint" in config:
                    hf_endpoint = config.get("hf_endpoint", "https://hf-api.gitee.com")
                    os.environ["HF_ENDPOINT"] = hf_endpoint
                    print(f"使用镜像站: {hf_endpoint}")
        except Exception as e:
            print(f"无法从配置文件读取token: {e}")
    
    # 确定URL基础
    if use_mirror:
        # 检查环境变量
        hf_endpoint = os.environ.get("HF_ENDPOINT", "https://hf-api.gitee.com")
        print(f"使用镜像站点: {hf_endpoint}")
        # 从环境变量中提取域名部分
        if "gitee.com" in hf_endpoint:
            # 更新为Gitee AI的正确URL格式
            hub_url = "https://ai.gitee.com"
            print(f"使用Gitee AI镜像URL: {hub_url}")
        else:
            # 默认使用与API相同的域名
            from urllib.parse import urlparse
            parsed_url = urlparse(hf_endpoint)
            domain = parsed_url.netloc
            hub_url = f"https://{domain}"
        print(f"使用Hub URL: {hub_url}")
    else:
        hub_url = "https://huggingface.co"
    
    # 测试网络连接
    print("\n1. 测试网络连接...")
    
    # 检查Hub URL
    try:
        import requests
        from urllib.parse import urlparse
        
        parsed_url = urlparse(hub_url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        print(f"  尝试连接到 {base_url} ...")
        response = requests.get(base_url, timeout=10)
        
        if response.status_code == 200:
            print(f"  ✅ 成功连接到 {base_url}，响应状态码: {response.status_code}")
        else:
            print(f"  ❌ 连接到 {base_url} 返回异常状态码: {response.status_code}")
    except Exception as e:
        print(f"  ❌ 连接到 {base_url} 失败: {e}")
    
    # 检查API Endpoint
    try:
        api_url = hf_endpoint
        print(f"  尝试连接到API端点 {api_url} ...")
        response = requests.get(api_url, timeout=10)
        
        if response.status_code == 200:
            print(f"  ✅ 成功连接到API端点 {api_url}，响应状态码: {response.status_code}")
        else:
            print(f"  ❌ 连接到API端点 {api_url} 返回异常状态码: {response.status_code}")
    except Exception as e:
        print(f"  ❌ 连接到API端点 {api_url} 失败: {e}")
    
    # 如果提供了token，测试认证
    if token:
        print("\n2. 测试认证...")
        try:
            from huggingface_hub import HfApi
            
            api = HfApi(token=token, endpoint=os.environ.get("HF_ENDPOINT", None))
            
            # 获取用户信息
            print("  尝试获取用户信息...")
            user_info = api.whoami()
            
            if user_info:
                username = user_info.get("name", "未知用户")
                print(f"  ✅ 认证成功! 用户名: {username}")
            else:
                print("  ❌ 无法获取用户信息")
        except Exception as e:
            print(f"  ❌ 认证失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 如果提供了repo_id，测试仓库访问
    if repo_id:
        print(f"\n3. 测试仓库 {repo_id} 访问...")
        
        # 使用API检查仓库
        if token:
            try:
                from huggingface_hub import HfApi
                
                api = HfApi(token=token, endpoint=os.environ.get("HF_ENDPOINT", None))
                
                # 检查仓库是否存在
                print(f"  尝试通过API访问仓库 {repo_id}...")
                try:
                    repo_info = api.repo_info(repo_id=repo_id)
                    if repo_info:
                        print(f"  ✅ 仓库存在且可访问: {repo_id}")
                    else:
                        print(f"  ❓ 仓库信息为空: {repo_id}")
                except Exception as e:
                    print(f"  ❌ 通过API访问仓库失败: {e}")
                    
                    # 尝试创建仓库
                    if "404" in str(e):
                        print(f"  仓库不存在，尝试创建...")
                        try:
                            api.create_repo(repo_id=repo_id)
                            print(f"  ✅ 成功创建仓库: {repo_id}")
                        except Exception as create_e:
                            print(f"  ❌ 创建仓库失败: {create_e}")
            except Exception as e:
                print(f"  ❌ API访问出错: {e}")
        
        # 直接尝试访问仓库URL
        try:
            import requests
            
            repo_url = f"{hub_url}/{repo_id}"
            print(f"  尝试直接访问仓库URL: {repo_url}...")
            
            response = requests.get(repo_url, timeout=10)
            if response.status_code == 200:
                print(f"  ✅ 成功访问仓库URL: {repo_url}")
            else:
                print(f"  ❌ 访问仓库URL返回异常状态码: {response.status_code}")
                
                if response.status_code == 404:
                    print("  仓库可能不存在或您没有访问权限")
        except Exception as e:
            print(f"  ❌ 访问仓库URL失败: {e}")
    
    print("\n测试完成!")
    return True

def main():
    parser = argparse.ArgumentParser(description="模型工具 - 加载、推送和推理")
    parser.add_argument("--model_path", type=str, default="/root/data-tmp/codeparrot-ds",
                       help="模型目录路径")
    parser.add_argument("--repo_id", type=str, default="aokimi/codeparrot-ds",
                       help="Hugging Face Hub上的仓库ID")
    parser.add_argument("--token", type=str, default=None,
                       help="Hugging Face token(可选，默认从config.json读取)")
    parser.add_argument("--load", action="store_true", default=False,
                       help="加载模型并验证")
    parser.add_argument("--push", action="store_true", default=False,
                       help="推送模型到Hub")
    parser.add_argument("--infer", action="store_true", default=False,
                       help="执行推理测试")
    parser.add_argument("--test", action="store_true", default=False,
                       help="测试与Hub的连接和仓库状态")
    parser.add_argument("--all", action="store_true", default=False,
                       help="执行所有操作(加载+推送+推理)")
    
    args = parser.parse_args()
    
    # 如果没有指定任何操作，默认执行所有操作
    if not (args.load or args.push or args.infer or args.test):
        args.all = True
    
    # 执行指定操作
    if args.test:
        test_hub_connection(args.repo_id, args.token)
        return
    
    if args.load or args.all:
        model, tokenizer = load_model(args.model_path)
    
    if args.push or args.all:
        push_to_hub(args.model_path, args.repo_id, args.token)
    
    if args.infer or args.all:
        run_inference(args.model_path)

if __name__ == "__main__":
    main() 