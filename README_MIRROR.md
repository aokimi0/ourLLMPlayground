# Hugging Face 镜像站配置和使用指南

本指南帮助您配置和使用Hugging Face镜像站，特别是针对中国用户访问受限的情况。

## 什么是Hugging Face镜像站？

Hugging Face镜像站是Hugging Face官方站点的国内镜像，提供更快的访问速度和更稳定的连接，常用的镜像站包括：

1. **hf-mirror.com**: 社区维护的镜像站
2. **gitee.com**: 基于Gitee的Hugging Face镜像站，API地址为 https://hf-api.gitee.com

## 配置工具

我们提供了一个配置工具 `hf_mirror_setup.py` 来帮助您管理Hugging Face镜像站配置：

```bash
# 赋予执行权限
chmod +x hf_mirror_setup.py

# 直接运行进入交互式配置
./hf_mirror_setup.py

# 或者使用命令参数
./hf_mirror_setup.py set --endpoint=https://hf-api.gitee.com --home=/root/data-tmp
```

## 命令

配置工具支持以下命令：

### 检查镜像站连接状态

```bash
./hf_mirror_setup.py check
```

### 设置镜像站配置

```bash
# 使用参数设置
./hf_mirror_setup.py set --endpoint=https://hf-api.gitee.com --home=/root/data-tmp --token=你的token

# 或者进入交互式设置
./hf_mirror_setup.py set
```

### 登录到Hugging Face

```bash
./hf_mirror_setup.py login
```

### 列出您的仓库

```bash
./hf_mirror_setup.py list
```

## 环境变量说明

要使用Hugging Face镜像站，需要设置以下环境变量：

- `HF_ENDPOINT`: 镜像站的API地址，例如 https://hf-api.gitee.com
- `HF_HOME`: 缓存目录，例如 /root/data-tmp
- `HF_TOKEN` 和 `HUGGINGFACE_TOKEN`: 您的Hugging Face认证令牌

您可以在`.bashrc`或`.bash_profile`中添加以下内容以永久设置环境变量：

```bash
export HF_ENDPOINT=https://hf-api.gitee.com
export HF_HOME=/root/data-tmp
export HF_TOKEN=你的token
export HUGGINGFACE_TOKEN=你的token
```

## 训练和推理

我们的代码已经适配了镜像站的使用：

### 训练

使用`train`脚本进行训练，它会自动读取`config.json`中的镜像站配置：

```bash
./train
```

### 推理

使用`run_inference.py`进行推理：

```bash
# 默认会读取config.json中的镜像站配置
python run_inference.py --model_path=/root/data-tmp/codeparrot-ds

# 或者显式指定使用镜像站
python run_inference.py --model_path=/root/data-tmp/codeparrot-ds --use_mirror
```

### 推送模型到Hub

训练脚本会自动尝试使用`model_utils.py`中的`push_to_hub`函数将模型推送到Hugging Face Hub。这个函数已经适配了镜像站的使用。

如果遇到推送问题，可以手动推送：

```bash
cd /path/to/your/model
git init
echo '*.bin filter=lfs diff=lfs merge=lfs -text' > .gitattributes
echo '*.safetensors filter=lfs diff=lfs merge=lfs -text' >> .gitattributes
git add .gitattributes
git remote add origin https://ai.gitee.com/你的仓库名
git lfs install
git checkout -b main
git add --all
git commit -m "Add model"
git push -u origin main
```

## 故障排除

### 连接问题

如果无法连接到镜像站，请检查：

1. 网络连接是否正常
2. 输入的镜像站URL是否正确
3. 尝试使用不同的镜像站

### 认证问题

如果登录失败，请检查：

1. 令牌是否有效
2. 是否正确设置了`HF_TOKEN`和`HUGGINGFACE_TOKEN`环境变量

### 推送失败

如果推送模型到Hub失败，请尝试：

1. 使用完整的Git命令手动推送（如上所示）
2. 检查是否安装了Git LFS（`git lfs install`）
3. 确保您有推送到目标仓库的权限

## 参考资料

- [Hugging Face 官方文档](https://huggingface.co/docs)
- [hf-mirror.com 镜像站](https://hf-mirror.com)
- [Gitee Hugging Face 镜像站](https://ai.gitee.com) 