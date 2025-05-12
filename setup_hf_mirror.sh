#!/bin/bash
# Hugging Face 镜像站一键配置脚本

# 颜色设置
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # 恢复默认颜色

echo -e "${BLUE}===== Hugging Face 镜像站一键配置工具 =====${NC}"
echo "此脚本将帮助您配置 Hugging Face 镜像站，以便更快地访问模型和数据集。"
echo

# 检查是否安装了必要的工具
check_dependency() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}错误: 未找到 $1 命令。请先安装 $1${NC}"
        exit 1
    fi
}

check_dependency python3
check_dependency git
check_dependency pip

# 检查Python依赖
echo "检查Python依赖..."
pip install -q requests huggingface_hub

# 检查Git LFS
if ! command -v git-lfs &> /dev/null; then
    echo -e "${BLUE}Git LFS 未安装，正在尝试安装...${NC}"
    # 不同系统安装方法可能不同，这里提供通用方法
    if command -v apt-get &> /dev/null; then
        sudo apt-get install -y git-lfs
    elif command -v yum &> /dev/null; then
        sudo yum install -y git-lfs
    else
        echo -e "${RED}请手动安装 Git LFS，然后再运行此脚本${NC}"
        echo "安装指南: https://git-lfs.github.com/"
        exit 1
    fi
    git lfs install
fi

# 检查配置工具是否存在
if [ ! -f "hf_mirror_setup.py" ]; then
    echo -e "${RED}未找到 hf_mirror_setup.py 配置工具${NC}"
    exit 1
fi

# 确保hf_mirror_setup.py可执行
chmod +x hf_mirror_setup.py

echo
echo -e "${BLUE}镜像站选项:${NC}"
echo "1) Gitee AI 镜像 (ai.gitee.com)"
echo "2) hf-mirror 镜像 (hf-mirror.com)"
echo "3) 官方站点 (huggingface.co)"
echo "4) 交互式配置 (运行 hf_mirror_setup.py)"

read -p "请选择镜像站 [1-4]: " mirror_choice

case $mirror_choice in
    1)
        endpoint="https://hf-api.gitee.com"
        mirror_name="Gitee AI镜像"
        ;;
    2)
        endpoint="https://hf-mirror.com"
        mirror_name="hf-mirror镜像"
        ;;
    3)
        endpoint="https://huggingface.co"
        mirror_name="官方站点"
        ;;
    4)
        # 运行交互式配置
        ./hf_mirror_setup.py
        exit 0
        ;;
    *)
        echo -e "${RED}无效的选择，使用默认值 Gitee AI镜像${NC}"
        endpoint="https://hf-api.gitee.com"
        mirror_name="Gitee AI镜像"
        ;;
esac

# 设置默认缓存目录
default_home="/root/data-tmp"
read -p "请输入缓存目录路径 [默认: $default_home]: " hf_home
hf_home=${hf_home:-$default_home}

# 创建缓存目录
if [ ! -d "$hf_home" ]; then
    echo "创建缓存目录: $hf_home"
    mkdir -p "$hf_home"
fi

# 询问令牌
read -p "请输入Hugging Face令牌 (可选，按Enter跳过): " hf_token

# 使用配置工具更新设置
if [ -z "$hf_token" ]; then
    ./hf_mirror_setup.py set --endpoint="$endpoint" --home="$hf_home"
else
    ./hf_mirror_setup.py set --endpoint="$endpoint" --home="$hf_home" --token="$hf_token"
fi

# 设置环境变量并检查连接
./hf_mirror_setup.py check

# 创建或更新环境变量导出脚本
cat > hf_env.sh << EOF
#!/bin/bash
# Hugging Face 镜像站环境变量
export HF_ENDPOINT=$endpoint
export HF_HOME=$hf_home
EOF

if [ ! -z "$hf_token" ]; then
    echo "export HF_TOKEN=$hf_token" >> hf_env.sh
    echo "export HUGGINGFACE_TOKEN=$hf_token" >> hf_env.sh
fi

chmod +x hf_env.sh

echo
echo -e "${GREEN}======= 配置完成 =======${NC}"
echo "已选择: $mirror_name ($endpoint)"
echo "缓存目录: $hf_home"
echo
echo "使用以下命令加载环境变量:"
echo "  source ./hf_env.sh"
echo
echo "在新终端中，您可以运行:"
echo "  source $(pwd)/hf_env.sh"
echo
echo "建议将环境变量添加到 ~/.bashrc 中以永久生效:"
echo "  echo 'source $(pwd)/hf_env.sh' >> ~/.bashrc"
echo
echo -e "${BLUE}立即加载环境变量? [y/n]${NC} "
read -p "" load_now

if [[ $load_now == "y" || $load_now == "Y" ]]; then
    source ./hf_env.sh
    echo -e "${GREEN}环境变量已加载${NC}"
    
    # 如果有token，尝试登录
    if [ ! -z "$hf_token" ]; then
        echo -e "${BLUE}尝试登录Hugging Face...${NC}"
        ./hf_mirror_setup.py login
    fi
fi

echo
echo -e "${GREEN}现在您可以使用以下命令:${NC}"
echo " - 训练模型:  ./train"
echo " - 推理测试:  python run_inference.py --model_path=$hf_home/codeparrot-ds"
echo " - 查看日志:  ./train log"
echo " - 终止训练:  ./train kill"
echo
echo -e "${BLUE}Happy Modeling!${NC}" 