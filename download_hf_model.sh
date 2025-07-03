#!/bin/bash

# 设置 Hugging Face 的用户名和 Token
HF_USERNAME="alllifealone"
HF_TOKEN="hf_GjmDIiqXvgPVEFszbAAfLpPxuxBAatXsV"

# 检查 hfd 是否已安装
if ! command -v hfd &> /dev/null
then
    echo "hfd 工具未安装，请先运行 'pip install hfd'"
    exit 1
fi

# 检查输入参数
if [ $# -lt 1 ]; then
    echo "用法: bash download_hf_model.sh <model_name>"
    echo "示例: bash download_hf_model.sh bert-base-uncased"
    exit 1
fi

MODEL_NAME=$1

# 使用 hfd 下载模型
echo "正在下载模型：$MODEL_NAME"
hfd download $MODEL_NAME --hf_username $HF_USERNAME --hf_token $HF_TOKEN

# 判断是否下载成功
if [ $? -eq 0 ]; then
    echo "✅ 模型下载成功：$MODEL_NAME"
else
    echo "❌ 模型下载失败，请检查参数是否正确！"
    exit 1
fi
