#!/usr/bin/env bash

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
echo "GPU Detected: $GPU_NAME"

# 只匹配 RTX 50 系列（比如 RTX 5080, RTX 5090）
if [ ! -f "/root/project/install/installed.flag" ]; then
    echo "🚀 第一次启动，安装依赖..."

    if echo "$GPU_NAME" | grep -Eiq "RTX ?50[0-9]{2}"; then
        echo "✅ 检测到 RTX 50 系列及以上显卡，安装 requirements_up50.txt"
        echo "开始安装 PyTorch..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --progress-bar=on
        echo "安装 PyTorch成功..."
        echo "开始安装依赖..."
        pip install -r requirements_up50.txt
        echo "安装依赖成功..."
    else
        echo "⚠️ 非 RTX 50 系列显卡，安装 requirements_down50.txt"
        echo "开始安装依赖..."
        pip install -r requirements_down50.txt
        echo "安装依赖成功..."
    fi

    # 安装完成，打一个标记文件
    touch /root/project/install/installed.flag
else
    echo "✅ 依赖已安装，跳过安装步骤！"
fi
exec "$@"



