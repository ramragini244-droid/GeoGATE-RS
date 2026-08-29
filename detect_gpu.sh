#!/usr/bin/env bash

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
echo "GPU Detected: $GPU_NAME"

# 只匹配 RTX 50 系列（比如 RTX 5080, RTX 5090）
if echo "$GPU_NAME" | grep -Eiq "RTX ?50[0-9]{2}"; then
    echo "Detected RTX 50 系列及以上显卡，安装 requirementA.txt"
    # pip install -r requirementA.txt
else
    echo "非 RTX 50 系列显卡，安装 requirementB.txt"
    # pip install -r requirementB.txt
fi
