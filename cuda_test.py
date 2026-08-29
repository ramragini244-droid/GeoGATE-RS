import torch
print("GPU 名称:", torch.cuda.get_device_name(0)) # 例如：NVIDIA RTX 3090
print("CUDA 是否可用:", torch.cuda.is_available())
print("PyTorch 的 CUDA 版本:", torch.version.cuda)
print(torch.__version__)

import sys
print(sys.path)

from config.config import (
    my_config
)

pexels_api_key = my_config['resource']['pexels']['api_key']
print("Pexels API Key:", pexels_api_key)