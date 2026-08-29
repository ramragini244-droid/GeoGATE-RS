import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer,AutoConfig

print('beging load model....')
model_file = '/home/user01/PythonProject/CPM-9G-8B/FM9G4B-V/checkpoint/FM9G4B-V'
# model_file = '/home/user01/PythonProject/CPM-9G-8B/FM9G4B-V/finetune/output'
config = AutoConfig.from_pretrained(model_file)
model = AutoModel.from_pretrained(model_file, trust_remote_code=True,attn_implementation='sdpa', torch_dtype=torch.bfloat16)
model = model.eval().cuda()
# model = model.eval().cpu
print("end load model ...") 
print('beging load autokenizer ....')
tokenizer = AutoTokenizer.from_pretrained(model_file, trust_remote_code=True)

print('end load autokenizer ....')


# # 打印所有参数名
# for name, param in model.state_dict().items():
#     print(name, param.shape)
# 保存到文件
with open("state_dict_keys.txt", "w", encoding="utf-8") as f:
    for name, param in model.state_dict().items():
        f.write(f"{name}\n")
print()