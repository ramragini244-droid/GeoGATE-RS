# 读取数据集样本
import json
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer,AutoConfig
from tqdm import tqdm  # ✅ 引入 tqdm 库


print('beging load model....')
# model_file = '/home/user01/PythonProject/CPM-9G-8B/FM9G4B-V/checkpoint/FM9G4B-V'
# model_file = '/home/user01/PythonProject/MiniCPM-o/checkpoint/OpenBMB/MiniCPM-V-2_6'

model_file = '/home/user01/PythonProject/CPM-9G-8B/FM9G4B-V/finetune/output/total'
model = AutoModel.from_pretrained(model_file, trust_remote_code=True,attn_implementation='sdpa', torch_dtype=torch.bfloat16)
model = model.eval().cuda()
print("end load model ...") 
print('beging load autokenizer ....')
tokenizer = AutoTokenizer.from_pretrained(model_file, trust_remote_code=True)
print('end load autokenizer ....')


with open("/home/user01/PythonProject/CPM-9G-8B/FM9G4B-V/eval/MME-RealWorld/MME_RealWord_converted_data.json", "r") as f:
    dataset = json.load(f)

image_dir_path = "/home/user01/PythonProject/CPM-9G-8B/FM9G4B-V/DataSet/MME-RealWorld/remote_sensing/"

eval_image_num = None  # 改为你想测试的数量，如 100
# 截断数据集（如需要）
if eval_image_num is not None:
    dataset = dataset[:eval_image_num]  # ✅ 只保留前 eval_image_num 条
    
# 遍历推理
results = []
for item in tqdm(dataset, desc="Running Inference"):  # ✅ tqdm 包裹迭代器
    image_id = item["id"]
    image_file_path = item["image"]
    references = item["conversations"][1]['content']
    question = item["conversations"][0]['content'].removeprefix("<image>")
    
    image = Image.open(image_file_path).convert('RGB')

    # print(f"question: {question}, references: {references}")

    msgs = [{'role': 'user', 'content': [image, question]}]
    
    candidate = model.chat(image=None,msgs=msgs,tokenizer=tokenizer)
    
    results.append({
        "image_id": image_id,
        "image_file_path": image_file_path,
        "candidate": candidate.strip(),
        "references": references
    })

# 保存结果
output_path = "/home/user01/PythonProject/CPM-9G-8B/FM9G4B-V/eval/MME-RealWorld/results_cap.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"✅ 推理完成，结果已保存到：{output_path}")