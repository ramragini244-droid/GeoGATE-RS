import os
import json
import re
from tqdm import tqdm
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import time

# ===== 1. 预处理 =====
# 准确率评估文件（json格式）
'''
  {
    "Image": "FAIR1M/295.tif",
    "Text": "In satellite imagery, if a certain area has a regular shape and is distinctly different from the surrounding land use types, what could be the reason?",
    "Answer choices": [
      "(A) Naturally formed forest areas.",
      "(B) Artificially constructed airport facilities.",
      "(C) Traces left by the rerouting of a river.",
      "(D) Wildlife Activity Area."
    ],
    "Ground truth": "B",
    "Task": "Complex reasoning",
    "Subtask": "Anomaly Detection and Interpretation",
    "Question id": "Complex reasoning/Anomaly Detection and Interpretation/0005"
  },
'''
json_file = '/data/YHY/valid/subset_low/en/Complex_reasoning__Anomaly_Detection_and_Interpretation.json'

# 图像文件夹根路径，（拼接json中Image字段为具体图像路径）
image_file_dir = '/data/YHY/valid/images'
# 评估后推理结果保存路径
'''
    {
    "image_id": "FAIR1M/295.tif",
    "ground_truth": "B",
    "candidate": "B"
    },
'''
save_result_file = "/data/YHY/valid/results_choice_eval.json"

# 模型路径
model_file = '/data/YHY/fm9g4bv_remote_sensing/checkpoint/FM9G4BV-RS'

with open(json_file, "r", encoding="utf-8") as f:
    dataset = json.load(f)
print(f" {os.path.basename(json_file)} 加载完成，条目数: {len(dataset)}")

dataset = dataset[:1]  # 仅用于测试，正式运行时注释掉

# ===== 2. 加载模型 =====
model = AutoModel.from_pretrained(model_file, trust_remote_code=True,
                                  attn_implementation='sdpa',
                                  torch_dtype="auto").eval().cuda()
print(" 模型加载完成")
tokenizer = AutoTokenizer.from_pretrained(model_file, trust_remote_code=True)
print(" tokenizer 加载完成")

# ===== 3. 遍历推理 =====
results = []
correct_count = 0

start = time.time()  # 记录开始时间

for item in tqdm(dataset, desc="Running Inference"):
    image_file_path = os.path.join(image_file_dir, item["Image"])
    question = item["Text"]
    choices = item["Answer choices"]
    ground_truth = item["Ground truth"].strip().upper()  # "B"

    image = Image.open(image_file_path).convert('RGB')
    # 将问题和选项拼接给模型
    choice_text = "\n".join(choices)
    # prompt = f"{question}\nOptions:\n{choice_text}\nAnswer (A/B/C/D):"
    prompt = (
    f"{question}\n"
    f"Options:\n{choice_text}\n"
    f"Answer (Please ONLY respond with a single letter: A, B, C, or D, no additional text):"
    )

    msgs = [{'role': 'user', 'content': [image, prompt]}]

    # 推理
    candidate = model.chat(image=None, msgs=msgs, tokenizer=tokenizer)

    #  使用正则提取首个有效选项字母
    candidate_text = candidate.strip().upper() if candidate else ""
    match = re.search(r'\b([A-D])\b', candidate_text)
    candidate_ans = match.group(1) if match else ""

    results.append({
        "image_id": item["Image"],
        "ground_truth": ground_truth,
        "candidate": candidate_ans
    })

    if candidate_ans == ground_truth:
        correct_count += 1


end = time.time()  # 记录结束时间

print(f"耗时: {end - start:.4f} 秒")

# ===== 4. 计算准确率 =====
accuracy = correct_count / len(dataset) * 100
print(f" 总题数: {len(dataset)}")
print(f" 正确题数: {correct_count}")
print(f" 准确率: {accuracy:.2f}%")

# ===== 5. 保存结果 =====
with open(save_result_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f" 推理结果已保存到 {save_result_file}")
