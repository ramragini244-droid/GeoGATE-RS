import os
import json
from tqdm import tqdm
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import nltk

# ===== 1. 预处理 =====
# 字幕生成评估文件（json格式）
'''
  {
    "Question Type": "caption",
    "Image": "DOTA_v2_4096_4096/dota_v2_dota_v2_dota_v2_P3244.png",
    "Text": "Describe the image in detail.",
    "Ground truth": "This remote sensing image meticulously depicts a bustling port area, divided into nine sections arranged from left to right and top to bottom. From the top left to the top right corner, the image displays the ocean, with the top right corner specifically showing the port's water area, where several small boats and two large orange port cranes are visible.\n\nThe middle row from the center-left to the center-right continues to show the water area, with the center-right part representing the port where several ships are docked at the shore.\n\nThe bottom three sections primarily display various arrangements of containers. The left side mostly shows the maritime culture with a minor part of the port and a large parking lot. The containers in the middle section are arranged more compactly, while the bottom right corner shows part of the water area and the shore.\n\nOverall, this port area is highly busy, demonstrating a high level of organization and extensive cargo handling activities. The color and arrangement of the containers indicate a refined level of classification and management. Additionally, the port's facilities, such as cranes and the movement of trucks, highlight its function as a logistics hub.\n\nFrom complex reasoning, the efficient operation of this port is likely due to its well-organized structure and advanced logistics management system. Furthermore, the port's proximity to the water area plays a crucial role in international trade, enabling it to handle a significant volume of imports and exports. The visible number of vehicles and the dense arrangement of containers suggest that the port may continue to expand its transportation and cargo handling capacities in the future, especially in container shipping.",
    "Task": "Image caption",
    "Subtask": "Overall image caption with details",
    "Question id": "Image caption/Overall image caption with details/00013"
  },
'''
json_file = '/home/user01/PythonProject/CPM-9G-8B/FM9G4B-V/DataSet/00test_val/valid/en_caption.json'
# 图像文件夹根路径，（拼接json中Image字段为具体图像路径）
image_file_dir = '/home/user01/PythonProject/CPM-9G-8B/FM9G4B-V/DataSet/00test_val/valid/images'

# 评估后推理结果保存路径
save_result_file = "/home/user01/PythonProject/CPM-9G-8B/FM9G4B-V/DataSet/00test_val/results_cap_eval.json"

# 模型路径
model_file = '/home/user01/PythonProject/CPM-9G-8B/FM9G4B-V/checkpoint/FM9G4BV-RS'

with open(json_file, "r", encoding="utf-8") as f:
    dataset = json.load(f)
print(f" {os.path.basename(json_file)} 加载完成，条目数: {len(dataset)}")

# dataset = dataset[:10]  # 仅用于测试，正式运行时注释掉


# ===== 2. 加载模型 =====
model = AutoModel.from_pretrained(model_file, trust_remote_code=True,
                                  attn_implementation='sdpa',
                                  torch_dtype="auto").eval().cuda()
print(" 模型加载完成")
tokenizer = AutoTokenizer.from_pretrained(model_file, trust_remote_code=True)
print(" tokenizer 加载完成")

# ===== 3. 遍历推理 =====
results = []
for item in tqdm(dataset, desc="Running Inference"):
    image_file_path = os.path.join(image_file_dir, item["Image"])
    references = item["Ground truth"]
    question = item["Text"]

    image = Image.open(image_file_path).convert('RGB')
    msgs = [{'role': 'user', 'content': [image, question]}]

    # 生成答案
    candidate = model.chat(image=None, msgs=msgs, tokenizer=tokenizer)

    results.append({
        "image_id": item["Image"],
        "references": references,
        "candidate": candidate
    })

# 保存推理结果
with open(save_result_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f" 推理结果已保存到 {save_result_file}")

# ===== 4. 评估部分 =====
nltk.download('wordnet')
nltk.download('omw-1.4')
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
smoothie = SmoothingFunction().method1

def compute_scores(references, candidate):
    if isinstance(references, str):
        references = [references]
    refs_tokenized = [ref.split() for ref in references]
    cand_tokenized = candidate.split()

    bleu1 = sentence_bleu(refs_tokenized, cand_tokenized,
                         weights=(1.0, 0, 0, 0),
                         smoothing_function=smoothie)
    bleu2 = sentence_bleu(refs_tokenized, cand_tokenized,
                         weights=(0.5, 0.5, 0, 0),
                         smoothing_function=smoothie)
    bleu4 = sentence_bleu(refs_tokenized, cand_tokenized,
                         weights=(0.25, 0.25, 0.25, 0.25),
                         smoothing_function=smoothie)
    meteor = meteor_score(refs_tokenized, cand_tokenized)
    rouge_l = rouge.score(candidate, references[0])['rougeL'].fmeasure

    return bleu1, bleu2, bleu4, meteor, rouge_l

# 逐条计算
bleu1_scores, bleu2_scores, bleu4_scores = [], [], []
meteor_scores, rouge_l_scores = [], []

for res in tqdm(results, desc="Evaluating"):
    references = res["references"]
    candidate = res["candidate"]
    b1, b2, b4, meteor, rouge_l = compute_scores(references, candidate)
    bleu1_scores.append(b1)
    bleu2_scores.append(b2)
    bleu4_scores.append(b4)
    meteor_scores.append(meteor)
    rouge_l_scores.append(rouge_l)

# 平均分数
avg_bleu1 = sum(bleu1_scores) / len(bleu1_scores) * 100
avg_bleu2 = sum(bleu2_scores) / len(bleu2_scores) * 100
avg_bleu4 = sum(bleu4_scores) / len(bleu4_scores) * 100
avg_meteor = sum(meteor_scores) / len(meteor_scores) * 100
avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores) * 100

print(f"Average BLEU-1 Score: {avg_bleu1:.4f}")
print(f"Average BLEU-2 Score: {avg_bleu2:.4f}")
print(f"Average BLEU-4 Score: {avg_bleu4:.4f}")
print(f"Average METEOR Score: {avg_meteor:.4f}")
print(f"Average ROUGE-L Score: {avg_rouge_l:.4f}")
