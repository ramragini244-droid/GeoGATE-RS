import json
from tqdm import tqdm  # 确保已经引入 tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')  # 这个也是推荐下载的

rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# 加载推理结果文件
# with open("/home/user01/PythonProject/CPM-9G-8B/FM9G4B-V/eval/VRSBench/results_cap_minicpm.json", "r", encoding="utf-8") as f:
# with open("/home/user01/PythonProject/CPM-9G-8B/FM9G4B-V/eval/VRSBench/results_cap_origin.json", "r", encoding="utf-8") as f:
with open("/home/user01/PythonProject/CPM-9G-8B/FM9G4B-V/eval/VRSBench/results_cap_total_train.json", "r", encoding="utf-8") as f:
    results = json.load(f)

smoothie = SmoothingFunction().method1

def compute_scores(references, candidate):
    # 如果传入的是字符串，转成列表
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
    
    # METEOR
    meteor = meteor_score(refs_tokenized, cand_tokenized)

    # ROUGE-L 只和第一个参考比较，返回 fmeasure
    rouge_l = rouge.score(candidate, references[0])['rougeL'].fmeasure

    return bleu1, bleu2, bleu4, meteor, rouge_l

# 存储每条数据的三个 BLEU 分数
bleu1_scores = []
bleu2_scores = []
bleu4_scores = []
meteor_scores = []
rouge_l_scores = []

for res in tqdm(results, desc="Evaluating BLEU"):
    references = res["references"]
    candidate = res["candidate"]
    b1, b2, b4, meteor, rouge_l = compute_scores(references, candidate)

    bleu1_scores.append(b1)
    bleu2_scores.append(b2)
    bleu4_scores.append(b4)
    meteor_scores.append(meteor)
    rouge_l_scores.append(rouge_l)

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

