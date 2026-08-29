import os
import json
from tqdm import tqdm
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import nltk

json_file = '/home/user01/PythonProject/CPM-9G-8B/FM9G4B-V/DataSet/00test_val/results_cap_eval_zh.json'
chinese_flag = True  # 如果是中文数据集，设为True；英文数据集，设为False
# chinese_flag = False  # 如果是中文数据集，设为True；英文数据集，设为False

nltk.download('wordnet')
nltk.download('omw-1.4')
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
smoothie = SmoothingFunction().method1

with open(json_file, "r", encoding="utf-8") as f:
    results = json.load(f)
print(f" {os.path.basename(json_file)} 加载完成，条目数: {len(results)}")

def char_tokenize(text: str):
    return list(text.replace(" ", ""))  # 中文逐字分割

def word_tokenize(text: str):
    return text.split()  # 英文空格分词



def compute_scores(references, candidate):
    if isinstance(references, str):
        references = [references]
        
    # refs_tokenized = [ref.split() for ref in references]
    # cand_tokenized = candidate.split()

    if chinese_flag:
        refs_tokenized = [char_tokenize(ref) for ref in references]
        cand_tokenized = char_tokenize(candidate)
    else:
        refs_tokenized = [word_tokenize(ref) for ref in references]
        cand_tokenized = word_tokenize(candidate)
    print("##################################################################################")
    print(f"refs_tokenized: {refs_tokenized}, \n cand_tokenized: {cand_tokenized}")

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