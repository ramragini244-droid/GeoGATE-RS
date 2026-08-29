import os
import json
from tqdm import tqdm
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import nltk
import time

# ===== 1. 预处理 =====
chinese_flag = True   # True:中文评测, False:英文评测
# chinese_flag = False   # True:中文评测, False:英文评测

# 字幕生成评估文件（json格式）
# json_file = '/data/YHY/valid/en_caption.json'
json_file = '/data/YHY/valid//zh_caption.json'
# 图像文件夹根路径，（拼接json中Image字段为具体图像路径）
image_file_dir = '/data/YHY/valid/images'

# 评估后推理结果保存路径
if chinese_flag:
    save_result_file = "/data/YHY/valid/results_cap_eval_zh.json"
else:
    save_result_file = "/data/YHY/valid/results_cap_eval_en.json"

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

start = time.time()  # 记录开始时间

# ===== 3. 遍历推理 =====
results = []
for item in tqdm(dataset, desc="Running Inference"):
    image_file_path = os.path.join(image_file_dir, item["Image"])
    references = item["Ground truth"]
    question = item["Text"]
    if chinese_flag:
        question = question + (
            "\n要求："
            "①请用中文非常详细地描述这张遥感图像，从左到右、从上到下（左上部分，中上部分，右上部分，左中部分，中部，中右部分，左下部分，中下部分，右下部分）部分都要描述，"
            "描述要连贯、细致，不要用分点形式，而是用完整的段落或句子详细分析每个部分的特征和细节；在描述完所有区块后，再对整幅图像的整体概述，总结各区块之间的空间关系和整体特征；"
            "②描述内容要全面、条理清晰，字数不少于1000字，使读者通过文字能够清楚理解整幅图像的细节与整体结构；"
            "③请注意逻辑顺序和条理性，使读者能够通过文字清楚理解整幅图像的整体结构和细节分布。"
        )
    else:
        question = question + (
            "\nInstructions:"
            "①Please provide an extremely detailed description of this remote sensing image in English, describing each part sequentially from left to right and top to bottom "
            "(top-left, top-center, top-right, middle-left, center, middle-right, bottom-left, bottom-center, bottom-right). "
            "The description should be continuous and thorough, written in complete sentences or paragraphs, without using bullet points, and analyze the features and details of each part. "
            "After describing all parts, provide an overall summary of the entire image, highlighting the spatial relationships between parts and the overall characteristics; "
            "②The description should be comprehensive, logically organized, and at least 1000 words, allowing readers to clearly understand both the details and the overall structure of the image; "
            "③Please maintain logical order and clarity so that readers can fully comprehend the image's detailed and overall structure."
        )

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

def char_tokenize(text: str):
    return list(text.replace(" ", ""))  # 中文逐字分割

def word_tokenize(text: str):
    return text.split()  # 英文空格分词

def compute_scores(references, candidate):
    if isinstance(references, str):
        references = [references]

    if chinese_flag:
        refs_tokenized = [char_tokenize(ref) for ref in references]
        cand_tokenized = char_tokenize(candidate)
    else:
        refs_tokenized = [word_tokenize(ref) for ref in references]
        cand_tokenized = word_tokenize(candidate)
        
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
    
end = time.time()  # 记录结束时间

print(f"耗时: {end - start:.4f} 秒")

# 平均分数
avg_bleu1 = sum(bleu1_scores) / len(bleu1_scores) * 100
avg_bleu2 = sum(bleu2_scores) / len(bleu2_scores) * 100
avg_bleu4 = sum(bleu4_scores) / len(bleu4_scores) * 100
# avg_meteor = sum(meteor_scores) / len(meteor_scores) * 100
# avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores) * 100

print(f"Average BLEU-1 Score: {avg_bleu1:.4f}")
print(f"Average BLEU-2 Score: {avg_bleu2:.4f}")
print(f"Average BLEU-4 Score: {avg_bleu4:.4f}")
