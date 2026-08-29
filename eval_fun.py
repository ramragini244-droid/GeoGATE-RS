import json
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoModel, AutoTokenizer,AutoConfig

# model_file = '/home/user01/PythonProject/CPM-9G-8B/FM9G4B-V/checkpoint/FM9G4B-V'
# model = AutoModel.from_pretrained(model_file, trust_remote_code=True,attn_implementation='sdpa', torch_dtype=torch.bfloat16)
# model = model.eval().cuda()
# print("end load model ...") 
# print('beging load autokenizer ....')
# tokenizer = AutoTokenizer.from_pretrained(model_file, trust_remote_code=True)
# print('end load autokenizer ....')


# def is_equivalent(candidate: str, reference: str) -> bool:
# #     prompt = f"""
# # 请判断下面两个选项是否意思相同（忽略大小写、括号、格式，只看含义是否一致）：
# # 预测答案: {candidate}
# # 参考答案: {reference}
# # 请只回复“是”或“否”。
# # """
#     prompt = f"""
# 请判断以下两个句子的含义是否一致（忽略大小写、括号、标点符号、语言风格、表达方式等，只关注实际意思是否相同），如果意思基本一致但措辞不同，也请回答“是”；如果含义不同或存在歧义，请回答“否”。：
# 预测答案：{candidate}
# 参考答案：{reference}
# 请只回答“是”或“否”。
# """
#     msgs = [{'role': 'user', 'content': [prompt]}]
#     response = model.chat(image=None, msgs=msgs, tokenizer=tokenizer)
#     # 这里根据你模型返回的格式调整，保证只返回"是"或"否"
#     return response.strip() == "是"

def is_equivalent(candidate: str, reference: str) -> bool:
    # 去除前后空格、转为小写进行比较
    return candidate.strip().lower() == reference.strip().lower()

def main(json_file_path, force_rerun=False):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    total = len(json_data)
    correct_count = 0
    error_count = 0
    unprocessed_count = 0

    for item in tqdm(json_data, desc="Running Inference"):
        # ✅ 如果不强制重新判断，且已有结果，就跳过
        if not force_rerun and "is_correct" in item:
            unprocessed_count += 1
            if item["is_correct"]:
                correct_count += 1
            else:
                error_count += 1
            continue

        candidate = item['candidate']
        reference = item['references']
        correct = is_equivalent(candidate, reference)
        item["is_correct"] = correct  # 写回结果到json数据结构中

        if correct:
            correct_count += 1
        else:
            error_count += 1
        
        print(f"{item['image_id']}: 预测={candidate}，答案={reference}，正确？{correct}")

    accuracy = correct_count / total if total > 0 else 0
    print(f"总数: {total}, 正确: {correct_count}, 错误: {error_count}, 已有结果跳过: {unprocessed_count}, 准确率: {accuracy:.4f}")

    # 写回文件，覆盖原文件或写新文件
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    json_file_path = "/home/user01/PythonProject/CPM-9G-8B/FM9G4B-V/eval/MME-RealWorld/results_cap.json"
    # json_file_path = "/home/user01/PythonProject/CPM-9G-8B/FM9G4B-V/eval/MME-RealWorld/test.json"
    force_rerun = True  # 设置为 True 会全部重新判断
    main(json_file_path, force_rerun)
