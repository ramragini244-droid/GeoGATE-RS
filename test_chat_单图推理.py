import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer,AutoConfig

if __name__ == '__main__':
    # prompt = f"""执行目标检测任务，识别出你能识别的所有物体，并附上他们的坐标 """
    # prompt = f""" 图中有哪些物体？请给出检测框坐标。 List all the objects you see and their locations."""
    prompt = """
请检测下面一组图像中你能识别到的物体，并返回其对应的坐标信息。
只输出一个符合以下格式的 JSON 列表，格式示例如下：

[
  {
    "image_id": "image_001",
    "detections": [
      {"text": "xxx", "bbox": [100, 50, 300, 200]},
      ...
    ]
  },
  {
    "image_id": "image_002",
    "detections": [
      {"text": "xxx", "bbox": [200, 300, 400, 600]},
      ...
    ]
  }
]

其中:
- "image_id" 为每张图片的唯一标识, image_id 命名用 f"image_{idx+1:03d}"。
- "detections" 为该图片中所有检测到的物体列表。
- "bbox" 格式为 [左上角x, 左上角y, 右下角x, 右下角y]。
- 如果某张图片未检测到任何物体，"detections" 返回空列表。

无论是单张图还是多张图，都以列表形式返回，且只返回 JSON，不要输出其他任何内容。
"""
    
    print('beging load model....')
    # model_file = '/home/user01/PythonProject/CPM-9G-8B/FM9G4B-V/checkpoint/FM9G4B-V'
    model_file = '/home/user01/PythonProject/CPM-9G-8B/FM9G4B-V/finetune/output/total'
    model = AutoModel.from_pretrained(model_file, trust_remote_code=True,attn_implementation='sdpa', torch_dtype=torch.bfloat16)
    model = model.eval().cuda()
    # model = model.eval().cpu
    print("end load model ...") 
    print('beging load autokenizer ....')
    tokenizer = AutoTokenizer.from_pretrained(model_file, trust_remote_code=True)
    
    print('end load autokenizer ....')
    
    # image = Image.open('/home/user01/PythonProject/CPM-9G-8B/FM9G4B-V/DataSet/VRSBench/Images/Images_train/00075_0000.png').convert('RGB')

    image = Image.open('/home/user01/PythonProject/CPM-9G-8B/FM9G4B-V/Code/target_detection/down_load_data/VOC2007/images/000007.jpg').convert('RGB')


    msgs = [{'role': 'user', 'content': [image, prompt]}]
    res = model.chat(image=None,msgs=msgs,tokenizer=tokenizer)
    print(res)
