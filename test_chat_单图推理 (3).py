import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer,AutoConfig

if __name__ == '__main__':
    # prompt = f"""详细描述一下这张遥感图的主要内容是什么？ """
    prompt = f"""What color is the roof of the square building on the middle right area of the picture?\n(A) Black\n(B) White\n(C) Blue\n(D) Green\n(E) This image doesn't feature the color. """
    
    print('beging load model....')
    # model_file = '/home/user01/PythonProject/CPM-9G-8B/FM9G4B-V/checkpoint/FM9G4B-V'
    # model_file = '/home/user01/PythonProject/CPM-9G-8B/FM9G4B-V/finetune/output/total'
    model_file = '/home/user01/PythonProject/CPM-9G-8B/FM9G4B-V/quantize/output/FM9G4BV_int4'
    model = AutoModel.from_pretrained(model_file, trust_remote_code=True,attn_implementation='sdpa', torch_dtype=torch.bfloat16)
    # model = model.eval().cuda()
    model = model.eval()
    # model = model.eval().cpu
    print("end load model ...") 
    print('beging load autokenizer ....')
    tokenizer = AutoTokenizer.from_pretrained(model_file, trust_remote_code=True)
    
    print('end load autokenizer ....')
    
    image = Image.open('/home/user01/PythonProject/CPM-9G-8B/FM9G4B-V/DataSet/MME-RealWorld/remote_sensing/03553_Toronto.png').convert('RGB')


    msgs = [{'role': 'user', 'content': [image, prompt]}]
    res = model.chat(image=None,msgs=msgs,tokenizer=tokenizer)
    print(res)
