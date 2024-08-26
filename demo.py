import re
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
import torch
import streamlit as st
from peft import PeftModel


# 定义地址
path = './IEITYuan/Yuan2-2B-Mars-hf'

# 已训练模型路径
lora_path = './output/Yuan2.0-2B_lora_bf16/checkpoint-45'

# 定义模型数据类型
torch_dtype = torch.bfloat16 # A10
# torch_dtype = torch.float16 # P100

# 获取微调后模型，初始化tokenizer
print("Creat tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(path, add_eos_token=False, add_bos_token=False, eos_token='<eod>')
tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens=True)

print("Creat model...")
lora_model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch_dtype, trust_remote_code=True).cuda()
lora_model = PeftModel.from_pretrained(lora_model, model_id=lora_path)

template = '''
# 任务描述
假设你是一个文本学术化修改助手，能接收一段文本，并将其修改得更加学术化。

# 任务要求
输入的文本可能是一个句子，也可能是一个段落。你需要将其中的口语化或不恰当的表述修改为学术化的表述，并输出。

# 样例
输入：
我们改进了实验方法，让产量翻了一倍。
输出：
我们通过优化实验流程与技术手段，实现了产量显著提升，具体表现为产量相较于原方法增长了一倍。

# 当前文本
query

# 任务重述
请参考样例，按照任务要求，将当前文本修改得更加学术化。
'''


# 在聊天界面上显示模型的输出
st.chat_message("assistant").write(f"请输入待优化文本：")


# 如果用户在聊天输入框中输入了内容，则执行以下操作
if query := st.chat_input():

    # 在聊天界面上显示用户的输入
    st.chat_message("user").write(query)

    # 在聊天界面上显示模型的输出
    st.chat_message("assistant").write(f"正在优化文本，请稍候...")

    # 调用模型
    prompt = template.replace('query', query).strip()
    prompt += "<sep>"
    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
    outputs = lora_model.generate(inputs, do_sample=False, max_length=1024) # 设置解码方式和最大生成长度
    output = tokenizer.decode(outputs[0])
    pattern = r'<sep>(.*?)<eod>'
    response = re.findall(pattern, output, re.DOTALL)

    st.chat_message("assistant").write(response[0])
