import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = None

def load_model():
    # ZeroGPU에서는 실제 GPU가 @spaces.GPU 함수 호출 시점에만 할당되므로
    # bitsandbytes 4bit 로딩도 이 함수 안에서 처음 호출될 때 수행해야 한다.
    global model
    if model is not None:
        return model

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, "JunHwi/Joseon-Qwen")
    return model

@spaces.GPU
def generate(prompt, max_new_tokens=200):
    model = load_model()
    model.eval()
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=False,
    ).to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
        )
    return tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

import gradio as gr

def chat(message, history):
    return generate(message)

gr.ChatInterface(chat).launch()   # Colab에서는 share=True 로 임시 공개 링크 생성