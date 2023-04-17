import torch
import os
import sys
from typing import List
15022
import fire
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
from transformers import GenerationConfig, LlamaTokenizer, LlamaForCausalLM

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = '/alpaca_model'
LORA_WEIGHTS = "./lora-alpaca"
tokenizer = LlamaTokenizer.from_pretrained("chainyo/alpaca-lora-7b")
model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16, # 加载半精度
        device_map={"":0}, # 指定GPU 0
    )
model.eval()
# 加载LoRA权重
model = PeftModel.from_pretrained(model, LORA_WEIGHTS, torch_dtype=torch.float16)
model.half()
import os
import sys
from typing import List
15022
import fire
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
from transformers import GenerationConfig, LlamaTokenizer, LlamaForCausalLM


def t():
        def generate_prompt2(instruction: str, input_ctxt: str = None) -> str:
            if input_ctxt:
                return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        ### Instruction:
        {instruction}

        ### Input:
        {input_ctxt}

        ### Response:"""
            else:
                return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

        ### Instruction:
        {instruction}

        ### Response:"""

        instruction = "Give three tips for staying healthy."
        input_ctxt = None  # For some tasks, you can provide an input context to help the model generate a better response.
        generation_config = GenerationConfig(
            temperature=0.2,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=128,
        )
        prompt = generate_prompt2(instruction, input_ctxt)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )

        response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        print(response)
t()
# prompt = ""
# inp = tokenizer(prompt, max_length=512, return_tensors="pt").to("cuda")
# outputs = model.generate(input_ids=inp["input_ids"], max_new_tokens=256)
# print(tokenizer.decode(outputs[0]))