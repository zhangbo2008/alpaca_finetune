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

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"#这个库包很难装, 反复装. 网络不太好链接github可以vpn试试.
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)


def train(
    # model/data params
    base_model: str = "chainyo/alpaca-lora-7b",  # the only required argument
    data_path: str = "./alpaca_data_cleaned_test.json",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 128,  #每次128个才进行bp算法.
    micro_batch_size: int = 4,  #每次并行训练数量.
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    val_set_size: int = 2,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss#其实一般要开true.
    group_by_length: bool = True,  # faster, but produces an odd training loss curve
):
    print(
        f"Training Alpaca-LoRA model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"group_by_length: {group_by_length}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
    model = LlamaForCausalLM.from_pretrained(
        '/alpaca_model', #https://huggingface.co/chainyo/alpaca-lora-7b/tree/main 把这个文件夹都放到/alpaca_model文件夹里面.
        load_in_8bit=True,
        torch_dtype=torch.float16,
device_map=device_map,        
    )

    print('加载好了初始模型')
    print('int8模型的测试:')
    t()



    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):#把prompt进行编码.
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point) #prompt工程.
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:#如果不在inputs上训练.一般步奏这个逻辑.这个逻辑是mask了input.
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)



#=============2023-04-14,15点25
    #训练之前进行测试.
    print('peft模型的测试')
    if 1:
      def t():
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
    print('下面开始训练!!!!!!!!!!!!!!!!')
    data = load_dataset("json", data_files=data_path)

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train()

    model.save_pretrained(output_dir)

    print("\n If there's a warning about missing keys above, please disregard :)")
    print('训练完毕开始测试')
    print('重新加载模型.')
    from peft.tuners import LoraModel, PrefixEncoder, PromptEmbedding, PromptEncoder
    model=LoraModel.from_pretrained(output_dir)
    t()


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""


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
if __name__ == "__main__":
    fire.Fire(train)##    
