import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig,BitsAndBytesConfig
from transformers import LlamaForCausalLM, LlamaTokenizer
tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf",cache_dir="./cache/")

model = LlamaForCausalLM.from_pretrained(
    "decapoda-research/llama-7b-hf",
    load_in_8bit=True,
    quantization_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=5.0),
    torch_dtype=torch.float16,
    device_map={'': 0},cache_dir="./cache/"
)
model = PeftModel.from_pretrained(
    model, "tloen/alpaca-lora-7b", torch_dtype=torch.float16,cache_dir="./cache/",device_map={'': 0}
)


def evaluate(instruction, input=None, **kwargs):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    generation_config = GenerationConfig(
        temperature=0.7,
        top_p=1.0,
        num_beams=5,
        **kwargs,
    )
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=1024,
    )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output.split("### Response:")[1].strip()

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


if __name__ == "__main__":
    # testing code for readme
    for instruction in [
        "Tell me about alpacas.",
        "Tell me about the president of Mexico in 2019.",
        "Tell me about the king of France in 2019.",
        "List all Canadian provinces in alphabetical order.",
        "Write a Python program that prints the first 10 Fibonacci numbers.",
        "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",
        "Tell me five words that rhyme with 'shock'.",
        "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
        "Count up from 1 to 500."
    ]:
        print("Instruction:", instruction)
        print("Response:", evaluate(instruction))
        print()