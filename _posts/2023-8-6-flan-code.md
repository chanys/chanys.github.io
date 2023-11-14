---
layout: post
title: Code Example on Instruction Fine-tuning of llama2-7B using LoRA
---

In earlier articles we discussed [instruction fine-tuning](https://chanys.github.io/flan), [LoRA](https://chanys.github.io/lora/) and [quantization](https://chanys.github.io/qlora/).
We now tie these concepts and show an example code where we perform instruction fine-tuning of llama2-7B using LoRA.
This was done on a A5000 GPU with 24GB of ram.

### Imports
```
import argparse
import json
from types import SimpleNamespace as Namespace
from typing import Union

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, PeftConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
```

### Instruction Fine-tuning Prompt Templates
For instruction fine-tuning, we leverage the 52K examples collected from the [Stanford Alpaca project](https://github.com/tatsu-lab/stanford_alpaca).

```
prompt_input = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    )

prompt_no_input = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)
```

### Helper Methods to Generate and Tokenize Prompts
```
def tokenize(tokenizer, prompt, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=256,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < 256
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result

def generate_prompt(instruction: str, input: Union[None, str] = None, label: Union[None, str] = None):
    if input is None or input == "":
        prompt = prompt_no_input.format(instruction=instruction)
    else:
        prompt = prompt_input.format(instruction=instruction, input=input)

    if label is not None:
        prompt = f"{prompt}{label}"

    return prompt

def generate_and_tokenize_prompt(data_point, tokenizer):
    full_prompt = generate_prompt(instruction=data_point["instruction"], input=data_point["input"], label=data_point["output"])
    full_prompt_tokenized = tokenize(tokenizer, full_prompt)

    user_prompt = generate_prompt(instruction=data_point["instruction"], input=data_point["input"])
    user_prompt_tokenized = tokenize(tokenizer, user_prompt, add_eos_token=False)
    user_prompt_len = len(user_prompt_tokenized["input_ids"])

    full_prompt_tokenized["labels"] = [-100] * user_prompt_len + full_prompt_tokenized["labels"][user_prompt_len:]
    return full_prompt_tokenized
```

### Training Method
```
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def train(params):
    # e.g. tokenizer_id = "TinyPixel/Llama-2-7B-bf16-sharded"
    tokenizer = AutoTokenizer.from_pretrained(params.tokenizer_id)
    tokenizer.pad_token_id = (0)  # just something different from eos token

    dataset = load_dataset("json", data_files="../alpaca_data.json")
    train_data = dataset['train'].shuffle().map(generate_and_tokenize_prompt, fn_kwargs={"tokenizer": tokenizer})

    data_collator = transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)

    # e.g. model_id = "TinyPixel/Llama-2-7B-bf16-sharded"
    model = AutoModelForCausalLM.from_pretrained(params.model_id, load_in_8bit=True, device_map="auto")
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=params.lora.r,
        lora_alpha=params.lora.alpha,
        target_modules=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=params.train.train_batch_size,
        gradient_accumulation_steps=params.train.gradient_accumulation_steps,
        warmup_steps=5,
        max_steps=60,
        num_train_epochs=1,
        learning_rate=3e-4,
        output_dir="outputs",
        report_to=None
    )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data,
    )

    trainer.train()
    trainer.model.save_pretrained(params.model_dir)
    tokenizer.save_pretrained(params.model_dir)
```

### Example Inference Method
```
def inference(params):
    config = PeftConfig.from_pretrained(params.model_dir)

    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, load_in_8bit=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    model = PeftModel.from_pretrained(model, params.model_dir)
    model.eval()

    tokenizer.pad_token_id = model.config.pad_token_id

    instruction = "Tell me about deep learning and transformers."
    prompt = generate_prompt(instruction=instruction, input=None)
    inputs = tokenizer(prompt, return_tensors="pt")

    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(temperature=0.1, top_p=1.0, top_k=10, num_beams=3)

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=128,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    print(output)
```

### Main Controller Method
```
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--params", required=True)
    args = parser.parse_args()

    with open(args.params, "r", encoding="utf-8") as f:
        params = json.load(f, object_hook=lambda d: Namespace(**d))

    if args.mode == "train":
        train(params)

    elif args.mode == "inference":
        inference(params)
```

