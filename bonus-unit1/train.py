from enum import Enum
from functools import partial
import pandas as pd
import torch
import json
import re

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, TaskType, PeftModel, PeftConfig

import wandb


seed = 42
set_seed(seed)

# Define model token configurations for different model families
MODEL_CONFIGS = {
    "gemma": {
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "start_turn_token": "<start_of_turn>",
        "end_turn_token": "<end_of_turn>",
        "chat_template": "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{{ '<start_of_turn>' + message['role'] + '\n' + message['content'] | trim + '<end_of_turn><eos>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"
    },
    "smollm": {
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|im_end|>",
        "start_turn_token": "<|im_start|>",
        "end_turn_token": "<|im_end|>",
        "chat_template": "{% for message in messages %}{{ '<|im_start|>' + message['role'] }}{% if message['role'] == 'system' %}{{ '\n' + message['content'] + '<|im_end|>' }}{% else %}{{ '\n' + message['content'] + '<|im_end|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    },
    "llama": {
        "bos_token": "<|begin_of_text|>",
        "eos_token": "<|eot_id|>",
        "pad_token": "<|eot_id|>",  # Using EOS as pad token is common
        "start_turn_token": "<|begin_of_turn|>",
        "end_turn_token": "<|end_of_turn|>",
        "chat_template": "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|begin_of_turn|>system\n' + message['content'] + '<|end_of_turn|>' }}{% elif message['role'] == 'user' %}{{ '<|begin_of_turn|>user\n' + message['content'] + '<|end_of_turn|>' }}{% elif message['role'] == 'assistant' %}{{ '<|begin_of_turn|>assistant\n' + message['content'] + '<|end_of_turn|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|begin_of_turn|>assistant\n' }}{% endif %}"
    },
    "default": {  # Fallback configuration
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "</s>",
        "start_turn_token": "",
        "end_turn_token": "",
        "chat_template": "{% for message in messages %}{% if message['role'] == 'system' %}{{ message['content'] }}{% elif message['role'] == 'user' %}{{ 'User: ' + message['content'] }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant: ' }}{% endif %}"
    }
}

# Function to detect model family from model name
def detect_model_family(model_name):
    model_name_lower = model_name.lower()
    
    if "gemma" in model_name_lower:
        return "gemma"
    elif "smollm" in model_name_lower:
        return "smollm"
    elif "llama" in model_name_lower:
        return "llama"
    else:
        return "default"

model_name = "google/gemma-2-2b-it"
# model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
# model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
# model_name = "meta-llama/Llama-3.2-3B-Instruct"

# Detect the model family and get its configuration
model_family = detect_model_family(model_name)
model_config = MODEL_CONFIGS[model_family]
print(f"Detected model family: {model_family}")
print(f"Using configuration: {model_config}")

# Load tokenizer with the appropriate configuration
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define special tokens for function calling
class ChatmlSpecialTokens(str, Enum):
    tools = "<tools>"
    eotools = "</tools>"
    think = "<think>"
    eothink = "</think>"
    tool_call="<tool_call>"
    eotool_call="</tool_call>"
    tool_response="<tool_response>"
    eotool_response="</tool_response>"
    
    @classmethod
    def list(cls):
        return [c.value for c in cls]

# Apply appropriate special tokens and chat template based on model family
tokenizer.chat_template = model_config["chat_template"]

# Make sure model has the required tokens
if tokenizer.pad_token is None:
    tokenizer.pad_token = model_config["pad_token"]
if tokenizer.eos_token is None:
    tokenizer.eos_token = model_config["eos_token"]
if tokenizer.bos_token is None:
    tokenizer.bos_token = model_config["bos_token"]

# Add our special function calling tokens
additional_tokens = ChatmlSpecialTokens.list()
print(f"Adding special tokens: {additional_tokens}")
special_tokens_dict = {'additional_special_tokens': additional_tokens}
tokenizer.add_special_tokens(special_tokens_dict)

dataset_name = "Jofthomas/hermes-function-calling-thinking-V1"

# Create a more flexible preprocessing function that works with different model token structures
def preprocess(sample, tokenizer, model_config):
    messages = sample["messages"]
    first_message = messages[0]

    # Extract model-specific tokens
    start_token = model_config["start_turn_token"]
    end_token = model_config["end_turn_token"]

    # Instead of adding a system message, we merge the content into the first user message
    if first_message["role"] == "system":
        system_message_content = first_message["content"]
        # Merge system content with the first user message
        thinking_instruction = "Also, before making a call to a function take the time to plan the function to take. Make that thinking process between <think>{your thoughts}</think>\n\n"
        messages[1]["content"] = system_message_content + thinking_instruction + messages[1]["content"]
        # Remove the system message from the conversation
        messages.pop(0)

    # Apply the appropriate chat template
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

dataset = load_dataset(dataset_name)
dataset = dataset.rename_column("conversations", "messages")

# Use partial to pass the model_config to the preprocess function
preprocess_with_config = partial(preprocess, tokenizer=tokenizer, model_config=model_config)
dataset = dataset.map(preprocess_with_config, remove_columns="messages")
dataset = dataset["train"].train_test_split(0.1)
print("Dataset:\n", dataset)

# Let's look at how we formatted the dataset
print("\n", dataset["train"][8]["text"])

# Sanity check
print("pad_token", tokenizer.pad_token)
print("eos_token", tokenizer.eos_token)
print("bos_token", tokenizer.bos_token)

# Load model and apply token updates
model = AutoModelForCausalLM.from_pretrained(model_name,
                                          attn_implementation='eager',
                                          device_map="auto")

# Resize token embeddings to accommodate the new special tokens
model.resize_token_embeddings(len(tokenizer))
model.to(torch.bfloat16)


from peft import LoraConfig

# TODO: Configure LoRA parameters
# r: rank dimension for LoRA update matrices (smaller = more compression)
rank_dimension = 16
# lora_alpha: scaling factor for LoRA layers (higher = stronger adaptation)
lora_alpha = 64
# lora_dropout: dropout probability for LoRA layers (helps prevent overfitting)
lora_dropout = 0.05
peft_config = LoraConfig(r=rank_dimension,
                         lora_alpha=lora_alpha,
                         lora_dropout=lora_dropout,
                         target_modules=["gate_proj","q_proj","lm_head","o_proj","k_proj","embed_tokens","down_proj","up_proj","v_proj"],
                         task_type=TaskType.CAUSAL_LM)

username="Brainkite"
output_dir = f"{model_name.split('/')[-1]}-thinking-function_calling-V0" 
per_device_train_batch_size = 1
per_device_eval_batch_size = 1
gradient_accumulation_steps = 4
logging_steps = 5
learning_rate = 1e-4
max_grad_norm = 1.0
num_train_epochs=1
warmup_ratio = 0.1
lr_scheduler_type = "cosine"
max_seq_length = 1500
evaluation_steps = 100

training_arguments = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    save_strategy="no",
    eval_strategy="steps",
    eval_steps=evaluation_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    max_grad_norm=max_grad_norm,
    weight_decay=0.1,
    warmup_ratio=warmup_ratio,
    lr_scheduler_type=lr_scheduler_type,
    report_to=["wandb","tensorboard"],
    bf16=True,
    hub_private_repo=False,
    push_to_hub=False,
    num_train_epochs=num_train_epochs,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    packing=True,
    max_seq_length=max_seq_length,
)

wandb.init(
    project="function-calling-sft",
    name=output_dir,
    config=training_arguments.to_dict()
)

trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    peft_config=peft_config,
)

trainer.train()
trainer.save_model()

trainer.push_to_hub(f"{username}/{output_dir}")

# Set appropriate eos_token before pushing tokenizer
tokenizer.eos_token = model_config["eos_token"]
tokenizer.push_to_hub(f"{username}/{output_dir}", token=True)

## Validate model outputs

# Clear memory before loading validation model
del model
del trainer
import gc
gc.collect()
torch.cuda.empty_cache()  # Clear CUDA cache

# Load the model for validation
bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

peft_model_id = f"{username}/{output_dir}"
device = "auto"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,
                                             device_map="auto",
                                             )
tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
model.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(model, peft_model_id)
model.to(torch.bfloat16)
model.eval()

# Adjust test prompt based on model family
def create_test_prompt(model_family):
    base_content = "You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags.You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.Here are the available tools:<tools> [{'type': 'function', 'function': {'name': 'convert_currency', 'description': 'Convert from one currency to another', 'parameters': {'type': 'object', 'properties': {'amount': {'type': 'number', 'description': 'The amount to convert'}, 'from_currency': {'type': 'string', 'description': 'The currency to convert from'}, 'to_currency': {'type': 'string', 'description': 'The currency to convert to'}}, 'required': ['amount', 'from_currency', 'to_currency']}}}, {'type': 'function', 'function': {'name': 'calculate_distance', 'description': 'Calculate the distance between two locations', 'parameters': {'type': 'object', 'properties': {'start_location': {'type': 'string', 'description': 'The starting location'}, 'end_location': {'type': 'string', 'description': 'The ending location'}}, 'required': ['start_location', 'end_location']}}}] </tools>Use the following pydantic model json schema for each tool call you will make: {'title': 'FunctionCall', 'type': 'object', 'properties': {'arguments': {'title': 'Arguments', 'type': 'object'}, 'name': {'title': 'Name', 'type': 'string'}}, 'required': ['arguments', 'name']}For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows: <tool_call>{tool_call}</tool_call>Also, before making a call to a function take the time to plan the function to take. Make that thinking process between <think>{your thoughts}</think>\n\nHi, I need to convert 500 USD to Euros. Can you help me with that?"
    user_message = "Hi, I need to convert 500 USD to Euros. Can you help me with that?"
    
    config = MODEL_CONFIGS[model_family]
    
    if model_family == "gemma":
        return f"{config['bos_token']}{config['start_turn_token']}human\n{base_content}{config['end_turn_token']}{config['eos_token']}\n{config['start_turn_token']}model\n<think>"
    elif model_family == "smollm":
        return f"{config['start_turn_token']}system\n{base_content.split('Hi, I need')[0]}{config['end_turn_token']}{config['start_turn_token']}user\nHi, I need to convert 500 USD to Euros. Can you help me with that?{config['end_turn_token']}{config['start_turn_token']}assistant\n<think>"
    elif model_family == "llama":
        return f"{config['bos_token']}{config['start_turn_token']}system\n{base_content.split('Hi, I need')[0]}{config['end_turn_token']}{config['start_turn_token']}user\nHi, I need to convert 500 USD to Euros. Can you help me with that?{config['end_turn_token']}{config['start_turn_token']}assistant\n<think>"
    else:
        # Default format
        return f"System: {base_content.split('Hi, I need')[0]}\nUser: Hi, I need to convert 500 USD to Euros. Can you help me with that?\nAssistant: <think>"

prompt = create_test_prompt(model_family)

inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
inputs = {k: v.to("cuda") for k,v in inputs.items()}
outputs = model.generate(**inputs,
                         max_new_tokens=300,
                         do_sample=True,
                         top_p=0.95,
                         temperature=0.01,
                         repetition_penalty=1.0,
                         eos_token_id=tokenizer.eos_token_id)
generated_text = tokenizer.decode(outputs[0])
print(generated_text)

# Log the model output to wandb
wandb.log({
    "model_output": wandb.Html(f"<pre>{generated_text}</pre>"),
    "test_prompt": wandb.Html(f"<pre>{prompt}</pre>"),
    "model_family": model_family
})

# You can also log multiple examples in a table
examples = [
    {"prompt": prompt, "output": generated_text}
]

# For a more comprehensive view, create a table with multiple test cases
examples_table = wandb.Table(columns=["prompt", "output"])
examples_table.add_data(prompt, generated_text)

wandb.log({"generation_examples": examples_table})

# Finish the wandb run
wandb.finish()