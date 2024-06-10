from unsloth import FastLanguageModel
import torch
from datasets import load_dataset

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.



model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit", # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

action_prompt = "<s>[INST] You are assisting a user that is training as an attacker in a cyberattack simulator, you have to suggest to the user the next action to perform based on his needs and possibilities. A list of possible actions with their description is given here: . This is the summary of what happened so far: {}. Please write out the name of the action the user should choose next by looking at his resources and goal in this format: 'The next action to take should be [action] because you have / don't have / need access to [resource]'. The user's situation is: {} [/INST] {}"
summary_prompt = "{} Please write a summary of what was discussed so far. You should only describe what happened to the user so far, what the user wants and which actions were taken by the user. Do not write the reasons why actions were taken and do not explain what actions do. [/INST] Firstly, the user said that {}"

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    typ = examples["typ"]
    summary       = examples["summary"]
    question = examples["question"]
    outputs      = examples["output"]
    texts = []
    for typ, summary, question, output in zip(typ, summary, question, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = ''
        if typ == 'Action':
            text = action_prompt.format(summary, question, output)
        elif typ == 'Summary':
            text = summary_prompt.format(summary, output)
        text = text + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass


dataset = load_dataset("bazga/llmfinetune", split="train")
dataset = dataset.map(formatting_prompts_func, batched = True,)


trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, # Set num_train_epochs = 1 for full training runs
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)



# Save to q4_k_m GGUF
model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
