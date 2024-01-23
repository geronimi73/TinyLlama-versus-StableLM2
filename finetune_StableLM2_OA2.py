import torch, os, wandb, uuid, json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, set_seed
from accelerate import Accelerator
from datasets import load_dataset
from functools import partial

accelerator = Accelerator()

set_seed(42)

run_id = str(uuid.uuid4())
modelpath="models/stablelm-2-1_6b"
dataset_name="g-ronimo/oasst2_top1_en"
lr=0.00002
bs=1            # batch size
bs_eval=8        # batch size for evals
ga_steps=16     # gradient acc. steps
epochs=4
max_length=1600        # max. sample length w/o OOM on 24GB VRAM GPU
output_dir=f"out-{run_id}"

# Load model
model = AutoModelForCausalLM.from_pretrained(
    modelpath,    
    device_map={"": accelerator.process_index},
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2", 
    trust_remote_code=True,
)

# Load (slow) tokenizer, fast tokenizer sometimes ignores the added tokens
tokenizer = AutoTokenizer.from_pretrained(
    modelpath, 
    use_fast=False, 
    trust_remote_code=True
)    

# Set/add ChatML tokens
tokenizer.eos_token_id=tokenizer.encode("<|im_end|>")[0]
model.config.eos_token_id = tokenizer.eos_token_id

# Load dataset
dataset = load_dataset(dataset_name)
dataset = dataset["train"].train_test_split(test_size=0.1)

# Format (chatML) and tokenize dataset
templates=[
    "<|im_start|>assistant\n{msg}<|im_end|>",
    "<|im_start|>user\n{msg}<|im_end|>"
]
IGNORE_INDEX=-100

def tokenize(input, max_length):
    input_ids, attention_mask, labels = [], [], []

    for i,msg in enumerate(input["conversation"]):
        isHuman = msg["role"]=="user"
        msg_chatml=templates[isHuman].format(msg=msg["content"])
        msg_tokenized=tokenizer(msg_chatml, truncation=False, add_special_tokens=False)
    
        input_ids+=msg_tokenized["input_ids"]
        attention_mask+=msg_tokenized["attention_mask"]
        labels+=[IGNORE_INDEX]*len(msg_tokenized["input_ids"]) if isHuman else msg_tokenized["input_ids"]

    return {
        "input_ids": input_ids[:max_length],
        "attention_mask": attention_mask[:max_length],
        "labels": labels[:max_length],
    }

dataset_tokenized = dataset.map(
    partial(tokenize, max_length=max_length), 
    batched=False, 
    num_proc=os.cpu_count()//accelerator.num_processes,    # multithreaded
    remove_columns=dataset["train"].column_names  # don't need this anymore, we have tokens from here on
)

# collate function - to transform list of dictionaries [ {input_ids: [123, ..]}, {.. ] to single batch dictionary { input_ids: [..], labels: [..], attention_mask: [..] }
def collate(elements):
    tokens=[e["input_ids"] for e in elements]
    tokens_maxlen=max([len(t) for t in tokens])

    for i,sample in enumerate(elements):
        input_ids=sample["input_ids"]
        labels=sample["labels"]
        attention_mask=sample["attention_mask"]

        pad_len=tokens_maxlen-len(input_ids)

        input_ids.extend( pad_len * [tokenizer.pad_token_id] )   
        labels.extend( pad_len * [IGNORE_INDEX] )    
        attention_mask.extend( pad_len * [0] ) 

    batch={
        "input_ids": torch.tensor( [e["input_ids"] for e in elements] ),
        "labels": torch.tensor( [e["labels"] for e in elements] ),
        "attention_mask": torch.tensor( [e["attention_mask"] for e in elements] ),
    }

    return batch

steps_per_epoch=len(dataset_tokenized["train"])//(accelerator.num_processes*bs*ga_steps)

args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=bs,
    per_device_eval_batch_size=bs_eval,
    evaluation_strategy="steps",
    logging_steps=1,
    eval_steps=steps_per_epoch,    # eval once per epoch
    save_steps=steps_per_epoch,     # save once per epoch
    gradient_accumulation_steps=ga_steps,
    num_train_epochs=epochs,
    lr_scheduler_type="constant",
    optim="paged_adamw_32bit",      # val_loss will go nan with paged_adamw_8bit
    learning_rate=lr,
    group_by_length=False,
    bf16=True,        
    ddp_find_unused_parameters=False,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=collate,
    train_dataset=dataset_tokenized["train"],
    eval_dataset=dataset_tokenized["test"],
)

if accelerator.is_main_process:
    run = wandb.init(
        project="OA2-finetune",
        name=modelpath.split("/")[1]+"_"+dataset_name+f"_bs-{bs}_LR-{lr}_maxlen-{max_length}_{run_id}",
        config={
            "model_name": modelpath,
            "run_id": run_id,
            "dataset": dataset_name,
            "output_dir": output_dir,
            "lr": lr,
            "max_length": max_length,
            "train_batch_size": bs,
            "validation_batch_size": bs,
            "ga_steps": ga_steps,
            "training_args": args,
            "GPUs": accelerator.num_processes,
        }
    )

trainer.train()