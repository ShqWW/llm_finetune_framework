# -*- coding: utf-8 -*-
# time: 2023/6/1 17:19
# file: train_qlora.py
# author: zmfy
# email: shuxueslpi@163.com

import os
import argparse
from typing import List, Dict, Optional
from finetune_dataset import *

import torch
from loguru import logger
from datasets import load_dataset
import bitsandbytes as bnb
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict,
    prepare_model_for_kbit_training
)
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from get_config import get_cfg, get_train_args_dict

_compute_dtype_map = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16
}

os.environ["WANDB_DISABLED"] = "true"
def parse_args():
    parser = argparse.ArgumentParser(description='QLoRA')
    parser.add_argument('--cfg', type=str, required=True, help='配置py文件')
    return parser.parse_args()


def find_all_linear_names(model):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

class LoRATrainer(Trainer):
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """只保存adapter"""
        if output_dir is None:
            output_dir = self.args.output_dir
        self.model.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


def train(args):
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))

    hf_parser = HfArgumentParser(TrainingArguments)
    train_args, = hf_parser.parse_dict(get_train_args_dict(args))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Quantization
    q_config = BitsAndBytesConfig(load_in_4bit=True,
                                  bnb_4bit_quant_type='nf4',
                                  bnb_4bit_use_double_quant=True,
                                  bn3b_4bit_compute_dtype=_compute_dtype_map[args.compute_dtype])
    
    print('loading model')
    model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                      quantization_config=q_config,
                                      trust_remote_code=True, device_map={'': local_rank})
    print('quantizing model')

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # LoRA
    # target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['chatglm']
    print('find lora parameter')
    target_modules = find_all_linear_names(model)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias='none',
        inference_mode=False,
        task_type=TaskType.CAUSAL_LM
    )

    print('getting peft model')
    model = get_peft_model(model, lora_config)
    train_dataset = PreDataset(tokenizer=tokenizer, datapath=args.data_path, eos_token_id=args.eos_token_id, pad_token_id=args.pad_token_id, max_len = args.max_len)

    trainer = LoRATrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=train_dataset.collate_fn
    )
    
    print('start training')
    trainer.train()
    if local_rank==0:
        trainer.model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    args = parse_args()
    args = get_cfg(args)
    train(args)

