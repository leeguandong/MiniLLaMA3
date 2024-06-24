# coding=utf-8
import time
import os
import torch
import pandas as pd

from tqdm import tqdm
import numpy as np
from transformers import TrainingArguments
from config import ModelConfig, DPOPhiConfig, DPOQwenConfig, DPOMinillama3Config
from utils.functions import MyTrainerCallback
from utils.utils import print_model_parameters
from model.llm_model import PhiHandler, Minillama3Handler, QwenHandler
from model.dataset import get_dpo_dataset
from trl import DataCollatorForCompletionOnlyLM, DPOTrainer

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

tqdm.pandas()

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

handler_dict = {
    "Phi": PhiHandler,
    "Minillama3": Minillama3Handler,
    "Qwen": QwenHandler
}


def dpo_train() -> None:
    if ModelConfig.mode == 'Qwen':
        config = DPOQwenConfig()
    elif ModelConfig.mode == "Phi":
        config = DPOPhiConfig()
    elif ModelConfig.mode == "Minillama3":
        config = DPOMinillama3Config()
    else:
        raise TypeError("just support qwen/Phi/Minillama3!!!")

    HandlerClass = handler_dict.get(ModelConfig.mode)
    handler = HandlerClass(config)

    # step 1. 加载tokenizer
    tokenizer = handler.load_tokenizer()

    # step 2. 初始化模型
    model_train, model_ref = handler.get_dpo_model()
    print_model_parameters(model_train)

    # step 3. Load dataset
    train_dataset = get_dpo_dataset(file=config.train_file)
    eval_dataset = get_dpo_dataset(file=config.validation_file)

    # step 4. Define the training argument
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        overwrite_output_dir=getattr(config, "overwrite_output_dir", False),
        evaluation_strategy=getattr(config, "evaluation_strategy", "no"),
        do_train=getattr(config, "do_train", False),
        do_eval=getattr(config, "do_eval", False),

        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        auto_find_batch_size=getattr(config, "auto_find_batch_size", False),

        learning_rate=config.learning_rate,
        weight_decay=getattr(config, "weight_decay", 0),

        num_train_epochs=config.num_train_epochs,
        lr_scheduler_type=getattr(config, "lr_scheduler_type", "linear"),
        warmup_steps=getattr(config, "warmup_steps", 0),

        log_level=getattr(config, "log_level", "info"),
        logging_first_step=getattr(config, "logging_first_step", False),
        logging_steps=config.logging_steps,

        save_strategy=getattr(config, "save_strategy", "steps"),
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,

        optim=getattr(config, "optim", "adamw_torch"),
        bf16=config.bf16,
        fp16=config.fp16,
        eval_steps=getattr(config, "eval_steps", 2000),
        report_to=config.report_to,

        group_by_length=getattr(config, "group_by_length", False),
        ddp_find_unused_parameters=getattr(config, "ddp_find_unused_parameters", False),
        seed=config.seed,
    )

    # step 5.init collator
    empty_cuda_cahce = MyTrainerCallback()

    # step 6. Define the trainer
    trainer = DPOTrainer(
        model=model_train,
        ref_model=model_ref,
        args=training_args,
        beta=getattr(config, "beta", 0.1),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=[empty_cuda_cahce],
        max_length=config.max_seq_len * 2 + 16,  # 官方文档中也是max_prompt_length的2倍
        max_prompt_length=config.max_seq_len,
        max_target_length=config.max_seq_len
    )

    # step 7. train
    trainer.train(
        # resume_from_checkpoint=True
    )

    # step 8. eval
    eval_results = trainer.evaluate()
    print(f"Perplexity: {np.exp(eval_results['eval_loss']):.2f}")

    # step 9: save log
    loss_log = pd.DataFrame(trainer.state.log_history)
    loss_log.to_csv(f"{config.output_dir}/pre_train_log_{time.strftime('%Y%m%d-%H%M')}.csv")

    # Step 10: Save the model
    trainer.save_model(config.output_dir)


if __name__ == "__main__":
    dpo_train()

# sh train.sh sft.py
