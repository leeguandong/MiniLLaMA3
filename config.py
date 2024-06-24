import torch
from dataclasses import dataclass

PROJECT_ROOT = "/home/image_team/image_team_docker_home/lgd/e_commerce_llm/minillama3"


@dataclass
class PhiModelConfig:
    hidden_size: int = 960  # Self-attention的维度
    num_attention_heads: int = 16  # Multi-head

    num_hidden_layers = 24  # 总层数
    max_position_embeddings = 512  # 模型能够处理的序列长度
    intermediate_size = 4096  # FFN的维度
    num_key_value_heads: int = 8  # 这个值和num_attention_heads一致就是MHA，小的话就是GQA


@dataclass
class MiniLLaMA3ModelConfig:
    hidden_size: int = 256
    intermediate_size: int = 768
    num_attention_heads: int = 16

    num_hidden_layers: int = 4
    max_position_embeddings: int = 2048
    num_key_value_heads: int = 8  # GQA


@dataclass
class ModelConfig:
    mode: str = "Phi"  # PHI/Minillama3/Qwen


# --------------------------------------PT-------------------------------------
@dataclass
class PTQwenConfig:
    # ----------------------- TrainerArgs -------------------------------------
    output_dir: str = PROJECT_ROOT + '/weights/qwen_pt'
    evaluation_strategy: str = "steps"  # no

    per_device_train_batch_size: int = 8  # 默认是8
    per_device_eval_batch_size: int = 4  # 默认是8
    # 注意：计算梯度时相当于batch_size * gradient_accumulation_steps，说人话就是梯度累积步数>1时，等于增大n倍的batch_size
    gradient_accumulation_steps: int = 10  # 累积梯度更新步数,默认1
    auto_find_batch_size: bool = True  # 防止OOM

    learning_rate: float = 1e-4  # 5e-5
    weight_decay: float = 0.1  # 0

    num_train_epochs: int = 1  # 3
    lr_scheduler_type: str = "cosine"  # linear
    warmup_steps: int = 0  # 0

    log_level: str = "info"  # passive
    logging_first_step: bool = True  # False
    logging_steps: int = 20  # 500

    save_strategy: str = "steps"  # steps
    save_steps: int = 1000  # 500
    save_total_limit: int = 4  # 最多只保存4次

    bf16: bool = torch.cuda.is_bf16_supported()
    fp16: bool = not torch.cuda.is_bf16_supported()
    optim: str = "adamw_torch"  # adamw_torch
    eval_steps: int = 100  #
    report_to: str = "tensorboard"  # all
    ddp_find_unused_parameters: bool = False
    seed: int = 42

    # -----------------------------------------------------
    tokenizer_dir: str = PROJECT_ROOT + '/model/qwen/'  # tokenizer一般和model权重放在同一个文件夹

    train_file: str = PROJECT_ROOT + '/data/wiki_zh_simple_no_dulpticates_shuffle_train_dataset.parquet'
    validation_file: str = PROJECT_ROOT + '/data/wiki_zh_simple_no_dulpticates_shuffle_valid_dataset.parquet'
    test_file: str = PROJECT_ROOT + '/data/wiki_zh_simple_no_dulpticates_shuffle_test_dataset.parquet'

    # 从哪个模型开始微调，仅当traing 函数 is_finetune = True时生效
    # 微调记得冻结某些层或者调低学习率
    finetune_from_ckp_file = PROJECT_ROOT + '/model_save/chat_small_qwen.best.bin'

    max_seq_len: int = 320  # 最大句子长度，默认：256


@dataclass()
class PTPhiConfig:
    #  ----------------------- TrainerArgs -------------------------------------
    output_dir: str = PROJECT_ROOT + '/weights/phi_pt'
    evaluation_strategy: str = "steps"  # no

    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4  # 梯度累计步大小，省显存，但小模型没必要，用 1 收敛比较快  注意：计算梯度时相当于batch_size * gradient_accumulation_steps，说人话就是梯度累积步数>1时，等于增大n倍的batch_size

    learning_rate: float = 5e-4  # 5e-5
    weight_decay: float = 0.1  # 0

    num_train_epochs: int = 4  # 3
    warmup_steps: int = 1000  # 0 模型参数预热步数，预热样本数=warmup_steps * batch_size * gradient_accumulation_steps

    log_level: str = "info"  # passive
    logging_first_step: bool = True  # False
    logging_steps: int = 50  # 500

    save_strategy: str = "steps"  # steps
    save_steps: int = 2000  # 500
    save_total_limit: int = 3  # 最多只保存4次

    optim: str = "adafactor"  # adamw_torch
    eval_steps: int = 2000  #
    report_to: str = "tensorboard"  # all
    bf16: bool = torch.cuda.is_bf16_supported()
    fp16: bool = not torch.cuda.is_bf16_supported()
    seed: int = 42

    # ----------------------------------- ModelArgs-------------------------------------------
    tokenizer_dir: str = PROJECT_ROOT + '/weights/hf_tokenizer/tokenizer_wiki/'  # tokenizer一般和model权重放在同一个文件夹

    train_file: str = PROJECT_ROOT + '/data/wiki_zh_simple_no_dulpticates_shuffle_train_dataset.parquet'
    validation_file: str = PROJECT_ROOT + '/data/wiki_zh_simple_no_dulpticates_shuffle_valid_dataset.parquet'
    test_file: str = PROJECT_ROOT + '/data/wiki_zh_simple_no_dulpticates_shuffle_test_dataset.parquet'

    # 从哪个模型开始微调，仅当traing 函数 is_finetune = True时生效
    # 微调记得冻结某些层或者调低学习率
    finetune_from_ckp_file = PROJECT_ROOT + '/weights/phi_pt/chat_small_phi.best.bin'

    max_seq_len: int = 512  # 最大句子长度，默认：256


@dataclass
class PTMinillama3Config:
    # ----------------------- TrainerArgs -------------------------------------
    output_dir: str = PROJECT_ROOT + '/weights/minillama3_pt'
    overwrite_output_dir: bool = True  # False

    do_train: bool = True  # False
    do_eval: bool = True  # False

    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1  # 梯度累计步大小，省显存，但小模型没必要，用 1 收敛比较快
    num_train_epochs: int = 2
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "cosine"

    logging_steps: int = 50
    report_to: str = "tensorboard"
    save_steps: int = 1000
    save_total_limit: int = 4  # 最多只保存4次

    seed: int = 3407  # 42
    bf16: bool = torch.cuda.is_bf16_supported()
    fp16: bool = not torch.cuda.is_bf16_supported()
    eval_steps: int = 1000

    # ----------------------------------- ModelArgs-------------------------------------------
    tokenizer_dir: str = PROJECT_ROOT + '/weights/hf_tokenizer/Llama-2-7b-hf/'  # tokenizer一般和model权重放在同一个文件夹

    train_file: str = PROJECT_ROOT + '/data/wiki_zh_simple_no_dulpticates_shuffle_train_dataset.parquet'
    validation_file: str = PROJECT_ROOT + '/data/wiki_zh_simple_no_dulpticates_shuffle_valid_dataset.parquet'
    test_file: str = PROJECT_ROOT + '/data/wiki_zh_simple_no_dulpticates_shuffle_test_dataset.parquet'

    # 从哪个模型开始微调，仅当traing 函数 is_finetune = True时生效
    # 微调记得冻结某些层或者调低学习率
    finetune_from_ckp_file = PROJECT_ROOT + '/weights/minillama3_pt/chat_small_llama3.best.bin'
    max_seq_len: int = 2048  # 最大句子长度，默认：256


# --------------------------------------SFT-------------------------------------
@dataclass
class SFTPhiConfig:
    #  ----------------------- TrainerArgs -------------------------------------
    output_dir: str = PROJECT_ROOT + '/weights/phi_sft'
    evaluation_strategy: str = "steps"  # no

    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4  # 梯度累计步大小，省显存，但小模型没必要，用 1 收敛比较快  注意：计算梯度时相当于batch_size * gradient_accumulation_steps，说人话就是梯度累积步数>1时，等于增大n倍的batch_size

    learning_rate: float = 5e-5  # 5e-5
    weight_decay: float = 0.1  # 0

    num_train_epochs: int = 4  # 3
    warmup_steps: int = 1000  # 0 模型参数预热步数，预热样本数=warmup_steps * batch_size * gradient_accumulation_steps

    log_level: str = "info"  # passive
    logging_first_step: bool = True  # False
    logging_steps: int = 50  # 500

    save_strategy: str = "steps"  # steps
    save_steps: int = 2000  # 500
    save_total_limit: int = 3  # 最多只保存4次

    optim: str = "adafactor"  # adamw_torch
    eval_steps: int = 2000  #
    report_to: str = "tensorboard"  # all
    bf16: bool = torch.cuda.is_bf16_supported()
    fp16: bool = not torch.cuda.is_bf16_supported()
    seed: int = 42
    group_by_length: bool = True

    # ----------------------------------- ModelArgs-------------------------------------------
    tokenizer_dir: str = PROJECT_ROOT + '/weights/hf_tokenizer/tokenizer_wiki/'  # tokenizer一般和model权重放在同一个文件夹
    model_file: str = PROJECT_ROOT + '/weights/phi_pt/checkpoint-15164'  # T5/PHI/MINILLAMA3

    train_file: str = PROJECT_ROOT + '/data/belle_sft_data_zh_train_dataset.parquet'
    validation_file: str = PROJECT_ROOT + '/data/belle_sft_data_zh_valid_dataset.parquet'
    test_file: str = PROJECT_ROOT + '/data/belle_sft_data_zh_test_dataset.parquet'

    # 从哪个模型开始微调，仅当traing 函数 is_finetune = True时生效
    # 微调记得冻结某些层或者调低学习率
    finetune_from_ckp_file = PROJECT_ROOT + '/weights/phi_sft/chat_small_phi.best.bin'

    max_seq_len: int = 512  # 最大句子长度，默认：256


@dataclass
class SFTQwenConfig:
    pass


@dataclass
class SFTMinillama3Config:
    pass


# --------------------------------------DPO-------------------------------------
@dataclass
class DPOPhiConfig:
    #  ----------------------- TrainerArgs -------------------------------------
    output_dir: str = PROJECT_ROOT + '/weights/phi_dpo'
    evaluation_strategy: str = "steps"  # no

    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4  # 梯度累计步大小，省显存，但小模型没必要，用 1 收敛比较快  注意：计算梯度时相当于batch_size * gradient_accumulation_steps，说人话就是梯度累积步数>1时，等于增大n倍的batch_size

    learning_rate: float = 2e-5  # 5e-5
    weight_decay: float = 0.1  # 0

    num_train_epochs: int = 4  # 3
    warmup_steps: int = 1000  # 0 模型参数预热步数，预热样本数=warmup_steps * batch_size * gradient_accumulation_steps

    log_level: str = "info"  # passive
    logging_first_step: bool = True  # False
    logging_steps: int = 50  # 500

    save_strategy: str = "steps"  # steps
    save_steps: int = 2000  # 500
    save_total_limit: int = 3  # 最多只保存4次

    optim: str = "adafactor"  # adamw_torch
    eval_steps: int = 2000  #
    report_to: str = "tensorboard"  # all
    bf16: bool = torch.cuda.is_bf16_supported()
    fp16: bool = not torch.cuda.is_bf16_supported()
    seed: int = 42
    group_by_length: bool = False
    remove_unused_columns: bool = False

    beta: float = 0.1

    # ----------------------------------- ModelArgs-------------------------------------------
    tokenizer_dir: str = PROJECT_ROOT + '/weights/hf_tokenizer/tokenizer_wiki/'  # tokenizer一般和model权重放在同一个文件夹
    model_file: str = PROJECT_ROOT + '/weights/phi_sft'  # T5/PHI/MINILLAMA3

    train_file: str = PROJECT_ROOT + '/data/dpo_train.json'
    validation_file: str = PROJECT_ROOT + '/data/dpo_eval.json'
    test_file: str = PROJECT_ROOT + '/data/dpo_eval.json'

    # 从哪个模型开始微调，仅当traing 函数 is_finetune = True时生效
    # 微调记得冻结某些层或者调低学习率
    finetune_from_ckp_file = PROJECT_ROOT + '/weights/phi_sft/chat_small_phi.best.bin'

    max_seq_len: int = 512  # 最大句子长度，默认：256


@dataclass
class DPOQwenConfig:
    pass


@dataclass
class DPOMinillama3Config:
    pass


@dataclass
class RAGConfig:
    general_embedding_model: str = ""
    model_file: str = ""
    doc: str = ""
    doc_db: str = ""

    template: str = "请根据以下给出的背景知识回答问题，对于不知道的信息，直接回答“未找到相关答案”。\n以下为为背景知识：\n"
    max_new_token: int = 512
