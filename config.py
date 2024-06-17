from dataclasses import dataclass
from os.path import dirname, abspath

PROJECT_ROOT = "/home/image_team/image_team_docker_home/lgd/e_commerce_llm/minillama3/data/"


@dataclass()
class PTconfig:
    epochs: int = 8
    batch_size_per_gpu: int = 16

    learn_rate: float = 0.0001  # 最大 div_factor * learn_rate
    div_factor: int = 50

    mixed_precision: str = "bf16"  # 混合精度 ''no','fp16','bf16' or 'fp8'

    # 注意：计算梯度时相当于batch_size * gradient_accumulation_steps，说人话就是梯度累积步数>1时，等于增大n倍的batch_size
    gradient_accumulation_steps: int = 8  # 累积梯度更新步数

    warmup_steps: int = 1024  # 模型参数预热步数，预热样本数=warmup_steps * batch_size * gradient_accumulation_steps

    tokenizer_dir: str = PROJECT_ROOT + '/model_save/'  # tokenizer一般和model权重放在同一个文件夹
    model_file: str = PROJECT_ROOT + '/model_save/chat_small_t5.{}.bin'
    model_config_file: str = PROJECT_ROOT + '/model_save/model_config.json'

    train_file: str = PROJECT_ROOT + '/data/my_train_dataset.parquet'
    validation_file: str = PROJECT_ROOT + '/data/my_valid_dataset.parquet'
    test_file: str = PROJECT_ROOT + '/data/my_test_dataset.parquet'

    # 从哪个模型开始微调，仅当traing 函数 is_finetune = True时生效
    # 微调记得冻结某些层或者调低学习率
    finetune_from_ckp_file = PROJECT_ROOT + '/model_save/chat_small_t5.best.bin'

    # 训练状态保存，中断后可以从此处继续训练
    train_state_dir: str = PROJECT_ROOT + '/model_save/train_latest_state'
    output_dir: str = PROJECT_ROOT + '/model_save/pretrain'

    logging_steps: int = 50
    save_steps: int = 10000

    # dataset_cache_dir: str = PROJECT_ROOT + '/data/.cache'
    # trainer_log_file: str = PROJECT_ROOT + '/logs/trainer.log'

    keep_latest_n_ckp: int = 8  # 训练过程中，最多保留多少个分数最好的模型文件

    seed: int = 23333
    dataloader_buffer_size: int = 50000
    max_seq_len: int = 256  # 最大句子长度，默认：256
