import sys

sys.path.append("/home/image_team/image_team_docker_home/lgd/e_commerce_llm/minillama3/")

import os
import torch

os.environ['WANDB_DISABLED'] = 'true'  # 禁用 wandb，也可以不用这一条
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, \
    TrainingArguments, Trainer
from utils import print_model_parameters, kaiming_initialization, process_func


def mini_model():
    """
    模型配置
    :return:
    """
    hidden_size = 256
    # 中间层取 8/3 倍，按 128 向上取整 FFN维度
    intermediate_size = (int(hidden_size * 8 / 3 / 128) + 1) * 128

    # 只改动我们需要调整的参数，其余保持不变
    config = AutoConfig.for_model(
        model_type="llama",
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=16,  # GQA
        num_hidden_layers=4,
        num_key_value_heads=8)  # 分为 8 组
    """
       LlamaConfig {
      "attention_bias": false,                 # 不使用注意力偏置
      "attention_dropout": 0.0,                # 注意力层的 dropout 比例
      "bos_token_id": 1,                       # bos_token (begin of sentence) 的 id
      "eos_token_id": 2,                       # eos_token (end of sentence) 的 id
      "hidden_act": "silu",                    # 隐藏层激活函数类型，silu 即 SwiGLU
      "hidden_size": 256,                      # 隐藏层维度大小
      "initializer_range": 0.02,               # 权重初始化范围，会被后面的 Kaiming 初始化覆盖
      "intermediate_size": 768,                # 中间层大小，采用 8/3 倍而非 4 倍
      "max_position_embeddings": 2048,
      "model_type": "llama",
      "num_attention_heads": 16,
      "num_hidden_layers": 4,
      "num_key_value_heads": 8,
      "pretraining_tp": 1,
      "rms_norm_eps": 1e-06,
      "rope_scaling": null,
      "rope_theta": 10000.0,
      "tie_word_embeddings": false,            # 头尾 embedding 和 lm_head 是否共享权重
      "transformers_version": "4.40.0",
      "use_cache": true,
      "vocab_size": 32000
    }
    """
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)
    kaiming_initialization(model)

    if print_model:
        print_model_parameters(model)

    return model


def mini_tokenizer():
    """
    选用 LLaMA 2 的分词器，因为二代的词表比较小（32k），LLaMA 3 的词表太大了（128k），在 SLM 中会占用太多的参数比重，并且这只是个专有任务数据训练，没必要用太大的词表。词表大小可以在huggingface文件的config.json中查看，有个vocab_size
    :return:
    """
    if tokenizer_kind == "llama2":
        tokenizer = AutoTokenizer.from_pretrained(
            "/home/image_team/image_team_docker_home/lgd/e_commerce_llm/minillama3/Llama-2-7b-hf/")

        """
        LlamaTokenizerFast(name_or_path='NousResearch/Llama-2-7b-hf', vocab_size=32000, model_max_length=1000000000000000019884624838656, is_fast=True, padding_side='left', truncation_side='right', special_tokens={'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'pad_token': '<unk>'}, clean_up_tokenization_spaces=False),  added_tokens_decoder={
            0: AddedToken("<unk>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=True),
            1: AddedToken("<s>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=True),
            2: AddedToken("</s>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=True),
        }
        """
    else:
        raise TypeError(f"not support {tokenizer_kind} tokenizer")
    return tokenizer


def inference(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        input_text: str = "Once upon a time, ",
        max_new_tokens: int = 16
):
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=40,
        top_p=0.95,
        temperature=0.8
    )
    generated_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )
    # print(outputs)
    print(generated_text)


def mini_dataset(tokenizer):
    if dataset == "TinyStoriesV2":
        # 应用全部训练集，约 2.7 M,这里可以调整比例，我只用了 10%，约 270 K
        ds_train = load_dataset("/home/image_team/image_team_docker_home/lgd/e_commerce_llm/minillama3/TinyStoriesV2/",
                                split='train[:10%]')
        ds_val = load_dataset("/home/image_team/image_team_docker_home/lgd/e_commerce_llm/minillama3/TinyStoriesV2/",
                              split='validation')
        ds_train = ds_train.shuffle()

        """
        {'text': ['Once upon a time, there was a reliable otter named Ollie. He lived in a river with his family. They all loved to play and swim together.\nOne day, Ollie\'s mom said, "Ollie, hurry and get some fish for dinner!" Ollie swam fast to catch fish. He saw his friend, the duck. "Hi, Ollie!" said the duck. "Hi, duck!" said Ollie. "I need to hurry and catch fish for my family."\nWhile Ollie was catching fish, he found a big shiny stone. He thought, "This is not a fish, but it is so pretty!" Ollie took the shiny stone home to show his family. They all looked at the shiny stone and smiled. The shiny stone made everyone happy, and they forgot about the fish for dinner.',
          'One day, a little boy named Tim went to the park. He saw a big tiger. The tiger was not mean, but very easy to play with. Tim and the tiger played all day. They had lots of fun.\nThen, something unexpected happened. The tiger started to shake. Tim was scared. He did not know what was going on. But then, the tiger turned into a nice dog. Tim was very surprised.\nTim and the dog played together now. They were very happy. The dog was easy to play with too. At the end of the day, Tim went home with his new friend.']}
        """
    elif dataset == "TinyStoriesZh":
        # 应用全部训练集，约 2.7 M,这里可以调整比例，我只用了 10%，约 270 K
        ds_train = load_dataset("/home/image_team/image_team_docker_home/lgd/e_commerce_llm/minillama3/TinyStoriesV2/",
                                split='train[:10%]')
        ds_val = load_dataset("/home/image_team/image_team_docker_home/lgd/e_commerce_llm/minillama3/TinyStoriesV2/",
                              split='validation[:-2%]')
        ds_train = ds_train.shuffle()

        """
     {'story': '从前，有一个小女孩，名叫莉莉。她喜欢每周和妈妈一起去图书馆。图书馆是一座很大的建筑，里面有很多书。莉莉喜欢读有关公主和动物的书籍。\n有一天，莉莉和妈妈去图书馆，但有些不一样。那里没有人！莉莉感到孤独和悲伤。她向妈妈提到，她想念那些经常和她一起去图书馆的朋友们。\n突然，他们听到图书馆后面传来一声巨响。他们前去查看，发现一群友善的动物正在开派对！有兔子、松鼠，甚至还有一只友善的狐狸。他们一直躲在图书馆里。莉莉很高兴有新朋友一起读书。从那天起，莉莉和她的新朋友们每周都在图书馆度过了愉快的时光。'}
    Line 3: {'story': '从前，有一只大狮子。他喜欢大声吼叫。有一天，他遇到了一只小老鼠。老鼠被狮子的大吼声吓坏了。但狮子想了想，说道：“小老鼠，别害怕，我不会伤害你的。”\n老鼠感觉好多了，说：“谢谢你，狮子先生。我们可以成为朋友吗？”狮子微笑着说：“当然，我们可以成为朋友。”\n但有一天，狮子非常饿了。他看到小老鼠，心想：“也许我可以吃掉我的朋友。”于是，他追赶老鼠，抓住了它。老鼠叫道：“狮子先生，你为什么这么做？我以为我们是朋友呢！”\n狮子心里很难受，说：“对不起，小老鼠，我饿了，犯了一个错误。”但老鼠已经不见了，飞快地逃到了安全的地方。狮子很伤心，意识到做朋友比挨饿更重要。'}
        """

    from functools import partial
    # 创建一个新的函数，它有一个预设的 'tokenizer' 参数
    process_func_with_tokenizer = partial(process_func, tokenizer=mini_tokenizer)

    ds_train = ds_train.map(
        process_func_with_tokenizer,
        batched=True,
        num_proc=8,
        remove_columns=ds_train.column_names,
        desc='Running tokenizer on train_set: '
    )
    ds_val = ds_val.map(
        process_func_with_tokenizer,
        batched=True,
        num_proc=8,
        remove_columns=ds_val.column_names,
        desc='Running tokenizer on val_set: '
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    return data_collator, ds_train, ds_val


def mini_training(model, ds_train, ds_val, tokenizer, data_collator):
    training_args = TrainingArguments(
        output_dir='saves',  # 输出路径，包括模型检查点、中间文件等
        overwrite_output_dir=True,  # 是否覆写 output_dir
        do_train=True,  # 是否做训练
        do_eval=True,  # 是否做评估
        eval_steps=1000,  # 评估步骤间隔
        per_device_train_batch_size=4,  # 每设备批次
        gradient_accumulation_steps=1,  # 梯度累计步大小，省显存，但小模型没必要，用 1 收敛比较快
        learning_rate=1e-4,  # 学习率大小
        lr_scheduler_type='cosine',  # 学习率调度策略，LLM 训练一般都用余弦
        bf16=torch.cuda.is_bf16_supported(),  # 尝试配置 bf16
        fp16=not torch.cuda.is_bf16_supported(),  # bf16 不行就上 fp16
        logging_steps=50,  # 打印步骤间隔
        report_to=None,  # 日志输出目标，不想用 wandb 可以设置为 None
        num_train_epochs=2,  # 训练轮数，2 ~ 3 即可
        save_steps=1000,  # 检查点保存步骤间隔
        save_total_limit=2,  # output_dir 内留存的检查点最大数目
        seed=3407,  # 随机种子
    )

    trainer = Trainer(
        model=model,  # 模型实例
        args=training_args,  # 训练参数
        train_dataset=ds_train,  # 训练集
        eval_dataset=ds_val,  # 验证集（评估集）
        tokenizer=tokenizer,  # 分词器
        data_collator=data_collator,  # data collator
    )

    # 启动训练
    # 这里只 train 了 2 epochs，loss 收敛到了 1.6 左右
    trainer.train()
    model.save_pretrained(model_path="/home/image_team/image_team_docker_home/lgd/e_commerce_llm/minillama3/saves/")


if __name__ == "__main__":
    print_model = True
    tokenizer_kind = "llama2"
    dataset = "TinyStoriesZh"  # TinyStoriesV2

    # 模型
    model = mini_model()
    # Tokenizer
    mini_tokenizer = mini_tokenizer()
    # 加载数据
    data_collator, ds_train, ds_val = mini_dataset(tokenizer=mini_tokenizer)
    # 训练
    mini_training(model, ds_train, ds_val, mini_tokenizer, data_collator)
    # 前向推理
    inference(model, mini_tokenizer,
              "Once upon a time, in a beautiful garden, there lived a little rabbit named Peter Rabbit.",
              max_new_tokens=256)

# sh train.sh minillama3.py
