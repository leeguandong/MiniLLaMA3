import sys

sys.path.append("/home/image_team/image_team_docker_home/lgd/e_commerce_llm/minillama3/")

import os
import re
import ujson
import torch
import numpy as np
from rich import progress
import pyarrow.parquet as pq
from tools.raw_data_process import delete_file
from config import ModelConfig, DPOPhiConfig, DPOQwenConfig, DPOMinillama3Config
from model.llm_model import PhiHandler, Minillama3Handler, QwenHandler
from transformers import GenerationConfig

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

handler_dict = {
    "Phi": PhiHandler,
    "Minillama3": Minillama3Handler,
    "Qwen": QwenHandler
}

if ModelConfig.mode == 'Qwen':
    config = DPOQwenConfig()
elif ModelConfig.mode == "Phi":
    config = DPOPhiConfig()
elif ModelConfig.mode == "Minillama3":
    config = DPOMinillama3Config()
else:
    raise TypeError("just support qwen/phi/minillama3!!!")

HandlerClass = handler_dict.get(ModelConfig.mode)
handler = HandlerClass(config)

# step 1. 加载tokenizer
tokenizer = handler.load_tokenizer()

# step 2. 初始化模型
model = handler.get_sft_model().to(device)

gen_config = GenerationConfig(
    temperature=0.3,
    top_k=20,
    top_p=0.5,
    do_sample=True,
    num_beams=1,
    repetition_penalty=1.1,
    max_new_tokens=512,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)


def process_alpaca_gpt4_data(max_len: int = 512) -> None:
    ''''
    处理RM高质量回答部分
    数据集：https://huggingface.co/datasets/c-s-ale/alpaca-gpt4-data-zh
    '''

    read_file = PROJECT_ROOT + '/alpaca_gpt4_data_zh.json'
    save_file = PROJECT_ROOT + '/alpaca_gpt4_data_dop_chosen_zh.json'

    max_len += 8

    my_data = []

    with open(read_file, 'r', encoding='utf-8') as f:
        data = ujson.load(f)
        print('length of {} is {}'.format(read_file, len(data)))
        for item in progress.track(data):
            prompt = item['instruction']
            inputs = item['input']

            response = item['output']

            if len(response) > max_len: continue  # 超长的不要

            if len(inputs.strip()) > 0:
                prompt = f"{prompt}，{inputs}"

            if len(prompt) > max_len: continue

            if len(prompt) == 0 or len(response) == 0: continue

            my_data.append(
                {
                    'prompt': prompt,
                    'chosen': response
                }
            )

    print('length of {} is {}'.format(save_file, len(my_data)))

    with open(save_file, 'w', encoding='utf-8') as f:
        ujson.dump(my_data, f, indent=4, ensure_ascii=False)


def generate_alpaca_gpt4_reject_response(groups_cnt: int = 50000, max_len: int = 320, batch_size: int = 32) -> None:
    '''生成不是很满意的回答回答
    '''
    print('load model...')

    finetune_file = PROJECT_ROOT + '/alpaca_gpt4_data_dop_chosen_zh.json'
    save_rw_json_file = PROJECT_ROOT + '/alpaca_gpt4_data_dpo_zh.json'
    # save_rw_parquet_file = PROJECT_ROOT + '/data/my_rlhf_dataset.parquet'

    data = []
    with open(finetune_file, 'r', encoding='utf-8') as f:
        data = ujson.load(f)

    print('length of {} is {}'.format(save_rw_json_file, len(data)))

    model_outs = []
    batch_prompt = []
    process_item = []
    for i, item in progress.track(enumerate(data), total=len(data)):
        # 模型生成的答案为拒绝答案
        batch_prompt.append(f"{item['prompt']}[EOS]")
        process_item.append(item)

        if i % 500 == 0:
            print('process {} items.'.format(i))

        if len(batch_prompt) >= batch_size or i == len(data) - 1:
            encoded = tokenizer.batch_encode_plus(batch_prompt, truncation=False, padding=True)

            with torch.no_grad():
                input_ids = torch.LongTensor(encoded.input_ids).to(device)
                attention_mask = torch.LongTensor(encoded.attention_mask).to(device)

                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    # max_seq_len=512,
                    # search_type='greedy',
                    generation_config=gen_config
                )

                outputs = tokenizer.batch_decode(outputs.cpu().numpy(), clean_up_tokenization_spaces=True,
                                                 skip_special_tokens=True)

            model_outs.extend(outputs)

            batch_prompt = []

        if len(model_outs) % 2000 == 0:
            for i in range(len(model_outs)):
                process_item[i]['reject'] = model_outs[i]
            try:
                with open(PROJECT_ROOT + '/outs.ckp.json', 'w', encoding='utf-8') as f:
                    ujson.dump(process_item, f, indent=4, ensure_ascii=False)
            except Exception as e:
                print(e)

    for i in range(len(model_outs)):
        process_item[i]['reject'] = model_outs[i]

    with open(save_rw_json_file, 'w', encoding='utf-8') as f:
        ujson.dump(process_item, f, indent=4, ensure_ascii=False)

    # df = pd.DataFrame(data)
    # write_single_parquet_file(save_rw_parquet_file, df)


def replace_line(s: str) -> str:
    '''将双斜杠替换为单斜杠，既是 \\n 替换为 \n
    '''
    return re.sub('\\\\n', '\n', s)


def merge_rlhf_data(max_len: int = 512) -> None:
    ''''
    处理RM高质量回答部分
    数据集：https://huggingface.co/datasets/Skepsun/huozi_rlhf_data_json
    https://huggingface.co/datasets/beyond/rlhf-reward-single-round-trans_chinese
    '''
    my_data = []
    read_files = [
        PROJECT_ROOT + '/huozi_rlhf_data.json',
        PROJECT_ROOT + '/alpaca_gpt4_data_dpo_zh.json',
    ]
    save_file = PROJECT_ROOT + '/dpo_data.json'

    if os.path.exists(save_file):
        assert delete_file(save_file)

    max_len += 8  # for eos token

    for read_file in read_files:
        items = []
        with open(read_file, 'r', encoding='utf-8') as f:
            items = ujson.load(f)

        for item in progress.track(items):
            prompt, chosen, reject = item['prompt'], item['chosen'], item['reject']

            if len(prompt) > max_len or len(chosen) > max_len or len(reject) > max_len:
                continue

            # reject.strip() == chosen.strip()，这两个相同的也不要
            if len(prompt) == 0 or len(chosen) == 0 or len(reject) == 0 or reject.strip() == chosen.strip():
                continue

            my_data.append({
                'prompt': replace_line(prompt),
                'chosen': replace_line(chosen),
                'rejected': replace_line(reject),
            })

    read_files = [
        PROJECT_ROOT + '/train-00000-of-00001-789dc5dece0f1fc1.parquet',
        PROJECT_ROOT + '/test-00000-of-00001-8ecd46436fadcf7f.parquet',
    ]

    for read_file in read_files:
        pf = pq.read_table(read_file)
        for prompt, chosen, rejected in progress.track(zip(pf['prompt'], pf['chosen'], pf['rejected']),
                                                       total=pf.num_rows):

            prompt, chosen, rejected = prompt.as_py(), chosen.as_py(), rejected.as_py()

            if len(prompt) > max_len or len(chosen) > max_len or len(rejected) > max_len:
                continue

            if len(prompt) == 0 or len(chosen) == 0 or len(rejected) == 0 or rejected.strip() == chosen.strip():
                continue

            my_data.append({
                'prompt': replace_line(prompt),
                'chosen': replace_line(chosen),
                'rejected': replace_line(rejected),
            })
    print('length of {} is {}'.format(save_file, len(my_data)))

    with open(save_file, 'w', encoding='utf-8') as f:
        ujson.dump(my_data, f, indent=4, ensure_ascii=False)


def split_train_eval_dataset() -> None:
    '''划分数据集
    '''
    rw_json_file = PROJECT_ROOT + '/dpo_data.json'
    train_file = PROJECT_ROOT + '/dpo_train.json'
    eval_file = PROJECT_ROOT + '/dpo_eval.json'

    data = []

    with open(rw_json_file, 'r', encoding='utf-8') as f:
        data = ujson.load(f)

    np.random.shuffle(data)
    split_idx = int(len(data) * 0.99)

    train_data = data[0: split_idx]
    eval_data = data[split_idx:]

    print('train size: {}, eval size:{}'.format(len(train_data), len(eval_data)))

    with open(train_file, 'w', encoding='utf-8') as f:
        ujson.dump(train_data, f, indent=4, ensure_ascii=False)

    with open(eval_file, 'w', encoding='utf-8') as f:
        ujson.dump(eval_data, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    PROJECT_ROOT = "/home/image_team/image_team_docker_home/lgd/e_commerce_llm/minillama3/data/"

    # 1. 处理chosen文本
    # process_alpaca_gpt4_data()

    # 2. 生成rejected文本
    generate_alpaca_gpt4_reject_response()

    # 合并数据集
    # merge_rlhf_data()

    # 3. split train and eval dataset
    # split_train_eval_dataset()
