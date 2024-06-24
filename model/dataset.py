# coding=utf-8
from typing import Dict, List
import numpy as np
from datasets import Dataset, load_dataset


def get_pt_dataset(file: str,
                   tokenizer,
                   max_seq_len,
                   cache_dir: str = '.cache') -> Dataset:
    map_dtype = np.uint16 if len(tokenizer) < 65535 else np.uint32

    def token_to_id(samples: dict) -> dict:
        batch_txt = samples["text"]
        outputs = tokenizer(
            batch_txt,
            padding=False,
            return_attention_mask=False,
            truncation=True,
            max_length=max_seq_len
        )

        input_ids = [np.array(item, dtype=map_dtype) for item in outputs["input_ids"]]

        return {"input_ids": input_ids}

    def merge_texts(examples):
        # "text":xxxxx<Im_end>xxxx （最长为512）im_end来区分两个文本，我是尽量填充到最大长度的，qwen的tokenizer会截断并添加im_end
        return {'text': [prompt + "" + response for prompt, response in zip(examples['prompt'], examples['response'])]}

    dataset = load_dataset('parquet', data_files=file, split="train", cache_dir=cache_dir, keep_in_memory=False)
    dataset = dataset.map(merge_texts, batched=True)
    dataset = dataset.map(token_to_id, batched=True, num_proc=24, remove_columns=dataset.column_names,
                          desc='Running tokenizer on set: ')

    return dataset


def get_sft_dataset(file: str,
                    tokenizer,
                    max_seq_len,
                    cache_dir: str = '.cache'):
    map_dtype = np.uint16 if len(tokenizer) < 65535 else np.uint32

    instruction_template = "##提问:"
    response_template = "##回答:"

    # import pdb;pdb.set_trace()
    def batched_formatting_prompts_func(example: list[dict]) -> list[str]:
        batch_txt = []
        for i in range(len(example['prompt'])):
            text = f"{instruction_template}\n{example['prompt'][i]}\n{response_template}\n{example['response'][i]}[EOS]"
            batch_txt.append(text)

        # token to id
        outputs = tokenizer(batch_txt, return_attention_mask=False, truncation=True, max_length=max_seq_len)
        input_ids = [np.array(item, dtype=map_dtype) for item in outputs["input_ids"]]

        return {
            "input_ids": input_ids
        }

    dataset = load_dataset(path='parquet', data_files=file, split='train', cache_dir=cache_dir)
    dataset = dataset.map(batched_formatting_prompts_func, batched=True, num_proc=24,
                          remove_columns=dataset.column_names, desc='Running tokenizer on set: ').shuffle(23333)
    return dataset


def get_dpo_dataset(file: str,
                    cache_dir: str = '.cache'):
    def split_prompt_and_responses(samples: dict[str, str]) -> Dict[str, str]:
        prompts, chosens, rejects = [], [], []
        batch_size = len(samples['prompt'])
        for i in range(batch_size):
            # add an eos token for signal that end of sentence, using in generate.
            prompts.append(f"[BOS]{samples['prompt'][i]}[EOS]")
            chosens.append(f"[BOS]{samples['chosen'][i]}[EOS]")
            rejects.append(f"[BOS]{samples['rejected'][i]}[EOS]")

        return {
            'prompt': prompts,
            'chosen': chosens,
            'rejected': rejects,
        }

    dataset = load_dataset(path='json', data_files=file, split='train', cache_dir=cache_dir)
    dataset = dataset.map(split_prompt_and_responses, batched=True, num_proc=24,
                          desc='Running tokenizer on set: ').shuffle(2333)
    return dataset
