import sys

sys.path.append("/home/image_team/image_team_docker_home/lgd/e_commerce_llm/minillama3/")

import os
import re
import torch

from datasketch import MinHash, MinHashLSH
from collections import defaultdict
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers import TrainingArguments, TrainerCallback


# 保留中文和英文、下划线，不要标点符号
NON_CHAR = re.compile("[^[\u4E00-\u9FA5|A-Za-z_0-9]")


class MyTrainerCallback(TrainerCallback):
    log_cnt = 0

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        '''
        在打印 n 次日志后清除cuda缓存，适合低显存设备，能防止OOM
        '''
        self.log_cnt += 1
        if self.log_cnt % 2 == 0:
            torch.cuda.empty_cache()

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        '''
        在 on_epoch_end 时保存一次模型。
        TrainingArguments的 save_strategy 中 epoch 和 steps 不兼容。要实现每隔 save_steps 步保存一次检查点，考虑到磁盘空间大小，最多只保存最近N个检查点。
        '''
        # 设置should_save=True并返回即可
        control.should_save = True
        return control


def _get_doc_mini_hash(doc: list[str] | str, num_perm: int) -> MinHash:
    '''
    获取一段文本的mini hash
    '''
    mini_hash = MinHash(num_perm=num_perm)
    for s in doc:
        mini_hash.update(s.encode('utf-8'))
    return mini_hash


class DropDatasetDuplicate:

    def __init__(self, threshold: float = 0.85, num_perm: int = 256) -> None:
        '''
        获取一个数据集中所有重复（相似的超过threshold）的index，输入为：list[str]，一个str元素为一段文本(doc)
        如输入： [a, b, c, d, c, d, e] 返回：{4, 5} (后面两个 c, d 的index)
        '''
        self.similar_index_cluster = defaultdict(set)
        self.data_lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.num_perm = num_perm

    def add_doc(self, index: object, doc: str, ) -> set[int]:
        '''
        添加文档，
        index： 文档的索引
        doc: 文档本身
        '''

        # 只保留中文和英文、下划线，不要标点符号
        doc = ''.join(NON_CHAR.split(doc))
        # doc = [''.join(t) for t in list(ngrams(doc, 3))]

        doc_hash = _get_doc_mini_hash(doc, self.num_perm)
        close_duplicates = self.data_lsh.query(doc_hash)

        self.data_lsh.insert(index, doc_hash)

        # 所有相似的doc在similar_index_cluster中的key都是最早出现的idx
        # 如：data中索引inndex 2, 7, 8, 9, 10, 12 是相似的，则在similar_index_cluster中表现为 {2: {8, 9, 10, 12}}
        if len(close_duplicates) > 0:
            min_idx = min(close_duplicates)
            self.similar_index_cluster[min_idx].add(index)

    def get_duplicate_indexs(self):
        '''
        返回所有的重复文档索引
        '''
        similar_index_cluster = self.similar_index_cluster
        need_to_remove_idx = set()

        for key_idx in similar_index_cluster.keys():
            need_to_remove_idx |= similar_index_cluster[key_idx]

        return need_to_remove_idx


def get_path_of_suffix_files(root: str,
                             suffix: str,
                             with_create_time: bool = False) -> list:
    '''
        获取指定目录下下指定后缀的所有文件的绝对路径
    '''
    suffix_files = []
    for root, _, files in os.walk(root):
        for file in files:
            if file.endswith(suffix):
                full_path = '{}/{}'.format(root, file)
                if with_create_time:
                    suffix_files.append((full_path, os.path.getctime(full_path)))
                else:
                    suffix_files.append(full_path)

    return suffix_files

