import sys

sys.path.append("/home/image_team/image_team_docker_home/lgd/e_commerce_llm/minillama3/")

import re
import time
import ujson
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

from os.path import dirname, abspath, exists, isdir
from os import remove, mkdir, walk
from rich import progress
from fastparquet import ParquetFile, write
from opencc import OpenCC
from rich.console import Console
from rich.table import Table
from utils.functions import DropDatasetDuplicate, get_path_of_suffix_files

punctuation = set("!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：\n")
en_punctuation = ",().!;:"
zh_punctuation = "，（）。！；："


def count_my_parquet_data(parquet_file: str = None) -> None:
    '''
    统计dir目录下所有parquet数据集数据量
    '''
    my_data_files = []

    if not parquet_file:
        my_data_files = get_path_of_suffix_files(PROJECT_ROOT + '/data/my_data', '.parquet')
    elif isdir(parquet_file):
        my_data_files = get_path_of_suffix_files(parquet_file, '.parquet')
    elif parquet_file.endswith('.parquet'):
        my_data_files = [parquet_file]

    result = [['file_name', 'count']]
    all_cnt = 0
    for file in my_data_files:
        file_name = file.split('/')[-1]
        cur_cnt = 0
        pf = ParquetFile(file)

        for pf_chunk in pf:
            cur_cnt += pf_chunk.info['rows']

        all_cnt += cur_cnt
        result.append([file_name, cur_cnt])

    result.append(['汇总', all_cnt])

    # log.info(str(result), save_to_file=True)
    print(result)

    console = Console()
    table = Table(show_header=True, show_lines=True, )

    for col in result[0]:
        table.add_column(col)
    for i in range(1, len(result)):  # 跳过表头
        table.add_row(str(result[i][0]), str(result[i][1]))

    console.print(table)


def delete_file(file: str) -> bool:
    '''
    询问删除文件
    '''
    if exists(file):
        ans = input('delete file: {} ? Yes (y) or No (n)'.format(file))
        ans = ans.lower()
        if ans in ('yes', 'y'):
            remove(file)
            print('deleted.')
            return True
    return False


def convert_en_punctuation_to_zh_punct(sentence: str) -> str:
    '''
    将句子中的英文标点替换文中文标点
    '''
    n = len(zh_punctuation)
    for i in range(n):
        sentence = sentence.replace(en_punctuation[i], zh_punctuation[i])
    return sentence


def remove_duplicate_punctuation(sentence: str) -> str:
    '''
    删除句子中重复的标点符号、重复的空格，同时将换行变为特殊字符'\n'
    '''
    # 将空格（全角空格）替换为逗号, 可能会有重复的空客，下面删除重复标点会删除
    sentence = re.sub(' |　', '，', sentence)

    ans = ''
    n = len(sentence)
    p = 0
    while p < n:
        ans += sentence[p]

        while p + 1 < n and sentence[p] in punctuation and sentence[p + 1] in punctuation:
            p += 1
        p += 1

    return ans


def write_single_parquet_file(file_name: str,
                              data_frame: pd.DataFrame) -> None:
    '''
    将dataframe写到单独的parquet file中
    '''
    append = False
    if exists(file_name):
        append = True

    write(file_name, data_frame, compression='GZIP', append=append)


def read_and_write_template(read_file: str, write_to_file: str, call_back: object, group_cnt: int = 10000) -> None:
    '''
    处理数据读写模板，需要提供一个回调函数call_back，
    read_file: 原始数据文件
    write_to_file：处理后的要保存数据文件
    call_back：函数输入一个字符串，输出一个处理后的字典dict，如果输入的字符串为无效数据，请返回None
    group_cnt: parquet file分割行数
    如：
    >>> def call_back(inputs: str) -> dict:
    >>>     if check(inputs) not valid:
    >>>         return None
    ...
    ...    do something for inputs
    ...
    >>>     my_dict = {
    >>>             'prompt': inputs['p'],
    >>>             'response': inputs['a1'] + inputs['a2'],
    >>>             ...
    >>>         }
    >>>     return my_dict
    '''

    print('process file:{}'.format(read_file))
    start = time.time()

    raw_line_cnt = 0
    keep_line_cnt = 0

    with progress.open(read_file, 'r', encoding='utf-8') as f_read:
        cur_rows = []
        append = cur_rows.append
        for line in f_read:
            try:
                raw_line_cnt += 1

                write_dict = call_back(line)

                if write_dict is None: continue

                keep_line_cnt += 1
                append(write_dict)
                # ujson.dump(write_obj, f_write, indent=4, ensure_ascii=False)
                # ujson.dump(write_obj, f_write,  ensure_ascii=False,)
                # f_write.write('\n')

                if len(cur_rows) >= group_cnt:
                    df = pd.DataFrame(cur_rows)
                    write_single_parquet_file(write_to_file, df)
                    cur_rows = []
                    append = cur_rows.append

            except Exception as e:
                # log.error('处理文件异常：{}, content:{}'.format(str(e), line))
                print(line)
                raise e

        # end for
        # 处理末尾部分
        if len(cur_rows) > 0:
            df = pd.DataFrame(cur_rows)
            write_single_parquet_file(write_to_file, df)
            cur_rows = []

    end = time.time()

    print('原始文件:{}，共{}行，处理后剩余{}行，保存到文件：{}。耗时：{:.6}s' \
          .format(read_file, raw_line_cnt, keep_line_cnt, write_to_file, end - start))


def convert_wiki_to_simple_zh(buffer_size: int = 10000) -> None:
    '''
    将繁体wiki转换为简体Wiki
    '''
    raw_zh_wiki_file = PROJECT_ROOT + '/wiki.txt'
    save_zh_wiki_simple_file = PROJECT_ROOT + '/wiki.simple.txt'

    if exists(save_zh_wiki_simple_file):
        assert delete_file(save_zh_wiki_simple_file)

    cc = OpenCC('t2s')
    cur_rows = []
    append = cur_rows.append

    def procees_line(line: str) -> str:
        '''
        处理一行文本
        '''
        # 将繁体转换为简体
        line = cc.convert(line)

        line = re.sub(r"\「|\」|\｢|\｣|\『|\』", '\"', line)  # 将「」｢｣『』这些符号替换成引号
        line = re.sub(r"\，\）|\；\）", '）', line)  # 罗德·法尼(Rod Dodji Fanni，）
        line = re.sub(r"\（\，|\(\，", '（', line)  # 阿魯拉·基馬(Alula Girma (，

        line = convert_en_punctuation_to_zh_punct(line)  # 英文标点转换为中文标点
        line = remove_duplicate_punctuation(line)  # 删除中文空括号和重复的标点

        return line

    with progress.open(raw_zh_wiki_file, 'r', encoding='utf-8') as read_f:
        with open(save_zh_wiki_simple_file, 'a', encoding='utf-8') as write_f:
            for line in read_f:
                line = procees_line(line)
                if len(line.strip()) == 0: continue

                line = '{}\n'.format(line)
                append(line)

                if len(cur_rows) >= buffer_size:
                    write_f.writelines(cur_rows)
                    cur_rows = []
                    append = cur_rows.append

            if len(cur_rows) > 0:
                write_f.writelines(cur_rows)
                cur_rows = []


def remove_dataset_duplicate_rows(groups_cnt: int = 50000) -> None:
    '''
    使用mini_hash删除数据集中重复的部分
    '''
    from_parquet_files = PROJECT_ROOT + "wiki_zh_simple.parquet"

    save_file = PROJECT_ROOT + 'wiki_zh_simple_no_dulpticates.parquet'

    # 后续append写入，存在文件先删除
    if exists(save_file):
        assert delete_file(save_file)

    cur_rows = []
    all_cnt, keep_cnt = 0, 0
    row_index = -1
    drop_dataset_duplicate = DropDatasetDuplicate(threshold=0.85, num_perm=256)

    parquet_table = pq.read_table(from_parquet_files)
    all_cnt = parquet_table.num_rows

    # 先顺序遍历获取哪些行是重复的
    for prompt, response in progress.track(zip(parquet_table['prompt'], parquet_table['response']),
                                           total=parquet_table.num_rows):
        row_index += 1

        doc = f"{prompt.as_py()}{response.as_py()}"
        drop_dataset_duplicate.add_doc(index=row_index, doc=doc)

    row_index = -1
    need_to_drop_indexs = drop_dataset_duplicate.get_duplicate_indexs()

    # 再顺序遍历一遍，重复的行不添加到新的数据集
    for prompt, response in progress.track(zip(parquet_table['prompt'], parquet_table['response']),
                                           total=parquet_table.num_rows):
        row_index += 1  # 不管有没有跳过行, row_index都必须+1

        # 重复的行跳过
        if row_index in need_to_drop_indexs:
            continue

        cur_rows.append({'prompt': prompt.as_py(), 'response': response.as_py()})
        keep_cnt += 1

        if len(cur_rows) >= groups_cnt:
            df = pd.DataFrame(cur_rows)
            write_single_parquet_file(save_file, df)
            cur_rows = []

    # 处理末尾部分
    if len(cur_rows) > 0:
        df = pd.DataFrame(cur_rows)
        write_single_parquet_file(save_file, df)

    print("merge into file: {}, 全部数据共{}行，文档去重后剩余{}行".format(save_file, all_cnt, keep_cnt))


def shuffle_parquet_dataset(parquet_file: str,
                            shuffle_file: str,
                            seed: int = 23333,
                            groups_cnt: int = 65536) -> None:
    '''
    打乱一个parquet文件数据集
    '''
    if not exists(parquet_file):
        raise Exception('can not find parquet file: {}'.format(parquet_file))

    print('start shuffle...')
    pf = pq.read_table(parquet_file)
    df = pf.to_pandas()
    df = df.sample(frac=1.0, replace=False, random_state=seed, axis=0)

    if exists(shuffle_file):
        assert delete_file(shuffle_file)

    # 分块写入parquet，否则小内存读取直接OOM
    n = len(df)
    for i in range(0, n, groups_cnt):
        cur_group_df = df[i: i + groups_cnt]
        write_single_parquet_file(shuffle_file, cur_group_df)


def split_train_valid_test_datasets(source_parquet_file: str,
                                    max_len: int = 320,
                                    seed: int = 23333,
                                    train_ratio: float = 0.91,
                                    test_ratio: float = 0.0875,
                                    valid_ratio: float = 0.0025,
                                    groups_cnt: int = 50000) -> None:
    '''
    将原始数据拆分为训练集、测试集和验证集
    '''
    assert train_ratio + test_ratio + valid_ratio == 1.0

    # train_parquet_file = PROJECT_ROOT + 'wiki_zh_simple_no_dulpticates_shuffle_train_dataset.parquet'
    # test_parquet_file = PROJECT_ROOT + 'wiki_zh_simple_no_dulpticates_shuffle_test_dataset.parquet'
    # valid_parquet_file = PROJECT_ROOT + 'wiki_zh_simple_no_dulpticates_shuffle_valid_dataset.parquet'
    train_parquet_file = PROJECT_ROOT + 'belle_sft_data_zh_train_dataset.parquet'
    test_parquet_file = PROJECT_ROOT + 'belle_sft_data_zh_test_dataset.parquet'
    valid_parquet_file = PROJECT_ROOT + 'belle_sft_data_zh_valid_dataset.parquet'

    if exists(train_parquet_file): assert delete_file(train_parquet_file)
    if exists(test_parquet_file): assert delete_file(test_parquet_file)
    if exists(valid_parquet_file): assert delete_file(valid_parquet_file)

    np.random.seed(seed)

    train, test, valid = [], [], []

    parquet_table = pq.read_table(source_parquet_file)

    for prompt, response in progress.track(zip(parquet_table['prompt'], parquet_table['response']),
                                           total=parquet_table.num_rows):

        prompt, response = prompt.as_py(), response.as_py()
        rand = np.random.random()
        cur_data = {'prompt': ''.join(prompt[0: max_len]), 'response': ''.join(response[0: max_len])}

        if 0 <= rand < train_ratio:
            train.append(cur_data)
        elif train_ratio <= rand < train_ratio + test_ratio:
            test.append(cur_data)
        else:
            valid.append(cur_data)

        if len(train) >= groups_cnt:
            write_single_parquet_file(train_parquet_file, pd.DataFrame(train))
            train = []

        if len(test) >= groups_cnt:
            write_single_parquet_file(test_parquet_file, pd.DataFrame(test))
            test = []

        if len(valid) >= groups_cnt:
            write_single_parquet_file(valid_parquet_file, pd.DataFrame(valid))
            valid = []

    if len(train) > 0:
        write_single_parquet_file(train_parquet_file, pd.DataFrame(train))
        train = []

    if len(test) > 0:
        write_single_parquet_file(test_parquet_file, pd.DataFrame(test))
        test = []

    if len(valid) > 0:
        write_single_parquet_file(valid_parquet_file, pd.DataFrame(valid))
        valid = []


def process_zh_wiki_data_to_datset(groups_cnt: int = 10000,
                                   max_len: int = 512,
                                   seed: int = 23333) -> None:
    '''
    将Wiki中文数转换为问答数据集
    wiki 下载地址：https://dumps.wikimedia.org/zhwiki/
    将下载的bz2文件转换为wiki.txt参考：https://github.com/apertium/WikiExtractor
    '''
    raw_zh_wiki_file = PROJECT_ROOT + "wiki.txt"
    zhwiki_simple_file = PROJECT_ROOT + "wiki_zh_simple.parquet"

    # 删除已经存在的数据
    if exists(zhwiki_simple_file):
        assert delete_file(zhwiki_simple_file)

    # 将繁体转换为简体
    cc = OpenCC('t2s')
    all_cnt, keep_cnt = 0, 0

    # 构造问题的前缀
    prompt_prefix = [
        '什么是{}？',
        '介绍一下{}',
        '介绍一下什么是{}',
        '写一篇关于{}的介绍',
        '{}是什么？',
        '你知道{}吗？',
        '生成关于{}的介绍',
        '我想知道关于{}的详细信息',
        '你了解{}吗？',
        '请解释一下{}',
        '对于{}，你有什么了解或看法吗？',
        '请告诉我关于{}的信息',
        '请简要描述一下{}',
        '请提供有关{}的一些详细信息',
        '能否解释一下{}是什么?',
        '请分享一些关于{}的背景知识',
        '请简要概括一下{}',
        '能给我一些关于{}的背景资料吗?',
        '有关{}的信息可以分享一下吗？',
        '你能告诉我{}是什么吗？',
    ]

    def procees_line(line: str) -> str:
        '''
        处理一行文本
        '''
        # 将繁体转换为简体
        line = cc.convert(line)

        line = re.sub(r"\「|\」|\｢|\｣|\『|\』", '\"', line)  # 将「」｢｣『』这些符号替换成引号
        line = re.sub(r"\，\）|\；\）", '）', line)  # 罗德·法尼(Rod Dodji Fanni，）
        line = re.sub(r"\（\，|\(\，", '（', line)  # 阿魯拉·基馬(Alula Girma (，

        line = convert_en_punctuation_to_zh_punct(line)  # 英文标点转换为中文标点
        line = remove_duplicate_punctuation(line)  # 删除中文空括号和重复的标点

        return line

    np.random.seed(seed)
    choice = np.random.choice

    with progress.open(raw_zh_wiki_file, 'r', encoding='utf-8') as read_file:
        prompt = ''
        response = ''
        pre_line_len = 0
        cur_rows = []
        append = cur_rows.append
        for line in read_file:
            all_cnt += 1

            # prompt已经保存，但是仍有多余的行，这些行使得response的长度＞max_len，故跳过，不处理
            if len(prompt) == 0 and pre_line_len > 0:
                pre_line_len = len(line.strip())
                continue

            # 清洗一行
            line = procees_line(line)

            # 确定问题，pre_line_len是0，既是上一行是空行，则当前行是新的百科词条，设置为prompt
            if prompt == '' and line.endswith('：') and pre_line_len == 0:
                prompt = choice(prompt_prefix).format(line[0: -1])
                continue

            pre_line_len = len(line.strip())

            # 问题下来若干行为答案
            if prompt != '' and not line.endswith('：'):
                # 其实，pre_line_len已经是len(line.strip())了，如果len(line.strip())=0，既是当前行是0，则不管答案长度够不够，都需要保存了
                if len(response) + len(line) <= max_len and pre_line_len != 0:
                    response = '{}{}'.format(response, line)
                elif len(response) + len(line) > max_len or pre_line_len == 0:
                    # 长度超了或者当前的百科已经结束，保存一条样例
                    keep_cnt += 1
                    response = '{}{}'.format(response, line)
                    append({'prompt': prompt, 'response': ''.join(response[0: max_len])})
                    prompt = ''
                    response = ''

            # =groups_cnt保存到文件
            if len(cur_rows) >= groups_cnt:
                df = pd.DataFrame(cur_rows)
                write_single_parquet_file(zhwiki_simple_file, df)
                cur_rows = []
                append = cur_rows.append

        # end for
        if len(prompt) > 0 and len(response) > 0:
            keep_cnt += 1
            append({'prompt': prompt, 'response': response})

        if len(cur_rows) > 0:
            df = pd.DataFrame(cur_rows)
            write_single_parquet_file(zhwiki_simple_file, df)
            cur_rows = []

    print("merge into file: {}, 全部数据共{}行，清洗后剩余{}行".format(zhwiki_simple_file, all_cnt, keep_cnt))


def process_belle_knowledge_enhanced_dataset_for_finetune(max_len: int = 320, group_cnt: int = 50000) -> None:
    '''
    处理belle开源的知识增强数据集
    '''
    file_names = [
        "Belle_open_source_0.5M.json",
        'train_2M_CN.json',
        'generated_chat_0.4M.json',
    ]

    save_file = PROJECT_ROOT + '/belle_sft_data_zh.parquet'

    # 后续append写入，存在文件先删除
    if exists(save_file):
        assert delete_file(save_file)

    def process_function(line: str) -> dict:
        '''
        每行的处理函数
        '''
        item = ujson.loads(line)
        prompt = item['instruction']
        response = item['output']

        # 剔除翻译任务
        if 'translate' in prompt.lower(): return None
        for word in ('翻译', '英译', '译英', '中译', '译中', '汉译', '译汉'):
            if word in prompt:
                return None

        # 删除表格类任务
        if '表格' in prompt or '-----' in prompt or '-----' in response:
            return None

        if len(prompt) > max_len or len(response) > max_len:
            return None

        write_dict = {
            'prompt': prompt,
            'response': response
        }

        return write_dict

    for file in file_names:
        file = PROJECT_ROOT + file

        read_and_write_template(file, save_file, process_function)


if __name__ == "__main__":
    PROJECT_ROOT = "/home/image_team/image_team_docker_home/lgd/e_commerce_llm/minillama3/data/"

    # 1.tokenizer 训练
    # convert_wiki_to_simple_zh()

    # 2.pt 训练数据
    # 社区问答知识类 425w https://github.com/brightmart/nlp_chinese_corpus
    # 百度知道知识类 147w
    # 中国医药领域问答数据集
    # 金融问题
    # 知乎
    # 贝壳增强数据集
    # wiki百科
    # process_zh_wiki_data_to_datset(groups_cnt=10000, max_len=512)

    # 2.1 去重
    # remove_dataset_duplicate_rows(groups_cnt=50000)

    # 2.2 shuffle
    # shuffle_parquet_dataset(
    #     parquet_file=PROJECT_ROOT + 'wiki_zh_simple_no_dulpticates.parquet',
    #     shuffle_file=PROJECT_ROOT + "wiki_zh_simple_no_dulpticates.shuffle.parquet", seed=2333)

    # 2.3 划分数据集
    # split_train_valid_test_datasets(
    #     source_parquet_file=PROJECT_ROOT + "wiki_zh_simple_no_dulpticates.shuffle.parquet",
    #     max_len=320,
    #     groups_cnt=50000)

    # count_my_parquet_data(PROJECT_ROOT + 'wiki_zh_simple_no_dulpticates_shuffle_train_dataset.parquet')
    # count_my_parquet_data(PROJECT_ROOT + 'wiki_zh_simple_no_dulpticates_shuffle_test_dataset.parquet')
    # count_my_parquet_data(PROJECT_ROOT + 'wiki_zh_simple_no_dulpticates_shuffle_valid_dataset.parquet')

    # 3.sft belle数据制作
    # process_belle_knowledge_enhanced_dataset_for_finetune(max_len=320, group_cnt=50000)

    # 3.1 划分数据集
    split_train_valid_test_datasets(
        source_parquet_file=PROJECT_ROOT + "belle_sft_data_zh.parquet",
        max_len=320,
        groups_cnt=50000)
