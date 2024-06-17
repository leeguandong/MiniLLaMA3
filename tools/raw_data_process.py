import re
from os.path import dirname, abspath, exists, isdir
from os import remove, mkdir, walk
import pandas as pd
import numpy as np
from rich import progress
from fastparquet import ParquetFile, write
from opencc import OpenCC

punctuation = set("!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：\n")
en_punctuation = ",().!;:"
zh_punctuation = "，（）。！；："


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


def write_single_parquet_file(file_name: str, data_frame: pd.DataFrame) -> None:
    '''
    将dataframe写到单独的parquet file中
    '''
    append = False
    if exists(file_name):
        append = True

    write(file_name, data_frame, compression='GZIP', append=append)


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


def process_zh_wiki_data_to_datset(groups_cnt: int = 10000, max_len: int = 512, seed: int = 23333) -> None:
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


if __name__ == "__main__":
    PROJECT_ROOT = "/home/image_team/image_team_docker_home/lgd/e_commerce_llm/minillama3/data/"

    # 1.tokenizer 训练
    #convert_wiki_to_simple_zh()

    # 2.pt 训练数据
    process_zh_wiki_data_to_datset(groups_cnt=10000, max_len=512)
