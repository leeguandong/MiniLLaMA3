import json
import os

# 获取文件夹中所有的json文件
folder_path = "/home/image_team/image_team_docker_home/lgd/e_commerce_llm/minillama3/TinyStoriesZh/"  # 你的文件夹路径
json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

# 初始化一个列表来保存所有的故事
stories = []

# 遍历每个json文件
for json_file in json_files:
    file_path = os.path.join(folder_path, json_file)

    # 打开并读取json文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 如果数据是一个列表，遍历每个元素
    if isinstance(data, list):
        for item in data:
            # 获取故事并添加到列表中
            story = item.get('story', '')
            stories.append(story)
    # 如果数据是一个字典，直接获取故事
    elif isinstance(data, dict):
        story = data.get('story', '')
        stories.append(story)

# 将所有的故事保存到一个字典中
output_data = {'text': stories}

# 将字典写入到jsonl文件中
with open("/home/image_team/image_team_docker_home/lgd/e_commerce_llm/tokenizer_wiki/TinyStoriesZh/TinyStoriesZh-GPT4-train.jsonl", 'w', encoding='utf-8') as f:
    f.write(json.dumps(output_data, ensure_ascii=False) + '\n')