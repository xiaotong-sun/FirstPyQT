import os
import shutil
from tqdm import tqdm

# 源文件夹和目标文件夹的路径
source_folder = './data/query'
destination_folder = './data/search'

# 读取文件名列表
with open('./data/search.txt', 'r') as file:
    lines = file.readlines()

gap = len(lines) // 2
print(gap)

file_names = []
for i in range(0, gap):
    key = lines[i].strip()
    file_names.append(key)

# 遍历文件名列表，复制文件

for file_name in tqdm(file_names):
    source_file_path = os.path.join(source_folder, file_name)
    destination_file_path = os.path.join(destination_folder, file_name)

    # 复制文件
    shutil.copy(source_file_path, destination_file_path)
