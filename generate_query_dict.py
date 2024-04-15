"""
    这个文件用于生成一个固化的字典文件（query_dict.txt）
    该字典文件用于获取指定图片的utm值，
    同时，可以用来判断我们指定的查询图片是否位于./data/search路径中
"""

data_dict = {}
with open('./data/query_in_test.txt', 'r') as file:
    lines = file.readlines()

gap = len(lines) // 2

for i in range(0, gap):
    key = lines[i].strip()
    value = lines[i + gap].strip()
    data_dict[key] = value

with open("./data/query_dict.txt", "w") as output_file:
    for key, value in data_dict.items():
        output_file.write(f'{key}: {value}\n')
