data_dict = {}
with open('./data/queryMap.txt', 'r') as file:
    lines = file.readlines()

gap = len(lines) // 2

for i in range(0, gap):
    key = lines[i].strip()
    value = lines[i + gap].strip()
    data_dict[key] = value

with open("./data/query_dict.txt", "w") as output_file:
    for key, value in data_dict.items():
        output_file.write(f'{key}: {value}\n')
