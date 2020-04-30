import os

stats = {'mean': [], 'min': [], 'max': [], 'first_quartile': [], 'median': [], 'third_quartile': [], 'std': []}

feature_names = []
feature_stats = {}

with open("track_feature_stats.txt", 'r', encoding='utf-8') as file:
    temp = filter(None, (line.rstrip() for line in file))
    for line in temp:
        line = line.replace(",", ".")
        parts = line.split(" ")
        feature_names.append(parts[0])
        feature_stats[parts[0]] = parts[1:]

for name in feature_names:
    stats['std'].append(float(feature_stats[name][0]))
    stats['mean'].append(float(feature_stats[name][1]))
    stats['first_quartile'].append(float(feature_stats[name][2]))
    stats['median'].append(float(feature_stats[name][3]))
    stats['third_quartile'].append(float(feature_stats[name][4]))
    stats['min'].append(float(feature_stats[name][5]))
    stats['max'].append(float(feature_stats[name][6]))

print(stats['std'])
print(stats['mean'])
print(stats['first_quartile'])
print(stats['median'])
print(stats['third_quartile'])
print(stats['min'])
print(stats['max'])

with open("track_feature_stats.py", 'w') as target_file:
    target_file.writelines(["import numpy as np", os.linesep])
    target_file.writelines(["Stats = {", os.linesep])
    target_file.writelines(["'std': np.array(" + str(stats['std']) + "),", os.linesep])
    target_file.writelines(["'mean': np.array(" + str(stats['mean']) + "),", os.linesep])
    target_file.writelines(["'first_quartile': np.array(" + str(stats['first_quartile']) + "),", os.linesep])
    target_file.writelines(["'median': np.array(" + str(stats['median']) + "),", os.linesep])
    target_file.writelines(["'third_quartile': np.array(" + str(stats['third_quartile']) + "),", os.linesep])
    target_file.writelines(["'min': np.array(" + str(stats['min']) + "),", os.linesep])
    target_file.writelines(["'max': np.array(" + str(stats['max']) + "),", os.linesep])
    target_file.writelines(["}", os.linesep])
