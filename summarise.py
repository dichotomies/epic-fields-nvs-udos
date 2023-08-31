
from glob import glob
import os
import torch
import math

# task 1

root = 'outputs/task1/test'

print('Summary for Task 1:\n')
for action_type in ['within_action', 'outside_action', 'outside_action_easy']:
    print(f'Action Type: {action_type}')
    with open(os.path.join(root, f'{action_type}/neuraldiff/results.txt')) as f:
        x = f.read()
        print(x)

# task 2

motion_types = ['dynamic', 'dynamic_semistatic', 'dynamic_semistatic_no_body_parts']

root = 'outputs/task2/test/'

summary = {model_type: {} for model_type in ['neuraldiff', 'mg']}


vids = set(
        os.listdir(os.path.join(root, 'dynamic/within_action/neuraldiff')
    )
).intersection(
    set(os.listdir(os.path.join(root, 'dynamic/within_action/mg'))
    )
)

for model_type in summary:
    for motion_type in motion_types:
        map_scores = []
        dir = os.path.join(root, f'{motion_type}/within_action/{model_type}/')
        for vid in vids:
            results = torch.load(os.path.join(dir, vid, 'results.pt'))
            map_score = results['map']
            if math.isnan(map_score):
                map_score = 0
            map_scores.append(map_score)
        summary[model_type][motion_type] = torch.mean(torch.FloatTensor(map_scores))

print("Summary for Task 2:")
print("")
print(f"""Motion Type {' ':<5} Dynamic {' ':<5} Dynamic {' ':<5} Dynamic""")
print(f""" {' ':<30} +Semistatic {' ':<1} +Semistatic""")
print(f""" {' ':<44} -Body Parts""")
print('-----------')
for model_type in summary:
    print(f'{model_type:<18}', end='')
    for motion_type in summary[model_type]:
        print(f"{summary[model_type][motion_type]:.2f}{' ':<9}", end='')
    print()
