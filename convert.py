import pandas as pd
import random

# pd.options.display.max_rows = 1000000

datasets = ['electronic', 'food', 'clothing']
filenames = ['user_bundle']

# for dataset in datasets:
#     for filename in filenames:
#         df = pd.read_csv(f'./datasets/{dataset}/{filename}.csv', index_col=[0], usecols=['user ID', 'bundle ID'])
#         content = str(df)
#         print(content, file=open(f'./datasets/{dataset}/{filename}.txt', 'w'))

for dataset in datasets:
    with open(f"./datasets/{dataset}/user_bundle.txt", "r") as f:
        data = f.read().split('\n')

    random.shuffle(data)
    l = len(data) + 1

    train_data = data[:int(l * .70)]
    validation_data = data[int(l * .70):int(l * .80)]
    test_data = data[int(l * .80):]
    print('\n'.join(validation_data), file=open(f"./datasets/{dataset}/user_bundle_tune.txt", "w"))
    print('\n'.join(train_data), file=open(f"./datasets/{dataset}/user_bundle_train.txt", "w"))
    print('\n'.join(test_data), file=open(f"./datasets/{dataset}/user_bundle_test.txt", "w"))