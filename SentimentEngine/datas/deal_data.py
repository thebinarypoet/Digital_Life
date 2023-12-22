import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('data_modified_2.csv')

# 对每个标签进行随机采样删除1500条数据
random_state = 42  # 可以根据需要选择一个随机种子
labels_to_remove = [2]

np.random.seed(random_state)  # 设置随机种子

for label in labels_to_remove:
    # 获取当前标签的索引
    label_indices = df[df['labels'] == label].index

    # 随机选择1500条数据的索引
    random_indices = np.random.choice(label_indices, size=100, replace=False)

    # 删除选择的数据
    df = df.drop(random_indices)

# 保存修改后的DataFrame到新的CSV文件
df.to_csv('data_modified_2.csv', index=False)
