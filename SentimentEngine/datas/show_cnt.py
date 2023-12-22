import pandas as pd

# 读取CSV文件
df = pd.read_csv('train.csv')

# 打印每个标签对应的数据数量
label_counts = df['labels'].value_counts()
print("Label Counts:")
print(label_counts)
