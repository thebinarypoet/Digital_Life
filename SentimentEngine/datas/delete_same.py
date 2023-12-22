import pandas as pd

# 读取CSV文件
df = pd.read_csv('label_1.csv')

# 删除重复行
df.drop_duplicates(inplace=True)

# 将去重后的数据保存回CSV文件
df.to_csv('your_file_no_duplicates.csv', index=False)
