import pandas as pd
from sklearn.utils import shuffle

# 读取CSV文件
csv_file_path = 'eval.csv'  # 请替换成你的CSV文件路径
df = pd.read_csv(csv_file_path)

# 打乱行顺序
df_shuffled = shuffle(df)

# 保存打乱后的结果到新的CSV文件
shuffled_csv_file_path = 'eval.csv'  # 请替换成你想保存的新CSV文件路径
df_shuffled.to_csv(shuffled_csv_file_path, index=False)
