import pandas as pd

# 读取原始CSV文件
original_data = pd.read_csv('total_predict_poi_data.csv')

# 从原始数据中选择labels
selected_data = original_data[original_data['labels'] == 5]

# 将选择的数据保存到新的CSV文件
selected_data.to_csv('label_5.csv', index=False)

