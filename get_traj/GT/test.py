import pandas as pd

# 读取CSV文件，由于没有表头，我们使用header=None，并指定列名
df = pd.read_csv('./origonal/gt_n_hard.csv', header=None, sep=' ', names=['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])

# 确保时间戳列正确解析为浮点数
df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')

# 计算时间戳的差值
time_diffs = df['timestamp'].diff().fillna(0)

# 扩大差值10倍
scaled_diffs = time_diffs * 10

# 重新计算时间戳，从0开始，使用累加的方法
df['timestamp'] = scaled_diffs.cumsum()
# 如果需要保存到新的CSV文件中，取消注释下面的行
df.to_csv('modified_file.csv', index=False, header=False,sep=' ')
