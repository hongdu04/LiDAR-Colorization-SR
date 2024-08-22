import pandas as pd

# 读取CSV文件
filename = 'EvoLIOPose.csv'
data = pd.read_csv(filename, sep=' ', header=None)

# 提取原始时间戳
original_timestamps = data[0]

# 计算时间差
time_deltas = original_timestamps.diff().fillna(0)

# 设定新的开始时间戳
new_start_time = 0

# 计算新的时间戳
new_timestamps = new_start_time + time_deltas.cumsum()

# 将新的时间戳替换到数据中
data[0] = new_timestamps

# 保存到新的CSV文件
new_filename = 'adjusted_odometry_data.csv'
data.to_csv(new_filename, sep=' ', header=False, index=False)

print(f"调整后的数据已保存至 {new_filename}")
