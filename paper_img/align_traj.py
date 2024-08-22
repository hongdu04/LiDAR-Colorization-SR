from evo.core import metrics
from evo.core.units import Unit

from evo.tools import log
log.configure_logging(verbose=True, debug=True, silent=False)

import pprint
import numpy as np

from evo.tools import plot
import matplotlib.pyplot as plt

# temporarily override some package settings
from evo.tools.settings import SETTINGS
from evo.core import sync
import copy
from evo.tools import file_interface
import pandas as pd

ref_file = "./GT_1.csv"
est_file = "./adjusted_odometry_data.csv"
max_diff = 0.01
traj_ref = file_interface.read_tum_trajectory_file(ref_file)
traj_est = file_interface.read_tum_trajectory_file(est_file)
# traj_est.timestamps = traj_ref.timestamps

traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff)

traj_est_aligned = copy.deepcopy(traj_est)
traj_est_aligned.align(traj_ref, correct_scale=False, correct_only_scale=False)

print(traj_est_aligned.timestamps.shape,traj_est_aligned.positions_xyz.shape,traj_est_aligned.orientations_quat_wxyz.shape)




# 将这些数据合并为一个二维数组 (n, 8)
combined_data = np.hstack((traj_est_aligned.timestamps.reshape(-1, 1), traj_est_aligned.positions_xyz, traj_est_aligned.orientations_quat_wxyz))

# 使用 pandas 将数据保存为无表头、空格分隔的CSV文件
df = pd.DataFrame(combined_data)
df.to_csv('output_old.csv', sep=' ', index=False, header=False)