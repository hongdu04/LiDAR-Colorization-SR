import open3d as o3d
import numpy as np
import os

def calculate_average_point_count(folder_path):
    point_counts = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pcd"):
            file_path = os.path.join(folder_path, filename)
            pcd = o3d.io.read_point_cloud(file_path)
            points = np.asarray(pcd.points)
            # 过滤掉所有值为（0,0,0）的点
            filtered_points = points[np.any(points != 0, axis=1)]
            point_counts.append(len(filtered_points))
    
    if point_counts:
        average_points = np.mean(point_counts)
        print(f"Average point count (excluding (0,0,0)) across all PCD files: {average_points}")
    else:
        print("No PCD files found or they don't contain any points.")

# 替换为你的文件夹路径
folder_path = './kp_5m_15/'
calculate_average_point_count(folder_path)
