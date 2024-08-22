# import open3d as o3d

# # 加载点云数据
# point_cloud = o3d.io.read_point_cloud("./scans.pcd")

# # 使用Statistical Outlier Removal来移除噪声点
# def remove_noise_statistical_outlier(point_cloud, nb_neighbors=20, std_ratio=2.0):
#     cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
#     inlier_cloud = point_cloud.select_by_index(ind)
#     return inlier_cloud

# # 使用Radius Outlier Removal来移除噪声点
# def remove_noise_radius_outlier(point_cloud, nb_points=16, radius=0.05):
#     cl, ind = point_cloud.remove_radius_outlier(nb_points=nb_points, radius=radius)
#     inlier_cloud = point_cloud.select_by_index(ind)
#     return inlier_cloud

# # 选择一种方法来移除噪声，可以根据需要选择不同的方法

# # 方法1：统计滤波（Statistical Outlier Removal）
# cleaned_point_cloud = remove_noise_statistical_outlier(point_cloud, nb_neighbors=20, std_ratio=2.0)

# # 方法2：半径滤波（Radius Outlier Removal）
# # cleaned_point_cloud = remove_noise_radius_outlier(point_cloud, nb_points=16, radius=0.05)

# # 保存处理后的点云
# o3d.io.write_point_cloud("./cleaned_point_cloud.pcd", cleaned_point_cloud)

# # 可视化原始和去噪后的点云
# print("Original point cloud:")
# # o3d.visualization.draw_geometries([point_cloud])

# # print("Cleaned point cloud:")
# # o3d.visualization.draw_geometries([cleaned_point_cloud])







import open3d as o3d
import numpy as np
# 加载点云数据
point_cloud = o3d.io.read_point_cloud("cleaned_point_cloud.pcd")

# 体素网格降采样
def voxel_downsample(point_cloud, voxel_size=0.5):
    downsampled_point_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size)
    return downsampled_point_cloud

# 随机降采样
def random_downsample(point_cloud, sample_ratio=0.5):
    downsampled_point_cloud = point_cloud.random_down_sample(sample_ratio=sample_ratio)
    return downsampled_point_cloud

# 选择一种方法来降采样

# 方法1：体素网格降采样
downsampled_point_cloud = voxel_downsample(point_cloud, voxel_size=0.3)
downsampled_point_cloud


red_color = np.array([[1, 1, 1] for i in range(len(downsampled_point_cloud.points))])
downsampled_point_cloud.colors = o3d.utility.Vector3dVector(red_color)
# 方法2：随机降采样
# downsampled_point_cloud = random_downsample(point_cloud, sample_ratio=0.5)

# 保存降采样后的点云
o3d.io.write_point_cloud("./downsampled_point_cloud1.pcd", downsampled_point_cloud)

# 可视化原始和降采样后的点云
# print("Original point cloud:")
# o3d.visualization.draw_geometries([point_cloud])

# print("Downsampled point cloud:")
# o3d.visualization.draw_geometries([downsampled_point_cloud])
