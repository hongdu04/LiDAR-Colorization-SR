# import open3d as o3d
# import numpy as np
# import pandas as pd

# # 加载点云数据



# point_cloud = o3d.io.read_point_cloud("downsampled_point_cloud1.pcd")

# # 从CSV文件中读取轨迹数据
# trajectory = pd.read_csv("kp_1st.txt", delim_whitespace=True)
# trajectory.columns = ['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']

# trajectory_points = trajectory[['x', 'y', 'z']].values

# # 创建一个LineSet来表示轨迹
# def create_lineset_from_trajectory(traj, color=[1, 0, 0]):
#     points = traj
#     lines = [[i, i + 1] for i in range(len(traj) - 1)]
#     colors = [color for i in range(len(lines))]
    
#     line_set = o3d.geometry.LineSet()
#     line_set.points = o3d.utility.Vector3dVector(points)
#     line_set.lines = o3d.utility.Vector2iVector(lines)
#     line_set.colors = o3d.utility.Vector3dVector(colors)
    
#     return line_set

# # 创建轨迹的LineSet对象
# trajectory_lineset = create_lineset_from_trajectory(trajectory_points)

# # 标记起始点和结束点
# start_point = trajectory_points[0]
# end_point = trajectory_points[-1]

# # 增大球体的半径，使标记点更明显
# sphere_radius = 0.8  # 这里将半径设为0.5，视点云规模调整

# # 创建起始点和结束点的几何体（较大的球体）
# start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
# start_sphere.translate(start_point)
# start_sphere.paint_uniform_color([0, 1, 0])  # 绿色

# end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
# end_sphere.translate(end_point)
# end_sphere.paint_uniform_color([1, 0, 0])  # 红色

# # 可视化点云、轨迹和标记点
# o3d.visualization.draw_geometries([point_cloud, trajectory_lineset, start_sphere, end_sphere])





















# import open3d as o3d
# import numpy as np
# import pandas as pd

# # 加载点云数据
# point_cloud = o3d.io.read_point_cloud("downsampled_point_cloud1.pcd")

# white_color = np.array([[1, 1, 1] for i in range(len(point_cloud.points))])
# point_cloud.colors = o3d.utility.Vector3dVector(white_color)

# # 读取第一个轨迹CSV文件 (无表头，空格分隔)
# trajectory1 = pd.read_csv("output.csv", sep=' ', header=None)
# trajectory1.columns = ['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']

# # 读取第二个轨迹CSV文件 (无表头，空格分隔)
# trajectory2 = pd.read_csv("gt_1st.csv", sep=' ', header=None)
# trajectory2.columns = ['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']

# trajectory3 = pd.read_csv("output_old.csv", sep=' ', header=None)
# trajectory3.columns = ['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']

# # 提取轨迹中的位置数据
# trajectory_points1 = trajectory1[['x', 'y', 'z']].values
# trajectory_points2 = trajectory2[['x', 'y', 'z']].values
# trajectory_points3 = trajectory3[['x', 'y', 'z']].values
# # 创建一个LineSet来表示轨迹
# def create_lineset_from_trajectory(traj, color=[1, 0, 0], is_dashed=False):
#     points = traj
#     if is_dashed:
#         # 将虚线分段
#         lines = [[i, i + 1] for i in range(0, len(traj) - 1, 2)]
#     else:
#         lines = [[i, i + 1] for i in range(len(traj) - 1)]
#     colors = [color for _ in range(len(lines))]
    
#     line_set = o3d.geometry.LineSet()
#     line_set.points = o3d.utility.Vector3dVector(points)
#     line_set.lines = o3d.utility.Vector2iVector(lines)
#     line_set.colors = o3d.utility.Vector3dVector(colors)
    
#     return line_set

# # 创建两个轨迹的LineSet对象
# trajectory_lineset1 = create_lineset_from_trajectory(trajectory_points1, color=[1, 0, 0])  # 红色
# trajectory_lineset2 = create_lineset_from_trajectory(trajectory_points2, color=[0, 0, 1], is_dashed=True)  # 黑色虚线
# trajectory_lineset3 = create_lineset_from_trajectory(trajectory_points3, color=[0, 1, 0])  # 黑色虚线

# # 创建可视化窗口并设置背景颜色为白色
# vis = o3d.visualization.Visualizer()
# vis.create_window(window_name='Point Cloud with Trajectories', width=800, height=600)
# opt = vis.get_render_option()
# opt.background_color = np.array([0.0, 0.0, 0.0])  

# # 将点云和轨迹添加到可视化器中
# vis.add_geometry(point_cloud)
# vis.add_geometry(trajectory_lineset1)
# vis.add_geometry(trajectory_lineset2)
# vis.add_geometry(trajectory_lineset3)

# # 启动可视化
# vis.run()

# # 关闭窗口
# vis.destroy_window()





# import open3d as o3d
import numpy as np
# import pandas as pd

# # 加载点云数据
# point_cloud = o3d.io.read_point_cloud("downsampled_point_cloud1.pcd")

# # 确保点云的颜色属性为空或正确设置
# if not point_cloud.has_colors():
#     white_color = np.ones((len(point_cloud.points), 3))  # 使用白色
#     point_cloud.colors = o3d.utility.Vector3dVector(white_color)

# # 更新点云的颜色
# point_cloud.colors = o3d.utility.Vector3dVector(np.array([[1, 1, 1] for _ in range(len(point_cloud.points))]))

# # 读取轨迹文件并创建 LineSet 如之前代码
# # 读取第一个轨迹CSV文件 (无表头，空格分隔)
# trajectory1 = pd.read_csv("output.csv", sep=' ', header=None)
# trajectory1.columns = ['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']

# # 读取第二个轨迹CSV文件 (无表头，空格分隔)
# trajectory2 = pd.read_csv("gt_1st.csv", sep=' ', header=None)
# trajectory2.columns = ['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']

# trajectory3 = pd.read_csv("output_old.csv", sep=' ', header=None)
# trajectory3.columns = ['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']

# # 提取轨迹中的位置数据
# trajectory_points1 = trajectory1[['x', 'y', 'z']].values
# trajectory_points2 = trajectory2[['x', 'y', 'z']].values
# trajectory_points3 = trajectory3[['x', 'y', 'z']].values
# # 创建一个LineSet来表示轨迹
# def create_lineset_from_trajectory(traj, color=[1, 0, 0], is_dashed=False):
#     points = traj
#     if is_dashed:
#         # 将虚线分段
#         lines = [[i, i + 1] for i in range(0, len(traj) - 1, 2)]
#     else:
#         lines = [[i, i + 1] for i in range(len(traj) - 1)]
#     colors = [color for _ in range(len(lines))]
    
#     line_set = o3d.geometry.LineSet()
#     line_set.points = o3d.utility.Vector3dVector(points)
#     line_set.lines = o3d.utility.Vector2iVector(lines)
#     line_set.colors = o3d.utility.Vector3dVector(colors)
    
#     return line_set

# # 创建两个轨迹的LineSet对象
# trajectory_lineset1 = create_lineset_from_trajectory(trajectory_points1, color=[1, 0, 0])  # 红色
# trajectory_lineset2 = create_lineset_from_trajectory(trajectory_points2, color=[0, 0, 1], is_dashed=True)  # 黑色虚线
# trajectory_lineset3 = create_lineset_from_trajectory(trajectory_points3, color=[0, 1, 0])  # 黑色虚线
# # 创建可视化窗口并设置背景颜色
# vis = o3d.visualization.Visualizer()
# vis.create_window(window_name='Point Cloud with Trajectories', width=800, height=600)
# opt = vis.get_render_option()
# opt.background_color = np.array([1.0, 1.0, 1.0])  

# # 将点云和轨迹添加到可视化器中
# vis.add_geometry(point_cloud)
# vis.add_geometry(trajectory_lineset1)
# vis.add_geometry(trajectory_lineset2)
# vis.add_geometry(trajectory_lineset3)

# # 确保更新几何体
# vis.update_geometry(point_cloud)
# vis.update_geometry(trajectory_lineset1)
# vis.update_geometry(trajectory_lineset2)
# vis.update_geometry(trajectory_lineset3)

# # 启动可视化
# vis.run()

# # 关闭窗口
# vis.destroy_window()







import open3d as o3d

# 读取原始点云并转换为 PLY 格式保存
point_cloud = o3d.io.read_point_cloud("downsampled_point_cloud1.pcd")
o3d.io.write_point_cloud("converted_point_cloud.ply", point_cloud)

# 重新加载并设置颜色
point_cloud = o3d.io.read_point_cloud("converted_point_cloud.ply")
white_color = np.array([[1, 1, 1] for _ in range(len(point_cloud.points))])
point_cloud.colors = o3d.utility.Vector3dVector(white_color)

o3d.visualization.draw_geometries([point_cloud])

