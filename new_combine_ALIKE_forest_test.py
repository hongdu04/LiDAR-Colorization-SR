from deoldify.device_id import DeviceId
from deoldify import device

# choices:  CPU, GPU0...GPU7
device.set(device=DeviceId.GPU0)
from deoldify.visualize import *

plt.style.use("dark_background")
import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*?Your .*? set is empty.*?"
)
render_factor = 35
from deoldify.visualize import get_image_colorizer
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time

# 假设ModelImageVisualizer已经被正确修改并且已经加载
visualizer = get_image_colorizer(render_factor=render_factor, artistic=True)

import sys
import os

alike_module_path = os.path.abspath(os.path.join(os.getcwd(), "../ALIKE/"))
sys.path.append(alike_module_path)

from alike import ALike, configs


# 定义直接在代码中设置的参数
input_path = ""  # 图片目录、电影文件或"camera0"
model_config = "alike-l"  # 模型配置
device = "cuda"  # 运行设备
top_k = -1  # 检测顶部K个关键点
scores_th = 0.3  # 检测器分数阈值
n_limit = 1000  # 被检测的最大关键点数
output_dir = "./keypoints"  # 关键点的输出目录

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# image_loader = ImageLoader(input_path)
model_alike = ALike(
    **configs[model_config],
    device=device,
    top_k=top_k,
    scores_th=scores_th,
    n_limit=n_limit,
)

import open3d as o3d
import os
import json
import time
import importlib
from collections import OrderedDict
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as patches

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../SuperPointPretrainedNetwork')))
from superpoint import SuperPointFrontend
import copy
import glob
import logging
import argparse
import numpy as np
from tqdm import tqdm
import open3d as o3d
import json
import time


def tensor_to_np(tensor):
    tensor = tensor.cpu()
    np_img = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    return np_img


def sample(net, device, lr_image, scales):
    results = {}
    for scale in scales:
        lr = lr_image.unsqueeze(0).to(device)
        t1 = time.time()
        sr = net(lr, scale).detach().squeeze(0)
        t2 = time.time()
        sr_np = tensor_to_np(sr)
        results[scale] = sr_np
        # print(f"Scale {scale}, Time taken: {t2-t1:.3f}s")
    return results


def get_SR(img):

    lr_image_tensor = transform(img)

    scales = [2, 4]
    sr_images = sample(net, device, lr_image_tensor, scales)
    # print(img.shape)
    width = int(img.shape[0] * 0.5)
    height = int(img.shape[1] * 0.5)
    dim = (width, height)

    return sr_images, np.array(img)
    # return  np.array(img)


ckpt_path = "../checkpoint/carn.pth"
signal_dir = "./dataset/data_forest/signal/"
range_dir = "./dataset/data_forest/range/"
pcd_dir = "./dataset/data_forest/pcd/"
output_dir = "./kp_data/kp_forest_1000_35_r_r2_s_s2c_test_test"
img_output = "./kp_data/img_forest_1000_35_r_r2_s_s2c_test_test"
group = 1
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(img_output):
    os.makedirs(img_output)

module = importlib.import_module("model.{}".format("carn"))
net = module.Net(multi_scale=True, group=group)

state_dict = torch.load(ckpt_path)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    new_state_dict[k] = v

net.load_state_dict(new_state_dict)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)
transform = transforms.Compose([transforms.ToTensor()])


import numpy as np
import copy


class SimpleTracker(object):
    def __init__(self):
        self.pts_prev = None
        self.desc_prev = None

    def update(self, img, pts, desc):
        matched_points = []
        N_matches = 0
        current_frame_points = np.array([], dtype=int).reshape(0, 2)
        if self.pts_prev is None:
            self.pts_prev = pts
            self.desc_prev = desc
            current_frame_points = np.array(
                [(int(round(pt[0])), int(round(pt[1]))) for pt in pts]
            )
        else:
            matches = self.mnn_mather(self.desc_prev, desc)
            mpts1, mpts2 = self.pts_prev[matches[:, 0]], pts[matches[:, 1]]
            N_matches = len(matches)

            # 收集匹配的点坐标
            current_frame_points = np.array(
                [(int(round(p2[0])), int(round(p2[1]))) for p2 in mpts2]
            )

            self.pts_prev = pts

            self.pts_prev = pts
            self.desc_prev = desc

        return current_frame_points, N_matches

    def mnn_mather(self, desc1, desc2):
        sim = desc1 @ desc2.transpose()
        sim[sim < 0.8] = 0
        nn12 = np.argmax(sim, axis=1)
        nn21 = np.argmax(sim, axis=0)
        ids1 = np.arange(0, sim.shape[0])
        mask = ids1 == nn21[nn12]
        matches = np.stack([ids1[mask], nn12[mask]], axis=-1)
        return matches


import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import open3d as o3d
from tqdm import tqdm
import threading
import concurrent.futures


def get_adjacent_coordinates_vectorized(out_signals):
    # 确保输入是一个 numpy 数组
    out_signals = np.array(out_signals)

    # 计算 floor 和 ceil 坐标
    floor_coords = np.floor(out_signals).astype(int)
    ceil_coords = np.ceil(out_signals).astype(int)

    # 分别获取所有 x 和 y 的 floor 和 ceil 值
    x_floor, y_floor = floor_coords[:, 0], floor_coords[:, 1]
    x_ceil, y_ceil = ceil_coords[:, 0], ceil_coords[:, 1]

    # 生成四个相邻坐标的矩阵
    adjacent_coords = np.stack(
        [
            np.column_stack([x_floor, y_floor]),
            np.column_stack([x_ceil, y_floor]),
            np.column_stack([x_floor, y_ceil]),
            np.column_stack([x_ceil, y_ceil]),
        ],
        axis=1,
    )

    # 改变数组形状为 (8, 2)
    adjacent_coords = adjacent_coords.reshape(-1, 2)

    return adjacent_coords


font = cv2.FONT_HERSHEY_SIMPLEX

# 设置文字的大小
font_scale = 0.5

# 设置文字的厚度
thickness = 1

# 获取文字框的大小


tracker_range = SimpleTracker()
tracker_range_2X = SimpleTracker()
tracker_signal = SimpleTracker()
tracker_signal_color = SimpleTracker()
tracker_signal2X = SimpleTracker()
tracker_signal2X_color = SimpleTracker()


def adjust_gamma(image, gamma=1.0):
    # 建立一个映射表
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")

    # 应用gamma校正使用映射表
    return cv2.LUT(image, table)


def remove_duplicates_ndarray(arrays):
    seen = set()
    unique_arrays = []
    for arr in arrays:
        # 将numpy数组转换为元组
        tup = tuple(arr.tolist())
        if tup not in seen:
            unique_arrays.append(arr)  # 添加原始numpy数组
            seen.add(tup)
    return unique_arrays


def process_image(img, color, label, model, tracker, background_img, font, font_scale, thickness, text_y_offset, results, index):
    start_time = time.time()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pred = model(img, sub_pixel=True)
    keypoints = pred["keypoints"]
    descriptors = pred["descriptors"]
    out, N_matches = tracker.update(img, keypoints, descriptors)
    
    # print(f"Compare {label}: {N_matches}, {len(keypoints)}")
    out[:, 0] *= 2  # Adjust based on your specific needs
    out[:, 1] *= 1
    out = get_adjacent_coordinates_vectorized(out)

    # for pt in out:
    #     cv2.circle(background_img, (pt[0], pt[1]), 1, color, -1)
    
    # (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
    # text_position = (background_img.shape[1] - text_width - 10, text_height + text_y_offset)
    # cv2.putText(background_img, label, text_position, font, font_scale, color, thickness, cv2.LINE_AA)
    
    results[index] = out
    print('step process image', time.time() - start_time)

def detect_keypoints_and_save_pcd(
    range_img,
    range_2t,
    ori_sgn_img,
    sng_2t_color,
    model_alike,
    pcd_path,
    output_path,
    img_output_path,
):

    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
    ]  # Different colors for different images

    background_img = ori_sgn_img.copy()

    # Using threading for parallel processing
    threads = []
    results = [None] * 4  # Pre-allocated list to store results from threads

    # Start threads for each image processing task
    threads.append(threading.Thread(target=process_image, args=(range_img, colors[0], "range", model_alike, tracker_range, background_img, font, font_scale, thickness, 10, results, 0)))
    threads.append(threading.Thread(target=process_image, args=(range_2t, colors[0], "range2X", model_alike, tracker_range_2X, background_img, font, font_scale, thickness, 60, results, 1)))
    threads.append(threading.Thread(target=process_image, args=(ori_sgn_img, colors[1], "signal", model_alike, tracker_signal, background_img, font, font_scale, thickness, 20, results, 2)))
    threads.append(threading.Thread(target=process_image, args=(sng_2t_color, colors[4], "signal2X_color", model_alike, tracker_signal2X_color, background_img, font, font_scale, thickness, 50, results, 3)))

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    # Combine results from all threads
    keypoints_all_images = []
    for result in results:
        keypoints_all_images.extend(result)

    print(f"Total keypoints detected: {len(keypoints_all_images)}")
    # cv2.imwrite(img_output_path, background_img)  # Save the image with keypoints

    keypoints_all_images = remove_duplicates_ndarray(keypoints_all_images)

    # # Process and save point cloud
    # pcd = o3d.io.read_point_cloud(pcd_path)
    # points = np.asarray(pcd.points)
    # height, width = 128, 2048
    # data = points.reshape((height, width, -1))

    # keypoint_area_pcds = []
    # for kpt in keypoints_all_images:
    #     x, y = kpt
    #     if 0 <= x < width and 0 <= y < height:
    #         keypoint_area_pcds.append(data[y, x])

    # keypoint_cloud = o3d.geometry.PointCloud()
    # keypoint_cloud.points = o3d.utility.Vector3dVector(keypoint_area_pcds)
    # o3d.io.write_point_cloud(output_path, keypoint_cloud)


# Function to process each image
def process_transformed_image(img, watermarked, post_process, render_factor, result, index, visualizer):
    start_time = time.time()
    transformed_img = visualizer.plot_transformed_image_from_array(
        img, watermarked=watermarked, post_process=post_process, render_factor=render_factor
    )
    result[index] = transformed_img
    print('step process_transformed_image', time.time() - start_time)

# Function to run the image processing with threading
def run_transformed_images_in_threads(ori_sgn_img, sng_2t, visualizer):
    start_time = time.time()
    # List to store results from threads
    results = [None, None]

    # Create threads for each image processing task
    thread_1 = threading.Thread(target=process_transformed_image, args=(ori_sgn_img, False, False, 11, results, 0, visualizer))
    thread_2 = threading.Thread(target=process_transformed_image, args=(sng_2t, False, False, 11, results, 1, visualizer))

    # Start the threads
    thread_1.start()
    thread_2.start()

    # Wait for both threads to finish
    thread_1.join()
    thread_2.join()

    # Extract the results
    Ori_sgn_color = results[0]
    sng_2t_color = results[1]

    print('step run_transformed_images_in_threads', time.time() - start_time)

    return Ori_sgn_color, sng_2t_color


sum_time = 0.0
cnt = 0

for i in tqdm(range(622)):  # 假设ID从0到1463

    signal_path = os.path.join(signal_dir, f"image_{i}.jpg")
    range_path = os.path.join(range_dir, f"image_{i}.jpg")
    pcd_path = os.path.join(pcd_dir, f"pointcloud_{i}.pcd")
    output_path = os.path.join(output_dir, f"keypoint_{i}.pcd")
    img_output_path = os.path.join(img_output, f"kp_img_{i}.jpg")

    start_time = time.time()

    signal_img = Image.open(signal_path).convert("RGB")
    range_img = Image.open(range_path).convert("RGB")
    signal_img = signal_img.resize((1024, 128))
    range_img = range_img.resize((1024, 128))
    signal_img = np.array(signal_img)
    range_img = np.array(range_img)
    sr_sgn_img, ori_sgn_img = get_SR(signal_img)
    sng_4t, sng_2t = sr_sgn_img[4], sr_sgn_img[2]
    # print(ori_sgn_img.shape, sng_4t.shape, sng_2t.shape)

    # range_2t
    original_height, original_width = range_img.shape[:2]
    new_width = int(original_width * 2)
    new_height = int(original_height * 2)
    range_2t = cv2.resize(
        range_img, (new_width, new_height), interpolation=cv2.INTER_LINEAR
    )

    # 进行Gamma校正

    range_img = adjust_gamma(range_img, gamma=3)
    sng_2t = adjust_gamma(sng_2t, gamma=1.5)
    ori_sgn_img = adjust_gamma(ori_sgn_img, gamma=1.5)

    # # step visualizer Ori_sgn_color
    # start_time_s = time.time()
    # Ori_sgn_color = visualizer.plot_transformed_image_from_array(
    #     ori_sgn_img, watermarked=False, post_process=False, render_factor=11
    # )
    # end_time_s = time.time()
    # print("step visualizer Ori_sgn_color", end_time_s - start_time_s)
    # # step visualizer sng_2t_color
    # start_time_s2c = time.time()
    # sng_2t_color = visualizer.plot_transformed_image_from_array(
    #     sng_2t, watermarked=False, post_process=False, render_factor=11
    # )
    # end_time_s2c = time.time()
    # print("step visualizer sng_2t_color", end_time_s2c - start_time_s2c)

    Ori_sgn_color, sng_2t_color = run_transformed_images_in_threads(ori_sgn_img, sng_2t, visualizer)

    # step import images

    # images = [range_img, ori_sgn_img, Ori_sgn_color, sng_2t, sng_2t_color]
    start_time_imgs = time.time()
    images = [range_img, range_2t, ori_sgn_img, sng_2t_color]
    end_time_imgs = time.time()
    print("step images", end_time_imgs - start_time_imgs)

    # setp def
    start_time_def = time.time()
    detect_keypoints_and_save_pcd(
        range_img,
        range_2t,
        ori_sgn_img,
        # Ori_sgn_color,
        # sng_2t,
        sng_2t_color,
        model_alike,
        pcd_path,
        output_path,
        img_output_path,
    )
    end_time_def = time.time()
    print("step detect_keypoints_and_save_pcd", end_time_def - start_time_def)

    elapsed_time = time.time() - start_time

    sum_time += elapsed_time
    cnt += 1
    print(f"Processing time: {elapsed_time:.4f}.")

print(f"Averate processing time: {sum_time/cnt}")
