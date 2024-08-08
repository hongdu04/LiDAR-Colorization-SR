import argparse
import matplotlib.pyplot as plt
import os
import timeit

from colorizers import *

colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_siggraph17 = siggraph17(pretrained=True).eval()

colorizer_eccv16.cuda()
colorizer_siggraph17.cuda()

import sys
import os

alike_module_path = os.path.abspath(os.path.join(os.getcwd(), "../ALIKE/"))
sys.path.append(alike_module_path)

from alike import ALike, configs

# if __name__ == '__main__':
# logging.basicConfig(level=logging.INFO)

# 定义直接在代码中设置的参数
input_path = ""  # 图片目录、电影文件或"camera0"
model_config = "alike-l"  # 模型配置
device = "cuda"  # 运行设备
top_k = -1  # 检测顶部K个关键点
scores_th = 0.3  # 检测器分数阈值
n_limit = 500  # 被检测的最大关键点数
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
        print(f"Scale {scale}, Time taken: {t2-t1:.3f}s")
    return results


def get_SR(img):

    # lr_image = Image.open(img_path).convert("RGB")
    lr_image_tensor = transform(img)

    scales = [2, 4]
    sr_images = sample(net, device, lr_image_tensor, scales)
    print(img.shape)
    width = int(img.shape[0] * 0.5)
    height = int(img.shape[1] * 0.5)
    dim = (width, height)

    return sr_images, np.array(img)
    # return  np.array(img)


ckpt_path = "../checkpoint/carn.pth"
signal_dir = "./dataset/data_forest/signal/"
range_dir = "./dataset/data_forest/range/"
pcd_dir = "./dataset/data_forest/pcd/"
output_dir = "./kp_data/kp_forest_colorful_eccv"
img_output = "./kp_data/img_forest_colorful_eccv"
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


# 假设model_alike是你的关键点检测模型，pcd_path和output_path已经定义好
def detect_keypoints_and_save_pcd(
    range_img,
    ori_sgn_img,
    Ori_sgn_color,
    sng_2t,
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
    ]  # 不同的颜色对应不同的图片
    keypoints_all_images = []
    keypoints_counts = 0
    background_img = ori_sgn_img
    range_img = cv2.cvtColor(range_img, cv2.COLOR_BGR2RGB)
    pred_range = model_alike(range_img, sub_pixel=True)
    kpts_range = pred_range["keypoints"]
    desc_range = pred_range["descriptors"]
    # print("check:",type(kpts_range),len(kpts_range),print(kpts_range[0:10]))
    out_range, N_matches_1 = tracker_range.update(range_img, kpts_range, desc_range)
    print("Compare", N_matches_1, len(kpts_range))
    # out_range = np.round(out_range).astype(int)
    out_range[:, 0] = out_range[:, 0] * 2
    out_range[:, 1] = out_range[:, 1] * 1
    out_range = get_adjacent_coordinates_vectorized(out_range)
    for pt in out_range:
        cv2.circle(
            background_img, (pt[0], pt[1]), 1, colors[0], -1
        )  # 在ori_sgn_img上绘制关键点
        # (text_width, text_height), baseline = cv2.getTextSize("", font, font_scale, thickness)
        (text_width, text_height), _ = cv2.getTextSize(
            "range", font, font_scale, thickness
        )
        text_position = (background_img.shape[1] - text_width - 10, text_height + 10)
        cv2.putText(
            background_img,
            "range",
            text_position,
            font,
            font_scale,
            colors[0],
            thickness,
            cv2.LINE_AA,
        )
    keypoints_all_images.extend(out_range)

    ori_sgn_img = cv2.cvtColor(ori_sgn_img, cv2.COLOR_BGR2RGB)
    pred_signal = model_alike(ori_sgn_img, sub_pixel=True)
    kpts_signal = pred_signal["keypoints"]
    desc_signal = pred_signal["descriptors"]
    out_signal, N_matches_2 = tracker_signal.update(
        ori_sgn_img, kpts_signal, desc_signal
    )
    print("Compare", N_matches_2, len(kpts_signal))
    out_signal[:, 0] = out_signal[:, 0] * 2
    out_signal[:, 1] = out_signal[:, 1] * 1
    out_signal = np.round(out_signal).astype(int)
    for pt in out_signal:
        cv2.circle(
            background_img, (pt[0], pt[1]), 1, colors[1], -1
        )  # 在ori_sgn_img上绘制关键点
        (text_width, text_height), _ = cv2.getTextSize(
            "signal", font, font_scale, thickness
        )
        text_position = (background_img.shape[1] - text_width - 10, text_height + 20)
        cv2.putText(
            background_img,
            "signal",
            text_position,
            font,
            font_scale,
            colors[1],
            thickness,
            cv2.LINE_AA,
        )
    keypoints_all_images.extend(out_signal)

    Ori_sgn_color = cv2.cvtColor(Ori_sgn_color, cv2.COLOR_BGR2RGB)
    pred_sgn_color = model_alike(Ori_sgn_color, sub_pixel=True)
    kpts_sgn_color = pred_sgn_color["keypoints"]
    desc_sgn_color = pred_sgn_color["descriptors"]
    out_signal_color, N_matches_3 = tracker_signal_color.update(
        Ori_sgn_color, kpts_sgn_color, desc_sgn_color
    )
    print("Compare", N_matches_3, len(kpts_sgn_color))
    out_signal_color[:, 0] = out_signal_color[:, 0] * 2
    out_signal_color[:, 1] = out_signal_color[:, 1] * 1
    # out_signal_color = np.round(out_signal_color).astype(int)
    out_signal_color = get_adjacent_coordinates_vectorized(out_signal_color)
    for pt in out_signal_color:
        cv2.circle(
            background_img, (pt[0], pt[1]), 1, colors[2], -1
        )  # 在ori_sgn_img上绘制关键点
        (text_width, text_height), _ = cv2.getTextSize(
            "signal color", font, font_scale, thickness
        )
        text_position = (background_img.shape[1] - text_width - 10, text_height + 30)
        cv2.putText(
            background_img,
            "signal color",
            text_position,
            font,
            font_scale,
            colors[2],
            thickness,
            cv2.LINE_AA,
        )
    keypoints_all_images.extend(out_signal_color)

    sng_2t = cv2.cvtColor(sng_2t, cv2.COLOR_BGR2RGB)
    pred_sgn_2t = model_alike(sng_2t, sub_pixel=True)
    kpts_sgn_2t = pred_sgn_2t["keypoints"]
    desc_sgn_2t = pred_sgn_2t["descriptors"]
    out_sgn2t, N_matches_4 = tracker_signal2X.update(sng_2t, kpts_sgn_2t, desc_sgn_2t)
    print("Compare", N_matches_4, len(kpts_sgn_2t))
    out_sgn2t[:, 0] = out_sgn2t[:, 0] * 1
    out_sgn2t[:, 1] = out_sgn2t[:, 1] * 0.5
    out_sgn2t = np.round(out_sgn2t).astype(int)
    for pt in out_sgn2t:
        cv2.circle(
            background_img, (pt[0], pt[1]), 1, colors[3], -1
        )  # 在ori_sgn_img上绘制关键点
        (text_width, text_height), _ = cv2.getTextSize(
            "signal2X", font, font_scale, thickness
        )
        text_position = (background_img.shape[1] - text_width - 10, text_height + 40)
        cv2.putText(
            background_img,
            "signal2X",
            text_position,
            font,
            font_scale,
            colors[3],
            thickness,
            cv2.LINE_AA,
        )
    keypoints_all_images.extend(out_sgn2t)

    sng_2t_color = cv2.cvtColor(sng_2t_color, cv2.COLOR_BGR2RGB)
    pred_sgn_2t_color = model_alike(sng_2t_color, sub_pixel=True)
    kpts_sgn_2t_color = pred_sgn_2t_color["keypoints"]
    desc_sgn_2t_color = pred_sgn_2t_color["descriptors"]
    out_sgn2t_color, N_matches_5 = tracker_signal2X_color.update(
        sng_2t_color, kpts_sgn_2t_color, desc_sgn_2t_color
    )
    print("Compare", N_matches_5, len(kpts_sgn_2t_color))
    out_sgn2t_color[:, 0] = out_sgn2t_color[:, 0] * 1
    out_sgn2t_color[:, 1] = out_sgn2t_color[:, 1] * 0.5
    print(len(out_sgn2t_color))
    # out_sgn2t_color = np.round(out_sgn2t_color).astype(int)
    out_sgn2t_color = get_adjacent_coordinates_vectorized(out_sgn2t_color)
    print(len(out_sgn2t_color))
    for pt in out_sgn2t_color:
        cv2.circle(
            background_img, (pt[0], pt[1]), 1, colors[4], -1
        )  # 在ori_sgn_img上绘制关键点
        (text_width, text_height), _ = cv2.getTextSize(
            "signal2X_color", font, font_scale, thickness
        )
        text_position = (background_img.shape[1] - text_width - 10, text_height + 50)
        cv2.putText(
            background_img,
            "signal2X_color",
            text_position,
            font,
            font_scale,
            colors[4],
            thickness,
            cv2.LINE_AA,
        )
    # keypoints_counts = keypoints_counts + len(kpts_sgn_2t_color)
    keypoints_all_images.extend(out_sgn2t_color)

    print(f"Total keypoints detected: {len(keypoints_all_images)}")
    cv2.imwrite(img_output_path, background_img)  # 保存带有关键点的图像

    # print(f"before keypoints average: {keypoints_counts}")
    keypoints_all_images = remove_duplicates_ndarray(keypoints_all_images)

    # 打印关键点的平均数
    print(f"Ori_img_color and SR_2t_color keypoints average: {keypoints_counts}")

    # 以下是处理点云并保存的代码
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)
    height, width = 128, 2048
    data = points.reshape((height, width, -1))

    keypoint_area_pcds = []
    for kpt in keypoints_all_images:
        x, y = kpt
        # for dx in range(-1, 2):
        #     for dy in range(-1, 2):
        if 0 <= x < width and 0 <= y < height:
            keypoint_area_pcds.append(data[y, x])

    keypoint_cloud = o3d.geometry.PointCloud()
    keypoint_cloud.points = o3d.utility.Vector3dVector(keypoint_area_pcds)
    o3d.io.write_point_cloud(output_path, keypoint_cloud)


for i in tqdm(range(622)):  # 假设ID从0到1463

    signal_path = os.path.join(signal_dir, f"image_{i}.jpg")
    range_path = os.path.join(range_dir, f"image_{i}.jpg")
    pcd_path = os.path.join(pcd_dir, f"pointcloud_{i}.pcd")
    output_path = os.path.join(output_dir, f"keypoint_{i}.pcd")
    img_output_path = os.path.join(img_output, f"kp_img_{i}.jpg")

    signal_img = Image.open(signal_path).convert("RGB")
    range_img = Image.open(range_path).convert("RGB")
    signal_img = signal_img.resize((1024, 128))
    range_img = range_img.resize((1024, 128))
    signal_img = np.array(signal_img)
    range_img = np.array(range_img)
    sr_sgn_img, ori_sgn_img = get_SR(signal_img)
    sng_4t, sng_2t = sr_sgn_img[4], sr_sgn_img[2]
    print(ori_sgn_img.shape, sng_4t.shape, sng_2t.shape)

    # 进行Gamma校正

    range_img = adjust_gamma(range_img, gamma=3)
    sng_2t = adjust_gamma(sng_2t, gamma=1.5)
    ori_sgn_img = adjust_gamma(ori_sgn_img, gamma=1.5)

    # colorful to array
    (tens_l_orig, tens_l_rs) = preprocess_img(ori_sgn_img, HW=(256, 256))
    tens_l_rs = tens_l_rs.cuda()
    img_bw = postprocess_tens(
        tens_l_orig, torch.cat((0 * tens_l_orig, 0 * tens_l_orig), dim=1)
    )
    Ori_sgn_color = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())

    (tens_l_orig_s, tens_l_rs_s) = preprocess_img(sng_2t, HW=(256, 256))
    tens_l_rs_s = tens_l_rs_s.cuda()
    img_bw_s = postprocess_tens(
        tens_l_orig_s, torch.cat((0 * tens_l_orig_s, 0 * tens_l_orig_s), dim=1)
    )
    sng_2t_color = postprocess_tens(tens_l_orig_s, colorizer_eccv16(tens_l_rs).cpu())

    images = [range_img, ori_sgn_img, Ori_sgn_color, sng_2t, sng_2t_color]

    detect_keypoints_and_save_pcd(
        range_img,
        ori_sgn_img,
        Ori_sgn_color,
        sng_2t,
        sng_2t_color,
        model_alike,
        pcd_path,
        output_path,
        img_output_path,
    )


# 注意：确保model_alike函数、pcd_path和output_path已经正确定义
