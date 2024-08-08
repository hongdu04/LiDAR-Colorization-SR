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
scores_th = 0.4  # 检测器分数阈值
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
signal_dir = "./dataset/road/signal/"
range_dir = "./dataset/road/range/"
pcd_dir = "./dataset/road/pcd/"
output_dir = "./kp_data/kp_testtesttest_colorful_eccv"
group = 1
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


module = importlib.import_module("model.{}".format("carn"))
net = module.Net(multi_scale=True, group=group)

state_dict = torch.load(ckpt_path)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    new_state_dict[k] = v

net.load_state_dict(new_state_dict)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
net = net.to(device)
transform = transforms.Compose([transforms.ToTensor()])

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import open3d as o3d
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors


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
):

    keypoints_all_images = []
    keypoints_counts = 0

    range_img = cv2.cvtColor(range_img, cv2.COLOR_BGR2RGB)
    pred_range = model_alike(range_img, sub_pixel=True)
    kpts_range = pred_range["keypoints"]
    kpts_range = np.round(kpts_range).astype(int)
    keypoints_counts = keypoints_counts + len(kpts_range)
    keypoints_all_images.extend(kpts_range)

    ori_sgn_img = cv2.cvtColor(ori_sgn_img, cv2.COLOR_BGR2RGB)
    pred_signal = model_alike(ori_sgn_img, sub_pixel=True)
    kpts_signal = pred_signal["keypoints"]
    kpts_signal = np.round(kpts_signal).astype(int)
    # keypoints_counts.append(len(kpts_signal))
    keypoints_counts = keypoints_counts + len(kpts_signal)
    keypoints_all_images.extend(kpts_signal)

    Ori_sgn_color = cv2.cvtColor(Ori_sgn_color, cv2.COLOR_BGR2RGB)
    pred_sgn_color = model_alike(Ori_sgn_color, sub_pixel=True)
    kpts_sgn_color = pred_sgn_color["keypoints"]
    kpts_sgn_color = np.round(kpts_sgn_color).astype(int)
    keypoints_counts = keypoints_counts + len(kpts_sgn_color)
    keypoints_all_images.extend(kpts_sgn_color)

    sng_2t = cv2.cvtColor(sng_2t, cv2.COLOR_BGR2RGB)
    pred_sgn_2t = model_alike(sng_2t, sub_pixel=True)
    kpts_sgn_2t = pred_sgn_2t["keypoints"]
    kpts_sgn_2t[:, 0] = kpts_sgn_2t[:, 0] * 0.5
    kpts_sgn_2t[:, 1] = kpts_sgn_2t[:, 1] * 0.5
    kpts_sgn_2t = np.round(kpts_sgn_2t).astype(int)
    keypoints_counts = keypoints_counts + len(kpts_sgn_2t)
    keypoints_all_images.extend(kpts_sgn_2t)

    sng_2t_color = cv2.cvtColor(sng_2t_color, cv2.COLOR_BGR2RGB)
    pred_sgn_2t_color = model_alike(sng_2t_color, sub_pixel=True)
    kpts_sgn_2t_color = pred_sgn_2t_color["keypoints"]
    kpts_sgn_2t_color[:, 0] = kpts_sgn_2t_color[:, 0] * 0.5
    kpts_sgn_2t_color[:, 1] = kpts_sgn_2t_color[:, 1] * 0.5
    kpts_sgn_2t_color = np.round(kpts_sgn_2t_color).astype(int)
    keypoints_counts = keypoints_counts + len(kpts_sgn_2t_color)
    keypoints_all_images.extend(kpts_sgn_2t_color)

    print(f"before keypoints average: {keypoints_counts}")
    keypoints_all_images = remove_duplicates_ndarray(keypoints_all_images)

    # # 调整关键点的尺寸映射
    # scale_x, scale_y = 128 / img.shape[0], 1024 / img.shape[1]
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # pred = model_alike(img_rgb, sub_pixel=True)
    # kpts = preadmin
    # d['keypoints']

    # 如果是第二张图片（SR_2t_color），需要调整关键点坐标
    # if scale_x != 1 and scale_y != 1:
    #     kpts[:, 0] = kpts[:, 0] * scale_x
    #     kpts[:, 1] = kpts[:, 1] * scale_y

    # kpts_int = np.round(kpts).astype(int)
    # keypoints_counts.append(len(kpts_int))
    # keypoints_all_images.extend(kpts_int)

    # 打印关键点的平均数
    print(f"Ori_img_color and SR_2t_color keypoints average: {keypoints_counts}")

    # 以下是处理点云并保存的代码
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)
    height, width = 128, 1024
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


for i in tqdm(range(1464)):

    signal_path = os.path.join(signal_dir, f"image_{i}.jpg")
    range_path = os.path.join(range_dir, f"image_{i}.jpg")
    pcd_path = os.path.join(pcd_dir, f"pointcloud_{i}.pcd")
    output_path = os.path.join(output_dir, f"keypoint_{i}.pcd")

    signal_img = Image.open(signal_path).convert("RGB")
    range_img = Image.open(range_path).convert("RGB")
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
    )
