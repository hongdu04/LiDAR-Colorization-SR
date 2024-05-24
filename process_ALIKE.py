#!/usr/bin/env python
# coding: utf-8
from deoldify.device_id import DeviceId
from deoldify import device
#choices:  CPU, GPU0...GPU7
device.set(device=DeviceId.GPU0)
from deoldify.visualize import *
plt.style.use('dark_background')
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")
render_factor=35
from deoldify.visualize import get_image_colorizer
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import sys
import os
import json
import importlib
from collections import OrderedDict
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image
import argparse
import cv2
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as patches
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../SuperPointPretrainedNetwork')))
from superpoint import SuperPointFrontend
import copy
import glob
import logging
from tqdm import tqdm
import open3d as o3d  


# 假设ModelImageVisualizer已经被正确修改并且已经加载
visualizer = get_image_colorizer(render_factor=render_factor, artistic=True)
alike_module_path = os.path.abspath(os.path.join(os.getcwd(), '../ALIKE/'))
sys.path.append(alike_module_path)

from alike import ALike, configs

# if __name__ == '__main__':
# logging.basicConfig(level=logging.INFO)
num_samples = 500
def farthest_point_downsample(pcd, num_samples):
    return pcd.farthest_point_down_sample(num_samples=num_samples)
# output_dir = './keypoints'  # 关键点的输出目录


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
            current_frame_points = np.array([(int(round(pt[0])), int(round(pt[1]))) for pt in pts])
        else:
            matches = self.mnn_mather(self.desc_prev, desc)
            mpts1, mpts2 = self.pts_prev[matches[:, 0]], pts[matches[:, 1]]
            N_matches = len(matches)
            # 收集匹配的点坐标
            current_frame_points = np.array([(int(round(p2[0])), int(round(p2[1]))) for p2 in mpts2])
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
        mask = (ids1 == nn21[nn12])
        matches = np.stack([ids1[mask], nn12[mask]], axis=-1)
        return matches

class SuperColorPCSampling(object):
    def __init__(self) -> None:
        self.ckpt_path = '../checkpoint/carn.pth'
        self.signal_dir = './dataset/data_hard/signal_o/'
        self.range_dir  = './dataset/data_hard/range/'
        self.pcd_dir = './dataset/data_hard/pcd/'  
        self.output_dir = 'kp_hard_35_1000'
        self.img_output = "img_hard_35_1000"
        # 定义直接在代码中设置的参数
        self.input_path = ''           # 图片目录、电影文件或"camera0"
        self.model_config = 'alike-l'  # 模型配置
        self.top_k = -1                # 检测顶部K个关键点
        self.scores_th = 0.8           # 检测器分数阈值
        self.n_limit = 1500           # 被检测的最大关键点数
        self.group = 1
        self.weights_path = './superpoint_v1.pth'
        
        
        self.device_gn = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.img_output):
            os.makedirs(self.img_output)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # image_loader = ImageLoader(input_path)
        # keypoint extractor
        self.model_alike = ALike(**configs[self.model_config],
                        device=self.device_gn,
                        top_k=self.top_k,
                        scores_th=self.scores_th,
                        n_limit=self.n_limit)
        
        
        self.model_superpoint = SuperPointFrontend(weights_path=self.weights_path,
                                    nms_dist=2,
                                    conf_thresh=0.025,
                                    nn_thresh=0.7,
                                    cuda=True)
        # super resolution
        module = importlib.import_module("model.{}".format('carn'))
        self.net = module.Net(multi_scale=True, group=self.group)

        state_dict = torch.load(self.ckpt_path)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k] = v
        self.net.load_state_dict(new_state_dict)
        self.net = self.net.to(self.device_gn)
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        # 设置文字的大小
        self.font_scale = 0.5
        # 设置文字的厚度
        self.thickness = 1
        # 获取文字框的大小
        self.tracker_range = SimpleTracker()
        self.tracker_signal = SimpleTracker()
        self.tracker_signal_color = SimpleTracker()
        self.tracker_signal2X = SimpleTracker()
        self.tracker_signal2X_color = SimpleTracker()
            
    def tensor_to_np(self, tensor):
        tensor = tensor.cpu()
        np_img = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        return np_img

    def sample(self, net, lr_image, scales):
        results = {}
        for scale in scales:
            lr = lr_image.unsqueeze(0).to(self.device_gn)
            t1 = time.time()
            sr = net(lr, scale).detach().squeeze(0)
            t2 = time.time()
            sr_np = self.tensor_to_np(sr)
            results[scale] = sr_np
            print(f"Scale {scale}, Time taken: {t2-t1:.3f}s")
        return results
    
    def get_SR(self, img, scales=[1,2]):
        # lr_image = Image.open(img_path).convert("RGB")
        lr_image_tensor = self.transform(img)

        sr_images = self.sample(self.net, lr_image_tensor, scales)
        print(img.shape)
        width = int(img.shape[0] * 0.5)
        height = int(img.shape[1] * 0.5)
        dim = (width, height)
            
        return sr_images, np.array(img)

    def get_adjacent_coordinates_vectorized(self, out_signals):
        # 确保输入是一个 numpy 数组
        out_signals = np.array(out_signals)
        
        # 计算 floor 和 ceil 坐标
        floor_coords = np.floor(out_signals).astype(int)
        ceil_coords = np.ceil(out_signals).astype(int)
        
        # 分别获取所有 x 和 y 的 floor 和 ceil 值
        x_floor, y_floor = floor_coords[:, 0], floor_coords[:, 1]
        x_ceil, y_ceil = ceil_coords[:, 0], ceil_coords[:, 1]
        
        # 生成四个相邻坐标的矩阵
        adjacent_coords = np.stack([
            np.column_stack([x_floor, y_floor]),
            np.column_stack([x_ceil, y_floor]),
            np.column_stack([x_floor, y_ceil]),
            np.column_stack([x_ceil, y_ceil])
        ], axis=1)
        
        # 改变数组形状为 (8, 2)
        adjacent_coords = adjacent_coords.reshape(-1, 2)
        
        return adjacent_coords

    def adjust_gamma(self, image, gamma=1.0):
        # 建立一个映射表
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

        # 应用gamma校正使用映射表
        return cv2.LUT(image, table)


    def remove_duplicates_ndarray(self, arrays):
        seen = set()
        unique_arrays = []
        for arr in arrays:
            # 将numpy数组转换为元组
            tup = tuple(arr.tolist())
            if tup not in seen:
                unique_arrays.append(arr)  # 添加原始numpy数组
                seen.add(tup)
        return unique_arrays
    
    def keypoint_extractor(self, img, model, tracker, super_res=-1, save_plot=False, bg_img=np.array([]), colors=[]):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pred_range = model(img, sub_pixel=True)
        kpts_range = pred_range['keypoints']
        desc_range = pred_range['descriptors']
        keypoints, N_matches_1 = tracker.update(img, kpts_range, desc_range)
        # print("Compare",N_matches_1,len(kpts_range))
        # keypoints_coor = self.get_adjacent_coordinates_vectorized(keypoints)
        keypoints_coor = copy.deepcopy(keypoints)
        keypoints_coor = np.round(keypoints_coor).astype(int)

        if super_res > 0:
            keypoints_coor[:, 0] = keypoints_coor[:, 0] / super_res
            keypoints_coor[:, 1] = keypoints_coor[:, 1] / super_res
            keypoints_coor = np.round(keypoints_coor).astype(int)

        if save_plot:
            for pt in keypoints_coor:
                    cv2.circle(bg_img, (pt[0], pt[1]), 1, colors[0], -1)  # 在ori_sgn_img上绘制关键点
                    # (text_width, text_height), baseline = cv2.getTextSize("", font, font_scale, thickness)
                    (text_width, text_height), _ = cv2.getTextSize("range", self.font, self.font_scale, self.thickness)
                    text_position = (bg_img.shape[1] - text_width - 10, text_height + 10)
                    cv2.putText(bg_img, "range",text_position, self.font, self.font_scale, colors[0], self.thickness, cv2.LINE_AA)
        return keypoints_coor
    
    
    def keypoint_extractor_SP(self, img, model_SP, tracker, super_res=-1, save_plot=False, bg_img=np.array([]), colors=[]):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        kpts_range, desc_range, heatmap = model_SP.run(img)             

        keypoints, N_matches_1 = tracker.update(img, kpts_range, desc_range)
        # print("Compare",N_matches_1,len(kpts_range))
        # keypoints_coor = self.get_adjacent_coordinates_vectorized(keypoints)
        keypoints_coor = copy.deepcopy(keypoints)
        keypoints_coor = np.round(keypoints_coor).astype(int)

        if super_res > 0:
            keypoints_coor[:, 0] = keypoints_coor[:, 0] / super_res
            keypoints_coor[:, 1] = keypoints_coor[:, 1] / super_res
            keypoints_coor = np.round(keypoints_coor).astype(int)

        if save_plot:
            for pt in keypoints_coor:
                    cv2.circle(bg_img, (pt[0], pt[1]), 1, colors[0], -1)  # 在ori_sgn_img上绘制关键点
                    # (text_width, text_height), baseline = cv2.getTextSize("", font, font_scale, thickness)
                    (text_width, text_height), _ = cv2.getTextSize("range", self.font, self.font_scale, self.thickness)
                    text_position = (bg_img.shape[1] - text_width - 10, text_height + 10)
                    cv2.putText(bg_img, "range",text_position, self.font, self.font_scale, colors[0], self.thickness, cv2.LINE_AA)
        return keypoints_coor

    
    def get_nearby_points_efficient(self, keypoints, region_size=3):
        """
        获取每个关键点附近的点，去除重复的点（高效实现）。
        
        参数:
        - keypoints: np.ndarray, 关键点坐标数据，形状为[N, 2]
        - region_size: int, 区域大小（默认3，即3x3区域）
        
        返回:
        - unique_points: np.ndarray, 去重后的所有点坐标，形状为[M, 2]
        """
        # 计算半径
        radius = region_size // 2

        # 生成3x3区域的偏移
        offsets = np.arange(-radius, radius + 1)
        grid_x, grid_y = np.meshgrid(offsets, offsets)
        grid_offsets = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

        # 扩展关键点
        keypoints_expanded = keypoints[:, np.newaxis, :]

        # 计算所有临近点
        nearby_points = keypoints_expanded + grid_offsets

        # 重塑为二维数组
        nearby_points = nearby_points.reshape(-1, 2)

        # 去除重复点
        unique_points = np.unique(nearby_points, axis=0)

        return unique_points

    def detect_keypoints_and_save_pcd(self, images, ori_img,
            model_alike, pcd_path, output_path, img_output_path):
            
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]  # 不同的颜色对应不同的图片
            keypoints_all_images = []
            bg_img = copy.deepcopy(ori_img)

            for img_res in images:
                super_res = img_res[0]
                img = copy.deepcopy(img_res[1])
                tracker = img_res[2]
                print(f"-------{super_res}-------")
                out_keypoints = self.keypoint_extractor(img, model_alike, tracker,  super_res=super_res,  bg_img=bg_img, colors=colors)
                print(f"---- keypoints number: {np.array(out_keypoints).shape}")
                keypoints_all_images.extend(out_keypoints)


            # print(f"before keypoints average: {keypoints_counts}")
            keypoints_all_images = self.remove_duplicates_ndarray(keypoints_all_images)
            print(f">>>> keypoints number: {np.array(keypoints_all_images).shape}")
            
            keypoints_all_images = self.get_nearby_points_efficient(np.array(keypoints_all_images))
            print(f">>>>22222222 keypoints number: {np.array(keypoints_all_images).shape}")
            


            # # 打印关键点的平均数
            # # print(f"Ori_img_color and SR_2t_color keypoints average: {keypoints_counts}")

            # # 以下是处理点云并保存的代码
            # pcd = o3d.io.read_point_cloud(pcd_path)
            # combined_cloud = o3d.geometry.PointCloud()
            # # voxel_pcd = voxel_downsample(pcd, voxel_size)
            # farthest_pcd = farthest_point_downsample(pcd, num_samples)
            # farthest_pcd.colors = o3d.utility.Vector3dVector(np.tile([0, 1, 1], (len(farthest_pcd.points), 1))) 
            # points = np.asarray(pcd.points)
            # height, width = 128, 1024
            # data = points.reshape((height, width, -1))
            
            # keypoint_area_pcds = []
            # for kpt in keypoints_all_images:
            #     x, y = kpt
            #     if 0 <= x  < width and 0 <= y  < height:
            #         keypoint_area_pcds.append(data[y , x ])

            # keypoint_cloud = o3d.geometry.PointCloud()
            # keypoint_cloud.points = o3d.utility.Vector3dVector(keypoint_area_pcds)
            # keypoint_cloud.colors = o3d.utility.Vector3dVector(np.tile([1, 0, 0], (len(keypoint_cloud.points), 1))) 
            # combined_cloud = keypoint_cloud + farthest_pcd
            # # combined_points = np.concatenate((np.asarray(keypoint_cloud.points), np.asarray(farthest_pcd.points)), axis=0)
            # # combined_cloud.points = o3d.utility.Vector3dVector(combined_points)
            # o3d.io.write_point_cloud(output_path, combined_cloud)

            pcd = o3d.io.read_point_cloud(pcd_path)
            points = np.asarray(pcd.points)
            height, width = 128, 1024
            data = points.reshape((height, width, -1))
            
            keypoint_area_pcds = []
            for kpt in keypoints_all_images:
                x, y = kpt
                if 0 <= x  < width and 0 <= y  < height:
                    keypoint_area_pcds.append(data[y , x ])

            keypoint_cloud = o3d.geometry.PointCloud()
            keypoint_cloud.points = o3d.utility.Vector3dVector(keypoint_area_pcds)
            o3d.io.write_point_cloud(output_path, keypoint_cloud)



    def detect_keypoints_and_save_pcd_SP(self, images, ori_img,
            model_superpoint, pcd_path, output_path, img_output_path):
            
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]  # 不同的颜色对应不同的图片
            keypoints_all_images = []
            bg_img = copy.deepcopy(ori_img)

            for img_res in images:
                super_res = img_res[0]
                img = copy.deepcopy(img_res[1])
                tracker = img_res[2]
                print(f"-------{super_res}-------")
                out_keypoints = self.keypoint_extractor_SP(img, model_superpoint, tracker,  super_res=super_res,  bg_img=bg_img, colors=colors)
                print(f"---- keypoints number: {np.array(out_keypoints).shape}")
                keypoints_all_images.extend(out_keypoints)


            # print(f"before keypoints average: {keypoints_counts}")
            keypoints_all_images = self.remove_duplicates_ndarray(keypoints_all_images)
            print(f">>>> keypoints number: {np.array(keypoints_all_images).shape}")

            pcd = o3d.io.read_point_cloud(pcd_path)
            points = np.asarray(pcd.points)
            height, width = 128, 1024
            data = points.reshape((height, width, -1))
            
            keypoint_area_pcds = []
            for kpt in keypoints_all_images:
                x, y = kpt
                if 0 <= x  < width and 0 <= y  < height:
                    keypoint_area_pcds.append(data[y , x ])

            keypoint_cloud = o3d.geometry.PointCloud()
            keypoint_cloud.points = o3d.utility.Vector3dVector(keypoint_area_pcds)
            o3d.io.write_point_cloud(output_path, keypoint_cloud)

    def run(self):
        for i in range(2376):  # 假设ID从0到1463
            signal_path = os.path.join(self.signal_dir, f'image_{i}.jpg')
            range_path = os.path.join(self.range_dir, f'image_{i}.jpg')
            pcd_path = os.path.join(self.pcd_dir, f'pointcloud_{i}.pcd')
            output_path = os.path.join(self.output_dir, f'keypoint_{i}.pcd')
            img_output_path = os.path.join(self.img_output, f'kp_img_{i}.jpg')
            
    
            signal_img = Image.open(signal_path).convert("RGB")
            range_img = Image.open(range_path).convert("RGB")
            signal_img = np.array(signal_img)
            range_img = np.array(range_img)

            # super resolution, scales [2]
            scales = [2]
            sr_sgn_imgs, ori_sgn_img = self.get_SR(signal_img, scales=scales)
            sng_scales_imgs = [copy.deepcopy(sr_sgn_imgs[scale]) for scale in scales]
            print(ori_sgn_img.shape, sng_scales_imgs[0].shape)

            sr_rng_imgs, ori_rng_img = self.get_SR(range_img, scales=scales)
            rng_scales_imgs = [copy.deepcopy(sr_rng_imgs[scale]) for scale in scales]


            # 进行Gamma校正
            range_2t_img = self.adjust_gamma(rng_scales_imgs[0], gamma=3)
            sng_2t = self.adjust_gamma(sng_scales_imgs[0], gamma=1.5)
            ori_sgn_img = self.adjust_gamma(ori_sgn_img, gamma=1.5)
            
            # start_color = time.time()
            # ori_sgn_color =  visualizer.plot_transformed_image_from_array(ori_sgn_img, watermarked = False, post_process = False, render_factor= 35)
            sng_2t_color = visualizer.plot_transformed_image_from_array(sng_2t, watermarked = False, post_process = False, render_factor= 35)
            # print(f"color 2 images: {time.time() - start_color}")

            images = [
                [2, range_2t_img, SimpleTracker()],
                [2, sng_2t, SimpleTracker()],
                # [0, ori_sgn_color, SimpleTracker()],
                [2, sng_2t_color, SimpleTracker()]
            ]

            # trackers= [
            #     [2, SimpleTracker()],
            #     [2, SimpleTracker()],
            #     [-1, SimpleTracker()],
            #     [-1, SimpleTracker()]
            # ]
            
            # images = [
            #     range_img,
            #     ori_sgn_img,
            #     ori_sgn_color,
            #     sng_2t,
            #     sng_2t_color
            #     ]

            # self.detect_keypoints_and_save_pcd_new(range_2t_img,
            #     ori_sgn_img,
            #     # ori_sgn_color,
            #     sng_2t,
            #     # sng_2t_color, 
            #     self.model_alike, pcd_path, output_path, img_output_path)
            self.detect_keypoints_and_save_pcd(images, ori_sgn_img,
                self.model_alike, pcd_path, output_path, img_output_path)    

if __name__ == '__main__':
    scp = SuperColorPCSampling()
    scp.run()



