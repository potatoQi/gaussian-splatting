#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:
    '''
    看完这个类, 我对各种文件分别是干嘛的有了了解了:
    所以这个类能通过数据集文件夹, 也就是 COLMAP 导出数据的文件夹里, 封装出下面这几个东西:
        1. 点云对象 (BasicPointCloud)
        2. 点云文件路径 (str)
        3. 训练集相机对象列表 (list)
        4. 测试集相机对象列表 (list)
        5. 训练集相机归一化参数 (dict)
    其中数据集文件夹中, iamges/ 是所有视角的图片, sparse/0/ 下有下面几个文件:
        points3D.bin: COLMAP 导出的点云数据
        points3D.ply: 经过 Scene 类处理过后的点云数据
        images.bin: 相机外参, 其实就是一个 dict: key 是图像的 id, value 是一个结构体, 包括: 相机坐标, 相机旋转, 图像name, 内参id
        cameras.bin: 相机内参, 其实就是一个 dict: key 是相机的 id, value 是一个结构体, 包括: # NOTE: ?

    说完数据集文件夹, 说说输出文件夹:
    根目录下有这些文件:
    point_cloud/: 这个文件夹下存着不同迭代点的 .ply 文件
    input.ply: 初始点云数据
    cameras.json: 测试集和训练集相机列表里所有相机对象的参数
    exposure.json: 每张图像对应的曝光补偿系数
    chkpntxxx.pth: ckpt 文件
    '''

    gaussians : GaussianModel

    def __init__(
        self,
        args : ModelParams,             # 配置
        gaussians : GaussianModel,      # GaussianModel 对象
        load_iteration=None,            # 从 point_cloud/ 文件夹中的第几次 iteration 中加载模型
        shuffle=True,                   # 是否打乱 train_ca,eras & test_cameras
        resolution_scales=[1.0]         # 分辨率列表
    ):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path  # 输出路径
        self.loaded_iter = None             # 从历史点云里加载的 iteration 数字
        self.gaussians = gaussians          # GaussianModel 对象的引用

        if load_iteration:
            # -1 表示加载最近一次的 iteration
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}     # 用来存训练集
        self.test_cameras = {}      # 用来存测试集

        # sparse 是 COLMAP 导出的数据格式文件夹
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            # sceneLoadTypeCallbacks 是一个 dict, 其中 "Colmap" 这个 key 对应的 value 是一个函数
            # 这个函数负责把 COLMAP 导出的数据读进来，封装成一个统一的 SceneInfo 对象
            # 这个对象里有: 点云对象(BasicPointCloud), 点云文件路径(str), 训练集相机对象列表(list), 测试集相机对象列表(list), 训练集相机归一化参数(dict)
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path,       # 数据根目录
                args.images,            # 图片文件夹 (默认是 "")
                args.depths,            # 深度图文件夹 (默认是 "")
                args.eval,              # 是否要划分 test_cam_names_list 出来
                args.train_test_exp     # 是否启用 Synthetic‐NeRF 训练评估模式 & 启用曝光补偿
            )
        # Blender（NeRF Synthetic）格式的数据
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.depths, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        # 如果没有从历史点云里加载模型, 说明这是第一次训练模型, 那么就把一些元数据写入到输出路径下
        if not self.loaded_iter:
            # 打开 /sparse/0/points3D.ply 文件, 逐字节拷贝到输出路径下的 input.ply 文件
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            # 把测试集相机列表和训练集相机列表放入 camlist
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            # 把所有相机对象参数写入到 json 文件中
            json_cams = []
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        # 这玩意就是场景尺度, 这个数字是根据训练集相机的 bbox 计算出来的, 也就是训练集相机的包围盒的半径
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # 对于每一个分辨率比例, 都分配一套训练集相机和测试集相机
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)

        # 这里决定了高斯点云模型（self.gaussians）是 从磁盘已经保存的某次迭代结果加载，还是从原始稀疏点云重新初始化
        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(
                    self.model_path,
                    "point_cloud",
                    "iteration_" + str(self.loaded_iter),
                    "point_cloud.ply"
                ),
                args.train_test_exp
            )
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)

    def save(self, iteration):
        # 保存路径: model_path/point_cloud/iteration_xxx/point_cloud.ply
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        # 构造一个 dict, 记录每张图像对应的曝光补偿系数
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }
        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
