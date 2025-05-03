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

import torch
import traceback
import socket
import json
from scene.cameras import MiniCam

host = "127.0.0.1"
port = 6009

conn = None
addr = None

listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def init(wish_host, wish_port):
    global host, port, listener
    host = wish_host
    port = wish_port
    # listener 负责接收 (host, port) 的信息
    listener.bind((host, port))
    # 切换为监听模式, 表示它不再主动发出连接请求, 而是被动等待客户端的连接请求, 即后台不断接收客户端传来的请求, 但是当 listener.accept() 一次才会取走一个连接请求
    listener.listen()
    # 这句话的作用就是当我调用 listener.accept() 但是没有连接进来时, 不会等待, 而是立即抛出错误
    listener.settimeout(0)

def try_connect():
    global conn, addr, listener
    try:
        conn, addr = listener.accept()
        print(f"\nConnected by {addr}")
        # 把连接超时抛出错误的等待时间变为无限大, 因为此时已经连接成功了, 我只需关注当前这次的连接, 后续的连接先不管先
        conn.settimeout(None)
    except Exception as inst:
        pass

def send(message_bytes, verify):
    # message_bytes: [h w c], uint8, ndarray
    # verify: str, 数据集根目录
    global conn
    if message_bytes != None:
        conn.sendall(message_bytes) # 发送图像数据
    conn.sendall(len(verify).to_bytes(4, 'little')) # 由于前端不知道数据集目录 str 的长度, 所以需要先发送一个 4 字节的长度, 让前端知道要接收多少字节的数据
    conn.sendall(bytes(verify, 'ascii'))    # 发送数据集目录

def read():
    global conn
    messageLength = conn.recv(4)    # 读取 4 字节的数据, 代表后面要接收多少字节的数据
    messageLength = int.from_bytes(messageLength, 'little') # 将 4 字节的数据转换为整数
    message = conn.recv(messageLength)  # 接收数据, 直到接收到指定长度的数据为止
    return json.loads(message.decode("utf-8"))  # 将接收到的数据转换为字符串, 然后再转换为字典格式

def receive():
    message = read()    # 从前端接收数据, 返回一个字典

    width = message["resolution_x"]
    height = message["resolution_y"]

    if width != 0 and height != 0:
        try:
            do_training = bool(message["train"])
            do_shs_python = bool(message["shs_python"])
            do_rot_scale_python = bool(message["rot_scale_python"])
            keep_alive = bool(message["keep_alive"])
            scaling_modifier = message["scaling_modifier"]

            fovy = message["fov_y"]
            fovx = message["fov_x"]
            znear = message["z_near"]
            zfar = message["z_far"]
            world_view_transform = torch.reshape(torch.tensor(message["view_matrix"]), (4, 4)).cuda()
            world_view_transform[:,1] = -world_view_transform[:,1]
            world_view_transform[:,2] = -world_view_transform[:,2]
            full_proj_transform = torch.reshape(torch.tensor(message["view_projection_matrix"]), (4, 4)).cuda()
            full_proj_transform[:,1] = -full_proj_transform[:,1]
            custom_cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform) # 相机视角

        except Exception as e:
            print("")
            traceback.print_exc()
            raise e
        
        return custom_cam, do_training, do_shs_python, do_rot_scale_python, keep_alive, scaling_modifier
    else:
        return None, None, None, None, None, None