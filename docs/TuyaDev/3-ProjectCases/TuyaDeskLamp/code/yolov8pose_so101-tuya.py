#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import argparse
import os
import cv2
import numpy as np
import torch

import socket
import json
import threading

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.utils.control_utils import init_keyboard_listener
from math import ceil

# --- 全局控制变量 ---
# 0: 未启动/停止, 1: 正在运行
SYSTEM_STATE = 0 
# Socket配置 (请修改为 Tuya 设备的 IP)
SERVER_IP = '127.0.0.1' 
SERVER_PORT = 8888
TRACKING_ENABLED = False # 涂鸦下发的跟随开关 (PROP_BOOL)
MANUAL_MODE = False      # 是否处于遥控模式
CURRENT_CMD = "stop"     # 当前遥控指令
REMOTE_SMOOTH_ALPHA = 0.15  # 遥控平滑系数：0.1-0.2 之间。越小越丝滑，但延迟感略增。
# --------------------

# 人体姿态检测，使用YOLOv8-Pose模型
from rknnlite.api import RKNNLite as RKNN
CLASSES = ['person']
nmsThresh = 0.4
objectThresh = 0.5

# 中心区域的定义，用于判断是否需要移动机械臂
CENTER_X_MIN = 0.3  # 画面宽度的30%
CENTER_X_MAX = 0.7  # 画面宽度的70%
CENTER_Y_MIN = 0.3  # 画面高度的30%
CENTER_Y_MAX = 0.7  # 画面高度的70%

# 启动时的舵机位置
HOME_POSE_DEG = {
    'shoulder_pan.pos': 0.0,    # 左右正中
    'shoulder_lift.pos': 30.0,  # 手稍微抬起
    'elbow_flex.pos': -60.0,    # 肘略弯
    'wrist_flex.pos': 20.0,     # 手腕稍微向下
    'wrist_roll.pos': 70.0,     # 手腕水平
    'gripper.pos': 0.0,         # 夹爪先不动
}

# 归位时的舵机位置
ZORE_POSE_DEG = {
    'shoulder_pan.pos': -10.0,  # 左右正中
    'shoulder_lift.pos': -5.0,  # 手稍微抬起
    'elbow_flex.pos': -5.0,     # 肘略弯
    'wrist_flex.pos': -70.0,     # 手腕稍微向下
    'wrist_roll.pos': 60.0,     # 手腕水平
    'gripper.pos': 0.0,         # 夹爪先不动
}

EMA_ALPHA = 0.15  # 0.15~0.35 越小越稳但越慢

# === 辅助函数 ===

def build_home_pose(joint_names):
    """
    根据 joint_names 构造一个 Home Pose 向量（torch.Tensor, shape=[1, action_dim], 单位：度）
    没在 HOME_POSE_DEG 里定义的关节就默认为 0 度。
    """
    import torch

    action_dim = len(joint_names)
    home = torch.zeros(1, action_dim, dtype=torch.float32)
    for i, name in enumerate(joint_names):
        if name in HOME_POSE_DEG:
            home[0, i] = HOME_POSE_DEG[name]
        else:
            home[0, i] = 0.0
    return home  # 确保返回 home


def build_zero_pose(joint_names):
    """
    根据 joint_names 构造一个 Home Pose 向量（torch.Tensor, shape=[1, action_dim], 单位：度）
    没在 ZORE_POSE_DEG 里定义的关节就默认为 0 度。
    """
    import torch

    action_dim = len(joint_names)
    home = torch.zeros(1, action_dim, dtype=torch.float32)
    for i, name in enumerate(joint_names):
        if name in ZORE_POSE_DEG:
            home[0, i] = ZORE_POSE_DEG[name]
        else:
            home[0, i] = 0.0
    return home

def letterbox_resize(image, size, bg_color):
    """
    按照目标大小调整图像，保持长宽比不变。
    """
    if isinstance(image, str):
        image = cv2.imread(image)
    target_width, target_height = size
    image_height, image_width, _ = image.shape
    aspect_ratio = min(target_width / image_width, target_height / image_height)
    new_width = int(image_width * aspect_ratio)
    new_height = int(image_height * aspect_ratio)
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    result_image = np.ones((target_height, target_width, 3), dtype=np.uint8) * bg_color
    offset_x = (target_width - new_width) // 2
    offset_y = (target_height - new_height) // 2
    result_image[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = image
    return result_image, aspect_ratio, offset_x, offset_y

class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax, keypoint):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.keypoint = keypoint

def IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    innerWidth = xmax - xmin
    innerHeight = ymax - ymin

    innerWidth = innerWidth if innerWidth > 0 else 0
    innerHeight = innerHeight if innerHeight > 0 else 0

    innerArea = innerWidth * innerHeight

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    total = area1 + area2 - innerArea

    return innerArea / total

def NMS(detectResult):
    predBoxs = []

    sort_detectboxs = sorted(detectResult, key=lambda x: x.score, reverse=True)

    for i in range(len(sort_detectboxs)):
        xmin1 = sort_detectboxs[i].xmin
        ymin1 = sort_detectboxs[i].ymin
        xmax1 = sort_detectboxs[i].xmax
        ymax1 = sort_detectboxs[i].ymax
        classId = sort_detectboxs[i].classId

        if sort_detectboxs[i].classId != -1:
            predBoxs.append(sort_detectboxs[i])
            for j in range(i + 1, len(sort_detectboxs), 1):
                if classId == sort_detectboxs[j].classId:
                    xmin2 = sort_detectboxs[j].xmin
                    ymin2 = sort_detectboxs[j].ymin
                    xmax2 = sort_detectboxs[j].xmax
                    ymax2 = sort_detectboxs[j].ymax
                    iou = IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2)
                    if iou > nmsThresh:
                        sort_detectboxs[j].classId = -1
    return predBoxs

def process(out,keypoints,index,model_w,model_h,stride,scale_w=1,scale_h=1):
    xywh=out[:,:64,:]
    conf=sigmoid(out[:,64:,:])
    out=[]
    for h in range(model_h):
        for w in range(model_w):
            for c in range(len(CLASSES)):
                if conf[0,c,(h*model_w)+w]>objectThresh:
                    xywh_=xywh[0,:,(h*model_w)+w] #[1,64,1]
                    xywh_=xywh_.reshape(1,4,16,1)
                    data=np.array([i for i in range(16)]).reshape(1,1,16,1)
                    xywh_=softmax(xywh_,2)
                    xywh_ = np.multiply(data, xywh_)
                    xywh_ = np.sum(xywh_, axis=2, keepdims=True).reshape(-1)

                    xywh_temp=xywh_.copy()
                    xywh_temp[0]=(w+0.5)-xywh_[0]
                    xywh_temp[1]=(h+0.5)-xywh_[1]
                    xywh_temp[2]=(w+0.5)+xywh_[2]
                    xywh_temp[3]=(h+0.5)+xywh_[3]

                    xywh_[0]=((xywh_temp[0]+xywh_temp[2])/2)
                    xywh_[1]=((xywh_temp[1]+xywh_temp[3])/2)
                    xywh_[2]=(xywh_temp[2]-xywh_temp[0])
                    xywh_[3]=(xywh_temp[3]-xywh_temp[1])
                    xywh_=xywh_*stride

                    xmin=(xywh_[0] - xywh_[2] / 2) * scale_w
                    ymin = (xywh_[1] - xywh_[3] / 2) * scale_h
                    xmax = (xywh_[0] + xywh_[2] / 2) * scale_w
                    ymax = (xywh_[1] + xywh_[3] / 2) * scale_h
                    keypoint=keypoints[...,(h*model_w)+w+index]
                    keypoint[...,0:2]=keypoint[...,0:2]//1
                    box = DetectBox(c,conf[0,c,(h*model_w)+w], xmin, ymin, xmax, ymax,keypoint)
                    out.append(box)

    return out
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def go_to_home_pose(robot, joint_names, duration=2.0, steps=50):
    """
    从当前姿态“平滑地”插值到 Home Pose。
    """
    import torch
    action_dim = len(joint_names)
    home = build_home_pose(joint_names)
    obs = robot.get_observation()
    current = torch.zeros(1, action_dim, dtype=torch.float32)
    for i, name in enumerate(joint_names):
        if name in obs:
            current[0, i] = float(obs[name])
        else:
            current[0, i] = 0.0
    for k in range(steps):
        alpha = float(k + 1) / float(steps)
        action = (1.0 - alpha) * current + alpha * home
        send_action_to_robot(robot, action)
        time.sleep(duration / steps)
    return action  # 需要返回action


def go_to_zero_pose(robot, joint_names, duration=2.0, steps=50):
    """
    从当前姿态“平滑地”插值到 Zero Pose（归位）。
    """
    import torch
    action_dim = len(joint_names)
    zero_pose = build_zero_pose(joint_names)
    obs = robot.get_observation()
    current = torch.zeros(1, action_dim, dtype=torch.float32)
    for i, name in enumerate(joint_names):
        if name in obs:
            current[0, i] = float(obs[name])
        else:
            current[0, i] = 0.0
    for k in range(steps):
        alpha = float(k + 1) / float(steps)
        action = (1.0 - alpha) * current + alpha * zero_pose
        send_action_to_robot(robot, action)
        time.sleep(duration / steps)

def move_arm_based_on_pose(current_action, pan_center, shoulder_center, elbow_center, wrist_roll_center, ex, ey, box_width, frame_width):
    """
    根据检测到的人体位置偏差，控制整个机械臂。
    """
    target_action = current_action.clone()

        # --- 引入控制死区 (Dead Zone) ---
    # 如果误差在±DEAD_ZONE像素以内，就认为已经对准，不进行移动，防止抖动。
    DEAD_ZONE_X = 25  # 水平方向死区（像素），可根据画面宽度调整
    DEAD_ZONE_Y = 35  # 垂直方向死区（像素），可根据画面高度调整

    if abs(ex) > DEAD_ZONE_X:
        # 水平方向控制 (Pan)
        pan_gain = 0.01
        pan_offset = ex * pan_gain
        # *** 修改点：将减号改回加号来修正方向 ***
        target_action[0, 0] = np.clip(current_action[0, 0] + pan_offset, pan_center - 90, pan_center + 90)

    # 垂直方向控制 (Shoulder and Elbow)
    if abs(ey) > DEAD_ZONE_Y:
        shoulder_gain = 0.01
        elbow_gain = 0.01

        shoulder_offset = ey * shoulder_gain
        elbow_offset = ey * elbow_gain
        
        target_action[0, 1] = np.clip(current_action[0, 1] + shoulder_offset, shoulder_center - 45, shoulder_center + 45)
        target_action[0, 2] = np.clip(current_action[0, 2] + elbow_offset, elbow_center - 45, elbow_center + 45)

    # 距离控制暂时保持注释，以避免干扰
    '''
    distance_gain = 0.1 
    base_width = frame_width / 3 
    distance_error = box_width - base_width
    distance_offset = distance_error * distance_gain
    target_action[0, 2] += distance_offset
    target_action[0, 3] -= distance_offset 
    '''
    return target_action

def send_action_to_robot(robot, action):
    """
    将“关节角度（度）”映射成 LeRobot 的键值 dict，然后 robot.send_action()
    """
    # 如果 action 是 torch.Tensor 类型
    if isinstance(action, torch.Tensor):
        action_np = action.detach().cpu().numpy().reshape(-1)
    else:
        # 如果 action 是 numpy.ndarray 类型，直接 reshape
        action_np = action.reshape(-1)

    joint_names = list(robot.action_features.keys())

    if len(joint_names) != len(action_np):
        print(f"[WARN] action dim = {len(action_np)}, but robot has {len(joint_names)} action_features:")
        print("       joint_names =", joint_names)
        n = min(len(joint_names), len(action_np))
    else:
        n = len(joint_names)

    # 创建字典，将 action_np 转换为字典格式
    robot_action = {
        joint_names[i]: float(action_np[i])
        for i in range(n)
    }
    print("send_action_to_robot:", robot_action)
    robot.send_action(robot_action)


pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                         [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                         [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                         [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]],dtype=np.uint8)
kpt_color  = pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
            [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
limb_color = pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]


def search_for_person(robot, joint_names, search_pan, search_dir, pan_center, SEARCH_PAN_RANGE, SEARCH_STEP_DEG, target_action):
    """
    Function to perform the search for a person.
    - It scans the environment by moving the robot's arm until a person is detected.
    - Once the robot finishes the scan, it will continue scanning in the opposite direction.

    Arguments:
    - robot: The robot object controlling the arm.
    - joint_names: List of joint names to control.
    - search_pan: The current pan position of the robot's arm.
    - search_dir: Direction in which the robot is searching (1 for right, -1 for left).
    - pan_center: The center pan position where the robot starts scanning.
    - SEARCH_PAN_RANGE: The maximum range for the search.
    - SEARCH_STEP_DEG: The step in degrees for each frame during the search.
    - target_action: The target action representing the robot's movement.

    Returns:
    - The updated target_action with the new shoulder_pan position.
    """
    # Update search pan to simulate the scanning motion
    search_pan += search_dir * SEARCH_STEP_DEG

    # Check if the pan position is beyond the range and reverse direction if necessary
    if search_pan > pan_center + SEARCH_PAN_RANGE:
        search_pan = pan_center + SEARCH_PAN_RANGE
        search_dir = -1.0  # Change direction to left
    elif search_pan < pan_center - SEARCH_PAN_RANGE:
        search_pan = pan_center - SEARCH_PAN_RANGE
        search_dir = 1.0  # Change direction to right

    # Update the shoulder pan position in the target action
    idx_map = {n: i for i, n in enumerate(joint_names)}
    if 'shoulder_pan.pos' in idx_map:
        target_action[0, idx_map['shoulder_pan.pos']] = float(search_pan)

    # Send the action to the robot
    send_action_to_robot(robot, target_action)

    return target_action, search_pan, search_dir

# === 新增：Socket Client 线程 ===
def socket_client_thread(host, port):
    global SYSTEM_STATE, TRACKING_ENABLED, MANUAL_MODE, CURRENT_CMD
    while True:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client_socket.connect((host, port))
            while True:
                data = client_socket.recv(1024)
                if not data: break
                
                # 处理粘包：依靠 C 端发送的 \n 分割消息
                msgs = data.decode('utf-8').strip().split('\n')
                for one_msg in msgs:
                    try:
                        msg = json.loads(one_msg)
                        # 处理跟随开关
                        if 'dpid' in msg and msg['dpid'] == 1:
                            TRACKING_ENABLED = (msg['value'] == True or msg['value'] == "true")
                            if TRACKING_ENABLED:
                                SYSTEM_STATE = 1
                                MANUAL_MODE = False
                        
                        # 处理遥控指令
                        elif msg.get('type') == 'control':
                            cmd = msg.get('command')
                            if not TRACKING_ENABLED:
                                # 收到指令即进入手动模式
                                MANUAL_MODE = True
                                SYSTEM_STATE = 1
                                CURRENT_CMD = cmd
                                print(f"[Socket] Received Command: {cmd}")
                    except:
                        continue
        except:
            time.sleep(3)
        finally:
            client_socket.close()

# === 主循环 ===

def main():
    global CURRENT_CMD, SYSTEM_STATE, TRACKING_ENABLED, MANUAL_MODE

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./model/yolov8_pose.rknn', help='模型路径，使用姿态检测模型')
    parser.add_argument('--cam-id', type=int, default=11)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--port', type=str, default='/dev/ttyACM0')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--show', action='store_true', help='是否显示预览图像')
    opt = parser.parse_args()

    # 1. 启动 Socket 线程
    print(f"启动 Socket 客户端线程，目标: {SERVER_IP}:{SERVER_PORT}")
    t = threading.Thread(target=socket_client_thread, args=(SERVER_IP, SERVER_PORT), daemon=True)
    t.start()

    # 加载YOLOv8 Pose模型
    print("加载YOLOv8 Pose模型...")
    pose_rknn = RKNN(verbose=False)
    assert pose_rknn.load_rknn(opt.model_path) == 0, "加载模型失败"
    assert pose_rknn.init_runtime() == 0, "初始化失败"

    # 打开摄像头
    cap = cv2.VideoCapture(opt.cam_id)
    if not cap.isOpened():
        print("无法打开摄像头")
        return



    # 初始化机器人
    robot_cfg = SO101FollowerConfig(
        port=opt.port,
        id="follower_arm",
        cameras={},
        use_degrees=True,
    )
    robot = SO101Follower(robot_cfg)
    robot.connect()

    joint_names = list(robot.action_features.keys())
    action_dim = len(joint_names)

    # 建立索引映射
    idx = {name: i for i, name in enumerate(joint_names)}

    print("开始跟踪循环...")
    
    # 启动时回到Home位置
    target_action = go_to_home_pose(robot, joint_names, duration=2.0, steps=50)

    print("Home target_action (deg):", target_action[0, :].numpy())

    # 用 Home Pose 作为中心（pan_center / shoulder_center 等）
    idx_map = {n: i for i, n in enumerate(joint_names)}
    pan_center = HOME_POSE_DEG.get('shoulder_pan.pos', 0.0)
    shoulder_center = HOME_POSE_DEG.get('shoulder_lift.pos', 0.0)
    elbow_center = HOME_POSE_DEG.get('elbow_flex.pos', 0.0)
    wrist_roll_center = HOME_POSE_DEG.get('wrist_roll.pos', 0.0)

    print(f"Centers: pan={pan_center:.1f}, shoulder={shoulder_center:.1f}, "
          f"elbow={elbow_center:.1f}, wrist_roll={wrist_roll_center:.1f}")


    # 6) 键盘监听（可选，headless 环境只是提示 warning）
    listener, events = init_keyboard_listener()

    period = 1.0 / opt.fps
    print("Start face-pose-follow control loop... (Ctrl+C to stop)")

    # ===== 搜索模式状态 =====
    last_time = time.time()
    miss_count = 0
    hit_count = 0
    MISS_TO_SEARCH = 8  # 连续多少帧没检测到脸 -> 进入搜索
    HIT_TO_CONFIRM_TRACK = 3  # 连续多少帧检测到目标 -> 才开始移动和跟踪
    last_face_time = time.time()  # 最近一次检测到脸的时间
    LOST_TIMEOUT = 3.0  # 连续 2 秒都没检测到脸，才开始扫描（建议 1.5~3.0）
    search_mode = False

    search_dir = 1.0  # +1 向右扫, -1 向左扫
    search_pan = pan_center  # 当前搜索时的 pan 目标
    SEARCH_PAN_RANGE = 45.0  # 搜索左右扫的范围（相对 pan_center）
    SEARCH_STEP_DEG = 2.0  # 每帧扫多少度（与你 fps 有关）

    ex_f, ey_f = 0.0, 0.0

    step = 0
    last_box = None
    print("等待 Tuya 指令启动 (SYSTEM_STATE=0)...")
    try:
        while True:
            # 计算两帧之间的时间差 dt
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            if SYSTEM_STATE == 0:
                if target_action is not None:
                    go_to_zero_pose(robot, joint_names, duration=1.2)
                    target_action = None
                time.sleep(0.2); continue

            if target_action is None:
                target_action = go_to_home_pose(robot, joint_names)
                smoothed_action = target_action.clone()

            ret, frame = cap.read()
            if not ret: break
            h, w = frame.shape[:2]

            # --- 分支一：遥控模式 (核心优化：平滑 + dt 控制) ---
            if MANUAL_MODE and not TRACKING_ENABLED:
                # 速度定义：每秒移动的角度 (SO101 建议 8-15 度/秒)
                speed = 18.0 
                step = speed * dt 

                 # --- 接收到退出时执行复位 ---
                if CURRENT_CMD == "exit":
                    print("[Remote] Exit received: Resetting to Home Pose...")
                    # 执行阻塞式复位
                    target_action = go_to_home_pose(robot, joint_names, duration=1.0)
                    # 同步平滑变量，防止复位后有剧动
                    smoothed_action = target_action.clone()
                    # 彻底退出手动模式，允许进入跟随或待机
                    MANUAL_MODE = False
                    CURRENT_CMD = "idle" 

                if CURRENT_CMD == "stop":
                    pass
                
                elif CURRENT_CMD != "idle":
                    if CURRENT_CMD == "turn_left":
                        target_action[0, idx['shoulder_pan.pos']] += step
                    elif CURRENT_CMD == "turn_right":
                        target_action[0, idx['shoulder_pan.pos']] -= step
                    elif CURRENT_CMD == "forward":
                        target_action[0, idx['shoulder_lift.pos']] += step * 0.8
                        target_action[0, idx['elbow_flex.pos']] -= step * 0.4
                    elif CURRENT_CMD == "backward":
                        target_action[0, idx['shoulder_lift.pos']] -= step * 0.8
                        target_action[0, idx['elbow_flex.pos']] += step * 0.5

                    # 安全限幅
                    target_action[0, idx['shoulder_pan.pos']] = torch.clamp(target_action[0, idx['shoulder_pan.pos']], -70, 70)
                    target_action[0, idx['shoulder_lift.pos']] = torch.clamp(target_action[0, idx['shoulder_lift.pos']], -40, 70)
                    target_action[0, idx['elbow_flex.pos']] = torch.clamp(target_action[0, idx['elbow_flex.pos']], -100, 20)

                    # --- 指令平滑滤波器 (EMA) ---
                    # 让实际发送的指令追赶目标位置，消除卡顿感
                if smoothed_action is not None:
                    smoothed_action = REMOTE_SMOOTH_ALPHA * target_action + (1 - REMOTE_SMOOTH_ALPHA) * smoothed_action
                    send_action_to_robot(robot, smoothed_action)

                if opt.show:
                    cv2.putText(frame, f"MANUAL: {CURRENT_CMD}", (20, 40), 1, 1.5, (0, 255, 0), 2)


            # 场景 B: 跟随模式 (TRACKING_ENABLED=True)
            elif TRACKING_ENABLED:
                infer_img, ar, off_x, off_y = letterbox_resize(frame, (640, 640), 56)
                infer_img = infer_img[..., ::-1]  # BGR -> RGB
                infer_img = np.expand_dims(infer_img, axis=0)
                results = pose_rknn.inference(inputs=[infer_img])
                outputs = []
                keypoints = results[3]  # 提取关键点
                for x in results[:3]:
                    layer_idx, stride = 0, 0
                    if x.shape[2] == 20:
                        stride, layer_idx = 32, 20*4*20*4 + 20*2*20*2
                    elif x.shape[2] == 40:
                        stride, layer_idx = 16, 20*4*20*4
                    elif x.shape[2] == 80:
                        stride, layer_idx = 8, 0
                    feature = x.reshape(1, 65, -1)
                    outputs += process(feature, keypoints, layer_idx,
                                x.shape[3], x.shape[2], stride)
                predbox = NMS(outputs)

                best_box = None
                if predbox:
                    # 寻找最大的检测框
                    best_box = max(predbox, key=lambda box: (box.xmax - box.xmin) * (box.ymax - box.ymin))


                if best_box:
                    search_mode = False
                    hit_count += 1
                    miss_count = 0
                    last_face_time = time.time()
                    if hit_count >= HIT_TO_CONFIRM_TRACK:
                        xmin = int((best_box.xmin - off_x) / ar)
                        ymin = int((best_box.ymin - off_y) / ar)
                        xmax = int((best_box.xmax - off_x) / ar)
                        ymax = int((best_box.ymax - off_y) / ar)
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        cv2.putText(frame, f"{CLASSES[best_box.classId]}:{best_box.score:.2f}",
                                    (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 0, 255), 2, cv2.LINE_AA)

                        kpts = best_box.keypoint.reshape(-1, 3)
                        kpts[..., 0] = (kpts[..., 0] - off_x) / ar
                        kpts[..., 1] = (kpts[..., 1] - off_y) / ar

                        # 画点
                        for k, (x_k, y_k, conf) in enumerate(kpts):
                            if x_k != 0 and y_k != 0:
                                cv2.circle(frame, (int(x_k), int(y_k)), 5,
                                        [int(c) for c in kpt_color[k]], -1)

                        # 画骨架
                        for k, sk in enumerate(skeleton):
                            pos1 = (int(kpts[sk[0]-1, 0]), int(kpts[sk[0]-1, 1]))
                            pos2 = (int(kpts[sk[1]-1, 0]), int(kpts[sk[1]-1, 1]))
                            if 0 in pos1 + pos2:
                                continue
                            cv2.line(frame, pos1, pos2,
                                    [int(c) for c in limb_color[k]], 2, cv2.LINE_AA)

                        # 当人物在中心区域时，进入正常跟踪模式
                        search_mode = False
                        hit_count += 1
                        miss_count = 0
                        last_face_time = time.time()
                        # === 新增逻辑：只有当连续命中次数达到阈值后，才移动机械臂 ===
                        if hit_count >= HIT_TO_CONFIRM_TRACK:
                            center_x = (xmin + xmax) / 2
                            center_y = (ymin + ymax) / 2
                            box_width = xmax - xmin
                            
                            ex = center_x - w / 2
                            ey = center_y - h / 2

                            # 使用 EMA 平滑误差
                            ex_f = EMA_ALPHA * ex + (1 - EMA_ALPHA) * ex_f
                            ey_f = EMA_ALPHA * ey + (1 - EMA_ALPHA) * ey_f

                            # 移动机械臂来调整摄像头位置
                            target_action = move_arm_based_on_pose(target_action, pan_center, shoulder_center, elbow_center, wrist_roll_center, ex_f, ey_f, box_width, w)
                            send_action_to_robot(robot, target_action)
                        if opt.show:
                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                else:
                    hit_count = 0; miss_count += 1
                    if miss_count > 10 and (time.time() - last_face_time > 2.0):
                        search_mode = True

                if search_mode:
                    target_action, search_pan, search_dir = search_for_person(
                        robot, joint_names, search_pan, search_dir, pan_center, 
                        SEARCH_PAN_RANGE, SEARCH_STEP_DEG, target_action
                    )
                    if opt.show:
                        cv2.putText(frame, "SEARCHING...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            # 场景 C: 待机 (两者都关)
            else:
                # 保持当前姿态或缓慢回正
                if opt.show: cv2.putText(frame, "STANDBY", (20, 50), 1, 2, (0, 255, 255), 2)

            if opt.show:
                cv2.imshow("Arm Control System", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        try:
            # 程序结束时，回归零位置
            go_to_zero_pose(robot, joint_names, duration=2.0, steps=50)
        except KeyboardInterrupt:
            print("程序被中断，回到初始位置")
        cap.release()
        cv2.destroyAllWindows()
        pose_rknn.release()
        robot.disconnect()
        if opt.show:
            cv2.destroyAllWindows()
        print("Resources released, exit.")

if __name__ == '__main__':
    main()