# Tuya 云端联动

本文档介绍如何实现 LeRobot 机械臂与 Tuya 涂鸦平台的云端联动控制。通过 Socket TCP 通信，实现涂鸦 APP 远程控制机械臂跟随人体运动和手动遥控功能。

## 1. 系统架构

### 1.1 整体架构图

```
┌──────────────────────────────────────────────────────────────┐
│                    Tuya 云平台系统架构                       │
└──────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────┐
│  用户端                                       │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│  │ Tuya APP │───│ 云端    │──│ 机械臂  │  │
│  │  (手机)  │    │ Tuya IoT │    │ 控制指令 │  │
│  └─────────┘    └─────────┘    └─────────┘  │
└──────────────────────────────────────────────────────────┘
                                    │
                            ┌──────────────────┐
                            │    Socket TCP     │
                            │   8888 端口      │
                            └────────┬─────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────┐
│ LeRobot 控制系统（运行在 RK3576 开发板上）                 │
│                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐│
│  │  Socket 客户端 │──│  模式选择    │──│  机械臂    ││
│  │  接收云端指令  │  │  跟随/遥控   │  │  执行动作  ││
│  └──────────────┘  └──────────────┘  └────────────┘│
│         │                │                                │
│         │                ▼                                │
│         │        ┌──────────────┐                      │
│         │        │  YOLOv8-Pose │                      │
│         └───────▶│  人体检测     │                      │
│                 └──────────────┘                      │
│                         │                              │
│                         ▼                              │
│                 ┌──────────────────┐                  │
│                 │  跟随/遥控控制  │                  │
│                 │  算法         │                  │
│                 └──────────────────┘                  │
└──────────────────────────────────────────────────────┘
```

### 1.2 数据流向图

```
摄像头图像 ──▶ YOLOv8-Pose ──┐
       │                    │
       │                    ▼
       │              检测到的人体
       │              关键点坐标
       │                    │
       ▼                    ▼
┌─────────────────────────────────┐
│       模式选择                   │
└─────────────────────────────────┘
              │
              ├──────────────┐
              │              │
              ▼              ▼
        ┌────────┐     ┌──────────┐
        │跟随模式│     │ 遥控模式 │
        │TRACKING│     │ 收到指令 │
        │ ENABLED│     │ CURRENT │
        │  =True │     │ CMD     │
        └────┬───┘     └────┬───┘
              │              │
              ▼              │
        ┌────────────┐     │
        │ 计算目标    │     │
        │ 位置偏差   │     │
        └────┬───────┘     │
              │              │
              │              ▼
              │        ┌──────────┐
              │        │ 处理指令  │
              │        │ 命令     │
              │        └────┬─────┘
              │             │
              └──────┬──────┘
                     │
                     ▼
              ┌──────────────────┐
              │  PID 控制算法   │
              │ EMA 滤波平滑   │
              └────────┬─────────┘
                      │
                      ▼
              ┌──────────────────┐
              │  关节角度目标    │
              │ [6个舵机角度]   │
              └────────┬─────────┘
                      │
                      ▼
              ┌──────────────────┐
              │ LeRobot 机械臂    │
              │ 执行动作控制     │
              └──────────────────┘
```

### 1.3 核心模块说明

| 模块 | 功能 | 关键参数 |
|------|------|---------|
| **Socket 客户端线程** | 接收涂鸦云端指令 | IP:PORT |
| **人体检测线程** | 实时检测人体姿态 | NMS/置信度阈值 |
| **模式切换** | 跟随/遥控模式切换 | TRACKING_ENABLED |
| **PID 控制** | 计算舵机控制量 | Kp=0.01 |
| **EMA 滤波** | 平滑关键点坐标 | alpha=0.15 |
| **LeRobot API** | 发送关节控制指令 | 6个舵机角度 |

## 2. 快速开始

### 2.1 修改配置

编辑代码文件顶部配置区域：

```python
# Socket 配置（★必须修改★）
SERVER_IP = '192.168.1.100'  # 涂鸦设备 IP 地址
SERVER_PORT = 8888              # 通信端口

# 运行模式
TRACKING_ENABLED = True          # True=跟随模式，False=遥控模式
SYSTEM_STATE = 1                 # 1=运行，0=停止

# 跟随参数
EMA_ALPHA = 0.15               # 平滑系数
objectThresh = 0.5              # 检测置信度阈值
```

### 2.2 启动程序

```bash
# 运行夹爪控制版本
python duckyclaw-tuya.py --show --cam-id 0 --port /dev/ttyACM0

# 运行姿态跟随版本
python yolov8pose_so101-tuya.py --show --cam-id 0 --port /dev/ttyACM0
```

### 2.3 命令行参数

| 参数 | 说明 | 默认值 |
|-----|------|--------|
| `--show` | 显示摄像头预览 | 关闭 |
| `--cam-id` | 摄像头设备号 | 0 |
| `--port` | 机械臂串口 | `/dev/ttyACM0` |
| `--width` | 图像宽度 | 640 |
| `--height` | 图像高度 | 480 |
| `--fps` | 帧率 | 30 |

## 3. Socket 通信详解

### 3.1 通信流程图

```
┌──────────┐
│ Tuya 设备 │
└────┬─────┘
     │
     │ TCP 连接
     ▼
┌────────────────────────────────────┐
│ LeRobot 控制系统                    │
└────┬───────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│ Socket 客户端线程                    │
│                                     │
│ 1. 创建 TCP Socket                  │
│ 2. 连接服务器                      │
│ 3. 接收 JSON 指令                 │
│ 4. 解析指令类型                   │
│    ├─ dpid=1 → 跟随开关             │
│    └─ type=control → 遥控指令       │
└─────────────────────────────────────┘
```

### 3.2 JSON 指令格式

#### 3.2.1 跟随开关指令

```json
// 开启跟随模式
{
  "dpid": 1,
  "value": true
}

// 关闭跟随模式
{
  "dp1": 1,
  "value": false
}
```

**参数说明**：
- `dpid`：功能 ID（固定为 1）
- `value`：`true`=开启跟随，`false`=关闭跟随

#### 3.2.2 遥控指令

```json
{
  "type": "control",
  "command": "forward"
}
```

**支持的命令**：
| 命令 | 说明 |
|------|------|
| `forward` | 前进 |
| `backward` | 后退 |
| `left` | 左转 |
| `right` | 右转 |
| `gripper_open` | 打开夹爪 |
| `gripper_close` | 关闭夹爪 |
| `stop` | 停止 |

### 3.3 客户端代码实现

```python
def socket_client_thread(host, port):
    """Socket 客户端线程，接收涂鸦指令"""
    global SYSTEM_STATE, TRACKING_ENABLED, MANUAL_MODE, CURRENT_CMD
    
    while True:
        # 1. 创建 Socket
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        try:
            # 2. 连接服务器
            client.connect((host, port))
            print(f"已连接 {host}:{port}")
            
            while True:
                # 3. 接收数据
                data = client.recv(1024)
                
                # 4. 粘包处理（按\n 分割消息）
                msgs = data.decode('utf-8').strip().split('\n')
                
                for one_msg in msgs:
                    try:
                        msg = json.loads(one_msg)
                        
                        # 5. 指令路由
                        if 'dpid' in msg:
                            # 跟随开关
                            TRACKING_ENABLED = msg['value']
                            SYSTEM_STATE = 1
                            
                        elif msg.get('type') == 'control':
                            # 遥控指令
                            CURRENT_CMD = msg['command']
                            MANUAL_MODE = True
                    except:
                        continue
                        
        except Exception as e:
            print(f"连接失败: {e}")
            time.sleep(3)  # 重试间隔
        finally:
            client.close()
```

### 3.4 启动 Socket 线程

```python
# 在 main() 函数中
t = threading.Thread(
    target=socket_client_thread,
    args=(SERVER_IP, SERVER_PORT),
    daemon=True
)
t.start()  # 启动接收线程
```

## 4. 人体检测详解

### 4.1 YOLOv8-Pose 模型

```python
from rknnlite.api import RKNNLite

# 加载 RKNN 模型
pose_rknn = RKNNLite()
pose_rknn.load_rknn('./model/yolov8_pose.rknn')
pose_rknn.init_runtime()
```

### 4.2 图像预处理流程

```
摄像头图像
    │
    ▼
letterbox_resize(640x640)
    │
    ▼
BGR → RGB
    │
    ▼
模型推理
    │
    ▼
17 个关键点坐标
```

### 4.3 关键点检测

YOLOv8-Pose 输出 17 个关键点：

| 索引 | 名称 | 说明 |
|------|------|------|
| 0 | nose | 鼻子 |
| 1-2 | eye | 眼睛 |
| 3-4 | ear | 耳朵 |
| 5-6 | shoulder | 肩膀 |
| 7-8 | elbow | 肘部 |
| 9-10 | wrist | 手腕 |
| 11-12 | hip | 髋部 |
| 13-14 | knee | 膝盖 |
| 15-16 | ankle | 脚踝 |

### 4.4 跟踪目标选择

```python
# 计算肩膀中心点作为跟踪目标
left_shoulder = keypoints[5]   # 左肩坐标
right_shoulder = keypoints[6]  # 右肩坐标
center_x = (left_shoulder.x + right_shoulder.x) / 2
center_y = (left_shoulder.y + right_shoulder.y) / 2
```

**选择肩膀中心的原因**：
- ✅ 稳定性好
- ✅ 不易被遮挡
- ✅ 反映整体运动趋势

## 5. 控制算法

### 5.1 死区控制

防止机械臂在目标附近抖动：

```python
# 定义中心区域
CENTER_X_MIN = 0.3   # 画面 30%
CENTER_X_MAX = 0.7   # 画面 70%

# 死区参数
DEAD_ZONE_X = 25  # 水平死区（像素）
DEAD_ZONE_Y = 35  # 垂直死区（像素）
```

**工作原理**：
```
目标位置 ──▶ 死区判断
                │
                ├─ 死区内 → 不控制（保持不动）
                │
                └─ 死区外 → PID 控制
```

### 5.2 PID 控制原理

```python
def move_arm_based_on_pose(current_action, ex, ey):
    """根据偏差计算控制量"""
    
    # 比例增益
    kp = 0.01
    
    # 计算控制量
    pan_offset = ex * kp      # 水平偏差 × 增益
    shoulder_offset = ey * kp  # 垂直偏差 × 增益
    
    # 应用到关节
    action[0] = np.clip(
        current_action[0] + pan_offset,  # 水平关节
        -90, 90                  # 限幅
    )
    
    return action
```

### 5.3 EMA 滤波

指数移动平均滤波，减少抖动：

```python
EMA_ALPHA = 0.15  # 平滑系数

def ema_filter(current, previous):
    """平滑关键点坐标"""
    return EMA_ALPHA * current + (1 - EMA_ALPHA) * previous
```

**不同平滑系数效果**：

| 值 | 效果 |
|----|------|
| 0.05 | 最平滑，响应慢 |
| 0.15 | 推荐值 |
| 0.30 | 响应快，可能抖动 |
| 0.50 | 最快，可能不稳定 |

## 6. 机械臂控制

### 6.1 舵机定义

SO101 机械臂有 6 个舵机：

| 舵机 | 范围 | 说明 |
|------|------|------|
| shoulder_pan | -90° ~ +90° | 底座旋转 |
| shoulder_lift | -45° ~ +45° | 肩部抬起 |
| elbow_flex | -90° ~ 0° | 肘部弯曲 |
| wrist_flex | -70° ~ +70° | 手腕弯曲 |
| wrist_roll | 0° ~ 180° | 手腕旋转 |
| gripper | 0° ~ 30° | 夹爪开合 |

### 6.2 姿态预设

```python
# 初始姿态
HOME_POSE_DEG = {
    'shoulder_pan.pos': 0.0,    # 正中
    'shoulder_lift.pos': 30.0,   # 抬起
    'elbow_flex.pos': -60.0,  # 弯曲
    'wrist_flex.pos': 20.0,   # 向下
    'wrist_roll.pos': 70.0,   # 水平
    'gripper.pos': 0.0,       # 夹爪关闭
}

# 归位姿态
ZORE_POSE_DEG = {
    'shoulder_pan.pos': -10.0,
    'shoulder_lift.pos': -5.0,
    'elbow_flex.pos': -5.0,
    'wrist_flex.pos': -70.0,
    'widget_roll.pos': 60.0,
    'gripper.pos': 0.0,
}
```

### 6.3 发送到机械臂

```python
def send_action_to_robot(robot, action):
    """发送关节角度到机械臂"""
    
    # 转换为关节字典
    joint_names = list(robot.action_features.keys())
    robot_action = {
        name: angle 
        for name, angle in zip(joint_names, action)
    }
    
    # 发送动作
    robot.send_action(robot_action)
```

## 7. 调试技巧

### 7.1 查看连接状态

程序启动时打印：
```
启动 Socket 客户端线程，目标: 192.168.1.100:8888
```

### 7.2 查看控制指令

接收到的指令会打印：
```
[Socket] Received Command: forward
[Socket] Received Command: stop
```

### 7.3 查看关节角度

发送控制指令时打印：
```python
send_action_to_robot: {
    'shoulder_pan.pos': 15.0,
    'shoulder_lift.pos': 30.0,
    ...
}
```

### 7.4 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 连接失败 | IP 错误 | 修改 SERVER_IP |
| 机械臂不动 | 串口权限 | sudo 运行 |
| 抖动严重 | 平滑系数过大 | 减小 EMA_ALPHA |
| 跟随不准 | 增益过高 | 减小 kp |

## 8. 扩展功能

### 8.1 添加新命令

在 `process_remote_command` 中添加：
```python
elif cmd == "custom":
    target_action[0, 3] += 10  # 第 4 个关节 +10°
```

### 8.2 自定义检测区域

修改中心区域定义：
```python
# 缩小检测区域
CENTER_X_MIN = 0.4
CENTER_X_MAX = 0.6
```

### 8.3 数据上报

```python
def report_status(sock):
    """上报状态到云端"""
    data = json.dumps({
        'type': 'status',
        'angles': action.tolist()
    })
    sock.send(data.encode())
```

## 9. 总结

**系统特点**：
- ✅ Socket TCP 实时通信
- ✅ 云端指令控制
- ✅ 本地人体检测
- ✅ PID 闭环控制
- ✅ EMA 滤波平滑

**性能指标**：
- 检测帧率：25-30 FPS
- 控制延迟：< 50ms
- 关节控制精度：0.1°

**适用场景**：
- 智能跟随拍摄
- 人机交互控制
- 远程医疗辅助
- 工业检测

**下一步优化**：
- 添加姿态记忆
- 手势控制
- 多人识别
- 语音控制集成