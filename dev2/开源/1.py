import re
from maix import camera, display, image, time, nn, app
from maix.v1.machine import UART
import struct, math
import cv2
import numpy as np

# ============== 全局配置 ==============
# 全局变量

CAM_WIDTH, CAM_HEIGHT = 240, 160
CENTER_REGION_RATIO = 0.5  # 中心区域占屏幕的比例
detector = nn.YOLOv5(model="/root/models/yolov5.mud", dual_buff=True)
READ_HEADER = [0xff, 0xfe, 3]  # 串口数据包头
WHITE_THRESHOLD = [(100, 255)]  # 白色阈值

path_points = []
trail_thickness = 3
MAX_POINTS = 100
last_cx, last_cy = 160, 120      # 上一帧坐标
LASER_ALPHA   = 0.3         # 惯性系数：越大越平滑
LASER_JUMP_TH = 90           # 像素跳变门限

laser_cx=0
laser_cy=0

RED_THRESHOLDS = [(0, 100, 19, 127, 0, 127)]
BLUE_THRESHOLDS = [(50, 80, -1, 127, -47, -11)]
# 圆环识别阈值

# ============== 卡尔曼滤波器初始化 (仅用于X轴预测) ==============
class XAxisKalmanFilter:
    def __init__(self, process_noise=0.01, measurement_noise=0.1):
        # 状态向量: [x, vx] (位置和速度)
        self.state = np.zeros((2, 1), dtype=np.float32)
        
        # 状态转移矩阵 (假设匀速模型)
        self.transition_matrix = np.array([
            [1, 1],  # x_next = x + vx
            [0, 1]   # vx_next = vx
        ], dtype=np.float32)
        
        # 测量矩阵 (只测量位置)
        self.measurement_matrix = np.array([[1, 0]], dtype=np.float32)
        
        # 过程噪声协方差
        self.process_noise_cov = np.eye(2, dtype=np.float32) * process_noise
        
        # 测量噪声协方差
        self.measurement_noise_cov = np.eye(1, dtype=np.float32) * measurement_noise
        
        # 误差协方差矩阵
        self.error_cov = np.eye(2, dtype=np.float32)
        
        # 是否已初始化
        self.initialized = False
    
    def init(self, x):
        """初始化滤波器"""
        self.state = np.array([[x], [0]], dtype=np.float32)  # 初始速度为0
        self.error_cov = np.eye(2, dtype=np.float32) * 1000  # 高初始不确定性
        self.initialized = True
    
    def predict(self):
        """预测下一状态"""
        if not self.initialized:
            return None
            
        # 预测状态
        self.state = self.transition_matrix @ self.state
        # 预测误差协方差
        self.error_cov = self.transition_matrix @ self.error_cov @ self.transition_matrix.T + self.process_noise_cov
        
        return self.state[0, 0]  # 返回预测的x
    
    def update(self, x):
        """使用新测量值更新滤波器"""
        if not self.initialized:
            self.init(x)
            return x
            
        # 预测
        self.predict()
        
        # 测量向量 (只包含x坐标)
        measurement = np.array([[x]], dtype=np.float32)
        
        # 计算卡尔曼增益
        S = self.measurement_matrix @ self.error_cov @ self.measurement_matrix.T + self.measurement_noise_cov
        K = self.error_cov @ self.measurement_matrix.T @ np.linalg.inv(S)
        
        # 更新状态估计
        innovation = measurement - self.measurement_matrix @ self.state
        self.state = self.state + K @ innovation
        
        # 更新误差协方差
        I = np.eye(2, dtype=np.float32)
        self.error_cov = (I - K @ self.measurement_matrix) @ self.error_cov
        
        return self.state[0, 0]  # 返回滤波后的x

# 创建卡尔曼滤波器实例
kf = XAxisKalmanFilter(process_noise=0.02, measurement_noise=0.1)

# ============== 类初始化 ==============
class SerialComm:               #串口类
    def __init__(self):
        self.uart = UART("/dev/ttyS0", 115200)
        time.sleep_ms(100)  # 等待串口初始化

    def send_detection_data(self, mode, data):
        
        """发送检测数据到串口"""
        if mode == 1:
            print(data)
            packed_data = struct.pack(">BBBHHBB", 0xb3, 0xb3, 0xb2, data[0], data[1], data[2], 0x5b)
            self.uart.write(packed_data)
        elif mode == 2:
            packed_data = struct.pack(">BBBHHBB", 0xb3, 0xb3, 0xb2, data[0], data[1], 0x5a, 0x5b)
            self.uart.write(packed_data)

    def read_detect(self):
        """从串口读取模式切换指令"""
        data = self.uart.read(40)
        if data and len(data) >= READ_HEADER[2]:
            if data[0] == READ_HEADER[0] and data[READ_HEADER[2] - 1] == READ_HEADER[1]:
                return data[1]
        time.sleep_ms(1)
        return None


# ============== 视觉处理函数 ==============
def initialize_camera(width=160, height=160, skip_frames=30):
    """初始化摄像头"""
    cam = camera.Camera(width, height, fps=60)
    cam.skip_frames(skip_frames)
    return cam

def display_fps(frame, prev_frame_time, position_x=5, position_y=5):
    """
    在图像上显示当前帧率
    :param frame: 当前帧图像
    :param prev_frame_time: 上一帧的时间
    :param position: 显示位置 (x, y)
    :param color: 文本颜色 (B, G, R)
    :return: 当前帧时间 (用于下一次调用)
    """
    # 计算当前帧时间
    current_frame_time = time.time()
    
    # 计算帧率
    fps = 1.0 / (current_frame_time - prev_frame_time)
    
    # 在图像上显示FPS
    fps_text = f"FPS: {fps:.1f}"
    frame.draw_string(position_x, position_y, fps_text, image.COLOR_BLUE, scale=1)
    
    return current_frame_time

# 摄像头位置补偿参数（根据实际安装位置调整）
CAMERA_OFFSET_X = 0  # 水平方向补偿值（像素）
CAMERA_OFFSET_Y = 0  # 垂直方向补偿值（像素）
CAMERA_ANGLE_COMP = 1.0  # 视角角度补偿系数（1.0为无补偿）

# 相对坐标系参数
REF_POINT_X = CAM_WIDTH // 2  # 参考点X坐标（图像中心）
REF_POINT_Y = CAM_HEIGHT // 2  # 参考点Y坐标（图像中心）

# ============== 坐标转换函数 ==============
def transform_coordinates(x, y):
    """
    将原始坐标转换为相对坐标并进行补偿
    :param x: 原始x坐标
    :param y: 原始y坐标
    :return: (trans_x, trans_y) 转换后的坐标
    """
    # 1. 视角角度补偿
    angle_comp_x = (x - REF_POINT_X) * CAMERA_ANGLE_COMP
    angle_comp_y = (y - REF_POINT_Y) * CAMERA_ANGLE_COMP
    
    # 2. 位置偏移补偿
    comp_x = angle_comp_x + REF_POINT_X + CAMERA_OFFSET_X
    comp_y = angle_comp_y + REF_POINT_Y + CAMERA_OFFSET_Y
    
    # 3. 边界保护
    comp_x = max(0, min(CAM_WIDTH - 1, comp_x))
    comp_y = max(0, min(CAM_HEIGHT - 1, comp_y))
    
    return int(comp_x), int(comp_y)


def is_in_center_region(img, x, y, width, height):
    """检查点是否在中心区域"""
    center_x, center_y = x + width // 2, y + height // 2

    region_size_x = int(240 * CENTER_REGION_RATIO) - 30
    region_size_y = int(160 * CENTER_REGION_RATIO) - 30
    msg1 = ""
    msg1 += "cen_x:{} \ncen_Y:{}".format(center_x, center_y)
    img.draw_string(0, 70, msg1, image.COLOR_RED, scale=1.0)
    if region_size_x <= center_x <= region_size_x + 120 and region_size_y <= center_y <= region_size_y + 120:
        return 1

    return 0

def  Get_yolov5(img):
    LineData = [0, 0, 0, 0]
    objs = detector.detect(img, conf_th = 0.5, iou_th = 0.45)

    if not objs:
        return None
    obj = max(objs, key=lambda obj: obj.w * obj.h)
    if obj:
        img.draw_rect(obj.x, obj.y, obj.w, obj.h, color = image.COLOR_RED)
        msg = f'{detector.labels[obj.class_id]}: {obj.score:.2f}'
        img.draw_string(obj.x, obj.y, msg, color = image.COLOR_RED)
        ROI_flag = is_in_center_region(img, obj.x, obj.y, obj.w, obj.h)
        if ROI_flag == 1:
            img.draw_string(obj.x, obj.y, msg, color = image.COLOR_GREEN)
            img.draw_rect(obj.x, obj.y, obj.w, obj.h, color = image.COLOR_GREEN)
            center_xx = obj.x + obj.w/2
            center_yy = obj.y + obj.h/2
            img.draw_circle(int(center_xx), int(center_yy), 5, color=image.COLOR_RED, thickness=-1)
            LineData = [center_xx, center_yy, 3]

            return LineData
def is_get_yolov5_objects(img):
    """使用YOLOv5检测物体并返回结果"""
    objs = detector.detect(img, conf_th = 0.5, iou_th = 0.45)
    if not objs:
        return 0
    obj = max(objs, key=lambda obj: obj.w * obj.h)
    if obj:
        img.draw_rect(obj.x, obj.y, obj.w, obj.h, color = image.COLOR_RED)
        msg = f'{detector.labels[obj.class_id]}: {obj.score:.2f}'
        img.draw_string(obj.x, obj.y, msg, color = image.COLOR_RED)
        ROI_flag = is_in_center_region(img, obj.x, obj.y, obj.w, obj.h)
        if ROI_flag == 1:
            img.draw_string(obj.x, obj.y, msg, color = image.COLOR_GREEN)
            img.draw_rect(obj.x, obj.y, obj.w, obj.h, color = image.COLOR_GREEN)
        return 1
    return 0


def get_yolov5_objects(img):
    """使用YOLOv5检测物体并返回结果"""
    objs = detector.detect(img, conf_th = 0.5, iou_th = 0.45)
    if not objs:
        return None
    obj = max(objs, key=lambda obj: obj.w * obj.h)
    if obj:
        img.draw_rect(obj.x, obj.y, obj.w, obj.h, color = image.COLOR_RED)
        msg = f'{detector.labels[obj.class_id]}: {obj.score:.2f}'
        img.draw_string(obj.x, obj.y, msg, color = image.COLOR_RED)
        y_center_x, y_center_y = obj.x + obj.w // 2, obj.y + obj.h // 2
        msg1 = ""
        msg1 += "cen_x:{} \ncen_Y:{}".format(y_center_x, y_center_y)
        img.draw_string(0, 120, msg1, image.COLOR_RED, scale=1.0)
        return y_center_x, y_center_y, 0
    
    return 0

ffff = 0
ffff1 = 0

def clamp_coordinates(x, y, width, height):
    """
    限制坐标在图像范围内
    :param x: 原始x坐标
    :param y: 原始y坐标
    :param width: 图像宽度
    :param height: 图像高度
    :return: (clamped_x, clamped_y) 限制后的坐标
    """
    # 添加安全边界 (5%的边界)
    safe_margin_x = int(width * 0.05)
    safe_margin_y = int(height * 0.05)
    
    # 限制x坐标在安全范围内
    clamped_x = max(safe_margin_x, min(width - safe_margin_x - 1, x))
    
    # 限制y坐标在安全范围内
    clamped_y = max(safe_margin_y, min(height - safe_margin_y - 1, y))
    
    # 如果坐标被限制，绘制警告标记
    if clamped_x != x or clamped_y != y:
        return int(clamped_x), int(clamped_y), True
    
    return int(x), int(y), False

def get_rect_center(img):
    """
    检测图像中的矩形并返回其中心点坐标
    :param img: 输入图像
    :return: [x, y, 2] 格式的矩形中心点数据，未检测到时返回 None
    """
    global last_cx, last_cy, ffff, kf
    
    # 存储上一次检测到的坐标
    if ffff == 0 and is_get_yolov5_objects(img) == 0:
        #print(11111111)
        return 0
    ffff = 1
    
    # 1. 图像预处理
    try:
        img_cv = image.image2cv(img, ensure_bgr=False, copy=False)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)
    except Exception as e:
        print(f"图像预处理失败: {e}")
        return None
    
    # 2. 轮廓检测
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    max_area = 0
    best_rect = None
    target_center = None
    
    # 3. 寻找最佳矩形
    for cnt in contours:
        # 跳过小面积轮廓
        area = cv2.contourArea(cnt)
        if area < 500:
            continue
            
        # 多边形逼近
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        # 只处理四边形
        if len(approx) == 4:
            # 获取最小外接矩形
            rect = cv2.minAreaRect(cnt)
            if area > max_area:
                max_area = area
                best_rect = rect
                target_center = (int(rect[0][0]), int(rect[0][1]))
    
    # 4. 检测到矩形 - 直接使用检测值
    if best_rect is not None:
        # 绘制检测结果
        box = cv2.boxPoints(best_rect)
        box = np.int0(box)
        
        # 绘制矩形边框
        for i in range(4):
            start = (int(box[i][0]), int(box[i][1]))
            end = (int(box[(i+1)%4][0]), int(box[(i+1)%4][1]))
            img.draw_line(start[0], start[1], end[0], end[1], 
                         color=image.COLOR_GREEN, thickness=2)
        
        # 绘制原始中心点
        raw_cx, raw_cy = target_center
        #raw_cx, raw_cy = transform_coordinates(raw_cx, raw_cy)
        img.draw_circle(raw_cx, raw_cy, 5, color=image.COLOR_RED, thickness=-1)
        msg1 = ""
        msg1 += "center_x:{} \ncenter_Y:{}".format(raw_cx, raw_cy)
        img.draw_string(0, 120, msg1, image.COLOR_RED, scale=1.0)
        
        # 更新卡尔曼滤波器
        kf.update(raw_cx)
        
        # 保存当前坐标作为上一次检测值
        last_cx, last_cy = raw_cx, raw_cy
        
        # 应用坐标限幅
        clamped_x, clamped_y, was_clamped = clamp_coordinates(raw_cx, raw_cy, CAM_WIDTH, CAM_HEIGHT)
        
        # 返回检测到的坐标
        return [clamped_x, clamped_y, 2]
    
    # 5. 未检测到矩形 - 仅对X轴进行预测
    # 使用卡尔曼滤波器预测X坐标
    pred_x = kf.predict()
    
    # 如果预测成功，使用预测的X坐标和上一次的Y坐标
    if pred_x is not None:
        # 应用坐标限幅
        clamped_x, clamped_y, was_clamped = clamp_coordinates(pred_x, last_cy, CAM_WIDTH, CAM_HEIGHT)
        
        # 绘制预测位置（黄色）
        img.draw_circle(clamped_x, last_cy, 5, color=image.COLOR_YELLOW, thickness=-1)
        img.draw_string(clamped_x+10, last_cy, f"Pred-X: {clamped_x}", image.COLOR_YELLOW, scale=1)
        
        # 如果坐标被限制，显示警告
        if was_clamped:
            img.draw_string(clamped_x+10, last_cy+15, "Clamped", image.COLOR_YELLOW, scale=1)
        
        return [clamped_x, last_cy, 2]
    
    # 6. 如果预测失败，使用上一次的坐标
    if last_cx is not None and last_cy is not None:
        # 应用坐标限幅
        clamped_x, clamped_y, was_clamped = clamp_coordinates(last_cx, last_cy, CAM_WIDTH, CAM_HEIGHT)
        
        # 绘制保持位置（紫色）
        img.draw_circle(clamped_x, clamped_y, 5, color=image.COLOR_PURPLE, thickness=-1)
        img.draw_string(clamped_x+10, clamped_y, "Last", image.COLOR_PURPLE, scale=1)
        
        return [clamped_x, clamped_y, 2]
    
    # 7. 所有情况都失败，返回默认值
    return [CAM_WIDTH // 2, CAM_HEIGHT // 2, 2]

def get_rect_center2(img):
    """
    检测图像中的矩形并返回其中心点坐标（带惯性滤波）
    :param img: 输入图像
    :return: [x, y, 2] 格式的矩形中心点数据，未检测到时返回 None
    """
    global last_cx, last_cy, ffff1
    if ffff1 == 0 and is_get_yolov5_objects(img) == 0:
        return 0
    ffff1 = 0
    # 1. 图像预处理
    try:
        img_cv = image.image2cv(img, ensure_bgr=False, copy=False)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)
    except Exception as e:
        print(f"图像预处理失败: {e}")
        return None
    # 2. 轮廓检测
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    max_area = 0
    best_rect = None
    target_center = None
    
    # 3. 寻找最佳矩形
    for cnt in contours:
        # 跳过小面积轮廓
        area = cv2.contourArea(cnt)
        if area < 500:
            continue
            
        # 多边形逼近
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        # 只处理四边形
        if len(approx) == 4:
            # 获取最小外接矩形
            rect = cv2.minAreaRect(cnt)
            if area > max_area:
                max_area = area
                best_rect = rect
                target_center = (int(rect[0][0]), int(rect[0][1]))
    
    # 4. 未检测到矩形
    if best_rect is None:
        return None
    
    # 5. 绘制检测结果
    box = cv2.boxPoints(best_rect)
    box = np.int0(box)
    
    # 绘制矩形边框
    for i in range(4):
        start = (int(box[i][0]), int(box[i][1]))
        end = (int(box[(i+1)%4][0]), int(box[(i+1)%4][1]))
        img.draw_line(start[0], start[1], end[0], end[1], 
                     color=image.COLOR_GREEN, thickness=2)
    
    # 绘制原始中心点
    raw_cx, raw_cy = target_center
    img.draw_circle(raw_cx, raw_cy, 5, color=image.COLOR_RED, thickness=-1)
    msg1 = ""
    msg1 += "center_x:{} \ncenter_Y:{}".format(raw_cx, raw_cy)
    #print(msg1)
    img.draw_string(0, 120, msg1, image.COLOR_RED, scale=1.0)
    # 8. 返回结果
    return [raw_cx, raw_cy, 2]

def mode():
    pass

# ============== 主程序 ==============
def main():
    # 初始化硬件
    comm = SerialComm()
    cam = initialize_camera(CAM_WIDTH, CAM_HEIGHT)  # 物料识别摄像头
    cam2 = camera.Camera(detector.input_width(), detector.input_height(), detector.input_format())
    disp = display.Display()
    
    # 状态变量
    #current_mode = comm.read_detect()
    current_mode = 2
    prev_frame_time = time.time()  # 用于FPS计算
    
    # 模式处理函数映射
    mode_handlers = {
        0: lambda f: mode(),  # 模式1: 检测
        1: lambda f: get_rect_center2(f),  # 模式0: 检测
        2: lambda f: get_rect_center(f),  # 模式1: 检测
        3: lambda f: get_yolov5_objects(f),  # 模式1: 检测
        
    }
    
    # 模式发送类型映射
    send_modes = {
        1: 1,  # 模式0发送类型1
        2: 1,  # 模式1发送类型1
        3: 1,  # 模式2发送类型1
    }

    while not app.need_exit():
        # 读取串口模式
        new_mode = comm.read_detect()
        if new_mode is not None:
            current_mode = new_mode
            print(f"Switched to mode {current_mode}")
        
        # 读取摄像头帧
        frame = cam.read()
        # 在帧上显示FPS
        prev_frame_time = display_fps(frame, prev_frame_time)
        
        # 根据当前模式处理图像
        if current_mode in mode_handlers:
            handler = mode_handlers[current_mode]
            data = handler(frame)
            
            # 发送检测结果
            if data:
                #print(data)
                comm.send_detection_data(send_modes[current_mode], data)

            # 显示结果
            disp.show(frame)
            
            # 显示稳定状态

        else:
            # 无模式时显示原始图像
            disp.show(frame)


if __name__ == "__main__":
    main()