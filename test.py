from queue import Queue
from threading import Event, Thread

import cv2
import mss
import numpy as np
import pyvirtualcam
import torch

from configs import Config
from utils import get_identities

cfg = Config()
# 共享配置

# 创建事件和队列
stop_event = Event()
frame_queue = Queue(maxsize=cfg.FRAME_QUEUE_SIZE)
identities_queue = Queue(maxsize=cfg.NUM_MAX_FRAMES)
if cfg.ANALYSIS_ENABLED:
    from freq_ts.device.models import DualStreamVideoModel
    model = DualStreamVideoModel()
    model.load_state_dict(torch.load("/home/naku/桌面/fold0_best_auc.pth")['model'])
    model.eval()


def screen_capture(stop_flag):
    """生产者：帧捕获与分发"""
    with pyvirtualcam.Camera(width=1920, height=1080, fps=5) as cam:
        print(f'虚拟摄像头已启动: {cam.device}')
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            while not stop_flag.is_set():
                # 捕获帧
                screenshot = sct.grab(monitor)
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                frame = cv2.flip(frame, 1)

                cam.send(frame)
                cam.sleep_until_next_frame()

                if cfg.DETECT_ENABLED:
                    frame_queue.put(frame.copy())  # 必须深拷贝


def detect_worker():
    """消费者：视频级身份检测，确保场景一致"""
    faces = []
    while not stop_event.is_set() or not frame_queue.empty():
        try:
            frame = frame_queue.get(timeout=1)
            # 在此处添加您的分析逻辑（示例：运动检测）
            identities_dict = get_identities(frame, faces)
            if identities_dict:
                identities_queue.put(identities_dict.copy())
        except Exception as e:
            print(f"分析错误: {str(e)}")

def analyze_worker():
    global model
    print()
    identities_dict = identities_queue.get()
    print()
    cv2.waitKey(10)

# 启动线程
capture_thread = Thread(target=screen_capture, args=(stop_event,))
detect_thread = Thread(target=detect_worker)
analyze_thread = Thread(target=analyze_worker)

capture_thread.start()
detect_thread.start()
analyze_thread.start()
# 主线程管理
try:
    while True:
        pass
except KeyboardInterrupt:
    stop_event.set()
    capture_thread.join()
    detect_thread.join()
    analyze_thread.join()
    cv2.destroyAllWindows()
    print("系统已安全停止")