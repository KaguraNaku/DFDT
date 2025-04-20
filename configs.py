import torch

class Config:
    DETECT_ENABLED = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crop_gpu_id = 0 if torch.cuda.is_available() else -1
    FRAME_QUEUE_SIZE = 30
    face_threshold = 0.7
    max_size = 256

    ANALYSIS_ENABLED = True
    NUM_MAX_FRAMES = 64
    # 与face_threshold用途不同。face_threshold提供人脸检测成功率和检测质量的调整
    # identity_threshold提供场景一致性的调整。但过高的identity_threshold可能使得场景分割过于杂乱，处理麻烦与场景样本量不够。
    # 较高校验阈值以保证场景一致性。防止场景跳变产生的误判。
    identity_threshold = 0.9

