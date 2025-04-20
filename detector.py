# np.array([H,W,C]) => BGR

import numpy as np
import torch


from configs import Config
from PIL import Image
from insightface.app import FaceAnalysis

app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
cfg = Config()

app.prepare(ctx_id=cfg.crop_gpu_id, det_size=(cfg.max_size, cfg.max_size))





# 需要谨慎评估对齐带来的影响。注意力机制不依赖对齐。
def crop_face(img):
    faces = app.get(img)
    # rimg = app.draw_on(img, faces)
    # print()
    frame = []
    for face in faces:
        box = face['bbox']
        kps = face['kps']
        embedding = face['embedding']
        # numpy => PIL Image => numpy
        cropped = np.array(Image.fromarray(img).crop(box))
        # 已经在aligned的基础上提取了embedding。因此所有操作均基于对齐后的embedding，无需额外对齐操作。
        frame.append({
            'box': box,
            'kps': kps,
            'embedding': embedding,
            'cropped': cropped,
        })
    return frame





