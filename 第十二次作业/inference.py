import cv2
import numpy as np
import torch
import mediapipe as mp
import json
from model import SkeletonTransformer
import os
# ========== 配置 ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_FRAMES = 30           # 必须与训练时 preprocess.py 中的值一致
MODEL_PATH = "skeleton_transformer.pth"
LABEL_MAP_PATH = "label_map.json"

# 加载标签映射（反转：id -> name）
with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)
id_to_name = {v: k for k, v in label_map.items()}

# 初始化 MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    min_detection_confidence=0.5)

# 加载模型（结构与训练时完全一致）
model = SkeletonTransformer(num_classes=6).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ========== 预处理函数（与 preprocess.py 完全一致） ==========
def extract_pose_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if results.pose_landmarks:
            vec = []
            for lm in results.pose_landmarks.landmark:
                vec.extend([lm.x, lm.y, lm.z, lm.visibility])
            frames.append(vec)
        else:
            frames.append([0.0] * 132)
    cap.release()
    return np.array(frames, dtype=np.float32)

def resample_sequence(frames, target_len):
    if len(frames) == 0:
        return np.zeros((target_len, 132), dtype=np.float32)
    orig_idx = np.linspace(0, len(frames)-1, len(frames))
    targ_idx = np.linspace(0, len(frames)-1, target_len)
    new_seq = np.zeros((target_len, 132), dtype=np.float32)
    for j in range(132):
        new_seq[:, j] = np.interp(targ_idx, orig_idx, frames[:, j])
    return new_seq

def normalize_skeleton(seq):
    hip_x = (seq[:, 23*4] + seq[:, 24*4]) / 2.0
    hip_y = (seq[:, 23*4+1] + seq[:, 24*4+1]) / 2.0
    shoulder_width = np.sqrt(
        (seq[:, 11*4] - seq[:, 12*4])**2 +
        (seq[:, 11*4+1] - seq[:, 12*4+1])**2
    )
    shoulder_width = np.where(shoulder_width < 1e-6, 1e-6, shoulder_width)
    for k in range(33):
        seq[:, k*4]     = (seq[:, k*4]     - hip_x) / shoulder_width
        seq[:, k*4 + 1] = (seq[:, k*4 + 1] - hip_y) / shoulder_width
    return seq

# ========== 推理 ==========
def inference(video_path):
    if not os.path.exists(video_path):
        print(f"错误：视频 {video_path} 不存在")
        return

    # 1. 提取骨架
    raw = extract_pose_from_video(video_path)
    # 2. 重采样到 30 帧
    seq = resample_sequence(raw, TARGET_FRAMES)
    # 3. 归一化
    seq = normalize_skeleton(seq)
    # 4. 转张量 [1, 30, 132]
    x = torch.from_numpy(seq).unsqueeze(0).to(DEVICE)

    # 5. 预测
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred_id = torch.max(probs, dim=1)

    pred_class = id_to_name[pred_id.item()]
    confidence = conf.item()

    print(f"Predicted class: {pred_class}")
    print(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    # 改成你的测试视频路径
    import sys
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "/home/fntq/cv-course/badminton_storke_video/demo.mp4"
    inference(video_path)