import cv2
import numpy as np
import torch
import mediapipe as mp
import matplotlib.pyplot as plt
import os
from model import SkeletonTransformer

# ========== 配置 ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_FRAMES = 30
MODEL_PATH = "/home/fntq/cv-course/skeleton_transformer.pth"   # 已修正为真实路径
VIDEO_PATH = "/home/fntq/cv-course/badminton_storke_video/demo.mp4"
OUTPUT_PLOT = "/home/fntq/cv-course/badminton_storke_video/attention_demo.png"

# 标签硬编码
label_map = {
    "forehand_drive": 0,
    "forehand_lift": 1,
    "forehand_net_shot": 2,
    "forehand_clear": 3,
    "backhand_drive": 4,
    "backhand_net_shot": 5
}
id_to_name = {v: k for k, v in label_map.items()}

# 检查文件
if not os.path.exists(MODEL_PATH):
    print(f"错误：模型文件 {MODEL_PATH} 不存在！")
    exit(1)
if not os.path.exists(VIDEO_PATH):
    print(f"错误：视频文件 {VIDEO_PATH} 不存在！")
    exit(1)

# 加载模型
model = SkeletonTransformer(num_classes=6).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ========== 预处理函数 ==========
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                    min_detection_confidence=0.5)

def extract_pose(video_path):
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

def resample(frames, target_len):
    if len(frames) == 0:
        return np.zeros((target_len, 132), dtype=np.float32)
    orig = np.linspace(0, len(frames) - 1, len(frames))
    targ = np.linspace(0, len(frames) - 1, target_len)
    new_seq = np.zeros((target_len, 132), dtype=np.float32)
    for j in range(132):
        new_seq[:, j] = np.interp(targ, orig, frames[:, j])
    return new_seq

def normalize(seq):
    hip_x = (seq[:, 23*4] + seq[:, 24*4]) / 2.0
    hip_y = (seq[:, 23*4+1] + seq[:, 24*4+1]) / 2.0
    sw = np.sqrt((seq[:, 11*4] - seq[:, 12*4])**2 + (seq[:, 11*4+1] - seq[:, 12*4+1])**2)
    sw = np.where(sw < 1e-6, 1e-6, sw)
    for k in range(33):
        seq[:, k*4] = (seq[:, k*4] - hip_x) / sw
        seq[:, k*4+1] = (seq[:, k*4+1] - hip_y) / sw
    return seq

# ========== 提取并预处理 ==========
raw = extract_pose(VIDEO_PATH)
seq = resample(raw, TARGET_FRAMES)
seq = normalize(seq)
x = torch.from_numpy(seq).unsqueeze(0).to(DEVICE)
x.requires_grad = True

# ========== 前向传播与梯度计算 ==========
logits = model(x)
probs = torch.softmax(logits, dim=1)
pred_id = logits.argmax(1).item()
pred_name = id_to_name[pred_id]
confidence = probs[0, pred_id].item()

model.zero_grad()
logits[0, pred_id].backward()

grad = x.grad[0]  # [30, 132]
importance = torch.norm(grad, dim=1).cpu().numpy()

# ========== 绘图 ==========
plt.figure(figsize=(10, 4))
plt.bar(range(1, TARGET_FRAMES + 1), importance, color='steelblue')
plt.xlabel("Frame Index")
plt.ylabel("Gradient Magnitude")
plt.title(f"Predicted: {pred_name}  (confidence: {confidence:.2f})")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=150)
print(f"注意力图已保存至: {OUTPUT_PLOT}")
print(f"预测类别: {pred_name}, 置信度: {confidence:.2f}")