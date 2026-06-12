import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from dataset import BadmintonDataset
from model import SkeletonTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

BATCH_SIZE = 16
EPOCHS = 80
LR = 1e-3

# 数据加载
full_train_set = BadmintonDataset("X_train.npy", "y_train.npy")
test_set = BadmintonDataset("X_test.npy", "y_test.npy")

# 计算类别权重
labels = full_train_set.y
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

val_size = int(0.1 * len(full_train_set))
train_size = len(full_train_set) - val_size
train_set, val_set = random_split(
    full_train_set, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False)

model = SkeletonTransformer(num_classes=6).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

for epoch in range(EPOCHS):
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # 数据增强1：噪声
        X_batch = X_batch + torch.randn_like(X_batch) * 0.01
        # 增强2：随机时间偏移
        shift = torch.randint(-3, 4, (X_batch.size(0),), device=device)
        for i in range(X_batch.size(0)):
            if shift[i] != 0:
                X_batch[i] = torch.roll(X_batch[i], shifts=shift[i].item(), dims=0)
                if shift[i] > 0:
                    X_batch[i, :shift[i]] = X_batch[i, shift[i]].unsqueeze(0)
                else:
                    X_batch[i, shift[i]:] = X_batch[i, shift[i]].unsqueeze(0)
        # 增强3：随机丢失连续2帧（20%概率）
        if torch.rand(1).item() < 0.2:
            start = torch.randint(0, X_batch.size(1)-2, (1,)).item()
            X_batch[:, start:start+2, :] = 0.0

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * X_batch.size(0)
        train_correct += (logits.argmax(1) == y_batch).sum().item()
        train_total += y_batch.size(0)

    train_acc = train_correct / train_total

    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val, y_val = X_val.to(device), y_val.to(device)
            logits = model(X_val)
            loss = criterion(logits, y_val)
            val_loss += loss.item() * X_val.size(0)
            val_correct += (logits.argmax(1) == y_val).sum().item()
            val_total += y_val.size(0)
    val_acc = val_correct / val_total

    scheduler.step()

    print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
          f"Train Loss: {train_loss/train_total:.4f} | Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss/val_total:.4f} | Val Acc: {val_acc:.4f}")

# 测试
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for X_test_batch, y_test_batch in test_loader:
        X_test_batch = X_test_batch.to(device)
        logits = model(X_test_batch)
        preds = logits.argmax(1).cpu()
        y_true.extend(y_test_batch.numpy())
        y_pred.extend(preds.numpy())

test_acc = accuracy_score(y_true, y_pred)
print(f"\n测试准确率: {test_acc:.4f}")

target_names = ["正手平抽", "正手挑球", "正手网前球",
                "正手高远球", "反手平抽", "反手网前球"]
print("\n混淆矩阵:\n", confusion_matrix(y_true, y_pred))
print("\n分类报告:\n", classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

torch.save(model.state_dict(), "skeleton_transformer.pth")
print("模型已保存")