# train.py
# -*- coding: utf-8 -*-
"""
最小可跑训练脚本：
- 使用内置的 ToyDataset 随机生成“看起来合理”的训练样本（含有序标签 j_left, j_right）
- 训练 Hand2GripperModel：有序左/右交叉熵 + 成对相容性分类 + 互斥正则 + 距离先验(可选) + 接触一致(可选)
- 打印单指准确率、成对有序准确率

替换真实数据时：
- 把 ToyDataset 换为你的 Dataset，返回以下字段（PyTorch tensor）：
  color[B,3,H,W], bbox[B,4], keypoints_3d[B,21,3], contact[B,21], is_right[B], j_left[B], j_right[B]
"""
import os
import math
import random
import cv2
import numpy as np
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from models.model import Hand2GripperModel
from processor_config import DataManager


# ------------------------------
# 样例数据集（可跑通）
# ------------------------------
class ToyHand2GripperDataset(Dataset):
    def __init__(self, length=2000, H=480, W=640, seed=123):
        super().__init__()
        self.length = length
        self.H, self.W = H, W
        random.seed(seed)
        torch.manual_seed(seed)

    def __len__(self):
        return self.length

    def _make_keypoints(self):
        """
        生成一个粗略合理的手部关键点云：
        - 腕部(0)在原点附近
        - 其他点从正态分布采样再缩放
        """
        kp = torch.randn(21, 3) * 0.05
        kp[0] = 0.0  # wrist
        return kp

    def _make_contact_from_pair(self, j_left, j_right, N=21):
        """
        根据 (j_left, j_right) 生成一个“有峰值”的 contact 向量
        """
        c = torch.zeros(N)
        c[j_left] = 1.0
        c[j_right] = 1.0
        # 给左右相邻关节一点扩散
        if j_left - 1 >= 0: c[j_left - 1] += 0.3
        if j_left + 1 < N: c[j_left + 1] += 0.3
        if j_right - 1 >= 0: c[j_right - 1] += 0.3
        if j_right + 1 < N: c[j_right + 1] += 0.3
        # 噪声
        c += 0.05 * torch.rand(N)
        return c.clamp(min=0.0)

    def __getitem__(self, idx):
        H, W = self.H, self.W
        color = torch.rand(3, H, W)
        # bbox 随机但确保面积
        x1 = random.uniform(0, W * 0.2)
        y1 = random.uniform(0, H * 0.2)
        x2 = random.uniform(W * 0.5, W * 0.9)
        y2 = random.uniform(H * 0.5, H * 0.9)
        bbox = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)

        keypoints_3d = self._make_keypoints()  # [21,3]
        # 随机左右手
        is_right = torch.tensor(1 if random.random() > 0.5 else 0, dtype=torch.long)

        # 有序标签 (j_left, j_right), 且 j_left != j_right
        j_left = random.randint(0, 20)
        j_right = random.randint(0, 20)
        while j_right == j_left:
            j_right = random.randint(0, 20)

        contact = self._make_contact_from_pair(j_left, j_right)  # [21]

        sample = {
            "color": color,                         # [3,H,W]
            "bbox": bbox,                           # [4]
            "keypoints_3d": keypoints_3d,           # [21,3]
            "contact": contact,                     # [21]
            "is_right": is_right,                   # []
            "j_left": torch.tensor(j_left),         # []
            "j_right": torch.tensor(j_right),       # []
        }
        return sample

class Hand2GripperDataset(Dataset):
    def __init__(self, raw_samples_root_dir: str) -> None:
        super().__init__()
        self.raw_samples_root_dir = raw_samples_root_dir
        self.full_sample_paths = []
        for samples_id in os.listdir(raw_samples_root_dir):  # for each video
            data_manager = DataManager(samples_id)
            color_image_dir = data_manager.labeling_app_config.labeling_app_color_image_dir
            depth_npy_dir = data_manager.labeling_app_config.labeling_app_depth_npy_dir
            labeling_app_results_dir = data_manager.labeling_app_config.labeling_app_results_dir
            color_image_files = sorted(os.listdir(color_image_dir))
            depth_npy_files = sorted(os.listdir(depth_npy_dir))
            labeling_app_results_files = sorted(os.listdir(labeling_app_results_dir))
            assert len(color_image_files) == len(depth_npy_files) == len(labeling_app_results_files), "Number of color images, depth npy files, and labeling app results must be the same"
            for color_image_file, depth_npy_file, labeling_app_results_file in zip(color_image_files, depth_npy_files, labeling_app_results_files):
                sample_path = {
                    'color_image_path': os.path.join(color_image_dir, color_image_file),
                    'depth_npy_path': os.path.join(depth_npy_dir, depth_npy_file),
                    'labeling_app_results_path': os.path.join(labeling_app_results_dir, labeling_app_results_file),
                }
                self.full_sample_paths.append(sample_path)
        print(f"Loaded {len(self.full_sample_paths)} samples")

    def __len__(self):
        return len(self.full_sample_paths)
    
    def __getitem__(self, idx):
        sample_path = self.full_sample_paths[idx]
        labeling_app_results = np.load(sample_path['labeling_app_results_path'])
        color_image = cv2.imread(sample_path['color_image_path'])
        depth_npy = np.load(sample_path['depth_npy_path'])
        # print(labeling_app_results['is_right_hand'], labeling_app_results['gripper_left_joint_id'], labeling_app_results['gripper_right_joint_id'])
        sample_item = {
            'color': torch.from_numpy(color_image).permute(2, 0, 1).float() / 255.0,
            'depth': torch.from_numpy(depth_npy).float(),
            'bbox': torch.from_numpy(labeling_app_results['bbox']).int(),
            'keypoints_3d': torch.from_numpy(labeling_app_results['joints']).float(),
            'contact': torch.from_numpy(labeling_app_results['contact_joint_out']).float(),
            'is_right': torch.from_numpy(labeling_app_results['is_right_hand']).long(),
            'j_left': torch.from_numpy(labeling_app_results['gripper_left_joint_id']).long(),
            'j_right': torch.from_numpy(labeling_app_results['gripper_right_joint_id']).long(),
        }

        return sample_item

# ------------------------------
# 损失函数与指标
# ------------------------------
@dataclass
class LossWeights:
    ce_pair: float = 1.0
    mutex: float = 0.1
    dist: float = 0.2
    contact_align: float = 0.2


def distance_prior_loss(prob_left, prob_right, kp_xyz_norm, d_min=0.2, d_max=3.5):
    """
    基于期望爪距的先验约束：
    E[d] = Σ_i Σ_j P_left(i) P_right(j) ||p_i - p_j||
    超出 [d_min, d_max] 的部分用线性 ReLU 惩罚
    """
    # pairwise 距离 [B,21,21]
    dist = torch.cdist(kp_xyz_norm, kp_xyz_norm, p=2)  # [B,21,21]
    # 期望距离
    E = (prob_left.unsqueeze(2) * prob_right.unsqueeze(1) * dist).sum(dim=(1, 2))  # [B]
    loss = F.relu(d_min - E) + F.relu(E - d_max)
    return loss.mean()


def contact_align_loss(prob, contact, eps=1e-6):
    """
    让 prob 与 contact 分布一致（soft-target CE）
    contact 正则化为分布；若全0，则退化为均匀分布
    """
    tgt = contact.clamp(min=0.0)
    s = tgt.sum(dim=1, keepdim=True)  # [B,1]
    tgt = torch.where(s > 0, tgt / (s + eps), torch.full_like(tgt, 1.0 / tgt.shape[1]))
    # KL(tgt || prob) == - sum tgt * log(prob) + const
    loss = -(tgt * (prob + eps).log()).sum(dim=1).mean()
    return loss


def compute_losses(out, gt_left, gt_right, kp_xyz_norm, contact, lw: LossWeights):
    """
    out: model outputs dict
    gt_left/right: [B]
    kp_xyz_norm: [B,21,3] (已规范化)
    contact: [B,21]
    """
    logits_left = out["logits_left"]
    logits_right = out["logits_right"]
    S_pair = out["S_pair"]

    # 有序单指 CE
    ce_left = F.cross_entropy(logits_left, gt_left)
    ce_right = F.cross_entropy(logits_right, gt_right)

    # 成对相容性作为 441 类分类（左=i, 右=j）
    B, N, _ = S_pair.shape
    S_flat = S_pair.reshape(B, -1)                   # [B, N*N]
    idx_pos = gt_left * N + gt_right              # [B]
    ce_pair = F.cross_entropy(S_flat, idx_pos)

    # 互斥：鼓励左右不选同一关节（分布层面）
    prob_left = F.softmax(logits_left, dim=-1)
    prob_right = F.softmax(logits_right, dim=-1)
    mutex = (prob_left * prob_right).sum(dim=-1).mean()

    # 距离先验（可选）
    dist_loss = distance_prior_loss(prob_left, prob_right, kp_xyz_norm)

    # 接触一致（可选，注意真实训练可做随机遮蔽增强）
    contact_loss_left = contact_align_loss(prob_left, contact)
    contact_loss_right = contact_align_loss(prob_right, contact)
    contact_loss = 0.5 * (contact_loss_left + contact_loss_right)

    loss = ce_left + ce_right \
         + lw.ce_pair * ce_pair \
         + lw.mutex * mutex \
         + lw.dist * dist_loss \
         + lw.contact_align * contact_loss

    loss_dict = {
        "loss": loss.item(),
        "ce_left": ce_left.item(),
        "ce_right": ce_right.item(),
        "ce_pair": ce_pair.item(),
        "mutex": mutex.item(),
        "dist": dist_loss.item(),
        "contact": contact_loss.item(),
    }
    return loss, loss_dict


@torch.no_grad()
def eval_metrics(out, gt_left, gt_right):
    """
    计算有序准确率：
    - left_acc: argmax_left == gt_left
    - right_acc: argmax_right == gt_right
    - pair_acc: (pred_pair == (gt_left, gt_right)) 有序命中
    """
    logits_left = out["logits_left"]
    logits_right = out["logits_right"]
    pred_pair = out["pred_pair"]  # [B,2]

    pred_left = logits_left.argmax(dim=-1)
    pred_right = logits_right.argmax(dim=-1)

    left_acc = (pred_left == gt_left).float().mean().item()
    right_acc = (pred_right == gt_right).float().mean().item()
    pair_acc = ((pred_pair[:, 0] == gt_left) & (pred_pair[:, 1] == gt_right)).float().mean().item()
    return {"left_acc": left_acc, "right_acc": right_acc, "pair_acc": pair_acc}


# ------------------------------
# 训练主函数
# ------------------------------
def main():
    # 配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64
    epochs = 3
    lr = 3e-4
    train_ratio = 0.8
    print(f"Device: {device}")

    # 数据
    # train_ds = ToyHand2GripperDataset(length=2000)
    # val_ds = ToyHand2GripperDataset(length=400, seed=999)
    # train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    # val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    # breakpoint()
    full_dataset = Hand2GripperDataset("/home/yutian/projs/Hand2Gripper/hand2gripper/raw")
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 模型/优化器
    model = Hand2GripperModel().to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    lw = LossWeights()

    for ep in range(1, epochs + 1):
        model.train()
        meter = {"loss": 0.0}
        for k in ["ce_left", "ce_right", "ce_pair", "mutex", "dist", "contact"]:
            meter[k] = 0.0

        for step, batch in enumerate(train_loader, 1):
            color = batch["color"].to(device)                # [B,3,H,W]
            bbox = batch["bbox"].to(device)                  # [B,4]
            keypoints_3d = batch["keypoints_3d"].to(device)  # [B,21,3]
            contact = batch["contact"].to(device)            # [B,21]
            is_right = batch["is_right"].to(device)          # [B]
            gt_left = batch["j_left"].to(device)             # [B]
            gt_right = batch["j_right"].to(device)           # [B]

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                out = model(color, bbox, keypoints_3d, contact, is_right)
                # 注意：模型内部会对 keypoints 做居中+尺度归一；这里为了距离先验，要得到同样的规范化版本
                # 复用 forward 的逻辑：简单起见，外部再规范化一次（与模型内部一致）
                kp = keypoints_3d.clone()
                wrist = kp[:, 0:1, :]
                kp = kp - wrist
                scale = kp.norm(dim=-1).mean(dim=1, keepdim=True).clamp(min=1e-6)
                kp_norm = kp / scale.unsqueeze(-1)

                loss, loss_items = compute_losses(out, gt_left, gt_right, kp_norm, contact, lw)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            for k, v in loss_items.items():
                meter[k] += v

        # 训练期均值
        n_batch = len(train_loader)
        train_log = {k: v / n_batch for k, v in meter.items()}

        # 验证
        model.eval()
        eval_meter = {"left_acc": 0.0, "right_acc": 0.0, "pair_acc": 0.0}
        with torch.no_grad():
            for batch in val_loader:
                color = batch["color"].to(device)
                bbox = batch["bbox"].to(device)
                keypoints_3d = batch["keypoints_3d"].to(device)
                contact = batch["contact"].to(device)
                is_right = batch["is_right"].to(device)
                gt_left = batch["j_left"].to(device)
                gt_right = batch["j_right"].to(device)

                out = model(color, bbox, keypoints_3d, contact, is_right)
                m = eval_metrics(out, gt_left, gt_right)
                for k in eval_meter:
                    eval_meter[k] += m[k]

        n_val = len(val_loader)
        eval_log = {k: v / n_val for k, v in eval_meter.items()}

        print(f"[Epoch {ep}] "
              f"loss={train_log['loss']:.4f} "
              f"CE(L/R,P)={train_log['ce_left']:.3f}/{train_log['ce_right']:.3f}/{train_log['ce_pair']:.3f} "
              f"mutex={train_log['mutex']:.3f} dist={train_log['dist']:.3f} contact={train_log['contact']:.3f} | "
              f"val_acc(L/R/pair)={eval_log['left_acc']:.3f}/{eval_log['right_acc']:.3f}/{eval_log['pair_acc']:.3f}")

    # 可选：保存权重
    torch.save(model.state_dict(), "hand2gripper_ckpt.pt")
    print("Saved to hand2gripper_ckpt.pt")


if __name__ == "__main__":
    main()
