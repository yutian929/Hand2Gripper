# train.py
# -*- coding: utf-8 -*-
"""
三元输出 (base, left, right) 的训练脚本。
- 支持两种数据源：
  1) 真实数据: Hand2GripperDataset(需 processor_config.DataManager)
  2) 玩具数据: ToyTripleDataset (--use_toy 开关)
- 模型: from models.model import Hand2GripperModel  (你的三元输出版 model.py)
- 损失:
  * 三个有序CE (base/left/right)
  * 三元联合分类 (基于联合打分 comb 的 21^3 类)
  * 接触一致 (soft-target)
  * 距离上界先验 (限制过远，允许闭合)

用法示例：
  python train.py --dataset_root /path/to/raw --epochs 10 --batch_size 64
  python train.py --use_toy --epochs 3
"""
import os
import random
import argparse
from dataclasses import dataclass

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split

from models.model import Hand2GripperModel


# ------------------------------
# 实用函数
# ------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------
# 早停和稳定性检测
# ------------------------------
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, metric='triple_acc'):
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.history = []
        
    def __call__(self, score):
        self.history.append(score)
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            
        return self.early_stop
    
    def is_stable(self, window_size=5, stability_threshold=0.001):
        """检查最近window_size个epoch的准确率是否稳定"""
        if len(self.history) < window_size:
            return False
        
        recent_scores = self.history[-window_size:]
        variance = np.var(recent_scores)
        return variance < stability_threshold


# ------------------------------
# 玩具数据集（兜底）
# ------------------------------
class ToyTripleDataset(Dataset):
    def __init__(self, length=2400, H=480, W=640, seed=123, p_nosol=0.1, p_lr_equal=0.2):
        super().__init__()
        self.length = length
        self.H, self.W = H, W
        self.p_nosol = p_nosol
        self.p_lr_equal = p_lr_equal
        random.seed(seed)
        torch.manual_seed(seed)

    def __len__(self):
        return self.length

    def _make_keypoints(self):
        kp = torch.randn(21, 3) * 0.05
        kp[0] = 0.0  # wrist near 0
        return kp

    def _make_contact_from_triplet(self, j_base, j_left, j_right, N=21):
        c = torch.zeros(N)
        for j in [j_base, j_left, j_right]:
            c[j] += 1.0
            if j - 1 >= 0: c[j - 1] += 0.3
            if j + 1 < N: c[j + 1] += 0.3
        c += 0.05 * torch.rand(N)
        return c.clamp(min=0.0)

    def __getitem__(self, idx):
        H, W = self.H, self.W
        color = torch.rand(3, H, W)
        # bbox
        x1 = random.uniform(0, W * 0.2)
        y1 = random.uniform(0, H * 0.2)
        x2 = random.uniform(W * 0.5, W * 0.9)
        y2 = random.uniform(H * 0.5, H * 0.9)
        bbox = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)

        keypoints_3d = self._make_keypoints()
        is_right = torch.tensor(1 if random.random() > 0.5 else 0, dtype=torch.long)

        # 生成 (base, left, right)
        if random.random() < self.p_nosol:
            j_base = j_left = j_right = 0  # [0,0,0] 无解/闭合占位
        else:
            j_base = random.randint(0, 20)
            if random.random() < self.p_lr_equal:
                j_left = j_right = random.randint(0, 20)
            else:
                j_left = random.randint(0, 20)
                j_right = random.randint(0, 20)

        contact = self._make_contact_from_triplet(j_base, j_left, j_right)

        return {
            "color": color,                           # [3,H,W]
            "bbox": bbox,                             # [4]
            "joints": keypoints_3d,             # [21,3]
            "contact_joint_out": contact,                       # [21]
            "is_right": is_right,                     # []
            "gripper_base_joint_id": torch.tensor(j_base),           # []
            "gripper_left_joint_id": torch.tensor(j_left),           # []
            "gripper_right_joint_id": torch.tensor(j_right),         # []
        }


# ------------------------------
# 真实数据集（参考你的实现）
# ------------------------------
class Hand2GripperDataset(Dataset):
    """
    假定 processor_config.DataManager 提供:
      - labeling_app_config.labeling_app_color_image_dir
      - labeling_app_config.labeling_app_depth_npy_dir
      - labeling_app_config.labeling_app_results_dir
    结果文件应包含 keys:
      - 'bbox' [4], 'joints' [21,3], 'contact_joint_out' [21],
        'is_right_hand' [], 'gripper_left_joint_id' [], 'gripper_right_joint_id' [], 'gripper_base_joint_id' []
    """
    def __init__(self, raw_samples_root_dir: str):
        super().__init__()
        from processor_config import DataManager  # 延迟导入, 以便 --use_toy 时不依赖
        self.full_sample_paths = []
        self.root = raw_samples_root_dir
        for samples_id in os.listdir(raw_samples_root_dir):  # each video/session
            dm = DataManager(samples_id)
            color_dir = dm.labeling_app_config.labeling_app_color_image_dir
            depth_dir = dm.labeling_app_config.labeling_app_depth_npy_dir
            res_dir = dm.labeling_app_config.labeling_app_results_dir
            color_files = sorted(os.listdir(color_dir))
            depth_files = sorted(os.listdir(depth_dir))
            res_files = sorted(os.listdir(res_dir))
            assert len(color_files) == len(depth_files) == len(res_files), \
                "color/depth/results 数量不一致"
            for cf, df, rf in zip(color_files, depth_files, res_files):
                self.full_sample_paths.append({
                    "color": os.path.join(color_dir, cf),
                    "depth": os.path.join(depth_dir, df),
                    "res": os.path.join(res_dir, rf),
                })
        print(f"[Hand2GripperDataset] Loaded {len(self.full_sample_paths)} samples from {raw_samples_root_dir}")

    def __len__(self):
        return len(self.full_sample_paths)

    def __getitem__(self, idx):
        p = self.full_sample_paths[idx]
        # 读标签 (npz)
        lab = np.load(p["res"])
        # 读图像
        color_bgr = cv2.imread(p["color"], cv2.IMREAD_COLOR)
        if color_bgr is None:
            raise FileNotFoundError(p["color"])
        color = torch.from_numpy(color_bgr[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0  # BGR->RGB
        # 读深度 (当前模型未用)
        _ = np.load(p["depth"])

        # tensors
        bbox = torch.from_numpy(lab["bbox"]).float()
        joints = torch.from_numpy(lab["joints"]).float()
        contact_joint_out = torch.from_numpy(lab["contact_joint_out"]).float()
        is_right = torch.as_tensor(lab["is_right_hand"]).long()
        gripper_left_joint_id = torch.as_tensor(lab["gripper_left_joint_id"]).long()
        gripper_right_joint_id = torch.as_tensor(lab["gripper_right_joint_id"]).long()
        gripper_base_joint_id = torch.as_tensor(lab["gripper_base_joint_id"]).long()

        return {
            "color": color,                 # [3,H,W]
            "bbox": bbox,                   # [4]
            "joints": joints,   # [21,3]
            "contact_joint_out": contact_joint_out,             # [21]
            "is_right": is_right,           # []
            "gripper_base_joint_id": gripper_base_joint_id,               # []
            "gripper_left_joint_id": gripper_left_joint_id,               # []
            "gripper_right_joint_id": gripper_right_joint_id,             # []
            "laebling_app_results_file_path": p["res"],
        }


# ------------------------------
# 损失 / 指标
# ------------------------------
@dataclass
class LossWeights:
    triple_ce: float = 1.0
    contact_align: float = 0.2
    dist_upper: float = 0.1  # 仅限制过远，允许闭合/相等


def build_comb_logits(out):
    """按模型三元联合打分构造 comb: [B,21,21,21]"""
    lb = out["logits_base"]
    ll = out["logits_left"]
    lr = out["logits_right"]
    S_bl = out["S_bl"]
    S_br = out["S_br"]
    S_lr = out["S_lr"]
    comb = (
        lb[:, :, None, None] +
        ll[:, None, :, None] +
        lr[:, None, None, :] +
        S_bl[:, :, :, None] +
        S_br[:, :, None, :] +
        S_lr[:, None, :, :]
    )
    return comb


def contact_align_loss_three(prob_base, prob_left, prob_right, contact, eps=1e-6):
    """把三路概率平均后与 contact 对齐（soft-target CE）。"""
    tgt = contact.clamp(min=0.0)
    s = tgt.sum(dim=1, keepdim=True)
    tgt = torch.where(s > 0, tgt / (s + eps), torch.full_like(tgt, 1.0 / tgt.shape[1]))
    avg_prob = (prob_base + prob_left + prob_right) / 3.0
    loss = -(tgt * (avg_prob + eps).log()).sum(dim=1).mean()
    return loss


def distance_upper_prior(prob_left, prob_right, kp_xyz_norm, d_max=3.5):
    """只限制过远抓取（允许闭合/相等）。"""
    dist = torch.cdist(kp_xyz_norm, kp_xyz_norm, p=2)  # [B,21,21]
    E = (prob_left.unsqueeze(2) * prob_right.unsqueeze(1) * dist).sum(dim=(1, 2))  # [B]
    return F.relu(E - d_max).mean()


def compute_losses(out, gt_base, gt_left, gt_right, kp_xyz_norm, contact, lw: LossWeights):
    # 三个有序CE
    ce_base = F.cross_entropy(out["logits_base"], gt_base)
    ce_left = F.cross_entropy(out["logits_left"], gt_left)
    ce_right = F.cross_entropy(out["logits_right"], gt_right)

    # 三元联合分类: 21^3 类
    comb = build_comb_logits(out)                    # [B,21,21,21]
    B, N, _, _ = comb.shape
    comb_flat = comb.view(B, -1)                     # [B, 9261]
    idx_pos = gt_base * (N * N) + gt_left * N + gt_right
    ce_triple = F.cross_entropy(comb_flat, idx_pos)

    # 接触一致
    pb = F.softmax(out["logits_base"], dim=-1)
    pl = F.softmax(out["logits_left"], dim=-1)
    pr = F.softmax(out["logits_right"], dim=-1)
    contact_loss = contact_align_loss_three(pb, pl, pr, contact)

    # 距离上界先验（仅限制过远）
    dist_loss = distance_upper_prior(pl, pr, kp_xyz_norm)

    loss = (ce_base + ce_left + ce_right) + lw.triple_ce * ce_triple \
           + lw.contact_align * contact_loss + lw.dist_upper * dist_loss

    return loss, {
        "loss": loss.item(),
        "ce_base": ce_base.item(),
        "ce_left": ce_left.item(),
        "ce_right": ce_right.item(),
        "ce_triple": ce_triple.item(),
        "contact": contact_loss.item(),
        "dist_upper": dist_loss.item(),
    }


@torch.no_grad()
def eval_metrics(out, gt_base, gt_left, gt_right):
    """有序 Top-1 与三元联合准确率。"""
    pb = out["logits_base"].argmax(dim=-1)
    pl = out["logits_left"].argmax(dim=-1)
    pr = out["logits_right"].argmax(dim=-1)
    triple = out["pred_triple"]  # [B,3]

    base_acc = (pb == gt_base).float().mean().item()
    left_acc = (pl == gt_left).float().mean().item()
    right_acc = (pr == gt_right).float().mean().item()
    triple_acc = ((triple[:, 0] == gt_base) & (triple[:, 1] == gt_left) & (triple[:, 2] == gt_right)).float().mean().item()
    return {"base_acc": base_acc, "left_acc": left_acc, "right_acc": right_acc, "triple_acc": triple_acc}


# ------------------------------
# 训练主函数
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="", help="真实数据根目录（含各 samples_id 子目录）")
    parser.add_argument("--use_toy", action="store_true", help="使用玩具数据集（忽略 dataset_root）")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数，None表示自动训练直到收敛")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default="hand2gripper.pt")
    parser.add_argument("--patience", type=int, default=15, help="早停耐心值")
    parser.add_argument("--stability_window", type=int, default=10, help="稳定性检测窗口大小")
    parser.add_argument("--stability_threshold", type=float, default=0.005, help="稳定性阈值")
    parser.add_argument("--resume", type=str, default="", help="从预训练模型继续训练（模型文件路径）")
    parser.add_argument("--resume_optimizer", action="store_true", help="同时加载优化器状态（需要保存的优化器状态文件）")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 数据
    if args.use_toy or not args.dataset_root:
        train_ds = ToyTripleDataset(length=2400)
        val_ds = ToyTripleDataset(length=480, seed=999)
        print("Using toy dataset")
    else:
        full_ds = Hand2GripperDataset(args.dataset_root)
        total = len(full_ds)
        tsize = int(total * args.train_ratio)
        vsize = total - tsize
        train_ds, val_ds = random_split(full_ds, [tsize, vsize])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # 模型/优化器
    model = Hand2GripperModel(d_model=256, img_size=256).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    lw = LossWeights()
    
    # 加载预训练模型
    start_epoch = 0
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Loading pretrained model from {args.resume}")
            model.load_state_dict(torch.load(args.resume, map_location=device))
            print("Model loaded successfully")
            
            # 如果指定了加载优化器状态
            if args.resume_optimizer:
                optimizer_path = args.resume.replace('.pt', '_optimizer.pt')
                if os.path.exists(optimizer_path):
                    print(f"Loading optimizer state from {optimizer_path}")
                    opt.load_state_dict(torch.load(optimizer_path, map_location=device))
                    print("Optimizer state loaded successfully")
                else:
                    print(f"Warning: Optimizer state file not found: {optimizer_path}")
        else:
            print(f"Warning: Pretrained model file not found: {args.resume}")
            print("Starting training from scratch")
    
    # 早停机制
    early_stopping = EarlyStopping(
        patience=args.patience, 
        min_delta=0.001, 
        metric='triple_acc'
    )

    best_val = -1.0
    ep = 0
    
    # 训练循环
    while True:
        ep += 1
        
        # 如果指定了epochs，检查是否达到
        if args.epochs is not None and ep > args.epochs:
            print(f"Reached specified epochs: {args.epochs}")
            break
        model.train()
        meter = {k: 0.0 for k in ["loss", "ce_base", "ce_left", "ce_right", "ce_triple", "contact", "dist_upper"]}
        for step, batch in enumerate(train_loader, 1):
            color = batch["color"].to(device)                # [B,3,H,W]
            bbox = batch["bbox"].to(device).float()          # [B,4]
            joints = batch["joints"].to(device)  # [B,21,3]
            contact_joint_out = batch["contact_joint_out"].to(device)            # [B,21]
            is_right = batch["is_right"].to(device)          # [B] or [B,1]
            gt_base = batch["gripper_base_joint_id"].to(device).view(-1)    # [B]
            gt_left = batch["gripper_left_joint_id"].to(device).view(-1)    # [B]
            gt_right = batch["gripper_right_joint_id"].to(device).view(-1)  # [B]

            # print(keypoints_3d.shape, keypoints_3d[:,0,:])

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                out = model(color, bbox, joints, contact_joint_out, is_right.view(-1))
                # 关键点规范化（与模型内部一致）用于距离先验
                kp = joints.clone()
                wrist = kp[:, 0:1, :]
                kp = kp - wrist
                scale = kp.norm(dim=-1).mean(dim=1, keepdim=True).clamp(min=1e-6)
                kp_norm = kp / scale.unsqueeze(-1)

                loss, loss_items = compute_losses(out, gt_base, gt_left, gt_right, kp_norm, contact_joint_out, lw)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            for k, v in loss_items.items():
                meter[k] += v

        n_batch = len(train_loader)
        train_log = {k: v / n_batch for k, v in meter.items()}

        # 验证
        model.eval()
        eval_meter = {"base_acc": 0.0, "left_acc": 0.0, "right_acc": 0.0, "triple_acc": 0.0}
        with torch.no_grad():
            for batch in val_loader:
                color = batch["color"].to(device)
                bbox = batch["bbox"].to(device).float()
                joints = batch["joints"].to(device)
                contact_joint_out = batch["contact_joint_out"].to(device)
                is_right = batch["is_right"].to(device)
                gt_base = batch["gripper_base_joint_id"].to(device).view(-1)
                gt_left = batch["gripper_left_joint_id"].to(device).view(-1)
                gt_right = batch["gripper_right_joint_id"].to(device).view(-1)

                out = model(color, bbox, joints, contact_joint_out, is_right.view(-1))
                m = eval_metrics(out, gt_base, gt_left, gt_right)
                for k in eval_meter:
                    eval_meter[k] += m[k]

        n_val = len(val_loader)
        eval_log = {k: v / n_val for k, v in eval_meter.items()}

        print(f"[Epoch {ep}] "
              f"loss={train_log['loss']:.4f} "
              f"CE(B/L/R/T)={train_log['ce_base']:.3f}/{train_log['ce_left']:.3f}/{train_log['ce_right']:.3f}/{train_log['ce_triple']:.3f} "
              f"contact={train_log['contact']:.3f} dist_upper={train_log['dist_upper']:.3f} | "
              f"val_acc(B/L/R/T)={eval_log['base_acc']:.3f}/{eval_log['left_acc']:.3f}/{eval_log['right_acc']:.3f}/{eval_log['triple_acc']:.3f}")

        # 保存最好（三元联合准确率）
        if eval_log["triple_acc"] > best_val:
            best_val = eval_log["triple_acc"]
            torch.save(model.state_dict(), args.save)
            # 如果启用了优化器状态保存，同时保存优化器状态
            if args.resume_optimizer:
                optimizer_save_path = args.save.replace('.pt', '_optimizer.pt')
                torch.save(opt.state_dict(), optimizer_save_path)
                print(f"  >> Saved best to {args.save} and optimizer to {optimizer_save_path} (triple_acc={best_val:.3f})")
            else:
                print(f"  >> Saved best to {args.save} (triple_acc={best_val:.3f})")

        # 早停检查
        if early_stopping(eval_log["triple_acc"]):
            print(f"Early stopping at epoch {ep} (patience={args.patience})")
            break
            
        # 稳定性检查（仅在未指定epochs时）
        if args.epochs is None and ep >= args.stability_window:
            if early_stopping.is_stable(args.stability_window, args.stability_threshold):
                print(f"Training converged at epoch {ep} (accuracy stabilized)")
                break

    print("Done.")


if __name__ == "__main__":
    main()
