# model.py
# -*- coding: utf-8 -*-
"""
Hand-to-Gripper (2-finger) Mapping Model (Ordered Left/Right)

输入:
- color:        [B, 3, H, W]   原始彩色图 (float, 0~1)
- bbox:         [B, 4]         [x1, y1, x2, y2]，像素坐标
- keypoints_3d: [B, 21, 3]     人手3D关键点 (单位任意, 相对比例会被归一化)
- contact:      [B, 21]        每个关节的接触强度/概率 [0~1]
- is_right:     [B]            1=右手, 0=左手 (不镜像，仅作为特征输入)

输出(前向):
- logits_left:  [B, 21]        夹爪左指选择各关节的 logits（未归一化）
- logits_right: [B, 21]        夹爪右指选择各关节的 logits（未归一化）
- S_pair:       [B, 21, 21]    成对选择（左=i, 右=j）的相容性分数（有方向）
- pred_pair:    [B, 2]         (i_left*, j_right*)，按 comb 最大选出的有序关节对
- img_emb:      [B, D]         图像全局嵌入（可选可视化/检索）
- node_emb:     [B, 21, D]     节点编码后的表征（可复用/可解释）
"""
from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------
# 小型视觉骨干：TinyCNN -> 全局embedding
# ------------------------------
class TinyCNN(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        ch = [3, 32, 64, 128, 256]
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch[0], ch[1], 3, 2, 1), nn.BatchNorm2d(ch[1]), nn.ReLU(inplace=True),
            nn.Conv2d(ch[1], ch[1], 3, 1, 1), nn.BatchNorm2d(ch[1]), nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch[1], ch[2], 3, 2, 1), nn.BatchNorm2d(ch[2]), nn.ReLU(inplace=True),
            nn.Conv2d(ch[2], ch[2], 3, 1, 1), nn.BatchNorm2d(ch[2]), nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(ch[2], ch[3], 3, 2, 1), nn.BatchNorm2d(ch[3]), nn.ReLU(inplace=True),
            nn.Conv2d(ch[3], ch[3], 3, 1, 1), nn.BatchNorm2d(ch[3]), nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(ch[3], ch[4], 3, 2, 1), nn.BatchNorm2d(ch[4]), nn.ReLU(inplace=True),
            nn.Conv2d(ch[4], ch[4], 3, 1, 1), nn.BatchNorm2d(ch[4]), nn.ReLU(inplace=True),
        )
        self.proj = nn.Linear(ch[4], out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,3,256,256]
        return: [B, D]
        """
        x = self.conv1(x)  # B,32,128,128
        x = self.conv2(x)  # B,64,64,64
        x = self.conv3(x)  # B,128,32,32
        x = self.conv4(x)  # B,256,16,16
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)  # B,256
        x = self.proj(x)  # B,D
        return x


# ------------------------------
# 节点特征编码 + Transformer 编码
# ------------------------------
class HandNodeEncoder(nn.Module):
    """
    节点输入 = [xyz(3) | contact(1) | onehot_joint(21) | is_right(1)] 共 26 维
    """
    def __init__(self, in_dim: int = 26, hidden: int = 256, n_layers: int = 2, out_dim: int = 256):
        super().__init__()
        mlp = []
        dim = in_dim
        for _ in range(n_layers):
            mlp += [nn.Linear(dim, hidden), nn.ReLU(inplace=True)]
            dim = hidden
        mlp += [nn.Linear(hidden, out_dim)]
        self.mlp = nn.Sequential(*mlp)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_dim, nhead=8, dim_feedforward=out_dim * 2, batch_first=True, dropout=0.1, activation="gelu"
        )
        self.tr = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # FiLM：用图像 embedding 调制节点通道
        self.film_gamma = nn.Linear(out_dim, out_dim)
        self.film_beta = nn.Linear(out_dim, out_dim)

    def forward(self, node_feats: torch.Tensor, img_emb: torch.Tensor) -> torch.Tensor:
        """
        node_feats: [B,21,26]
        img_emb:    [B,D]
        return:     [B,21,D]
        """
        H = self.mlp(node_feats)  # [B,21,D]
        gamma = self.film_gamma(img_emb).unsqueeze(1)  # [B,1,D]
        beta = self.film_beta(img_emb).unsqueeze(1)    # [B,1,D]
        H = gamma * H + beta
        H = self.tr(H)  # [B,21,D]
        return H


# ------------------------------
# 成对选择解码器：有序（左/右）双查询 + 成对相容性
# ------------------------------
class PairDecoder(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        # 左/右 两个可学习的查询向量
        self.q_left = nn.Parameter(torch.randn(d_model))
        self.q_right = nn.Parameter(torch.randn(d_model))
        nn.init.normal_(self.q_left, mean=0.0, std=0.02)
        nn.init.normal_(self.q_right, mean=0.0, std=0.02)

        # 成对相容性：双线性形式 H_i^T W H_j
        self.W = nn.Parameter(torch.empty(d_model, d_model))
        nn.init.xavier_uniform_(self.W)

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        H: [B,21,D]
        返回:
          logits_left:  [B,21]
          logits_right: [B,21]
          S_pair:       [B,21,21]  (左=i, 右=j)
          pred_pair:    [B,2] (i_left*, j_right*), i!=j
        """
        B, N, D = H.shape
        # 边际打分
        logits_left = torch.einsum('bnd,d->bn', H, self.q_left)    # [B,N]
        logits_right = torch.einsum('bnd,d->bn', H, self.q_right)  # [B,N]

        # 成对相容性：S = H W H^T
        S_pair = torch.einsum('bnd,de,bme->bnm', H, self.W, H)     # [B,N,N]

        # 联合打分 comb = S + 边际
        comb = S_pair + logits_left.unsqueeze(2) + logits_right.unsqueeze(1)  # [B,N,N]
        # 禁止同一节点成对
        eye = torch.eye(N, device=H.device, dtype=torch.bool).unsqueeze(0)
        comb = comb.masked_fill(eye, float('-inf'))

        # 取全局最大 (i_left*, j_right*)
        comb_flat = comb.view(B, -1)  # [B, N*N]
        idx = torch.argmax(comb_flat, dim=1)  # [B]
        i_star = idx // N
        j_star = idx % N
        pred_pair = torch.stack([i_star, j_star], dim=1)  # [B,2]
        return logits_left, logits_right, S_pair, pred_pair


# ------------------------------
# 顶层模型
# ------------------------------
class Hand2GripperModel(nn.Module):
    def __init__(self, d_model: int = 256, img_size: int = 256):
        super().__init__()
        self.img_size = img_size
        self.backbone = TinyCNN(out_dim=d_model)
        self.encoder = HandNodeEncoder(in_dim=26, hidden=d_model, n_layers=2, out_dim=d_model)
        self.decoder = PairDecoder(d_model=d_model)

    @staticmethod
    def _normalize_keypoints_xyz(kp3d: torch.Tensor, is_right: torch.Tensor) -> torch.Tensor:
        """
        不做左右镜像；只做 wrist(0) 居中 + 全局尺度归一
        kp3d: [B,21,3], is_right: [B]
        return: [B,21,3]
        """
        kp = kp3d.clone()
        wrist = kp[:, 0:1, :]
        kp = kp - wrist
        dist = torch.norm(kp, dim=-1)
        scale = dist.mean(dim=1, keepdim=True).clamp(min=1e-6)
        kp = kp / scale.unsqueeze(-1)
        return kp

    @staticmethod
    def _build_node_features(kp_xyz_norm: torch.Tensor, contact: torch.Tensor, is_right: torch.Tensor) -> torch.Tensor:
        """
        组装节点特征: [xyz | contact | onehot_joint | is_right]
        kp_xyz_norm: [B,21,3]
        contact:     [B,21]
        is_right:    [B]
        return:      [B,21,26]
        """
        B, N, _ = kp_xyz_norm.shape
        onehot = torch.eye(N, device=kp_xyz_norm.device).unsqueeze(0).repeat(B, 1, 1)  # [B,21,21]
        contact_f = contact.unsqueeze(-1)  # [B,21,1]
        isr = is_right.view(B, 1, 1).repeat(1, N, 1).float()  # [B,21,1]
        feats = torch.cat([kp_xyz_norm, contact_f, onehot, isr], dim=-1)  # [B,21,26]
        return feats

    @staticmethod
    def _expand_bbox(bbox: torch.Tensor, H: int, W: int, scale: float = 1.2) -> torch.Tensor:
        """
        扩张bbox并裁剪到图像范围
        bbox: [B,4] (x1,y1,x2,y2)
        """
        x1, y1, x2, y2 = bbox.unbind(dim=1)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = (x2 - x1).clamp(min=1.0)
        h = (y2 - y1).clamp(min=1.0)
        w2 = w * scale / 2.0
        h2 = h * scale / 2.0
        nx1 = (cx - w2).clamp(min=0.0, max=W - 1.0)
        ny1 = (cy - h2).clamp(min=0.0, max=H - 1.0)
        nx2 = (cx + w2).clamp(min=0.0, max=W - 1.0)
        ny2 = (cy + h2).clamp(min=0.0, max=H - 1.0)
        return torch.stack([nx1, ny1, nx2, ny2], dim=1)

    def _crop_and_resize(self, color: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
        """
        color: [B,3,H,W], bbox:[B,4] 像素坐标
        返回: 手部裁剪并缩放至[self.img_size, self.img_size]
        """
        B, C, H, W = color.shape
        bbox = self._expand_bbox(bbox, H, W, scale=1.2)
        crops = []
        for b in range(B):
            x1, y1, x2, y2 = bbox[b]
            x1i = int(torch.floor(x1).item())
            y1i = int(torch.floor(y1).item())
            x2i = int(torch.ceil(x2).item())
            y2i = int(torch.ceil(y2).item())
            x2i = max(x2i, x1i + 1)
            y2i = max(y2i, y1i + 1)
            patch = color[b:b+1, :, y1i:y2i, x1i:x2i]  # [1,3,h,w]
            patch = F.interpolate(patch, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
            crops.append(patch)
        crop_img = torch.cat(crops, dim=0)  # [B,3,S,S]
        return crop_img

    def forward(self, color: torch.Tensor, bbox: torch.Tensor, keypoints_3d: torch.Tensor,
                contact: torch.Tensor, is_right: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        执行一次前向:
        """
        # 1) 图像裁剪 + 编码
        img_crop = self._crop_and_resize(color, bbox)              # [B,3,S,S]
        img_emb = self.backbone(img_crop)                          # [B,D]

        # 2) 关键点居中 + 尺度归一（不镜像）
        kp_xyz_norm = self._normalize_keypoints_xyz(keypoints_3d, is_right)  # [B,21,3]

        # 3) 组装节点特征并编码
        node_feats = self._build_node_features(kp_xyz_norm, contact, is_right)  # [B,21,26]
        H = self.encoder(node_feats, img_emb)  # [B,21,D]

        # 4) 成对解码（左/右有序）
        logits_left, logits_right, S_pair, pred_pair = self.decoder(H)

        return {
            "logits_left": logits_left,
            "logits_right": logits_right,
            "S_pair": S_pair,
            "pred_pair": pred_pair,
            "img_emb": img_emb,
            "node_emb": H,
        }


# ------------------------------
# 演示: 随机数据跑一次 forward
# ------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B, H, W = 2, 480, 640
    model = Hand2GripperModel(d_model=256, img_size=256).to(device)
    model.eval()

    color = torch.rand(B, 3, H, W, device=device)
    bbox = torch.tensor([[120.0, 80.0, 320.0, 360.0],
                         [ 50.0, 60.0, 300.0, 420.0]], device=device)
    keypoints_3d = torch.randn(B, 21, 3, device=device) * 0.05
    keypoints_3d[:, 0, :] = 0.0  # 腕部近零
    contact = torch.rand(B, 21, device=device)
    is_right = torch.tensor([1, 0], device=device)

    with torch.no_grad():
        out = model(color, bbox, keypoints_3d, contact, is_right)

    print("pred_pair:", out["pred_pair"])
    print("logits_left shape:", out["logits_left"].shape)
    print("logits_right shape:", out["logits_right"].shape)
    print("S_pair shape:", out["S_pair"].shape)
