"""
羊种特化ReID模型训练脚本（完整修复版）
修复重点：Triplet Loss维度对齐 + 数据加载稳定性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
from collections import Counter
import random


# ==================== 数据集类 ====================

class SheepReIDDataset(Dataset):
    """羊只ReID数据集 - 增强稳定性版本"""

    def __init__(self, data_root, split='train', transform=None):
        self.data_root = Path(data_root) / split
        self.transform = transform or self._default_transform(split)

        # 扫描数据集
        self.images = []
        self.labels = []
        self.label_to_name = {}
        self.name_to_label = {}

        # 确保数据集路径存在
        if not self.data_root.exists():
            raise ValueError(f"数据集路径不存在: {self.data_root}")

        # 获取所有羊只目录
        sheep_dirs = [d for d in sorted(self.data_root.iterdir()) if d.is_dir()]

        if len(sheep_dirs) == 0:
            raise ValueError(f"在 {self.data_root} 中未找到羊只目录")

        for label_idx, sheep_dir in enumerate(sheep_dirs):
            sheep_name = sheep_dir.name
            self.label_to_name[label_idx] = sheep_name
            self.name_to_label[sheep_name] = label_idx

            # 获取所有jpg图像
            image_paths = list(sheep_dir.glob('*.jpg')) + list(sheep_dir.glob('*.png'))

            if len(image_paths) == 0:
                print(f"⚠️ 警告: 羊只目录 {sheep_name} 中没有找到图像文件")
                continue

            for img_path in image_paths:
                self.images.append(str(img_path))
                self.labels.append(label_idx)

        if len(self.images) == 0:
            raise ValueError(f"在 {self.data_root} 中未找到任何图像文件")

        self.num_classes = len(self.label_to_name)

        print(f"✅ 加载{split}数据集:")
        print(f"   - 羊只数量: {self.num_classes}")
        print(f"   - 图像数量: {len(self.images)}")

        # 打印类别分布
        self._print_class_distribution()

    def _default_transform(self, split):
        """根据训练/验证设置不同的数据增强策略[1,3](@ref)"""
        if split == 'train':
            return T.Compose([
                T.Resize((256, 256)),  # 统一尺寸
                T.RandomCrop((224, 224)),  # 随机裁剪
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return T.Compose([
                T.Resize((224, 224)),  # 固定尺寸用于验证
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def _print_class_distribution(self):
        """打印类别分布信息"""
        label_counts = Counter(self.labels)
        print(f"   - 类别分布: ")
        for label, count in label_counts.items():
            sheep_name = self.label_to_name[label]
            print(f"     {sheep_name}: {count}张图像")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            img_path = self.images[idx]
            label = self.labels[idx]

            image = Image.open(img_path).convert('RGB')

            if self.transform:
                image = self.transform(image)

            return image, label

        except Exception as e:
            print(f"❌ 加载图像失败: {img_path}, 错误: {e}")
            # 返回一个随机图像作为备用
            dummy_image = torch.randn(3, 224, 224)
            return dummy_image, 0


# ==================== 修复版Triplet Loss ====================

class FixedTripletLoss(nn.Module):
    """修复版Triplet Loss - 解决维度对齐问题[7,8](@ref)"""

    def __init__(self, margin=0.3, hard_mining=True):
        super().__init__()
        self.margin = margin
        self.hard_mining = hard_mining
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, embeddings, labels):
        """
        修复版本：确保正负样本对数量一致[6](@ref)
        """
        # 计算所有样本对之间的欧式距离矩阵
        dist_mat = self._compute_distance_matrix(embeddings)

        if self.hard_mining:
            loss, num_valid = self._hard_example_mining_fixed(dist_mat, labels)
        else:
            loss, num_valid = self._batch_all_triplet_loss(dist_mat, labels)

        return loss, num_valid

    def _compute_distance_matrix(self, embeddings):
        """计算距离矩阵"""
        # L2归一化
        embeddings = F.normalize(embeddings, p=2, dim=1)
        # 欧式距离矩阵
        dist_mat = torch.cdist(embeddings, embeddings, p=2)
        return dist_mat

    def _hard_example_mining_fixed(self, dist_mat, labels):
        """修复版：硬样本挖掘，确保维度对齐[7](@ref)"""
        n = dist_mat.size(0)

        if n < 2:
            return torch.tensor(0.0, device=dist_mat.device), 0

        # 构建mask
        labels_expanded = labels.unsqueeze(0).expand(n, n)
        is_pos = labels_expanded == labels_expanded.t()
        is_neg = labels_expanded != labels_expanded.t()

        # 对角线设为False（排除自身比较）
        eye_mask = torch.eye(n, dtype=torch.bool, device=dist_mat.device)
        is_pos = is_pos & ~eye_mask

        dist_ap = torch.zeros(n, device=dist_mat.device)
        dist_an = torch.zeros(n, device=dist_mat.device)

        valid_count = 0

        for i in range(n):
            pos_mask = is_pos[i]
            neg_mask = is_neg[i]

            # 检查是否有有效的正样本和负样本
            if pos_mask.any() and neg_mask.any():
                # 最难正样本（距离最大）
                dist_ap[valid_count] = dist_mat[i][pos_mask].max()
                # 最难负样本（距离最小）
                dist_an[valid_count] = dist_mat[i][neg_mask].min()
                valid_count += 1

        if valid_count == 0:
            return torch.tensor(0.0, device=dist_mat.device), 0

        # 🔧 修复关键：只使用有效的样本
        dist_ap = dist_ap[:valid_count]
        dist_an = dist_an[:valid_count]

        # 计算triplet loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        num_valid = valid_count

        return loss, num_valid

    def _batch_all_triplet_loss(self, dist_mat, labels):
        """备选方案：批量所有有效三元组[6](@ref)"""
        n = dist_mat.size(0)
        losses = []

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if labels[i] == labels[j]:  # 正样本对
                    for k in range(n):
                        if labels[i] != labels[k]:  # 负样本对
                            loss_val = F.relu(dist_mat[i, j] - dist_mat[i, k] + self.margin)
                            if loss_val > 0:
                                losses.append(loss_val)

        if len(losses) == 0:
            return torch.tensor(0.0, device=dist_mat.device), 0

        losses = torch.stack(losses)
        return losses.mean(), len(losses)


# ==================== 简化版ReID模型 ====================

class LightweightReIDModel(nn.Module):
    """轻量级ReID模型"""

    def __init__(self, embedding_dim=128, num_classes=None):
        super().__init__()

        # 使用预训练的ResNet18作为backbone
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

        # 移除最后的全连接层
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # 自定义分类头
        self.embedding = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim)
        )

        if num_classes:
            self.classifier = nn.Linear(embedding_dim, num_classes)
        else:
            self.classifier = None

    def forward(self, x, return_feature=False):
        # 特征提取
        features = self.backbone(x)
        features = features.view(features.size(0), -1)

        # 生成embedding
        embeddings = self.embedding(features)
        embeddings = F.normalize(embeddings, p=2, dim=1)  # L2归一化

        if return_feature:
            return embeddings

        if self.classifier:
            logits = self.classifier(embeddings)
            return logits, embeddings
        else:
            return embeddings


# ==================== 训练函数 ====================

def train_epoch(model, train_loader, optimizer, triplet_loss, device):
    """修复版训练函数"""
    model.train()
    total_loss = 0
    total_valid_triplets = 0
    total_batches = 0

    pbar = tqdm(train_loader, desc='Training')

    for batch_idx, (images, labels) in enumerate(pbar):
        # 🔧 修复：确保batch中有足够的类别
        unique_labels = torch.unique(labels)
        if len(unique_labels) < 2:
            continue  # 跳过类别不足的batch

        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        embeddings = model(images, return_feature=True)

        # 计算损失
        loss, num_valid = triplet_loss(embeddings, labels)

        # 只有有效损失才进行反向传播
        if loss > 0:
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # 统计
        total_loss += loss.item()
        total_valid_triplets += num_valid
        total_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'valid_triplets': num_valid,
            'batch': f'{batch_idx+1}/{len(train_loader)}'
        })

    avg_loss = total_loss / total_batches if total_batches > 0 else 0
    return avg_loss, total_valid_triplets


def evaluate(model, val_loader, device):
    """评估模型性能"""
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Evaluating'):
            images = images.to(device)
            embeddings = model(images, return_feature=True)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)

    if len(all_embeddings) == 0:
        return {'rank1': 0.0, 'rank5': 0.0, 'rank10': 0.0}, 0.0

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # 计算CMC和mAP
    cmc, mAP = compute_cmc_map(all_embeddings, all_labels)
    return cmc, mAP


def compute_cmc_map(embeddings, labels, topk=20):
    """计算CMC和mAP指标[1](@ref)"""
    n = embeddings.size(0)

    if n == 0:
        return {'rank1': 0.0, 'rank5': 0.0, 'rank10': 0.0}, 0.0

    # 计算距离矩阵
    dist_mat = torch.cdist(embeddings, embeddings, p=2)

    ranks = []
    average_precisions = []

    for i in range(n):
        # 排除自己
        distances = dist_mat[i]
        distances[i] = float('inf')  # 将自己设为无穷大

        # 排序
        indices = torch.argsort(distances)
        sorted_labels = labels[indices]

        # 找到正样本的位置
        matches = (sorted_labels == labels[i])
        match_indices = matches.nonzero(as_tuple=True)[0]

        if len(match_indices) == 0:
            continue

        # 第一个正样本的rank
        first_match_rank = match_indices[0].item() + 1
        ranks.append(first_match_rank)

        # 计算AP
        num_relevant = len(match_indices)
        cumsum = matches.cumsum(dim=0).float()
        precision_at_k = cumsum / torch.arange(1, len(matches) + 1, dtype=torch.float32)
        ap = (precision_at_k[matches] * matches[matches]).sum() / num_relevant
        average_precisions.append(ap.item())

    if len(ranks) == 0:
        return {'rank1': 0.0, 'rank5': 0.0, 'rank10': 0.0}, 0.0

    ranks = torch.tensor(ranks)
    cmc = {
        'rank1': (ranks <= 1).float().mean().item(),
        'rank5': (ranks <= 5).float().mean().item(),
        'rank10': (ranks <= 10).float().mean().item(),
    }

    mAP = np.mean(average_precisions) if average_precisions else 0.0
    return cmc, mAP


# ==================== 主训练流程 ====================

def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description='训练羊种ReID模型')
    parser.add_argument('--data_root', type=str, default='/home/ultralytics/ultralytics/trackers/data2', help='数据集路径')
    parser.add_argument('--batch_size', type=int, default=16, help='批大小')  # 减小batch_size避免内存问题
    parser.add_argument('--num_epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--margin', type=float, default=0.3, help='Triplet Loss边界')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='权重衰减')
    parser.add_argument('--save_path', type=str, default='/home/ultralytics/ultralytics/trackers/sheep_reid_128d2.pth', help='模型保存路径')
    parser.add_argument('--device', type=str, default='cuda', help='设备: cuda 或 cpu')

    args = parser.parse_args()

    print("=" * 70)
    print("羊种特化ReID模型训练（修复版）".center(70))
    print("=" * 70)
    print(f"配置:")
    print(f"  数据路径: {args.data_root}")
    print(f"  批大小: {args.batch_size}")
    print(f"  训练轮数: {args.num_epochs}")
    print(f"  学习率: {args.lr}")
    print(f"  Triplet Margin: {args.margin}")
    print(f"  设备: {args.device}")
    print()

    # 设备设置
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    try:
        # 数据集加载
        train_dataset = SheepReIDDataset(args.data_root, split='train')
        val_dataset = SheepReIDDataset(args.data_root, split='val')

        if train_dataset.num_classes < 2:
            print("❌ 错误：训练集至少需要2个类别（2只羊）")
            return

        # 数据加载器（关键修复：drop_last=True）
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,  # 减少workers避免内存问题
            pin_memory=True,
            drop_last=True  # 🔧 修复：丢弃不完整的批次
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        # 模型初始化
        model = LightweightReIDModel(embedding_dim=128, num_classes=train_dataset.num_classes).to(device)

        # 损失函数
        triplet_loss = FixedTripletLoss(margin=args.margin, hard_mining=True)

        # 优化器
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=20,
            gamma=0.1
        )

        # 训练循环
        best_rank1 = 0.0
        history = {'train_loss': [], 'rank1': [], 'mAP': []}

        print("\n开始训练...")
        for epoch in range(args.num_epochs):
            print(f"\n{'=' * 70}")
            print(f"Epoch {epoch + 1}/{args.num_epochs}")
            print(f"{'=' * 70}")

            # 训练
            train_loss, valid_triplets = train_epoch(
                model, train_loader, optimizer, triplet_loss, device
            )

            print(f"训练损失: {train_loss:.4f}, 有效三元组: {valid_triplets}")

            # 验证
            cmc, mAP = evaluate(model, val_loader, device)

            print(f"CMC - Rank1: {cmc['rank1']:.4f}, Rank5: {cmc['rank5']:.4f}, Rank10: {cmc['rank10']:.4f}")
            print(f"mAP: {mAP:.4f}")

            # 学习率调整
            scheduler.step()

            # 保存历史
            history['train_loss'].append(train_loss)
            history['rank1'].append(cmc['rank1'])
            history['mAP'].append(mAP)

            # 保存最佳模型
            if cmc['rank1'] > best_rank1:
                best_rank1 = cmc['rank1']

                # 确保保存目录存在
                Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'rank1': cmc['rank1'],
                    'mAP': mAP,
                    'history': history,
                    'config': vars(args)
                }, args.save_path)

                print(f"✅ 保存最佳模型: {args.save_path} (Rank1={best_rank1:.4f})")

        print("\n" + "=" * 70)
        print(f"训练完成！最佳 Rank1: {best_rank1:.4f}")
        print("=" * 70)

    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()