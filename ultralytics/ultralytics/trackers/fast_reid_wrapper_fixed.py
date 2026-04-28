"""
FastReID 包装类 - 调试增强版
专门解决 0% 成功率问题

关键改进：
1. 详细的调试日志
2. 边界框验证
3. 图像格式检查
4. 错误追踪
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from pathlib import Path
import warnings
import traceback


class FastReIDWrapper:
    """FastReID模型包装器 - 调试增强版"""

    def __init__(self, model_path=None, device='cuda', fp16=True, verbose=True):
        """初始化FastReID模型"""
        self.verbose = verbose
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.fp16 = fp16 and device == 'cuda'
        self.model = None
        self.initialized = False

        # 统计信息（扩展）
        self.stats = {
            'total_calls': 0,
            'success_calls': 0,
            'failed_calls': 0,
            'total_boxes': 0,
            'valid_boxes': 0,
            'invalid_boxes': 0,
            'too_small_boxes': 0,
            'out_of_bounds_boxes': 0,
            'crop_failures': 0,
            'transform_failures': 0,
            'inference_failures': 0
        }

        # 尝试加载模型
        try:
            if model_path and Path(model_path).exists():
                self.model = self._load_custom_model(model_path)
                self._log(f"✅ 成功加载自定义模型: {model_path}")
            else:
                self.model = self._create_lightweight_model()
                self._log("⚠️  使用默认轻量级模型（未训练）")

            self.model = self.model.to(self.device)
            self.model.eval()

            if self.fp16:
                self.model = self.model.half()

            # 图像预处理
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            self.initialized = True
            self._log(f"✅ ReID 初始化成功 (device={self.device}, fp16={self.fp16})")

        except Exception as e:
            self._log(f"❌ ReID 初始化失败: {e}")
            traceback.print_exc()
            self.initialized = False

    def _log(self, message):
        """日志输出"""
        if self.verbose:
            print(f"[FastReID] {message}")

    def _create_lightweight_model(self):
        """创建轻量级ReID模型"""
        return LightweightReIDModel(embedding_dim=128)

    def _load_custom_model(self, model_path):
        """加载自定义训练的模型"""
        model = LightweightReIDModel(embedding_dim=128)
        checkpoint = torch.load(model_path, map_location='cpu')

        # 获取state_dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # 移除classifier层（只需要embedding）
        state_dict = {k: v for k, v in state_dict.items()
                      if not k.startswith('classifier.')}

        # 使用strict=False允许部分加载
        model.load_state_dict(state_dict, strict=False)

        return model

    def _parse_boxes(self, boxes):
        """
        解析边界框 - 增强调试版

        返回: (boxes_xyxy, debug_info)
        """
        debug_info = {
            'input_type': type(boxes).__name__,
            'input_shape': None,
            'parsed_count': 0,
            'errors': []
        }

        try:
            # 1. 转为numpy array
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.cpu().numpy()
                debug_info['input_type'] = 'torch.Tensor'
            elif isinstance(boxes, list):
                boxes = np.array(boxes)
                debug_info['input_type'] = 'list'
            elif not isinstance(boxes, np.ndarray):
                boxes = np.asarray(boxes)

            debug_info['input_shape'] = boxes.shape

            # 2. 确保是2D数组
            if boxes.ndim == 1:
                boxes = boxes.reshape(1, -1)
                debug_info['reshaped'] = True

            # 3. 只取前4列（x1, y1, x2, y2）
            if boxes.shape[1] >= 4:
                boxes = boxes[:, :4].astype(np.float32)
                debug_info['parsed_count'] = len(boxes)
            else:
                debug_info['errors'].append(f"列数不足: {boxes.shape[1]} < 4")
                return np.empty((0, 4), dtype=np.float32), debug_info

            return boxes, debug_info

        except Exception as e:
            debug_info['errors'].append(str(e))
            return np.empty((0, 4), dtype=np.float32), debug_info

    @torch.no_grad()
    def extract_features(self, img, boxes):
        """
        提取ReID特征 - 调试增强版

        Args:
            img: BGR图像 [H, W, 3]
            boxes: 边界框 [N, 4+]

        Returns:
            features: [N, 128]
        """
        self.stats['total_calls'] += 1

        # 检查初始化状态
        if not self.initialized:
            self._log("❌ 模型未初始化")
            self.stats['failed_calls'] += 1
            return self._get_zero_features(len(boxes) if hasattr(boxes, '__len__') else 0)

        # 检查图像
        if img is None:
            self._log("❌ 输入图像为 None")
            self.stats['failed_calls'] += 1
            return np.empty((0, 128), dtype=np.float32)

        try:
            # 图像信息
            img_h, img_w = img.shape[:2]

            # 在第一次调用时打印详细信息
            if self.stats['total_calls'] == 1:
                self._log(f"图像尺寸: {img_w}x{img_h}, 类型: {img.dtype}")
                self._log(f"边界框输入类型: {type(boxes)}")
                if hasattr(boxes, 'shape'):
                    self._log(f"边界框形状: {boxes.shape}")

            # 解析边界框
            boxes_xyxy, debug_info = self._parse_boxes(boxes)
            self.stats['total_boxes'] += len(boxes_xyxy)

            # 调试信息（仅前5次）
            if self.stats['total_calls'] <= 5:
                self._log(f"解析结果: {debug_info}")
                if len(boxes_xyxy) > 0:
                    self._log(f"前3个框: {boxes_xyxy[:3]}")

            if len(boxes_xyxy) == 0:
                self._log(f"⚠️  没有有效边界框")
                self.stats['failed_calls'] += 1
                return np.empty((0, 128), dtype=np.float32)

            # 裁剪和预处理
            crops = []
            valid_indices = []

            for idx, box in enumerate(boxes_xyxy):
                try:
                    x1, y1, x2, y2 = map(int, box)

                    # 边界检查
                    original_box = (x1, y1, x2, y2)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(img_w, x2), min(img_h, y2)

                    # 调试：检查边界修正
                    if idx < 3 and self.stats['total_calls'] <= 3:
                        self._log(f"框{idx}: 原始{original_box} -> 修正({x1},{y1},{x2},{y2})")

                    # 有效性检查
                    if x2 <= x1 or y2 <= y1:
                        self.stats['invalid_boxes'] += 1
                        if idx < 3 and self.stats['total_calls'] <= 3:
                            self._log(f"  ❌ 无效框: w={x2 - x1}, h={y2 - y1}")
                        continue

                    # 尺寸检查
                    box_w, box_h = x2 - x1, y2 - y1
                    if box_w < 16 or box_h < 16:
                        self.stats['too_small_boxes'] += 1
                        if idx < 3 and self.stats['total_calls'] <= 3:
                            self._log(f"  ❌ 框太小: {box_w}x{box_h}")
                        continue

                    # 裁剪
                    crop = img[y1:y2, x1:x2]

                    # 检查裁剪结果
                    if crop.size == 0:
                        self.stats['crop_failures'] += 1
                        if idx < 3 and self.stats['total_calls'] <= 3:
                            self._log(f"  ❌ 裁剪失败: 空图像")
                        continue

                    # BGR -> RGB
                    crop_rgb = crop[:, :, ::-1].copy()

                    # 预处理
                    try:
                        crop_tensor = self.transform(crop_rgb)
                        crops.append(crop_tensor)
                        valid_indices.append(idx)

                        if idx < 3 and self.stats['total_calls'] <= 3:
                            self._log(f"  ✅ 框{idx}处理成功: {box_w}x{box_h}")

                    except Exception as e:
                        self.stats['transform_failures'] += 1
                        if idx < 3 and self.stats['total_calls'] <= 3:
                            self._log(f"  ❌ 变换失败: {e}")
                        continue

                except Exception as e:
                    if idx < 3 and self.stats['total_calls'] <= 3:
                        self._log(f"  ❌ 处理框{idx}失败: {e}")
                    continue

            # 检查是否有有效框
            if len(crops) == 0:
                self._log(f"⚠️  没有成功处理的框 (总共{len(boxes_xyxy)}个)")
                self.stats['failed_calls'] += 1
                return self._get_zero_features(len(boxes_xyxy))

            # 批量推理
            try:
                batch = torch.stack(crops).to(self.device)
                if self.fp16:
                    batch = batch.half()

                # 提取特征
                features = self.model(batch, return_feature=True)

                # L2归一化
                features = features / (features.norm(dim=1, keepdim=True) + 1e-12)
                features_np = features.cpu().float().numpy()

                # 构造完整结果
                result = np.zeros((len(boxes_xyxy), 128), dtype=np.float32)
                for i, valid_idx in enumerate(valid_indices):
                    result[valid_idx] = features_np[i]

                self.stats['success_calls'] += 1
                self.stats['valid_boxes'] += len(valid_indices)

                if self.stats['total_calls'] <= 3:
                    self._log(f"✅ 成功提取 {len(valid_indices)}/{len(boxes_xyxy)} 个特征")

                return result

            except Exception as e:
                self.stats['inference_failures'] += 1
                self._log(f"❌ 推理失败: {e}")
                traceback.print_exc()
                self.stats['failed_calls'] += 1
                return self._get_zero_features(len(boxes_xyxy))

        except Exception as e:
            self._log(f"❌ 特征提取失败: {e}")
            traceback.print_exc()
            self.stats['failed_calls'] += 1
            return self._get_zero_features(len(boxes) if hasattr(boxes, '__len__') else 0)

    def _get_zero_features(self, n):
        """返回零向量作为降级方案"""
        return np.zeros((n, 128), dtype=np.float32)

    def inference(self, img, boxes):
        """兼容原始 BoT-SORT 接口"""
        return self.extract_features(img, boxes)

    def compute_distance(self, feat1, feat2):
        """计算余弦距离"""
        try:
            if feat1.ndim == 1:
                feat1 = feat1[np.newaxis, :]
            if feat2.ndim == 1:
                feat2 = feat2[np.newaxis, :]

            similarity = np.dot(feat1, feat2.T)
            distance = 1.0 - similarity

            if distance.shape == (1, 1):
                return distance[0, 0]

            return distance
        except Exception as e:
            self._log(f"⚠️  距离计算失败: {e}")
            return 1.0

    def print_stats(self):
        """打印详细统计信息"""
        print("\n" + "=" * 70)
        print("ReID 详细运行统计".center(70))
        print("=" * 70)

        # 基础统计
        print(f"\n📊 调用统计:")
        print(f"  总调用次数: {self.stats['total_calls']}")
        print(f"  成功调用: {self.stats['success_calls']}")
        print(f"  失败调用: {self.stats['failed_calls']}")

        if self.stats['total_calls'] > 0:
            success_rate = self.stats['success_calls'] / self.stats['total_calls'] * 100
            print(f"  成功率: {success_rate:.2f}%")

        # 边界框统计
        print(f"\n📦 边界框统计:")
        print(f"  总边界框: {self.stats['total_boxes']}")
        print(f"  有效边界框: {self.stats['valid_boxes']}")
        print(f"  无效边界框: {self.stats['invalid_boxes']}")
        print(f"  过小边界框: {self.stats['too_small_boxes']}")
        print(f"  越界边界框: {self.stats['out_of_bounds_boxes']}")

        if self.stats['total_boxes'] > 0:
            valid_rate = self.stats['valid_boxes'] / self.stats['total_boxes'] * 100
            print(f"  有效率: {valid_rate:.2f}%")

        # 失败详情
        print(f"\n❌ 失败详情:")
        print(f"  裁剪失败: {self.stats['crop_failures']}")
        print(f"  变换失败: {self.stats['transform_failures']}")
        print(f"  推理失败: {self.stats['inference_failures']}")

        print("=" * 70 + "\n")


class LightweightReIDModel(nn.Module):
    """轻量级ReID模型"""

    def __init__(self, embedding_dim=128):
        super().__init__()

        import torchvision.models as models

        try:
            backbone = models.resnet18(pretrained=True)
        except TypeError:
            from torchvision.models import ResNet18_Weights
            backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        self.embedding = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, x, return_feature=False):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        embeddings = self.embedding(features)
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)

        if return_feature:
            return embeddings
        return embeddings