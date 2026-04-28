# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import LOGGER
from .checks import check_version
from .metrics import bbox_iou, probiou
from .ops import xywhr2xyxyxyxy

TORCH_1_10 = check_version(torch.__version__, "1.10.0")


class TaskAlignedAssigner(nn.Module):
    """
    A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric, which combines both
    classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        num_classes (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        eps (float): A small value to prevent division by zero.
    """

    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
        """Initialize a TaskAlignedAssigner object with customizable hyperparameters."""
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Compute the task-aligned assignment. Reference code is available at
        https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py.

        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            anc_points (Tensor): shape(num_total_anchors, 2)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)

        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
            target_gt_idx (Tensor): shape(bs, num_total_anchors)
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        device = gt_bboxes.device

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.bg_idx),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )

        try:
            return self._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        except torch.OutOfMemoryError:
            # Move tensors to CPU, compute, then move back to original device
            LOGGER.warning("WARNING: CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU")
            cpu_tensors = [t.cpu() for t in (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)]
            result = self._forward(*cpu_tensors)
            return tuple(t.to(device) for t in result)

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Compute the task-aligned assignment. Reference code is available at
        https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py.

        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            anc_points (Tensor): shape(num_total_anchors, 2)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)

        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
            target_gt_idx (Tensor): shape(bs, num_total_anchors)
        """
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        # Assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # Normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """Get in_gts mask, (b, max_num_obj, h*w)."""
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """Compute alignment metric given predicted and ground truth bounding boxes."""
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # Get the scores of each grid for each gt cls
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """IoU calculation for horizontal bounding boxes."""
        return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        """
        Select the top-k candidates based on the given metrics.

        Args:
            metrics (Tensor): A tensor of shape (b, max_num_obj, h*w), where b is the batch size,
                              max_num_obj is the maximum number of objects, and h*w represents the
                              total number of anchor points.
            largest (bool): If True, select the largest values; otherwise, select the smallest values.
            topk_mask (Tensor): An optional boolean tensor of shape (b, max_num_obj, topk), where
                                topk is the number of top candidates to consider. If not provided,
                                the top-k values are automatically computed based on the given metrics.

        Returns:
            (Tensor): A tensor of shape (b, max_num_obj, h*w) containing the selected top-k candidates.
        """
        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        # (b, max_num_obj, topk)
        topk_idxs.masked_fill_(~topk_mask, 0)

        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            # Expand topk_idxs for each value of k and add 1 at the specified positions
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k: k + 1], ones)
        # count_tensor.scatter_add_(-1, topk_idxs, torch.ones_like(topk_idxs, dtype=torch.int8, device=topk_idxs.device))
        # Filter invalid bboxes
        count_tensor.masked_fill_(count_tensor > 1, 0)

        return count_tensor.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        Compute target labels, target bounding boxes, and target scores for the positive anchor points.

        Args:
            gt_labels (Tensor): Ground truth labels of shape (b, max_num_obj, 1), where b is the
                                batch size and max_num_obj is the maximum number of objects.
            gt_bboxes (Tensor): Ground truth bounding boxes of shape (b, max_num_obj, 4).
            target_gt_idx (Tensor): Indices of the assigned ground truth objects for positive
                                    anchor points, with shape (b, h*w), where h*w is the total
                                    number of anchor points.
            fg_mask (Tensor): A boolean tensor of shape (b, h*w) indicating the positive
                              (foreground) anchor points.

        Returns:
            (Tuple[Tensor, Tensor, Tensor]): A tuple containing the following tensors:
                - target_labels (Tensor): Shape (b, h*w), containing the target labels for
                                          positive anchor points.
                - target_bboxes (Tensor): Shape (b, h*w, 4), containing the target bounding boxes
                                          for positive anchor points.
                - target_scores (Tensor): Shape (b, h*w, num_classes), containing the target scores
                                          for positive anchor points, where num_classes is the number
                                          of object classes.
        """
        # Assigned target labels, (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)

        # Assigned target boxes, (b, max_num_obj, 4) -> (b, h*w, 4)
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]

        # Assigned target scores
        target_labels.clamp_(0)

        # 10x faster than F.one_hot()
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int64,
            device=target_labels.device,
        )  # (b, h*w, 80)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
        """
        Select positive anchor centers within ground truth bounding boxes.

        Args:
            xy_centers (torch.Tensor): Anchor center coordinates, shape (h*w, 2).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes, shape (b, n_boxes, 4).
            eps (float, optional): Small value for numerical stability. Defaults to 1e-9.

        Returns:
            (torch.Tensor): Boolean mask of positive anchors, shape (b, n_boxes, h*w).

        Note:
            b: batch size, n_boxes: number of ground truth boxes, h: height, w: width.
            Bounding box format: [x_min, y_min, x_max, y_max].
        """
        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
        # return (bbox_deltas.min(3)[0] > eps).to(gt_bboxes.dtype)
        return bbox_deltas.amin(3).gt_(eps)

    @staticmethod
    def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
        """
        Select anchor boxes with highest IoU when assigned to multiple ground truths.

        Args:
            mask_pos (torch.Tensor): Positive mask, shape (b, n_max_boxes, h*w).
            overlaps (torch.Tensor): IoU overlaps, shape (b, n_max_boxes, h*w).
            n_max_boxes (int): Maximum number of ground truth boxes.

        Returns:
            target_gt_idx (torch.Tensor): Indices of assigned ground truths, shape (b, h*w).
            fg_mask (torch.Tensor): Foreground mask, shape (b, h*w).
            mask_pos (torch.Tensor): Updated positive mask, shape (b, n_max_boxes, h*w).

        Note:
            b: batch size, h: height, w: width.
        """
        # Convert (b, n_max_boxes, h*w) -> (b, h*w)
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # (b, n_max_boxes, h*w)
            max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)

            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)

            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # (b, n_max_boxes, h*w)
            fg_mask = mask_pos.sum(-2)
        # Find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
        return target_gt_idx, fg_mask, mask_pos


class SimOTAAssigner(nn.Module):
    """
    SimOTA样本分配器 (YOLOX风格)

    核心特点：
    1. 动态k值：根据IoU自适应确定每个GT分配的正样本数量
    2. OT最优传输：使用cost矩阵进行最优匹配
    3. 适合密集重叠场景：动态k能更好处理羊群这种高密度场景

    论文: YOLOX: Exceeding YOLO Series in 2021
    """

    def __init__(self, topk=10, num_classes=80, iou_weight=3.0, cls_weight=1.0, eps=1e-9):
        """
        初始化SimOTA分配器

        Args:
            topk: 候选正样本数量上限（用于计算动态k）
            num_classes: 类别数量
            iou_weight: IoU在cost中的权重
            cls_weight: 分类在cost中的权重
            eps: 数值稳定项
        """
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.iou_weight = iou_weight
        self.cls_weight = cls_weight
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        SimOTA样本分配主函数

        Args:
            pd_scores (Tensor): 预测分类分数 shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): 预测框 shape(bs, num_total_anchors, 4) xyxy格式
            anc_points (Tensor): anchor中心点 shape(num_total_anchors, 2)
            gt_labels (Tensor): GT类别 shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): GT框 shape(bs, n_max_boxes, 4) xyxy格式
            mask_gt (Tensor): 有效GT mask shape(bs, n_max_boxes, 1)

        Returns:
            target_labels: shape(bs, num_total_anchors)
            target_bboxes: shape(bs, num_total_anchors, 4)
            target_scores: shape(bs, num_total_anchors, num_classes)
            fg_mask: shape(bs, num_total_anchors)
            target_gt_idx: shape(bs, num_total_anchors)
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        device = gt_bboxes.device

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.bg_idx),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )

        # 初始化输出
        target_labels = torch.full_like(pd_scores[..., 0], self.bg_idx)
        target_bboxes = torch.zeros_like(pd_bboxes)
        target_scores = torch.zeros_like(pd_scores)
        fg_mask = torch.zeros_like(pd_scores[..., 0])
        target_gt_idx = torch.zeros_like(pd_scores[..., 0])

        # 批次循环
        for batch_idx in range(self.bs):
            num_gt = mask_gt[batch_idx].sum().int().item()
            if num_gt == 0:
                continue

            # 提取当前batch数据
            gt_bbox_i = gt_bboxes[batch_idx, :num_gt]  # [num_gt, 4]
            gt_label_i = gt_labels[batch_idx, :num_gt, 0]  # [num_gt]
            pred_score_i = pd_scores[batch_idx]  # [num_anchors, num_classes]
            pred_bbox_i = pd_bboxes[batch_idx]  # [num_anchors, 4]

            # ===== 1. 初步筛选：只考虑在GT框内的anchor =====
            is_in_boxes_all = self.select_candidates_in_gts(anc_points, gt_bbox_i.unsqueeze(0)).squeeze(
                0)  # [num_gt, num_anchors]
            is_in_boxes_all = is_in_boxes_all.bool()  # 关键修改：转换为布尔类型

            # ===== 2. 计算IoU矩阵 =====
            overlaps = self.bbox_iou_batch(pred_bbox_i, gt_bbox_i)  # [num_anchors, num_gt]

            # ===== 3. 计算分类cost =====
            # 获取每个GT对应类别的预测分数
            gt_labels_idx = gt_label_i.long().unsqueeze(0).expand(pred_score_i.shape[0], -1)  # [num_anchors, num_gt]
            pred_scores_pos = torch.gather(pred_score_i, 1, gt_labels_idx)  # [num_anchors, num_gt]

            # 使用BCE loss作为分类cost
            cls_cost = F.binary_cross_entropy_with_logits(
                pred_scores_pos,
                torch.ones_like(pred_scores_pos),
                reduction='none'
            )  # [num_anchors, num_gt]

            # ===== 4. 构建cost矩阵（SimOTA核心）=====
            # cost = cls_weight * cls_cost + iou_weight * (1 - iou) + 大惩罚(不在框内)
            iou_cost = 1.0 - overlaps  # IoU越大cost越小
            cost_matrix = (
                    self.cls_weight * cls_cost +
                    self.iou_weight * iou_cost
            )  # [num_anchors, num_gt]

            # 不在GT框内的anchor给予巨大惩罚
            cost_matrix = cost_matrix.transpose(0, 1)  # [num_gt, num_anchors]
            cost_matrix = torch.where(is_in_boxes_all, cost_matrix, cost_matrix + 100000.0)

            # ===== 5. 动态k值匹配（SimOTA关键创新）=====
            matched_pred_idx, matched_gt_idx, matched_ious = self.dynamic_k_matching(
                cost_matrix, overlaps.transpose(0, 1), num_gt
            )

            # ===== 6. 设置正样本 =====
            # 在 SimOTAAssigner 的 forward 方法中，修改正样本标签赋值部分
            if len(matched_pred_idx) > 0:
                fg_mask[batch_idx, matched_pred_idx] = 1.0
                # 关键修改：将标签转换为目标张量的 dtype（半精度）
                target_labels[batch_idx, matched_pred_idx] = gt_label_i[matched_gt_idx].to(target_labels.dtype)
                target_bboxes[batch_idx, matched_pred_idx] = gt_bbox_i[matched_gt_idx].to(target_bboxes.dtype)
                target_gt_idx[batch_idx, matched_pred_idx] = matched_gt_idx.float().to(target_gt_idx.dtype)

                # 对齐分数使用 IoU 时同样确保类型一致
                target_scores[batch_idx, matched_pred_idx, gt_label_i[matched_gt_idx].long()] = matched_ious.to(
                    target_scores.dtype)

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def dynamic_k_matching(self, cost_matrix, iou_matrix, num_gt):
        """
        SimOTA动态k值匹配 - 核心算法

        思想：
        1. 为每个GT根据topk个anchor的IoU和动态确定正样本数k
        2. k = max(1, int(sum(topk_ious)))  # 高质量匹配的GT分配更多正样本
        3. 选择cost最小的k个anchor作为该GT的正样本

        Args:
            cost_matrix: [num_gt, num_anchors] cost矩阵
            iou_matrix: [num_gt, num_anchors] IoU矩阵
            num_gt: GT数量

        Returns:
            matched_pred_idx: 匹配的anchor索引列表
            matched_gt_idx: 对应的GT索引列表
            matched_ious: 对应的IoU值列表
        """
        matched_pred_idx = []
        matched_gt_idx = []
        matched_ious = []

        # 记录每个anchor是否已被分配
        anchor_matched_gt = {}  # anchor_idx -> (gt_idx, cost, iou)

        for gt_idx in range(num_gt):
            # ===== 动态确定k值 =====
            # 选择IoU最高的topk个候选anchor
            _, topk_idxs = torch.topk(iou_matrix[gt_idx], k=min(self.topk, iou_matrix.shape[1]), largest=True)
            topk_ious = iou_matrix[gt_idx, topk_idxs]

            # 动态k：IoU和决定正样本数量（至少为1）
            dynamic_k = max(1, int(topk_ious.sum().item()))

            # ===== 选择cost最小的dynamic_k个anchor =====
            _, pos_idxs = torch.topk(cost_matrix[gt_idx], k=dynamic_k, largest=False)

            # ===== 处理一个anchor匹配多个GT的情况 =====
            for pos_idx in pos_idxs:
                pos_idx_item = pos_idx.item()
                current_cost = cost_matrix[gt_idx, pos_idx].item()
                current_iou = iou_matrix[gt_idx, pos_idx].item()

                # 如果anchor未被分配，或当前匹配cost更低
                if pos_idx_item not in anchor_matched_gt or current_cost < anchor_matched_gt[pos_idx_item][1]:
                    anchor_matched_gt[pos_idx_item] = (gt_idx, current_cost, current_iou)

        # 整理最终匹配结果
        for anchor_idx, (gt_idx, _, iou) in anchor_matched_gt.items():
            matched_pred_idx.append(anchor_idx)
            matched_gt_idx.append(gt_idx)
            matched_ious.append(iou)

        if len(matched_pred_idx) > 0:
            return (
                torch.tensor(matched_pred_idx, device=cost_matrix.device, dtype=torch.long),
                torch.tensor(matched_gt_idx, device=cost_matrix.device, dtype=torch.long),
                torch.tensor(matched_ious, device=cost_matrix.device, dtype=cost_matrix.dtype)
            )
        else:
            return (
                torch.tensor([], device=cost_matrix.device, dtype=torch.long),
                torch.tensor([], device=cost_matrix.device, dtype=torch.long),
                torch.tensor([], device=cost_matrix.device, dtype=cost_matrix.dtype)
            )

    @staticmethod
    def bbox_iou_batch(box1, box2):
        """
        计算IoU矩阵

        Args:
            box1: [N, 4] xyxy格式
            box2: [M, 4] xyxy格式

        Returns:
            iou: [N, M]
        """
        lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N, M, 2]
        rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N, M, 2]
        wh = (rb - lt).clamp(min=0)  # [N, M, 2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N]
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M]
        union = area1[:, None] + area2 - inter  # [N, M]

        iou = inter / (union + 1e-9)
        return iou

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
        """
        判断anchor中心点是否在GT框内

        Args:
            xy_centers: [num_anchors, 2] anchor中心坐标
            gt_bboxes: [bs, num_gt, 4] GT框 xyxy格式

        Returns:
            [bs, num_gt, num_anchors] bool矩阵
        """
        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape

        # 分解为左上和右下
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # [bs*n_boxes, 1, 2]

        # 计算anchor到边界的距离
        bbox_deltas = torch.cat(
            (xy_centers[None] - lt, rb - xy_centers[None]),
            dim=2
        ).view(bs, n_boxes, n_anchors, -1)  # [bs, n_boxes, n_anchors, 4]

        # 所有距离都>0说明在框内
        return bbox_deltas.amin(3).gt_(eps)  # [bs, n_boxes, n_anchors]


# ============================================================
# 对比工具函数 - 添加到 SimOTA_tal.py 末尾
# ============================================================

def compare_assigners(pred_scores, pred_bboxes, anchor_points,
                      gt_labels, gt_bboxes, mask_gt,
                      assigner_type='TaskAligned', **kwargs):
    """
    对比不同样本分配策略的工具函数

    Args:
        assigner_type: 'TaskAligned' 或 'SimOTA'
        **kwargs: 传递给assigner的参数

    Returns:
        统计信息字典
    """
    if assigner_type == 'TaskAligned':
        assigner = TaskAlignedAssigner(**kwargs)
    elif assigner_type == 'SimOTA':
        assigner = SimOTAAssigner(**kwargs)
    else:
        raise ValueError(f"Unknown assigner type: {assigner_type}")

    # 执行分配
    target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = assigner(
        pred_scores, pred_bboxes, anchor_points, gt_labels, gt_bboxes, mask_gt
    )

    # 统计指标
    num_gt = mask_gt.sum().item()
    num_pos = fg_mask.sum().item()
    avg_pos_per_gt = num_pos / num_gt if num_gt > 0 else 0

    # 分析重叠情况
    if num_gt > 1:
        overlaps = bbox_iou(
            gt_bboxes[mask_gt.bool()],
            gt_bboxes[mask_gt.bool()],
            xywh=False
        )
        # 排除自身，找到与其他GT的最大IoU
        overlaps.fill_diagonal_(0)
        max_overlap = overlaps.max().item()
        overlap_pairs = (overlaps > 0.5).sum().item() // 2  # 除以2因为对称矩阵
    else:
        max_overlap = 0
        overlap_pairs = 0

    stats = {
        'assigner': assigner_type,
        'num_gt': num_gt,
        'num_positive': num_pos,
        'avg_pos_per_gt': avg_pos_per_gt,
        'max_gt_overlap': max_overlap,
        'overlap_pairs': overlap_pairs,
        'target_scores_sum': target_scores.sum().item()
    }

    return stats, (target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx)


class RotatedTaskAlignedAssigner(TaskAlignedAssigner):
    """Assigns ground-truth objects to rotated bounding boxes using a task-aligned metric."""

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """IoU calculation for rotated bounding boxes."""
        return probiou(gt_bboxes, pd_bboxes).squeeze(-1).clamp_(0)

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes):
        """
        Select the positive anchor center in gt for rotated bounding boxes.

        Args:
            xy_centers (Tensor): shape(h*w, 2)
            gt_bboxes (Tensor): shape(b, n_boxes, 5)

        Returns:
            (Tensor): shape(b, n_boxes, h*w)
        """
        # (b, n_boxes, 5) --> (b, n_boxes, 4, 2)
        corners = xywhr2xyxyxyxy(gt_bboxes)
        # (b, n_boxes, 1, 2)
        a, b, _, d = corners.split(1, dim=-2)
        ab = b - a
        ad = d - a

        # (b, n_boxes, h*w, 2)
        ap = xy_centers - a
        norm_ab = (ab * ab).sum(dim=-1)
        norm_ad = (ad * ad).sum(dim=-1)
        ap_dot_ab = (ap * ab).sum(dim=-1)
        ap_dot_ad = (ap * ad).sum(dim=-1)
        return (ap_dot_ab >= 0) & (ap_dot_ab <= norm_ab) & (ap_dot_ad >= 0) & (ap_dot_ad <= norm_ad)  # is_in_box


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)  # dist (lt, rb)


def dist2rbox(pred_dist, pred_angle, anchor_points, dim=-1):
    """
    Decode predicted rotated bounding box coordinates from anchor points and distribution.

    Args:
        pred_dist (torch.Tensor): Predicted rotated distance, shape (bs, h*w, 4).
        pred_angle (torch.Tensor): Predicted angle, shape (bs, h*w, 1).
        anchor_points (torch.Tensor): Anchor points, shape (h*w, 2).
        dim (int, optional): Dimension along which to split. Defaults to -1.

    Returns:
        (torch.Tensor): Predicted rotated bounding boxes, shape (bs, h*w, 4).
    """
    lt, rb = pred_dist.split(2, dim=dim)
    cos, sin = torch.cos(pred_angle), torch.sin(pred_angle)
    # (bs, h*w, 1)
    xf, yf = ((rb - lt) / 2).split(1, dim=dim)
    x, y = xf * cos - yf * sin, xf * sin + yf * cos
    xy = torch.cat([x, y], dim=dim) + anchor_points
    return torch.cat([xy, lt + rb], dim=dim)
