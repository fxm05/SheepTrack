# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import deque

import numpy as np
import torch

from .basetrack import TrackState
from .byte_tracker import BYTETracker, STrack
from .utils import matching
from .utils.gmc import GMC
from .utils.kalman_filter import KalmanFilterXYWH


class BOTrack(STrack):
    """
    融合FAM和TCM的扩展BOTrack类。

    特性：
        - ReID特征提取和平滑（FAM）
        - 观测历史记录和校正（TCM）
        - 加速度异常检测（TCM）
        - 长遮挡处理（TCM）
    """

    shared_kalman = KalmanFilterXYWH()

    def __init__(self, tlwh, score, cls, feat=None, feat_history=50):
        """初始化融合了FAM和TCM功能的BOTrack对象。"""
        # 坐标格式转换和验证
        if isinstance(tlwh, (list, tuple)):
            tlwh = np.array(tlwh, dtype=np.float32)
        elif isinstance(tlwh, torch.Tensor):
            tlwh = tlwh.cpu().numpy().astype(np.float32)

        if len(tlwh) < 4:
            raise ValueError(f"Invalid box: expected >=4 values, got {len(tlwh)}")

        box_4 = tlwh[:4]

        # 检测格式：xyxy vs tlwh
        is_xyxy = (box_4[2] > box_4[0] and box_4[3] > box_4[1] and
                   (box_4[2] - box_4[0]) > 5 and (box_4[3] - box_4[1]) > 5)

        if is_xyxy:
            x1, y1, x2, y2 = box_4
            tlwh_converted = np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)
        else:
            tlwh_converted = box_4.astype(np.float32)

        xywh_score = np.concatenate([tlwh_converted, [score]])
        super().__init__(xywh_score, score, cls)

        # ========== FAM: ReID特征管理 ==========
        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None and len(feat) > 0:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9  # 特征平滑系数

        # ========== TCM: 观测历史和校正 ==========
        self.observation_history = deque([], maxlen=50)
        self.velocity_history = deque([], maxlen=30)
        self.acceleration_history = deque([], maxlen=20)

        # TCM阈值参数
        self.max_occlusion_frames = 30
        self.acceleration_threshold = 15.0
        self.observation_correction_window = 10

        # TCM状态标记
        self.occlusion_count = 0
        self.need_correction = False
        self.last_observation = None

    def update_features(self, feat):
        """【FAM】更新ReID特征向量，使用指数移动平均平滑。"""
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat

        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat

        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def calculate_velocity(self, current_pos, previous_pos):
        """【TCM】计算速度向量。"""
        if previous_pos is None:
            return np.zeros(2)
        return current_pos[:2] - previous_pos[:2]

    def calculate_acceleration(self, current_velocity, previous_velocity):
        """【TCM】计算加速度向量。"""
        if previous_velocity is None or len(previous_velocity) == 0:
            return np.zeros(2)
        return current_velocity - previous_velocity

    def detect_acceleration_anomaly(self):
        """【TCM】检测加速度异常。"""
        if len(self.acceleration_history) < 3:
            return False

        recent_acc = np.array(list(self.acceleration_history)[-3:])
        acc_magnitude = np.linalg.norm(recent_acc, axis=1)

        # 检测加速度突然增大
        if np.max(acc_magnitude) > self.acceleration_threshold:
            return True

        # 检测加速度方向突变
        if len(self.acceleration_history) >= 2:
            acc_diff = np.linalg.norm(recent_acc[-1] - recent_acc[-2])
            if acc_diff > self.acceleration_threshold * 0.8:
                return True

        return False

    def observation_based_correction(self):
        """【TCM】基于观测历史的轨迹校正。"""
        if len(self.observation_history) < self.observation_correction_window:
            return

        recent_obs = list(self.observation_history)[-self.observation_correction_window:]
        positions = np.array([obs[:2] for obs in recent_obs])
        time_steps = np.arange(len(positions))

        if len(positions) >= 3:
            # 使用多项式拟合捕捉非线性运动
            poly_x = np.polyfit(time_steps, positions[:, 0], min(2, len(positions) - 1))
            poly_y = np.polyfit(time_steps, positions[:, 1], min(2, len(positions) - 1))

            # 预测当前位置
            predicted_x = np.polyval(poly_x, len(positions))
            predicted_y = np.polyval(poly_y, len(positions))

            # 校正mean状态
            self.mean[0] = predicted_x
            self.mean[1] = predicted_y

            # 重新计算速度
            if len(positions) >= 2:
                velocity_x = positions[-1, 0] - positions[-2, 0]
                velocity_y = positions[-1, 1] - positions[-2, 1]
                self.mean[4] = velocity_x
                self.mean[5] = velocity_y

            # 增加协方差反映不确定性
            self.covariance[:2, :2] *= 1.5

    def predict(self):
        """【TCM】预测未来状态，集成观测校正机制。"""
        mean_state = self.mean.copy()

        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0
            self.occlusion_count += 1
        else:
            self.occlusion_count = 0

        # 检测是否需要观测校正
        is_long_occlusion = self.occlusion_count > self.max_occlusion_frames
        is_acceleration_anomaly = self.detect_acceleration_anomaly()

        if is_long_occlusion or is_acceleration_anomaly:
            self.need_correction = True

        # 卡尔曼滤波预测
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

        # 应用观测校正
        if self.need_correction:
            self.observation_based_correction()
            self.need_correction = False

    def update(self, new_track, frame_id):
        """【FAM+TCM】更新轨迹：更新特征和观测历史。"""
        # FAM: 更新ReID特征
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        # TCM: 记录观测历史
        current_pos = new_track.tlwh_to_xywh(new_track.tlwh)
        self.observation_history.append(current_pos.copy())

        # TCM: 计算速度
        if self.last_observation is not None:
            velocity = self.calculate_velocity(current_pos, self.last_observation)
            self.velocity_history.append(velocity)

            # TCM: 计算加速度
            if len(self.velocity_history) >= 2:
                acceleration = self.calculate_acceleration(
                    self.velocity_history[-1],
                    self.velocity_history[-2]
                )
                self.acceleration_history.append(acceleration)

        self.last_observation = current_pos

        # 调用父类update
        super().update(new_track, frame_id)
        self.occlusion_count = 0

    def re_activate(self, new_track, frame_id, new_id=False):
        """【FAM+TCM】重新激活轨迹：更新特征和观测。"""
        # FAM: 更新特征
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        # TCM: 记录观测
        current_pos = new_track.tlwh_to_xywh(new_track.tlwh)
        self.observation_history.append(current_pos.copy())
        self.last_observation = current_pos

        super().re_activate(new_track, frame_id, new_id)

        # 重置状态
        self.occlusion_count = 0
        self.need_correction = False

    @property
    def tlwh(self):
        """返回当前边界框（tlwh格式）。"""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @staticmethod
    def multi_predict(stracks):
        """批量预测多个轨迹的状态。"""
        if len(stracks) <= 0:
            return

        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])

        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][6] = 0
                multi_mean[i][7] = 0

        multi_mean, multi_covariance = BOTrack.shared_kalman.multi_predict(
            multi_mean, multi_covariance
        )

        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """转换边界框格式：tlwh -> xywh。"""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret


class BOTSORT(BYTETracker):
    """
    融合FAM和TCM的BOTSORT跟踪器。

    功能：
        - 【FAM】ReID外观特征匹配
        - 【TCM】观测校正和异常检测
        - 全局运动补偿（GMC）
        - 两阶段级联匹配
    """

    def __init__(self, args, frame_rate=30):
        """初始化融合跟踪器。"""
        super().__init__(args, frame_rate)

        # 匹配阈值
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh

        # 【FAM】初始化ReID编码器
        if args.with_reid:
            try:
                from .fast_reid_wrapper_fixed import FastReIDWrapper

                reid_model_path = getattr(args, 'reid_model_path', 'sheep_reid_128d.pth')
                self.encoder = FastReIDWrapper(
                    model_path=reid_model_path,
                    device='cuda',
                    fp16=True,
                    verbose=False
                )
                print(f"✅ ReID encoder initialized: {reid_model_path}")
            except Exception as e:
                print(f"⚠️ ReID encoder failed: {e}")
                print("   Using pure IoU tracking")
                self.encoder = None
        else:
            self.encoder = None

        # GMC和调试标记
        self.gmc = GMC(method=args.gmc_method)
        self._debug_printed = False
        self._frame_count = 0

    def get_kalmanfilter(self):
        """返回卡尔曼滤波器实例。"""
        return KalmanFilterXYWH()

    def init_track(self, dets, scores, cls, img=None):
        """【FAM】初始化轨迹，提取ReID特征。"""
        self._frame_count += 1

        if len(dets) == 0:
            return []

        # 转换为numpy
        if isinstance(dets, torch.Tensor):
            dets = dets.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if isinstance(cls, torch.Tensor):
            cls = cls.cpu().numpy()

        scores = np.asarray(scores).flatten()
        cls = np.asarray(cls).flatten()

        # 取前4列
        if dets.ndim == 2 and dets.shape[1] > 4:
            dets_4col = dets[:, :4].copy()
        else:
            dets_4col = dets.copy()

        # 格式检测和转换（用于ReID）
        if len(dets_4col) > 0:
            sample_box = dets_4col[0]
            is_xyxy = (sample_box[2] > sample_box[0] and
                       sample_box[3] > sample_box[1] and
                       (sample_box[2] - sample_box[0]) > 5)

            if is_xyxy:
                dets_for_reid = dets_4col
                format_name = "xyxy"
            else:
                x, y, w, h = dets_4col[:, 0], dets_4col[:, 1], dets_4col[:, 2], dets_4col[:, 3]
                dets_for_reid = np.stack([x, y, x + w, y + h], axis=1)
                format_name = "tlwh"
        else:
            dets_for_reid = dets_4col
            format_name = "unknown"

        # 调试输出（前3帧）
        if self._frame_count <= 3:
            print(f"\n🔍 [Frame {self._frame_count}] init_track")
            print(f"  Boxes: {len(dets_4col)} ({format_name})")
            print(f"  First 3 boxes: {dets_4col[:3]}")
            if format_name == "tlwh":
                print(f"  Converted: {dets_for_reid[:3]}")
            print(f"  Scores: [{scores.min():.3f}, {scores.max():.3f}]")
            if img is not None:
                print(f"  Image: {img.shape}")

        # 【FAM】提取ReID特征
        features = None
        if self.args.with_reid and self.encoder is not None and img is not None:
            try:
                features = self.encoder.inference(img, dets_for_reid)

                if self._frame_count <= 3:
                    print(f"  ✅ ReID features: {features.shape}")
                    non_zero = np.count_nonzero(features.sum(axis=1))
                    print(f"     Non-zero: {non_zero}/{len(features)}")
                    for i in range(min(3, len(features))):
                        norm = np.linalg.norm(features[i])
                        print(f"     Feature {i}: L2={norm:.4f}")

                if len(features) != len(dets_4col):
                    if not self._debug_printed:
                        print(f"⚠️ Feature count mismatch")
                        self._debug_printed = True
                    features = None

            except Exception as e:
                if not self._debug_printed:
                    print(f"⚠️ ReID failed: {e}")
                    import traceback
                    traceback.print_exc()
                    self._debug_printed = True
                features = None

        # 创建BOTrack对象
        tracks = []
        for i, (box, score, c) in enumerate(zip(dets_4col, scores, cls)):
            feat = None
            if features is not None:
                feat = features[i]
                if np.linalg.norm(feat) < 1e-6:
                    feat = None

            try:
                track = BOTrack(
                    tlwh=box,
                    score=float(score),
                    cls=int(c),
                    feat=feat
                )
                tracks.append(track)
            except Exception as e:
                if self._frame_count <= 3 and i < 3:
                    print(f"  ❌ Track {i} failed: {e}")
                continue

        if self._frame_count <= 3:
            tracks_with_feat = sum(1 for t in tracks if t.curr_feat is not None)
            print(f"  ✅ Created {len(tracks)}/{len(dets_4col)} tracks")
            print(f"  📊 With features: {tracks_with_feat}/{len(tracks)}\n")

        return tracks

    def get_dists(self, tracks, detections):
        """【FAM】计算距离矩阵：IoU + ReID特征融合。"""
        dists = matching.iou_distance(tracks, detections)
        dists_mask = dists > self.proximity_thresh

        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)

        # 【FAM】ReID外观匹配
        if self.args.with_reid and self.encoder is not None:
            try:
                tracks_valid = sum(
                    1 for t in tracks
                    if hasattr(t, 'curr_feat') and t.curr_feat is not None
                    and np.linalg.norm(t.curr_feat) > 1e-6
                )

                dets_valid = sum(
                    1 for d in detections
                    if hasattr(d, 'curr_feat') and d.curr_feat is not None
                    and np.linalg.norm(d.curr_feat) > 1e-6
                )

                if tracks_valid > 0 and dets_valid > 0:
                    emb_dists = matching.embedding_distance(tracks, detections) / 2.0
                    iou_only = dists.copy()

                    emb_dists[emb_dists > self.appearance_thresh] = 1.0
                    emb_dists[dists_mask] = 1.0
                    dists = np.minimum(dists, emb_dists)

                    # 统计ReID改善效果
                    changed = np.sum(dists < iou_only)
                    if changed > 0:
                        print(f"🔍 Frame: ReID improved {changed} matches")

            except Exception:
                pass

        return dists

    def multi_predict(self, tracks):
        """【TCM】批量预测轨迹状态。"""
        BOTrack.multi_predict(tracks)

    def reset(self):
        """重置跟踪器状态。"""
        super().reset()
        self.gmc.reset_params()
        self._debug_printed = False
        self._frame_count = 0

        if self.encoder is not None and hasattr(self.encoder, 'print_stats'):
            print("\n" + "=" * 60)
            print("Tracking finished - ReID stats")
            print("=" * 60)
            self.encoder.print_stats()